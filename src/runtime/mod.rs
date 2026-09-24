//! The turn loop.
//!
//! Each turn: load the instruction fragments whose conditions hold, decide
//! whether the previous prefix is worth keeping, assemble a fresh context from
//! the chunk store, classify sensitivity, route to a model, call it, and act
//! on its reply. Acting means: Jev routes the intent to a tool, the policy
//! layer judges the call, leases protect written paths, and the result lands
//! in the store as a chunk for the next assembly to score.

pub mod background;
pub mod events;
pub mod instructions;
pub mod subgoals;

use crate::config::Config;
use crate::context::{AssembleInput, Assembler, Summarizer, TruncateSummarizer};
use crate::jev::{questions, Jev};
use crate::llm::{self, LlmClient, LlmUsage, Prompt, Step};
use crate::policy::{Policy, Verdict};
use crate::router::Router;
use crate::state::{tokens, Access, Chunk, ChunkId, ChunkStore, Kind, Sensitivity, Snapshot};
use crate::tools::{ArgBuilder, Delegator, HeuristicArgBuilder, ToolCall, ToolCx, ToolOutput, ToolRegistry};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::{watch, Mutex};

pub use events::{emit, print_plain, ContextRow, Event, EventSink, UsageSnapshot};
pub use instructions::Instructions;

/// Who answers when the policy says `ask`.
#[async_trait]
pub trait Permissioner: Send + Sync {
    async fn confirm(&self, call: &str, reasons: &[String]) -> bool;
}

/// Always yes or always no, for unattended runs.
pub struct AutoPermissioner(pub bool);

#[async_trait]
impl Permissioner for AutoPermissioner {
    async fn confirm(&self, _call: &str, _reasons: &[String]) -> bool {
        self.0
    }
}

/// Routes the question to whoever consumes events (the TUI).
pub struct EventPermissioner(pub EventSink);

#[async_trait]
impl Permissioner for EventPermissioner {
    async fn confirm(&self, call: &str, reasons: &[String]) -> bool {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.0.send(Event::Ask { call: call.to_string(), reasons: reasons.to_vec(), reply: tx }).is_err() {
            return false;
        }
        rx.await.unwrap_or(false)
    }
}

/// Prompts on the terminal.
pub struct StdinPermissioner;

#[async_trait]
impl Permissioner for StdinPermissioner {
    async fn confirm(&self, call: &str, reasons: &[String]) -> bool {
        let call = call.to_string();
        let reasons = reasons.join("; ");
        tokio::task::spawn_blocking(move || {
            use std::io::Write;
            print!("\nallow  {call}\n       [{reasons}]\n       run it? [y/N] ");
            std::io::stdout().flush().ok();
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).ok();
            matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
        })
        .await
        .unwrap_or(false)
    }
}

/// Everything sessions and sub-agents share.
pub struct Shared {
    pub cfg: Config,
    pub root: PathBuf,
    pub store: Arc<Mutex<ChunkStore>>,
    pub jev: Arc<Jev>,
    pub llm: Arc<dyn LlmClient>,
    pub args: Arc<dyn ArgBuilder>,
    pub policy: Arc<Policy>,
    pub router: Arc<Router>,
    pub assembler: Arc<Assembler>,
    pub instructions: Arc<Instructions>,
    pub subgoals: Arc<Mutex<subgoals::SubgoalRegistry>>,
    pub leases: Arc<subgoals::LeaseTable>,
    pub retrieval: watch::Sender<Arc<background::Retrieval>>,
    pub events: EventSink,
    pub permissioner: Arc<dyn Permissioner>,
    sub_counter: AtomicU32,
}

#[derive(Default, Clone, Debug)]
pub struct Totals {
    pub llm: LlmUsage,
    pub llm_cost: f64,
    pub turns: u32,
}

pub struct Session {
    pub shared: Arc<Shared>,
    pub id: String,
    pub turn: u32,
    tools: ToolRegistry,
    goal: String,
    last_order: Option<Vec<ChunkId>>,
    recent_paths: Vec<PathBuf>,
    model_override: Option<String>,
    max_turns: u32,
    pub totals: Totals,
    _progress: Option<tokio::task::JoinHandle<()>>,
}

fn new_session_id() -> String {
    let t = chrono::Utc::now().timestamp_millis() as u64;
    let h = blake3::hash(&t.to_le_bytes());
    hex::encode(&h.as_bytes()[..3])
}

impl Session {
    /// Open the main session for `root`.
    pub async fn open(cfg: Config, root: &Path, events: EventSink, permissioner: Arc<dyn Permissioner>) -> Result<Self> {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let state_path = root.join(&cfg.session.dir).join("state.redb");
        let store = ChunkStore::open(&state_path)?;
        let jev = Arc::new(Jev::from_config(&cfg.jev)?);
        let llm: Arc<dyn LlmClient> = match cfg.llm.provider.as_str() {
            "scripted" => Arc::new(llm::scripted::ScriptedClient::from_env()?),
            _ => match llm::anthropic::AnthropicClient::from_env(cfg.llm.fallbacks) {
                Ok(c) => Arc::new(c),
                Err(e) => {
                    emit(&events, Event::Info(format!("{e}; falling back to the scripted demo model")));
                    Arc::new(llm::scripted::ScriptedClient::from_env()?)
                }
            },
        };
        let scripted = llm.name() == "scripted";
        let args: Arc<dyn ArgBuilder> = if scripted { Arc::new(HeuristicArgBuilder) } else { Arc::new(crate::tools::args::LlmArgBuilder::new(llm.clone(), &cfg.llm.cheap)) };
        let summarizer: Arc<dyn Summarizer> = if scripted { Arc::new(TruncateSummarizer) } else { Arc::new(llm::LlmSummarizer::new(llm.clone(), &cfg.llm.cheap)) };
        let assembler = Arc::new(Assembler::new(cfg.budget.clone(), summarizer));
        let policy = Arc::new(Policy::load(&root, &cfg)?);
        let router = Arc::new(Router::new(&cfg)?);
        let instructions = Arc::new(Instructions::load(&root)?);
        let (retrieval, rx) = background::channel();
        let progress = background::spawn_progress_writer(rx, root.join(&cfg.session.dir).join("progress.md"));
        emit(&events, Event::Info(format!("jev transport: {} · model client: {} · {} chunks in store · {} instruction fragments · {} tools", jev.transport_name(), llm.name(), store.len(), instructions.len(), 7)));
        let shared = Arc::new(Shared {
            root: root.clone(),
            store: Arc::new(Mutex::new(store)),
            jev,
            llm,
            args,
            policy,
            router,
            assembler,
            instructions,
            subgoals: Arc::new(Mutex::new(subgoals::SubgoalRegistry::default())),
            leases: Arc::new(subgoals::LeaseTable::default()),
            retrieval,
            events,
            permissioner,
            sub_counter: AtomicU32::new(0),
            cfg,
        });
        let max_turns = shared.cfg.session.max_turns;
        Ok(Self {
            shared,
            id: new_session_id(),
            turn: 0,
            tools: ToolRegistry::builtin(),
            goal: String::new(),
            last_order: None,
            recent_paths: Vec::new(),
            model_override: None,
            max_turns,
            totals: Totals::default(),
            _progress: Some(progress),
        })
    }

    /// A read-only sub-agent on the worker model, sharing the store.
    fn sub(shared: Arc<Shared>, n: u32) -> Self {
        let worker = shared.cfg.llm.worker.clone();
        let max_turns = shared.cfg.session.subagent_turns;
        Self {
            shared,
            id: format!("sub-{n}"),
            turn: 0,
            tools: ToolRegistry::builtin().read_only(),
            goal: String::new(),
            last_order: None,
            recent_paths: Vec::new(),
            model_override: Some(worker),
            max_turns,
            totals: Totals::default(),
            _progress: None,
        }
    }

    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    pub async fn snapshot(&self) -> Snapshot {
        self.shared.store.lock().await.snapshot()
    }

    async fn append(&self, chunk: Chunk) -> Result<ChunkId> {
        self.shared.store.lock().await.append(chunk)
    }

    fn events(&self) -> &EventSink {
        &self.shared.events
    }

    fn usage_snapshot(&self) -> UsageSnapshot {
        let js = self.shared.jev.stats();
        UsageSnapshot {
            turns: self.totals.turns,
            llm: self.totals.llm,
            llm_cost: self.totals.llm_cost,
            jev_calls: js.calls,
            jev_questions: js.questions,
            jev_tokens: js.input_tokens,
            jev_cost: js.input_tokens as f64 * self.shared.cfg.jev.price_per_mtok / 1e6,
            memo_hits: js.memo_hits,
        }
    }

    /// Run one goal to completion or to the turn cap. Returns the final answer.
    pub async fn run_goal(&mut self, goal: &str) -> Result<String> {
        self.goal = goal.to_string();
        self.turn += 1;
        self.append(Chunk::new(Kind::UserTurn, goal, self.turn, &self.id)).await?;
        let mut query = goal.to_string();
        for _ in 0..self.max_turns {
            let step = self.turn_once(&query).await?;
            match step {
                Step::Done { answer } => {
                    self.append(Chunk::new(Kind::Assistant, answer.clone(), self.turn, &self.id)).await?;
                    emit(self.events(), Event::Done { session: self.id.clone(), answer: answer.clone() });
                    emit(self.events(), Event::Usage(self.usage_snapshot()));
                    return Ok(answer);
                }
                Step::Act { intent, note } => {
                    if !note.is_empty() {
                        self.append(Chunk::new(Kind::Reasoning, note, self.turn, &self.id)).await?;
                    }
                    self.act(&intent).await?;
                    self.turn += 1;
                    query = format!("{goal}\n\n(last action: {})", crate::state::truncate(&intent, 200));
                }
            }
        }
        let answer = format!("stopped after {} turns without a final answer", self.max_turns);
        emit(self.events(), Event::Done { session: self.id.clone(), answer: answer.clone() });
        Ok(answer)
    }

    /// Assemble the context for `query` without calling the model.
    pub async fn assemble(&mut self, query: &str) -> Result<crate::context::Context> {
        let s = &self.shared;
        let pinned = s.instructions.active(query, &self.recent_paths, &s.root, &s.jev, &self.id, self.turn).await?;
        let snap = self.snapshot().await;
        let frontier = s.router.frontier();
        let ctx = s
            .assembler
            .build(
                AssembleInput { snap: &snap, goal: &self.goal, query, session: &self.id, turn: self.turn, pinned, previous_order: self.last_order.as_deref(), prices: (frontier.input, frontier.cached) },
                &s.jev,
            )
            .await?;
        for c in &ctx.new_chunks {
            self.append(c.clone()).await?;
        }
        if let Some(cd) = &ctx.cache {
            emit(
                self.events(),
                Event::Decision {
                    point: "cache",
                    detail: format!("{} · kept {} · +{} tok · {} stale · reuse ${:.4} vs rebuild ${:.4}", if cd.reuse { "reuse prefix" } else { "rebuild" }, cd.kept, cd.delta_tokens, cd.stale, cd.reuse_cost, cd.rebuild_cost),
                    p: cd.p,
                },
            );
        }
        emit(
            self.events(),
            Event::Context {
                rows: ctx.items.iter().map(|i| ContextRow { id: i.id.short(), label: i.label.clone(), visibility: i.visibility, tokens: i.tokens, p: i.p, pinned: i.pinned }).collect(),
                tokens: ctx.tokens,
                budget: ctx.budget,
                scored: ctx.scored,
                hidden: ctx.hidden,
                dropped: ctx.dropped,
                reused: ctx.cache.as_ref().map(|c| c.reuse),
            },
        );
        Ok(ctx)
    }

    async fn sensitivity(&self, query: &str) -> Result<(Sensitivity, f64, String)> {
        let s = &self.shared;
        if let Some(t) = s.router.policy().classify(self.recent_paths.iter().map(PathBuf::as_path)) {
            return Ok((t, 1.0, format!("from {} touched paths", self.recent_paths.len())));
        }
        let (id, q) = questions::sensitivity();
        let paths: Vec<String> = self.recent_paths.iter().map(|p| p.display().to_string()).collect();
        let a = s.jev.ask_one(json!({ "subtask": query, "paths": paths }), &id, q).await?;
        let (tier, conf) = questions::sensitivity_from(&a);
        Ok((tier, conf, "jev estimate".into()))
    }

    async fn turn_once(&mut self, query: &str) -> Result<Step> {
        emit(self.events(), Event::TurnStart { session: self.id.clone(), turn: self.turn, query: query.to_string() });
        let ctx = self.assemble(query).await?;
        let s = self.shared.clone();

        let (tier, p, how) = self.sensitivity(query).await?;
        emit(self.events(), Event::Decision { point: "security", detail: format!("{tier} · {how}"), p });

        let model = match &self.model_override {
            Some(m) => {
                emit(self.events(), Event::Route { model: m.clone(), est: 0.0, p: 1.0, reason: "sub-agent on worker model".into(), alternatives: vec![] });
                m.clone()
            }
            None => {
                let plan = s.router.choose(query, ctx.tokens, 2_000, 4_000, tier, &s.jev).await?;
                emit(self.events(), Event::Route { model: plan.model.clone(), est: plan.est_cost, p: plan.p, reason: plan.reason.clone(), alternatives: plan.alternatives.iter().map(|r| (r.model.clone(), r.est)).collect() });
                plan.model
            }
        };

        let pinned: String = ctx.items.iter().filter(|i| i.pinned).map(|i| i.text.clone()).collect::<Vec<_>>().join("\n\n");
        let context: String = crate::context::Context { items: ctx.items.iter().filter(|i| !i.pinned).cloned().collect(), ..Default::default() }.render();
        let prompt = Prompt {
            system: llm::SYSTEM_PROMPT.to_string(),
            pinned,
            context,
            query: format!("Goal: {}\n\nCurrent turn: {}. What is your next action, or your final answer?", self.goal, self.turn),
        };
        let completion = s.llm.complete(&model, &prompt, s.cfg.llm.max_tokens, &s.cfg.llm.effort).await?;
        self.totals.llm.add(&completion.usage);
        if let Some(spec) = s.router.model(&model) {
            self.totals.llm_cost += completion.usage.cost(&spec.price());
        }
        self.totals.turns += 1;
        self.last_order = Some(ctx.order());
        let step = llm::parse_step(&completion.text);
        if let Step::Act { note, .. } = &step {
            if !note.is_empty() {
                emit(self.events(), Event::Note { model: model.clone(), text: note.clone() });
            }
        }
        emit(self.events(), Event::Usage(self.usage_snapshot()));
        Ok(step)
    }

    async fn act(&mut self, intent: &str) -> Result<()> {
        let s = self.shared.clone();
        let call = match self.tools.route(intent, &s.jev, s.args.as_ref(), &s.root).await {
            Ok(c) => c,
            Err(e) => {
                let body = format!("could not map the intent to a tool: {e}");
                self.append(Chunk::new(Kind::ToolOutput { call: ChunkId([0; 32]), tool: "none".into(), access: Access::Read, ok: false }, body.clone(), self.turn, &self.id)).await?;
                emit(self.events(), Event::ToolRan { tool: "none".into(), access: "read".into(), ok: false, tokens: tokens::count(&body), preview: body });
                return Ok(());
            }
        };
        emit(self.events(), Event::ToolPicked { tool: call.tool.clone(), intent: intent.to_string(), candidates: call.candidates.clone(), args: crate::state::truncate(&call.args.to_string(), 200) });
        let call_chunk = Chunk::new(Kind::ToolCall { tool: call.tool.clone(), args: call.args.clone(), intent: intent.to_string() }, format!("{intent}\n→ {}", call.describe()), self.turn, &self.id);
        let call_id = self.append(call_chunk).await?;

        let decision = s.policy.check(&call, &self.goal, &s.jev).await?;
        emit(self.events(), Event::Permit { verdict: decision.verdict.to_string(), reasons: decision.reasons.clone(), call: call.describe() });
        let allowed = match decision.verdict {
            Verdict::Allow => true,
            Verdict::Deny => false,
            Verdict::Ask => s.permissioner.confirm(&call.describe(), &decision.reasons).await,
        };
        let output = if allowed { self.execute(&call).await } else { ToolOutput { tool: call.tool.clone(), body: format!("not run: {} [{}]", decision.verdict, decision.reasons.join("; ")), access: call.access, ok: false, paths: vec![] } };
        let sens = s.router.policy().classify(output.paths.iter().map(PathBuf::as_path)).unwrap_or(Sensitivity::Standard);
        let out_chunk = Chunk::new(Kind::ToolOutput { call: call_id, tool: output.tool.clone(), access: output.access, ok: output.ok }, output.body.clone(), self.turn, &self.id)
            .with_paths(output.paths.clone())
            .with_sensitivity(sens);
        self.append(out_chunk).await?;
        emit(self.events(), Event::ToolRan { tool: output.tool.clone(), access: output.access.to_string(), ok: output.ok, tokens: tokens::count(&output.body), preview: crate::state::truncate(&output.body, 300) });

        for p in &output.paths {
            let abs = if p.is_absolute() { p.clone() } else { s.root.join(p) };
            if !self.recent_paths.contains(&abs) {
                self.recent_paths.push(abs);
            }
        }
        if self.recent_paths.len() > 24 {
            let n = self.recent_paths.len() - 24;
            self.recent_paths.drain(..n);
        }
        if output.access == Access::Write && output.ok {
            let snap = self.snapshot().await;
            let r = background::retrieve(&snap, &self.id, self.turn, &self.goal, output.paths.clone(), intent);
            let _ = s.retrieval.send(Arc::new(r));
        }
        Ok(())
    }

    async fn execute(&self, call: &ToolCall) -> ToolOutput {
        let s = &self.shared;
        let _guards = if call.access == Access::Write { s.leases.acquire(&call.footprint.paths).await } else { Vec::new() };
        let cx = ToolCx { root: s.root.clone(), delegator: Some(Arc::new(SubDelegator { shared: s.clone() })) };
        match self.tools.run(call, &cx).await {
            Ok(o) => o,
            Err(e) => ToolOutput { tool: call.tool.clone(), body: format!("error: {e}"), access: call.access, ok: false, paths: vec![] },
        }
    }
}

struct SubDelegator {
    shared: Arc<Shared>,
}

#[async_trait]
impl Delegator for SubDelegator {
    async fn delegate(&self, goal: &str) -> Result<String> {
        let s = &self.shared;
        if let Some(existing) = s.subgoals.lock().await.duplicate_of(goal, &s.jev).await? {
            emit(&s.events, Event::Subagent { id: format!("sub-{}", existing.id), goal: goal.into(), status: "deduplicated".into() });
            return Ok(match existing.status {
                subgoals::Status::Done(r) => format!("already done as sub-goal {} ({}):\n{r}", existing.id, existing.goal),
                subgoals::Status::Running => format!("already in flight as sub-goal {} ({})", existing.id, existing.goal),
            });
        }
        let idx = s.subgoals.lock().await.register(goal);
        let n = s.sub_counter.fetch_add(1, Ordering::Relaxed) + 1;
        emit(&s.events, Event::Subagent { id: format!("sub-{n}"), goal: goal.into(), status: "started".into() });
        let mut sub = Session::sub(s.clone(), n);
        let result = sub.run_goal(goal).await.unwrap_or_else(|e| format!("sub-agent failed: {e}"));
        s.subgoals.lock().await.finish(idx, &result);
        emit(&s.events, Event::Subagent { id: format!("sub-{n}"), goal: goal.into(), status: "done".into() });
        let chunk = Chunk::new(Kind::SubagentResult { goal: goal.into(), model: s.cfg.llm.worker.clone() }, result.clone(), sub.turn, &sub.id);
        s.store.lock().await.append(chunk)?;
        Ok(result)
    }
}

/// Build a session for one-off, model-free commands (`context`, `state`).
pub async fn open_quiet(cfg: Config, root: &Path) -> Result<(Session, tokio::sync::mpsc::UnboundedReceiver<Event>)> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut cfg = cfg;
    cfg.llm.provider = "scripted".into();
    let session = Session::open(cfg, root, tx, Arc::new(AutoPermissioner(false))).await?;
    Ok((session, rx))
}

pub fn ensure_root(root: &Path) -> Result<PathBuf> {
    if !root.is_dir() {
        return Err(anyhow!("{} is not a directory", root.display()));
    }
    Ok(root.canonicalize()?)
}
