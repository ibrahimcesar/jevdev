//! `jevdev`: a coding-agent harness built around Jev.

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use jevdev::config::{Config, CONFIG_FILE};
use jevdev::router::{cost_table, Price};
use jevdev::runtime::{self, print_plain, AutoPermissioner, Event, Permissioner, Session, StdinPermissioner};
use jevdev::state::{tokens::fmt_k, Chunk, ChunkId};
use jevdev::tools::ToolRegistry;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "jevdev", version, about = "A coding-agent harness built around Jev, TypeSafe's System One decision model.", long_about = None)]
struct Cli {
    /// Repository root. Defaults to the current directory.
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Write jevdev.toml, the default Cedar policy, and a sample AGENTS.md.
    Init {
        /// Overwrite existing files.
        #[arg(long)]
        force: bool,
    },
    /// Interactive terminal session (the default).
    Tui {
        /// A goal to start with.
        goal: Option<String>,
    },
    /// Run one goal with plain output.
    Run {
        /// What to accomplish, in plain words.
        goal: String,
        /// Answer every permission prompt with yes.
        #[arg(long)]
        yes: bool,
        /// Answer every permission prompt with no (unattended).
        #[arg(long)]
        no: bool,
        /// Emit events as JSON lines instead of plain text.
        #[arg(long)]
        json: bool,
    },
    /// Re-run context assembly over a recorded session, turn by turn, without calling a model.
    Replay {
        /// Session id (default: the most recent).
        #[arg(long)]
        session: Option<String>,
        /// Use this query at every turn instead of the recorded one.
        #[arg(long)]
        query: Option<String>,
        /// Print the ladder rows for every turn.
        #[arg(long)]
        rows: bool,
    },
    /// Assemble the context for a query and show the visibility ladder, without calling a model.
    Context { query: String },
    /// Tier 1: list tools. With an id, tier 2 (schema) or tier 3 (docs).
    Tools {
        id: Option<String>,
        #[arg(long)]
        docs: bool,
    },
    /// Evaluate the execution policy for a shell command.
    Policy { command: String },
    /// The routing arithmetic from the design notes, in millions of tokens.
    Cost {
        #[arg(long, default_value_t = 0.65)]
        x: f64,
        #[arg(long, default_value_t = 0.12)]
        y: f64,
        #[arg(long, default_value_t = 0.23)]
        z: f64,
    },
    /// List stored chunks, or show one.
    State {
        /// Show one chunk in full by its id prefix.
        show: Option<String>,
        /// Only this session.
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value_t = 60)]
        limit: usize,
    },
    /// Configured models, prices, and trust tiers.
    Models,
    /// Check keys, endpoints, policy, instructions, and the store.
    Doctor {
        /// Skip the network round trips.
        #[arg(long)]
        offline: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Let `jevdev run --json | head` end quietly instead of panicking on EPIPE.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).init();
    let cli = Cli::parse();
    let root = runtime::ensure_root(&cli.root.unwrap_or(std::env::current_dir()?))?;
    let cfg = Config::load(&root)?;
    match cli.command.unwrap_or(Command::Tui { goal: None }) {
        Command::Init { force } => init(&root, force),
        Command::Tui { goal } => jevdev::tui::run(cfg, &root, goal).await,
        Command::Run { goal, yes, no, json } => run(cfg, &root, &goal, yes, no, json).await,
        Command::Replay { session, query, rows } => replay(cfg, &root, session, query, rows).await,
        Command::Context { query } => context(cfg, &root, &query).await,
        Command::Tools { id, docs } => tools(id, docs),
        Command::Policy { command } => policy(cfg, &root, &command).await,
        Command::Cost { x, y, z } => {
            let f = cfg.model(&cfg.llm.frontier).map(|m| m.price()).unwrap_or(Price { input: 5.0, cached: 0.5, output: 25.0 });
            let w = cfg.model(&cfg.llm.worker).map(|m| m.price()).unwrap_or(Price { input: 3.0, cached: 0.3, output: 15.0 });
            println!("frontier {} (${}/{} per Mtok) · worker {} (${}/{} per Mtok)\n", cfg.llm.frontier, f.input, f.output, cfg.llm.worker, w.input, w.output);
            print!("{}", cost_table(x, y, z, &f, &w));
            Ok(())
        }
        Command::State { show, session, limit } => state(cfg, &root, show, session, limit).await,
        Command::Doctor { offline } => doctor(cfg, &root, offline).await,
        Command::Models => {
            println!("{:<22} {:<10} {:<9} {:>7} {:>7} {:>7}  description", "model", "tier", "trust", "in", "cached", "out");
            for m in &cfg.models {
                println!("{:<22} {:<10} {:<9} {:>7.2} {:>7.2} {:>7.2}  {}", m.id, format!("{:?}", m.tier).to_lowercase(), format!("{:?}", m.trust).to_lowercase(), m.input, m.cached, m.output, m.description);
            }
            println!("\nfrontier: {} · worker: {} · cheap: {} · routing: {}", cfg.llm.frontier, cfg.llm.worker, cfg.llm.cheap, cfg.llm.routing);
            Ok(())
        }
    }
}

async fn doctor(cfg: Config, root: &std::path::Path, offline: bool) -> Result<()> {
    use jevdev::jev::{Jev, Question};
    use std::time::Instant;
    let mut failures = 0u32;
    let ok = |name: &str, detail: String| println!("  ✓ {name:<13} {detail}");
    let warn = |name: &str, detail: String| println!("  ! {name:<13} {detail}");
    println!("jevdev {} · {}\n", jevdev::VERSION, root.display());

    let cfg_path = root.join(CONFIG_FILE);
    if cfg_path.exists() {
        ok("config", format!("{}", cfg_path.display()));
    } else {
        warn("config", "no jevdev.toml, using defaults (run `jevdev init`)".into());
    }

    // Jev
    match Jev::from_config(&cfg.jev) {
        Ok(jev) if jev.transport_name() == "local" => warn("jev", "local heuristics stand in for Jev; set TYPESAFE_API_KEY for the real model".into()),
        Ok(jev) => {
            if offline {
                ok("jev", format!("http · {} · {}", cfg.jev.endpoint, cfg.jev.model));
            } else {
                let t = Instant::now();
                match jev.ask_one(serde_json::json!({ "probe": "jevdev doctor health check" }), "probe", Question::noul("Is state.probe a health-check message from a developer tool?")).await {
                    Ok(a) => {
                        let s = jev.stats();
                        ok("jev", format!("{} · {} · {:?} · {} input tok · {}", cfg.jev.endpoint, cfg.jev.model, t.elapsed(), s.input_tokens, a.summary()));
                    }
                    Err(e) => {
                        failures += 1;
                        println!("  ✗ {:<13} {e:#}", "jev");
                    }
                }
            }
        }
        Err(e) => {
            failures += 1;
            println!("  ✗ {:<13} {e:#}", "jev");
        }
    }

    // Model
    if cfg.llm.provider == "scripted" {
        warn("model", "scripted demo client; set llm.provider = \"anthropic\" for a real model".into());
    } else {
        match jevdev::llm::anthropic::AnthropicClient::from_env(cfg.llm.fallbacks) {
            Ok(c) => {
                if offline {
                    ok("model", format!("{} · credentials from {}", c.base_url(), c.auth_source()));
                } else {
                    use jevdev::llm::LlmClient;
                    let t = Instant::now();
                    match c.small(&cfg.llm.cheap, "Reply with exactly: OK", "ping", 8).await {
                        Ok(text) => ok("model", format!("{} · {} via {} · {:?} · replied {:?}", c.base_url(), cfg.llm.cheap, c.auth_source(), t.elapsed(), text.trim())),
                        Err(e) => {
                            failures += 1;
                            println!("  ✗ {:<13} {} via {}: {e:#}", "model", cfg.llm.cheap, c.auth_source());
                        }
                    }
                }
            }
            Err(e) => {
                failures += 1;
                println!("  ✗ {:<13} {e:#}", "model");
            }
        }
    }
    let mut models_ok = true;
    for (role, id) in [("frontier", &cfg.llm.frontier), ("worker", &cfg.llm.worker), ("cheap", &cfg.llm.cheap)] {
        if cfg.model(id).is_none() {
            failures += 1;
            models_ok = false;
            println!("  ✗ {:<13} llm.{role} = {id} is not in [[models]], so it has no price or trust tier", "models");
        }
    }
    if models_ok {
        ok("models", format!("{} configured · frontier {} · worker {} · cheap {} · routing {}", cfg.models.len(), cfg.llm.frontier, cfg.llm.worker, cfg.llm.cheap, cfg.llm.routing));
    }

    // Policy
    match jevdev::policy::Policy::load(root, &cfg) {
        Ok(p) => {
            let rules = p.rules();
            let src = if root.join(&cfg.policy.file).exists() { cfg.policy.file.clone() } else { "built-in default".into() };
            ok("policy", format!("{} rules from {src}: {}", rules.len(), rules.join(", ")));
        }
        Err(e) => {
            failures += 1;
            println!("  ✗ {:<13} {e:#}", "policy");
        }
    }

    // Instructions
    match runtime::Instructions::load(root) {
        Ok(i) if i.is_empty() => warn("instructions", "no AGENTS.md (run `jevdev init` for a sample)".into()),
        Ok(i) => {
            let conds: Vec<String> = i.fragments().iter().map(|f| f.condition.describe()).collect();
            ok("instructions", format!("{} fragments: {}", i.len(), conds.join(" · ")));
        }
        Err(e) => {
            failures += 1;
            println!("  ✗ {:<13} {e:#}", "instructions");
        }
    }

    // Store (opened only if it exists; doctor leaves no files behind)
    let path = root.join(&cfg.session.dir).join("state.redb");
    if path.exists() {
        match jevdev::state::ChunkStore::open(&path) {
            Ok(s) => ok("store", format!("{} chunks in {}", s.len(), path.display())),
            Err(e) => {
                failures += 1;
                println!("  ✗ {:<13} {e:#}", "store");
            }
        }
    } else {
        ok("store", format!("none yet; the first session creates {}", path.display()));
    }
    ok("tools", format!("{} built in", ToolRegistry::builtin().len()));

    println!();
    if failures > 0 {
        println!("{failures} problem(s)");
        std::process::exit(1);
    }
    println!("all good");
    Ok(())
}

fn init(root: &std::path::Path, force: bool) -> Result<()> {
    let write = |rel: &str, content: &str| -> Result<()> {
        let p = root.join(rel);
        if p.exists() && !force {
            println!("kept    {rel}");
            return Ok(());
        }
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&p, content).with_context(|| format!("writing {rel}"))?;
        println!("wrote   {rel}");
        Ok(())
    };
    write(CONFIG_FILE, &Config::default_toml())?;
    write(".jevdev/exec.cedar", jevdev::policy::DEFAULT_POLICY)?;
    write("AGENTS.md", runtime::instructions::SAMPLE_AGENTS_MD)?;
    let gi = root.join(".gitignore");
    let has = std::fs::read_to_string(&gi).map(|s| s.lines().any(|l| l.trim() == ".jevdev/")).unwrap_or(false);
    if !has {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&gi)?;
        writeln!(f, "\n# jevdev session state\n.jevdev/")?;
        println!("updated .gitignore");
    }
    println!("\nnext: export TYPESAFE_API_KEY for Jev and ANTHROPIC_API_KEY for the model, then `jevdev`.");
    Ok(())
}

async fn run(cfg: Config, root: &std::path::Path, goal: &str, yes: bool, no: bool, json: bool) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let printer = tokio::spawn(async move {
        while let Some(e) = rx.recv().await {
            if json {
                println!("{}", e.to_json());
            } else {
                print_plain(&e);
            }
        }
    });
    let permissioner: Arc<dyn Permissioner> = if yes {
        Arc::new(AutoPermissioner(true))
    } else if no {
        Arc::new(AutoPermissioner(false))
    } else {
        Arc::new(StdinPermissioner)
    };
    let mut session = Session::open(cfg, root, tx, permissioner).await?;
    let _ = session.run_goal(goal).await?;
    drop(session);
    let _ = printer.await;
    Ok(())
}

async fn replay(cfg: Config, root: &std::path::Path, session: Option<String>, query: Option<String>, rows: bool) -> Result<()> {
    use jevdev::state::Kind;
    let (mut s, mut rx) = runtime::open_quiet(cfg, root).await?;
    let snap = s.snapshot().await;
    let all: Vec<&Chunk> = snap.iter().collect();
    let sid = match session {
        Some(id) => id,
        None => all.iter().rev().find(|c| matches!(c.kind, Kind::UserTurn)).map(|c| c.session.clone()).ok_or_else(|| anyhow::anyhow!("no recorded sessions; run a goal first"))?,
    };
    let goal = all.iter().find(|c| c.session == sid && matches!(c.kind, Kind::UserTurn)).map(|c| c.body.clone()).ok_or_else(|| anyhow::anyhow!("session {sid} has no user turn"))?;
    let max_turn = all.iter().filter(|c| c.session == sid).map(|c| c.turn).max().unwrap_or(0);
    println!("replaying session {sid} · {max_turn} turns · goal: {}\n", jevdev::state::truncate(&goal, 100));
    s.id = sid.clone();
    s.set_goal(&goal);
    let mut last_intent: Option<String> = None;
    for t in 1..=max_turn {
        // The store as it was when turn t was assembled: everything before the first chunk turn t produced.
        let cutoff = all.iter().position(|c| c.session == sid && c.turn == t && !matches!(c.kind, Kind::UserTurn | Kind::Summary { .. } | Kind::Instruction { .. })).unwrap_or(all.len());
        let prefix = snap.prefix(cutoff);
        let paths: Vec<std::path::PathBuf> = prefix.iter().filter(|c| c.session == sid).flat_map(|c| c.paths.clone()).collect();
        s.note_paths(paths);
        s.turn = t;
        let q = match (&query, &last_intent) {
            (Some(q), _) => q.clone(),
            (None, Some(i)) => format!("{goal}\n\n(last action: {})", jevdev::state::truncate(i, 200)),
            (None, None) => goal.clone(),
        };
        let ctx = s.assemble_snapshot(&prefix, &q).await?;
        let cache = ctx.cache.as_ref().map(|c| format!(" · cache {} p={:.2}", if c.reuse { "reuse" } else { "rebuild" }, c.p)).unwrap_or_default();
        println!("turn {t:<3} {:>6} / {} tok · jev {:<3} memo {:<3} hidden {:<3} dropped {:<3}{cache}", ctx.tokens, ctx.budget, ctx.scored, ctx.memo_hits, ctx.hidden, ctx.dropped);
        if rows {
            for i in &ctx.items {
                println!("         {:<5} {:>6} tok  p={:.2}  {}{}", i.visibility, i.tokens, i.p, if i.pinned { "📌 " } else { "" }, i.label);
            }
        }
        last_intent = all.iter().find(|c| c.session == sid && c.turn == t).and_then(|_| all.iter().find(|c| c.session == sid && c.turn == t && matches!(c.kind, Kind::ToolCall { .. }))).and_then(|c| match &c.kind {
            Kind::ToolCall { intent, .. } => Some(intent.clone()),
            _ => None,
        });
        while rx.try_recv().is_ok() {}
    }
    let js = s.shared.jev.stats();
    println!("\njev: {} calls · {} questions · {} input tok · {} request memo hits", js.calls, js.questions, js.input_tokens, js.memo_hits);
    Ok(())
}

async fn context(cfg: Config, root: &std::path::Path, query: &str) -> Result<()> {
    let (mut session, mut rx) = runtime::open_quiet(cfg, root).await?;
    session.turn = session.snapshot().await.last_turn() + 1;
    let ctx = session.assemble(query).await?;
    while let Ok(e) = rx.try_recv() {
        if matches!(e, Event::Info(_) | Event::Decision { .. } | Event::Context { .. }) {
            print_plain(&e);
        }
    }
    println!("\n--- rendered ({} tok) ---\n{}", fmt_k(ctx.tokens), ctx.render());
    Ok(())
}

fn tools(id: Option<String>, docs: bool) -> Result<()> {
    let reg = ToolRegistry::builtin();
    match id {
        None => {
            println!("tier 1 · {} snippets, the only thing the model ever sees about tools:\n", reg.len());
            for (id, s) in reg.snippets() {
                let t = reg.get(&id).unwrap();
                println!("  {:<12} {:<6} {s}", id, t.access().to_string());
            }
        }
        Some(id) => {
            let t = reg.get(&id).ok_or_else(|| anyhow::anyhow!("no tool {id}"))?;
            if docs {
                println!("tier 3 · {id}\n\n{}", t.docs());
            } else {
                println!("tier 2 · {id} schema, loaded only when Jev picks it:\n\n{}", serde_json::to_string_pretty(&t.schema())?);
            }
        }
    }
    Ok(())
}

async fn policy(cfg: Config, root: &std::path::Path, command: &str) -> Result<()> {
    let p = jevdev::policy::Policy::load(root, &cfg)?;
    let jev = jevdev::jev::Jev::from_config(&cfg.jev)?;
    let shell = ToolRegistry::builtin().get("shell").unwrap();
    let args = serde_json::json!({ "command": command });
    let call = jevdev::tools::ToolCall { tool: "shell".into(), args: args.clone(), intent: command.into(), access: shell.access(), footprint: shell.footprint(&args, root), candidates: vec![] };
    let d = p.check(&call, "evaluate policy", &jev).await?;
    println!("{}  {}", d.verdict, command);
    println!("reasons: {}", if d.reasons.is_empty() { "none (no policy matched)".into() } else { d.reasons.join("; ") });
    println!("attributes: {}", serde_json::to_string_pretty(&d.attrs)?);
    Ok(())
}

async fn state(cfg: Config, root: &std::path::Path, show: Option<String>, session: Option<String>, limit: usize) -> Result<()> {
    let path = root.join(&cfg.session.dir).join("state.redb");
    let store = jevdev::state::ChunkStore::open(&path)?;
    let snap = store.snapshot();
    if let Some(prefix) = show {
        let found: Vec<&Chunk> = snap.iter().filter(|c| c.id.hex().starts_with(&prefix)).collect();
        match found.as_slice() {
            [c] => {
                println!("{} · {} · session {} · {} tok · {}\n\n{}", c.id.hex(), c.label(), c.session, c.tokens, c.sensitivity, c.body);
            }
            [] => println!("no chunk starts with {prefix}"),
            many => println!("{} chunks match; be more specific", many.len()),
        }
        return Ok(());
    }
    let all: Vec<&Chunk> = snap.iter().filter(|c| session.as_ref().map(|s| &c.session == s).unwrap_or(true)).collect();
    println!("{} chunks in {}{}\n", all.len(), path.display(), session.map(|s| format!(" (session {s})")).unwrap_or_default());
    let skip = all.len().saturating_sub(limit);
    for c in all.iter().skip(skip) {
        println!("{}  {:<8} t{:<3} {:>6} tok  {:<10} {}", c.id.short(), c.session, c.turn, c.tokens, c.sensitivity.to_string(), jevdev::state::truncate(&c.label(), 60));
    }
    let _ = ChunkId::parse("");
    Ok(())
}
