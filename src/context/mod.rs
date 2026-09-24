//! Meta-attention: context as a per-query decision.
//!
//! Nothing is appended to a transcript. Each turn the assembler takes a
//! snapshot of the chunk store, prefilters candidates deterministically, asks
//! Jev for a visibility level per candidate in one batched call, renders each
//! chunk at that level (summaries are generated once and stored as chunks),
//! packs to the token budget, and orders the result so the provider cache
//! still hits when Jev says the previous prefix is worth keeping.

pub mod summary;

use crate::config::BudgetConfig;
use crate::jev::{questions, Jev};
use crate::state::{tokens, Chunk, ChunkId, Kind, Snapshot, Visibility};
use anyhow::Result;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub use summary::{Summarizer, TruncateSummarizer};

#[derive(Clone, Debug)]
pub struct ContextItem {
    pub id: ChunkId,
    pub label: String,
    pub kind: &'static str,
    pub text: String,
    pub tokens: u32,
    pub visibility: Visibility,
    pub p: f64,
    pub pinned: bool,
    pub turn: u32,
    seq: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CacheDecision {
    pub reuse: bool,
    pub p: f64,
    pub kept: usize,
    pub delta_tokens: u32,
    pub stale: usize,
    pub reuse_cost: f64,
    pub rebuild_cost: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Context {
    pub items: Vec<ContextItem>,
    pub tokens: u32,
    pub budget: u32,
    pub scored: usize,
    pub hidden: usize,
    pub dropped: usize,
    pub cache: Option<CacheDecision>,
    /// Older chunks whose visibility came from the memo instead of Jev.
    pub memo_hits: usize,
    /// Summary chunks generated during assembly; the caller appends them.
    pub new_chunks: Vec<Chunk>,
}

impl Context {
    /// The chunk ids in the order they were rendered.
    pub fn order(&self) -> Vec<ChunkId> {
        self.items.iter().map(|i| i.id).collect()
    }

    /// Render for the model: chronological, each chunk under a stable header.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.tokens as usize * 4);
        for it in &self.items {
            out.push_str(&format!("<chunk id=\"{}\" kind=\"{}\" turn=\"{}\" view=\"{}\">\n", it.id.short(), it.kind, it.turn, it.visibility));
            out.push_str(it.text.trim_end());
            out.push_str("\n</chunk>\n\n");
        }
        out
    }
}

pub struct AssembleInput<'a> {
    pub snap: &'a Snapshot,
    pub goal: &'a str,
    pub query: &'a str,
    pub session: &'a str,
    pub turn: u32,
    /// Instruction chunks whose condition holds this turn. Always shown in full.
    pub pinned: Vec<Chunk>,
    /// The order sent last turn, for the cache-reuse decision.
    pub previous_order: Option<&'a [ChunkId]>,
    /// Ignore memoised scores for older chunks this turn (after a "rebuild").
    pub rescore_all: bool,
    /// Price per million tokens: (uncached input, cached input).
    pub prices: (f64, f64),
}

pub struct Assembler {
    cfg: BudgetConfig,
    summarizer: Arc<dyn Summarizer>,
    /// Visibility of an older chunk, keyed on (chunk, goal). Recent chunks are
    /// re-scored every turn; older ones only when the goal changes or the
    /// cache decision said "rebuild".
    memo: moka::sync::Cache<(ChunkId, u64), (Visibility, f64)>,
}

fn hash64(s: &str) -> u64 {
    u64::from_le_bytes(blake3::hash(s.as_bytes()).as_bytes()[..8].try_into().unwrap())
}

impl Assembler {
    pub fn new(cfg: BudgetConfig, summarizer: Arc<dyn Summarizer>) -> Self {
        Self { cfg, summarizer, memo: moka::sync::Cache::new(200_000) }
    }

    pub fn memo_len(&self) -> u64 {
        self.memo.entry_count()
    }

    pub fn budget(&self) -> u32 {
        self.cfg.context_tokens
    }

    /// Deterministic prefilter: recent chunks of this session always go to Jev;
    /// older ones (any session) compete on keyword overlap with the query.
    fn candidates<'s>(&self, snap: &'s Snapshot, query: &str, session: &str, turn: u32) -> Vec<(usize, &'s Chunk)> {
        let recent_from = turn.saturating_sub(self.cfg.recent_turns);
        // Identical bodies across sessions (the same goal run twice) count once: the newest copy wins.
        let mut newest: HashMap<[u8; 32], usize> = HashMap::new();
        for (seq, c) in snap.iter().enumerate() {
            newest.insert(*blake3::hash(c.body.as_bytes()).as_bytes(), seq);
        }
        let mut recent = Vec::new();
        let mut older = Vec::new();
        for (seq, c) in snap.iter().enumerate() {
            if matches!(c.kind, Kind::Summary { .. } | Kind::Instruction { .. }) {
                continue;
            }
            if newest.get(blake3::hash(c.body.as_bytes()).as_bytes()) != Some(&seq) {
                continue;
            }
            if c.session == session && c.turn >= recent_from {
                recent.push((seq, c));
            } else {
                older.push((seq, c));
            }
        }
        let room = self.cfg.candidates.saturating_sub(recent.len());
        if room > 0 && !older.is_empty() {
            let mut scored: Vec<(f64, usize, &Chunk)> = older
                .into_iter()
                .map(|(seq, c)| {
                    let mut s = crate::jev::local::overlap(query, &c.body);
                    if c.session == session {
                        s += 0.05;
                    }
                    (s, seq, c)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(b.1.cmp(&a.1)));
            for (s, seq, c) in scored.into_iter().take(room) {
                if s > 0.0 || c.session == session {
                    recent.push((seq, c));
                }
            }
        }
        recent.sort_by_key(|(seq, _)| *seq);
        recent
    }

    /// Batch visibility questions so each request's state stays under the Jev
    /// state limit. Older chunks already scored for this goal come from the memo.
    async fn score(&self, jev: &Jev, input: &AssembleInput<'_>, cands: &[(usize, &Chunk)], recent_from: u32) -> Result<(HashMap<ChunkId, (Visibility, f64)>, usize)> {
        let goal_key = hash64(input.goal);
        let mut out = HashMap::new();
        let mut hits = 0usize;
        let mut to_score: Vec<&Chunk> = Vec::new();
        for (_, c) in cands {
            let older = c.session != input.session || c.turn < recent_from;
            if older && !input.rescore_all {
                if let Some(v) = self.memo.get(&(c.id, goal_key)) {
                    out.insert(c.id, v);
                    hits += 1;
                    continue;
                }
            }
            to_score.push(c);
        }
        let mut batches: Vec<Vec<&Chunk>> = vec![Vec::new()];
        let mut acc: u32 = 0;
        let per = |c: &Chunk| tokens::count(&c.preview(self.cfg.preview_chars)) + 40;
        for c in &to_score {
            let t = per(c);
            if acc + t > self.cfg.jev_state_tokens && !batches.last().unwrap().is_empty() {
                batches.push(Vec::new());
                acc = 0;
            }
            batches.last_mut().unwrap().push(c);
            acc += t;
        }
        let futs = batches.into_iter().filter(|b| !b.is_empty()).map(|batch| {
            let chunks: Vec<_> = batch
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id.hex(),
                        "kind": c.kind.tag(),
                        "turn": c.turn,
                        "pinned": c.pinned(),
                        "label": c.label(),
                        "preview": c.preview(self.cfg.preview_chars),
                    })
                })
                .collect();
            let state = json!({ "goal": input.goal, "query": input.query, "turn": input.turn, "chunks": chunks });
            let qs = batch.iter().enumerate().map(|(i, c)| questions::visibility(&c.id, i)).collect();
            async move { jev.ask(state, qs).await }
        });
        let responses = futures::future::try_join_all(futs).await?;
        for r in responses {
            for (id, a) in &r.answers {
                if let Some(hex) = id.strip_prefix(questions::VIS_PREFIX) {
                    if let Some(cid) = ChunkId::parse(hex) {
                        let v = questions::visibility_from(a);
                        self.memo.insert((cid, goal_key), v);
                        out.insert(cid, v);
                    }
                }
            }
        }
        Ok((out, hits))
    }

    async fn render(&self, snap: &Snapshot, c: &Chunk, level: Visibility, new_chunks: &mut Vec<Chunk>) -> Result<String> {
        match level {
            Visibility::Full => Ok(c.body.clone()),
            Visibility::Hide => Ok(String::new()),
            level => {
                if let Some(s) = snap.summary(&c.id, level) {
                    return Ok(s.body.clone());
                }
                if let Some(s) = new_chunks.iter().find(|s| matches!(&s.kind, Kind::Summary { of, level: l } if *of == c.id && *l == level)) {
                    return Ok(s.body.clone());
                }
                let text = self.summarizer.summarize(c, level).await?;
                new_chunks.push(Chunk::new(Kind::Summary { of: c.id, level }, text.clone(), c.turn, &c.session));
                Ok(text)
            }
        }
    }

    pub async fn build(&self, input: AssembleInput<'_>, jev: &Jev) -> Result<Context> {
        let snap = input.snap;
        let cands = self.candidates(snap, input.query, input.session, input.turn);
        let recent_from = input.turn.saturating_sub(self.cfg.recent_turns);
        let (scores, memo_hits) = self.score(jev, &input, &cands, recent_from).await?;
        let mut new_chunks = Vec::new();
        let mut items: Vec<ContextItem> = Vec::new();
        let mut hidden = 0usize;

        for (i, c) in input.pinned.iter().enumerate() {
            items.push(ContextItem {
                id: c.id,
                label: c.label(),
                kind: c.kind.tag(),
                text: c.body.clone(),
                tokens: c.tokens,
                visibility: Visibility::Full,
                p: 1.0,
                pinned: true,
                turn: c.turn,
                seq: i,
            });
        }
        let base = input.pinned.len();
        for (seq, c) in &cands {
            let (vis, p) = scores.get(&c.id).copied().unwrap_or((Visibility::Short, 0.3));
            if vis == Visibility::Hide {
                hidden += 1;
                continue;
            }
            let text = self.render(snap, c, vis, &mut new_chunks).await?;
            let t = tokens::count(&text);
            let label = if c.session == input.session { c.label() } else { format!("{} · session {}", c.label(), c.session) };
            items.push(ContextItem { id: c.id, label, kind: c.kind.tag(), text, tokens: t, visibility: vis, p, pinned: false, turn: c.turn, seq: base + seq });
        }

        // Pack: pinned first, then by probability per token, newest first on ties.
        let budget = self.cfg.context_tokens;
        let mut total: u32 = items.iter().filter(|i| i.pinned).map(|i| i.tokens).sum();
        let mut rest: Vec<ContextItem> = items.iter().filter(|i| !i.pinned).cloned().collect();
        rest.sort_by(|a, b| {
            let ka = a.p / (a.tokens.max(1) as f64).sqrt();
            let kb = b.p / (b.tokens.max(1) as f64).sqrt();
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal).then(b.turn.cmp(&a.turn))
        });
        let mut selected: Vec<ContextItem> = items.into_iter().filter(|i| i.pinned).collect();
        let mut dropped = 0usize;
        for it in rest {
            if total + it.tokens <= budget {
                total += it.tokens;
                selected.push(it);
            } else {
                dropped += 1;
            }
        }

        // Order: chronological, unless Jev says the previous prefix is worth keeping.
        let cache = match input.previous_order {
            Some(prev) if !prev.is_empty() => {
                let now: HashSet<ChunkId> = selected.iter().map(|i| i.id).collect();
                let prev_set: HashSet<ChunkId> = prev.iter().copied().collect();
                let kept = prev.iter().filter(|id| now.contains(id)).count();
                let stale = prev.len() - kept;
                let kept_tokens: u32 = selected.iter().filter(|i| prev_set.contains(&i.id)).map(|i| i.tokens).sum();
                let delta_tokens: u32 = selected.iter().filter(|i| !prev_set.contains(&i.id)).map(|i| i.tokens).sum();
                let (uncached, cached) = input.prices;
                let reuse_cost = (cached * kept_tokens as f64 + uncached * delta_tokens as f64) / 1e6;
                let rebuild_cost = uncached * (kept_tokens + delta_tokens) as f64 / 1e6;
                let previous: Vec<String> = prev.iter().filter_map(|id| snap.get(id).map(|c| c.label())).collect();
                let (qid, q) = questions::cache_reuse();
                let state = json!({
                    "query": input.query,
                    "goal": input.goal,
                    "previous": previous,
                    "prefix_tokens": kept_tokens,
                    "delta_tokens": delta_tokens,
                    "stale": stale,
                });
                let a = jev.ask_one(state, &qid, q).await?;
                let p = a.as_noul().unwrap_or(0.0);
                Some(CacheDecision { reuse: p >= 0.5 && kept > 0, p, kept, delta_tokens, stale, reuse_cost, rebuild_cost })
            }
            _ => None,
        };
        match &cache {
            Some(cd) if cd.reuse => {
                let prev = input.previous_order.unwrap();
                let pos: HashMap<ChunkId, usize> = prev.iter().enumerate().map(|(i, id)| (*id, i)).collect();
                selected.sort_by(|a, b| {
                    let ra = if a.pinned { (0, 0, a.seq) } else { pos.get(&a.id).map(|p| (1, *p, 0)).unwrap_or((2, 0, a.seq)) };
                    let rb = if b.pinned { (0, 0, b.seq) } else { pos.get(&b.id).map(|p| (1, *p, 0)).unwrap_or((2, 0, b.seq)) };
                    ra.cmp(&rb)
                });
            }
            _ => selected.sort_by(|a, b| (!a.pinned, a.seq).cmp(&(!b.pinned, b.seq))),
        }

        Ok(Context { items: selected, tokens: total, budget, scored: cands.len() - memo_hits, hidden, dropped, cache, memo_hits, new_chunks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jev::local::LocalTransport;
    use crate::state::ChunkStore;

    #[tokio::test]
    async fn assembles_within_budget_and_hides_irrelevant() {
        let mut store = ChunkStore::ephemeral();
        store.append(Chunk::new(Kind::UserTurn, "fix the login bug in auth session handling", 1, "s")).unwrap();
        store
            .append(Chunk::new(Kind::ToolOutput { call: ChunkId([0; 32]), tool: "grep".into(), access: crate::state::Access::Read, ok: true }, "auth/session.ts:12: validateSession(token)\nauth/session.ts:40: expire login", 1, "s"))
            .unwrap();
        store.append(Chunk::new(Kind::ToolOutput { call: ChunkId([1; 32]), tool: "cat".into(), access: crate::state::Access::Read, ok: true }, "unrelated recipe for pancakes with maple syrup and butter", 1, "old")).unwrap();
        let snap = store.snapshot();
        let jev = Jev::new(Arc::new(LocalTransport), "local");
        let asm = Assembler::new(BudgetConfig { context_tokens: 400, ..Default::default() }, Arc::new(TruncateSummarizer));
        let ctx = asm
            .build(AssembleInput { snap: &snap, goal: "fix login", query: "fix the login bug in auth session", session: "s", turn: 2, pinned: vec![], previous_order: None, rescore_all: false, prices: (5.0, 0.5) }, &jev)
            .await
            .unwrap();
        assert!(ctx.tokens <= 400);
        assert!(ctx.items.iter().any(|i| i.kind == "user"));
        assert!(!ctx.items.iter().any(|i| i.text.contains("pancakes") && i.visibility == Visibility::Full));
    }

    #[tokio::test]
    async fn older_chunks_hit_the_memo_on_the_next_turn() {
        let mut store = ChunkStore::ephemeral();
        for i in 0..6 {
            store.append(Chunk::new(Kind::UserTurn, format!("earlier note {i} about login sessions"), 1, "s")).unwrap();
        }
        let snap = store.snapshot();
        let jev = Jev::new(Arc::new(LocalTransport), "local");
        let asm = Assembler::new(BudgetConfig { recent_turns: 1, ..Default::default() }, Arc::new(TruncateSummarizer));
        let input = |turn| AssembleInput { snap: &snap, goal: "fix login", query: "fix login", session: "s", turn, pinned: vec![], previous_order: None, rescore_all: false, prices: (5.0, 0.5) };
        let first = asm.build(input(10), &jev).await.unwrap();
        assert_eq!(first.memo_hits, 0);
        let second = asm.build(input(11), &jev).await.unwrap();
        assert_eq!(second.memo_hits, 6);
        assert_eq!(second.scored, 0);
        let mut forced = input(12);
        forced.rescore_all = true;
        let third = asm.build(forced, &jev).await.unwrap();
        assert_eq!(third.memo_hits, 0);
    }
}
