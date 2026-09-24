//! Events the runtime emits for the TUI, the plain printer, and tests.

use crate::llm::LlmUsage;
use crate::state::Visibility;
use tokio::sync::{mpsc, oneshot};

#[derive(Clone, Debug)]
pub struct ContextRow {
    pub id: String,
    pub label: String,
    pub visibility: Visibility,
    pub tokens: u32,
    pub p: f64,
    pub pinned: bool,
}

#[derive(Clone, Debug, Default)]
pub struct UsageSnapshot {
    pub turns: u32,
    pub llm: LlmUsage,
    pub llm_cost: f64,
    pub jev_calls: u64,
    pub jev_questions: u64,
    pub jev_tokens: u64,
    pub jev_cost: f64,
    pub memo_hits: u64,
}

pub enum Event {
    Info(String),
    TurnStart { session: String, turn: u32, query: String },
    /// A Jev decision at one of the six points.
    Decision { point: &'static str, detail: String, p: f64 },
    Context { rows: Vec<ContextRow>, tokens: u32, budget: u32, scored: usize, memo_hits: usize, hidden: usize, dropped: usize, reused: Option<bool> },
    Route { model: String, est: f64, p: f64, reason: String, alternatives: Vec<(String, f64)> },
    /// Prose the model wrote before an action.
    Note { model: String, text: String },
    ToolPicked { tool: String, intent: String, candidates: Vec<(String, f64)>, args: String },
    Permit { verdict: String, reasons: Vec<String>, call: String },
    ToolRan { tool: String, access: String, ok: bool, tokens: u32, preview: String },
    Subagent { id: String, goal: String, status: String },
    Done { session: String, answer: String },
    Error(String),
    Usage(UsageSnapshot),
    /// A permission question for the human. Reply `true` to run.
    Ask { call: String, reasons: Vec<String>, reply: oneshot::Sender<bool> },
}

impl Event {
    /// A JSON object for `run --json` and logs. The `ask` reply channel is not included.
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            Event::Info(s) => json!({ "type": "info", "text": s }),
            Event::TurnStart { session, turn, query } => json!({ "type": "turn", "session": session, "turn": turn, "query": query }),
            Event::Decision { point, detail, p } => json!({ "type": "decision", "point": point, "detail": detail, "p": p }),
            Event::Context { rows, tokens, budget, scored, memo_hits, hidden, dropped, reused } => json!({
                "type": "context", "tokens": tokens, "budget": budget, "scored": scored, "memo_hits": memo_hits, "hidden": hidden, "dropped": dropped, "reused": reused,
                "rows": rows.iter().map(|r| json!({ "id": r.id, "label": r.label, "visibility": r.visibility.as_str(), "tokens": r.tokens, "p": r.p, "pinned": r.pinned })).collect::<Vec<_>>(),
            }),
            Event::Route { model, est, p, reason, alternatives } => json!({ "type": "route", "model": model, "est": est, "p": p, "reason": reason, "alternatives": alternatives.iter().map(|(m, c)| json!({ "model": m, "est": c })).collect::<Vec<_>>() }),
            Event::Note { model, text } => json!({ "type": "note", "model": model, "text": text }),
            Event::ToolPicked { tool, intent, candidates, args } => json!({ "type": "tool_picked", "tool": tool, "intent": intent, "args": args, "candidates": candidates.iter().map(|(t, p)| json!({ "tool": t, "p": p })).collect::<Vec<_>>() }),
            Event::Permit { verdict, reasons, call } => json!({ "type": "permit", "verdict": verdict, "reasons": reasons, "call": call }),
            Event::ToolRan { tool, access, ok, tokens, preview } => json!({ "type": "tool_ran", "tool": tool, "access": access, "ok": ok, "tokens": tokens, "preview": preview }),
            Event::Subagent { id, goal, status } => json!({ "type": "subagent", "id": id, "goal": goal, "status": status }),
            Event::Done { session, answer } => json!({ "type": "done", "session": session, "answer": answer }),
            Event::Error(s) => json!({ "type": "error", "text": s }),
            Event::Usage(u) => json!({
                "type": "usage", "turns": u.turns,
                "llm": { "input": u.llm.input, "cached_read": u.llm.cached_read, "cache_write": u.llm.cache_write, "output": u.llm.output, "cost": u.llm_cost },
                "jev": { "calls": u.jev_calls, "questions": u.jev_questions, "input_tokens": u.jev_tokens, "cost": u.jev_cost, "memo_hits": u.memo_hits },
            }),
            Event::Ask { call, reasons, .. } => json!({ "type": "ask", "call": call, "reasons": reasons }),
        }
    }
}

pub type EventSink = mpsc::UnboundedSender<Event>;

pub fn emit(sink: &EventSink, e: Event) {
    let _ = sink.send(e);
}

/// Print events as plain lines, for `jevdev run` and logs.
pub fn print_plain(e: &Event) {
    match e {
        Event::Info(s) => println!("· {s}"),
        Event::TurnStart { session, turn, query } => println!("\n── turn {turn} · session {session} ──\n  {}", crate::state::truncate(query, 120)),
        Event::Decision { point, detail, p } => println!("  jev {point:<12} {detail}  (p={p:.2})"),
        Event::Context { rows, tokens, budget, scored, memo_hits, hidden, dropped, reused } => {
            println!("  context {tokens}/{budget} tok · {scored} scored by jev · {memo_hits} from memo · {hidden} hidden · {dropped} dropped{}", match reused { Some(true) => " · prefix reused", Some(false) => " · rebuilt", None => "" });
            for r in rows {
                println!("    {:<5} {:>6} tok  p={:.2}  {}{}", r.visibility, r.tokens, r.p, if r.pinned { "📌 " } else { "" }, r.label);
            }
        }
        Event::Route { model, est, p, reason, .. } => println!("  route → {model}  est ${est:.4}  p={p:.2}  {reason}"),
        Event::Note { model, text } => println!("  {model}: {text}"),
        Event::ToolPicked { tool, intent, candidates, args } => {
            let c: Vec<String> = candidates.iter().map(|(t, p)| format!("{t} {p:.2}")).collect();
            println!("  intent: {}\n  tool → {tool} {args}  [{}]", crate::state::truncate(intent, 160), c.join(", "));
        }
        Event::Permit { verdict, reasons, call } => println!("  permit {verdict}  {call}  [{}]", reasons.join("; ")),
        Event::ToolRan { tool, access, ok, tokens, preview } => println!("  {tool} ({access}) {} {tokens} tok\n    {}", if *ok { "ok" } else { "failed" }, preview.replace('\n', "\n    ")),
        Event::Subagent { id, goal, status } => println!("  sub-agent {id} [{status}] {}", crate::state::truncate(goal, 100)),
        Event::Done { answer, .. } => println!("\n{answer}\n"),
        Event::Error(s) => eprintln!("error: {s}"),
        Event::Usage(u) => println!(
            "  usage: llm in {} (cached {}) out {} ${:.4} · jev {} calls / {} questions / {} tok ${:.4} · memo hits {}",
            u.llm.input, u.llm.cached_read, u.llm.output, u.llm_cost, u.jev_calls, u.jev_questions, u.jev_tokens, u.jev_cost, u.memo_hits
        ),
        Event::Ask { call, reasons, .. } => println!("  ask: {call}  [{}]", reasons.join("; ")),
    }
}
