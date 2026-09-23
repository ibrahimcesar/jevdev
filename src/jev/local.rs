//! Offline heuristic transport.
//!
//! This is not Jev. It answers the harness's questions with keyword overlap and
//! simple rules so the harness runs end to end without a key, and so tests are
//! deterministic. It recognises the question ids that [`super::questions`] emits
//! and falls back to uniform answers for anything else.

use super::{Answer, Question, SystemOneRequest, SystemOneResponse, Transport, Usage};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

#[derive(Default)]
pub struct LocalTransport;

const STOP: &[&str] = &[
    "the", "and", "for", "with", "that", "this", "from", "into", "what", "which", "where", "when", "how", "are", "was",
    "were", "will", "should", "would", "could", "have", "has", "not", "you", "your", "our", "its", "then", "than",
    "also", "just", "about", "does", "did", "can", "all", "any", "one", "two", "use", "using", "used",
];

pub fn terms(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| t.len() >= 3 && !STOP.contains(&t.as_str()))
        .collect()
}

/// Jaccard-style overlap in [0, 1], weighted toward the smaller set so a short
/// query against a long chunk can still score high.
pub fn overlap(a: &str, b: &str) -> f64 {
    let ta = terms(a);
    let tb = terms(b);
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f64;
    inter / (ta.len().min(tb.len()) as f64)
}

fn choice(choice: &str, p: f64, options: &BTreeMap<String, Option<String>>) -> Answer {
    let n = options.len().max(1);
    let rest = if n > 1 { (1.0 - p) / (n as f64 - 1.0) } else { 0.0 };
    let probabilities = options.keys().map(|k| (k.clone(), if k == choice { p } else { rest })).collect();
    Answer::Choice { choice: choice.to_string(), confidence: p, probabilities }
}

fn score(level: usize, p: f64, levels: &[String]) -> Answer {
    let n = levels.len().max(1);
    let rest = if n > 1 { (1.0 - p) / (n as f64 - 1.0) } else { 0.0 };
    let probabilities: BTreeMap<String, f64> = (0..n).map(|i| (i.to_string(), if i == level { p } else { rest })).collect();
    let score = probabilities.iter().map(|(k, v)| k.parse::<f64>().unwrap_or(0.0) * v).sum();
    let legend = levels.iter().enumerate().map(|(i, l)| (i.to_string(), l.clone())).collect();
    Answer::Score { score, confidence: p, probabilities, legend }
}

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

fn answer(id: &str, q: &Question, state: &Value) -> Answer {
    let query = format!("{} {}", s(state, "query"), s(state, "goal"));
    match q {
        Question::Choice { criteria, .. } => {
            if let Some(hex) = id.strip_prefix(super::questions::VIS_PREFIX) {
                let turn = state.get("turn").and_then(Value::as_u64).unwrap_or(0);
                let chunk = state
                    .get("chunks")
                    .and_then(Value::as_array)
                    .and_then(|a| a.iter().find(|c| s(c, "id") == hex));
                let Some(c) = chunk else { return choice("short", 0.4, criteria) };
                let kind = s(c, "kind");
                let ct = c.get("turn").and_then(Value::as_u64).unwrap_or(0);
                let pinned = c.get("pinned").and_then(Value::as_bool).unwrap_or(false);
                let ov = overlap(&query, s(c, "preview"));
                let (v, p) = if pinned || kind == "instruction" {
                    ("full", 0.98)
                } else if kind == "user" && ct + 1 >= turn {
                    ("full", 0.95)
                } else if ov >= 0.25 || ct == turn {
                    ("full", 0.55 + ov.min(0.4))
                } else if ov >= 0.1 || ct + 1 >= turn {
                    ("long", 0.6)
                } else if ov >= 0.03 || ct + 3 >= turn {
                    ("short", 0.6)
                } else {
                    ("hide", 0.7)
                };
                return choice(v, p, criteria);
            }
            match id {
                "route" => {
                    let costs = state.get("costs").and_then(Value::as_object);
                    let best = costs
                        .and_then(|m| {
                            m.iter()
                                .filter(|(k, _)| criteria.contains_key(*k))
                                .min_by(|a, b| a.1.as_f64().unwrap_or(f64::MAX).partial_cmp(&b.1.as_f64().unwrap_or(f64::MAX)).unwrap())
                                .map(|(k, _)| k.clone())
                        })
                        .or_else(|| criteria.keys().next().cloned())
                        .unwrap_or_default();
                    choice(&best, 0.7, criteria)
                }
                "tool" => {
                    let intent = s(state, "intent");
                    let mut best = (String::new(), -1.0);
                    for (k, d) in criteria {
                        let text = format!("{} {}", k.replace('_', " "), d.clone().unwrap_or_default());
                        let mut sc = overlap(intent, &text);
                        if intent.to_ascii_lowercase().contains(&k.replace('_', " ")) || intent.to_ascii_lowercase().contains(k.as_str()) {
                            sc += 0.5;
                        }
                        if sc > best.1 {
                            best = (k.clone(), sc);
                        }
                    }
                    let p = (0.35 + best.1).min(0.95);
                    choice(&best.0, p, criteria)
                }
                "permit" => {
                    let cmd = s(state, "command");
                    let access = s(state, "access");
                    let lower = cmd.to_ascii_lowercase();
                    let dangerous = ["rm -rf", "sudo", "curl", "wget", "ssh ", "scp ", "> /dev", "mkfs", "dd if=", ":(){"];
                    if dangerous.iter().any(|d| lower.contains(d)) {
                        choice("deny", 0.85, criteria)
                    } else if access == "read" {
                        choice("allow", 0.9, criteria)
                    } else {
                        choice("ask", 0.6, criteria)
                    }
                }
                _ => {
                    let first = criteria.keys().next().cloned().unwrap_or_default();
                    choice(&first, 1.0 / criteria.len().max(1) as f64, criteria)
                }
            }
        }
        Question::Score { criteria, .. } => {
            if id == "sensitivity" {
                let paths: Vec<String> = state
                    .get("paths")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).map(|p| p.to_ascii_lowercase()).collect())
                    .unwrap_or_default();
                let text = format!("{} {}", s(state, "subtask").to_ascii_lowercase(), paths.join(" "));
                let restricted = [".env", "secret", "infra/", ".pem", ".key", "id_rsa", "credential", "token", "password"];
                let open = ["docs/", "readme", ".md", "license", "changelog"];
                let level = if restricted.iter().any(|r| text.contains(r)) {
                    2
                } else if !paths.is_empty() && paths.iter().all(|p| open.iter().any(|o| p.contains(o))) {
                    0
                } else {
                    1
                };
                return score(level.min(criteria.len().saturating_sub(1)), 0.75, criteria);
            }
            score(criteria.len() / 2, 0.5, criteria)
        }
        Question::Noul { .. } => {
            let p = match id {
                "reuse" => {
                    let delta = state.get("delta_tokens").and_then(Value::as_f64).unwrap_or(0.0);
                    let prefix = state.get("prefix_tokens").and_then(Value::as_f64).unwrap_or(1.0).max(1.0);
                    let stale = state.get("stale").and_then(Value::as_f64).unwrap_or(0.0);
                    (1.0 - (delta / prefix).min(1.0) * 0.6 - (stale * 0.08).min(0.4)).clamp(0.05, 0.98)
                }
                "egress" => {
                    let script = s(state, "script").to_ascii_lowercase();
                    let hits = ["curl ", "wget ", "http://", "https://", "requests.", "urllib", "fetch(", "socket", "ssh ", "scp ", "nc ", "netcat", "axios", "reqwest", "net/http"];
                    if hits.iter().any(|h| script.contains(h)) {
                        0.92
                    } else {
                        0.06
                    }
                }
                "dup" => overlap(s(state, "new"), s(state, "existing")).clamp(0.02, 0.98),
                _ if id.starts_with("cond:") => {
                    let text = format!("{} {}", query, s(state, "paths"));
                    (0.15 + overlap(&q.instructions_text(), &text)).clamp(0.05, 0.95)
                }
                _ => 0.5,
            };
            Answer::Noul { noul: p }
        }
    }
}

#[async_trait]
impl Transport for LocalTransport {
    async fn system_one(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        let answers = req.questions.iter().map(|(id, q)| (id.clone(), answer(id, q, &req.state))).collect();
        Ok(SystemOneResponse {
            model: "local-heuristics".into(),
            answers,
            usage: Usage { input_tokens: super::value_tokens(&req.state) as u64, output_tokens: 0 },
        })
    }
    fn name(&self) -> &'static str {
        "local"
    }
}
