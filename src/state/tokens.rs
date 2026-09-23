//! Token estimation.
//!
//! A byte heuristic is enough for budgeting and cost lines. Swap in the
//! provider's `count_tokens` endpoint if exact accounting matters.

/// Roughly 3.8 bytes per token for mixed code and prose.
pub fn count(text: &str) -> u32 {
    ((text.len() as f64) / 3.8).ceil() as u32
}

pub fn fmt_k(tokens: u32) -> String {
    if tokens >= 1000 {
        format!("{:.1}k", tokens as f64 / 1000.0)
    } else {
        tokens.to_string()
    }
}
