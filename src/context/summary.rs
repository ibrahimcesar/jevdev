//! Summaries for the visibility ladder.
//!
//! A summary is generated once per (chunk, level) and stored as a chunk, so the
//! ladder never re-pays for the same compression.

use crate::state::{Chunk, Visibility};
use anyhow::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, chunk: &Chunk, level: Visibility) -> Result<String>;
}

/// Head-and-tail truncation with a line count. Cheap, deterministic, and good
/// enough for logs and search output; wrap an LLM for prose.
pub struct TruncateSummarizer;

pub fn target_chars(level: Visibility) -> usize {
    match level {
        Visibility::Short => 240,
        Visibility::Long => 1400,
        Visibility::Full | Visibility::Hide => usize::MAX,
    }
}

#[async_trait]
impl Summarizer for TruncateSummarizer {
    async fn summarize(&self, chunk: &Chunk, level: Visibility) -> Result<String> {
        let n = target_chars(level);
        let body = chunk.body.trim();
        if body.chars().count() <= n {
            return Ok(body.to_string());
        }
        let lines = body.lines().count();
        let head_n = n * 2 / 3;
        let tail_n = n - head_n;
        let head: String = body.chars().take(head_n).collect();
        let tail: String = {
            let v: Vec<char> = body.chars().collect();
            v[v.len().saturating_sub(tail_n)..].iter().collect()
        };
        Ok(format!("{head}\n[… {lines} lines, {} chars; {} view of {} …]\n{tail}", body.len(), level, chunk.label()))
    }
}
