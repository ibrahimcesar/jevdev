//! Frontier and helper models.
//!
//! The model never receives tool schemas. It sees the assembled context and
//! answers with either one `<act>` block, an intent in plain words that the
//! harness routes to a tool, or a `<done>` block with its final answer.

pub mod anthropic;
pub mod scripted;

use crate::context::Summarizer;
use crate::state::{Chunk, Visibility};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

pub const SYSTEM_PROMPT: &str = "You are a coding agent working inside a repository through a harness.\n\
You do not call tools directly and you never see tool schemas. When you need to act, write exactly one block:\n\n\
<act>what you want to do, in plain words, with every concrete value it needs: the path, the regex, the exact shell command, or the exact text to insert or replace (quote literal text)</act>\n\n\
and stop. The harness picks the tool, checks permissions, runs it, and shows you the result next turn as a chunk. One action per turn.\n\n\
When the goal is complete, or you cannot make progress, write:\n\n\
<done>your final answer for the user</done>\n\n\
Capabilities the harness can map your words to: read a file (optionally a line range); list files under a directory with an optional glob; search file contents with a regex; write a whole file; replace one exact string in a file; run a shell command; delegate a read-only research sub-task to a helper agent.\n\n\
Rules: read and search before editing; make small exact edits; run the tests when they exist; never touch credentials or environment files; never repeat an action whose result is already in your context. Context chunks carry an id, kind, turn and view; a \"short\" or \"long\" view is a summary of a chunk the harness chose not to show in full.";

#[derive(Clone, Debug, Default)]
pub struct Prompt {
    pub system: String,
    /// Conditional instruction fragments in force this turn.
    pub pinned: String,
    /// The rendered context.
    pub context: String,
    pub query: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LlmUsage {
    pub input: u64,
    pub cached_read: u64,
    pub cache_write: u64,
    pub output: u64,
}

impl LlmUsage {
    pub fn add(&mut self, o: &LlmUsage) {
        self.input += o.input;
        self.cached_read += o.cached_read;
        self.cache_write += o.cache_write;
        self.output += o.output;
    }
    pub fn cost(&self, p: &crate::router::Price) -> f64 {
        (p.input * self.input as f64 + p.cached * self.cached_read as f64 + p.input * 1.25 * self.cache_write as f64 + p.output * self.output as f64) / 1e6
    }
}

#[derive(Clone, Debug)]
pub struct Completion {
    pub text: String,
    pub usage: LlmUsage,
    pub stop: String,
    pub model: String,
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, model: &str, prompt: &Prompt, max_tokens: u32, effort: &str) -> Result<Completion>;
    /// A small, schema-free call for argument filling and summaries.
    async fn small(&self, model: &str, system: &str, user: &str, max_tokens: u32) -> Result<String>;
    fn name(&self) -> &'static str;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Act { intent: String, note: String },
    Done { answer: String },
}

fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<(&'a str, &'a str)> {
    let s = text.find(open)?;
    let rest = &text[s + open.len()..];
    let e = rest.find(close).unwrap_or(rest.len());
    Some((&rest[..e], &text[..s]))
}

/// Parse the model's reply. Prose without either block counts as done, so a
/// model that simply talks never loops forever.
pub fn parse_step(text: &str) -> Step {
    if let Some((inner, before)) = between(text, "<act>", "</act>") {
        return Step::Act { intent: inner.trim().to_string(), note: before.trim().to_string() };
    }
    if let Some((inner, _)) = between(text, "<done>", "</done>") {
        return Step::Done { answer: inner.trim().to_string() };
    }
    Step::Done { answer: text.trim().to_string() }
}

/// Summaries from the cheap model, for prose chunks the truncating summarizer
/// would mangle.
pub struct LlmSummarizer {
    llm: Arc<dyn LlmClient>,
    model: String,
    fallback: crate::context::TruncateSummarizer,
}

impl LlmSummarizer {
    pub fn new(llm: Arc<dyn LlmClient>, model: &str) -> Self {
        Self { llm, model: model.to_string(), fallback: crate::context::TruncateSummarizer }
    }
}

#[async_trait]
impl Summarizer for LlmSummarizer {
    async fn summarize(&self, chunk: &Chunk, level: Visibility) -> Result<String> {
        let target = crate::context::summary::target_chars(level);
        if chunk.body.chars().count() <= target {
            return Ok(chunk.body.clone());
        }
        let (what, n) = match level {
            Visibility::Short => ("one line", 60),
            _ => ("one short paragraph of the key facts, hits, errors, and conclusions", 300),
        };
        let system = format!("Summarise a coding agent's stored chunk in {what}, at most {n} words. Keep file paths, line numbers, identifiers, error messages, and numbers exact. No preamble.");
        let user = format!("Chunk ({}):\n\n{}", chunk.label(), crate::state::truncate(&chunk.body, 30_000));
        match self.llm.small(&self.model, &system, &user, 600).await {
            Ok(text) if !text.trim().is_empty() => Ok(text.trim().to_string()),
            _ => self.fallback.summarize(chunk, level).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_act_and_done() {
        assert_eq!(parse_step("Let me look.\n<act>read the file src/main.rs</act>"), Step::Act { intent: "read the file src/main.rs".into(), note: "Let me look.".into() });
        assert_eq!(parse_step("<done>All fixed.</done>"), Step::Done { answer: "All fixed.".into() });
        assert_eq!(parse_step("just prose"), Step::Done { answer: "just prose".into() });
    }
}
