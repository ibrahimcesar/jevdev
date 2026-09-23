//! A scripted model for offline runs and tests.
//!
//! Set `JEVDEV_SCRIPT=path` to a file whose replies are separated by lines
//! containing only `---`; each `complete` call returns the next one. Without a
//! script, it performs a small demo: list the repository, read the README, and
//! report.

use super::{Completion, LlmClient, LlmUsage, Prompt};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::sync::Mutex;

pub struct ScriptedClient {
    replies: Mutex<std::collections::VecDeque<String>>,
}

impl ScriptedClient {
    pub fn new(replies: Vec<String>) -> Self {
        Self { replies: Mutex::new(replies.into_iter().collect()) }
    }

    pub fn from_env() -> Result<Self> {
        if let Ok(path) = std::env::var("JEVDEV_SCRIPT") {
            let text = std::fs::read_to_string(&path).map_err(|e| anyhow!("reading JEVDEV_SCRIPT {path}: {e}"))?;
            let replies = text.split("\n---\n").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            return Ok(Self::new(replies));
        }
        Ok(Self::new(vec![
            "<act>list files under the repository root</act>".into(),
            "<act>read the file README.md</act>".into(),
            "<done>Demo run complete: listed the repository and read README.md. Set ANTHROPIC_API_KEY (or run `ant auth login`) and switch `llm.provider` to `anthropic` in jevdev.toml for a real model.</done>".into(),
        ]))
    }
}

#[async_trait]
impl LlmClient for ScriptedClient {
    async fn complete(&self, model: &str, prompt: &Prompt, _max_tokens: u32, _effort: &str) -> Result<Completion> {
        let next = self.replies.lock().unwrap().pop_front().unwrap_or_else(|| "<done>script exhausted</done>".into());
        let input = crate::state::tokens::count(&prompt.context) as u64 + crate::state::tokens::count(&prompt.system) as u64;
        Ok(Completion { usage: LlmUsage { input, output: crate::state::tokens::count(&next) as u64, ..Default::default() }, text: next, stop: "end_turn".into(), model: model.to_string() })
    }

    async fn small(&self, _model: &str, _system: &str, _user: &str, _max_tokens: u32) -> Result<String> {
        Err(anyhow!("scripted client has no helper model"))
    }

    fn name(&self) -> &'static str {
        "scripted"
    }
}
