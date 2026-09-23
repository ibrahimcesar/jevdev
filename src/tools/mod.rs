//! Tools with tiered disclosure.
//!
//! The model never sees a tool schema. It describes an intent in plain words;
//! Jev ranks the tier-1 snippets; the harness loads the tier-2 schema for the
//! top candidates only, fills the arguments, validates them against the
//! schema, and runs the typed call. Nothing from tiers two and three is ever
//! written into a chunk, so it cannot linger in context.

pub mod args;
pub mod builtin;

use crate::jev::{questions, Jev};
use crate::state::Access;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use args::{ArgBuilder, HeuristicArgBuilder};

/// What a call would touch, computed before it runs so policy can judge it.
#[derive(Clone, Debug, Default)]
pub struct Footprint {
    pub paths: Vec<PathBuf>,
    pub command: Option<String>,
}

/// Sub-agent hook, implemented by the runtime.
#[async_trait]
pub trait Delegator: Send + Sync {
    async fn delegate(&self, goal: &str) -> Result<String>;
}

pub struct ToolCx {
    pub root: PathBuf,
    pub delegator: Option<Arc<dyn Delegator>>,
}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub tool: String,
    pub args: Value,
    pub intent: String,
    pub access: Access,
    pub footprint: Footprint,
    /// Jev's ranking of tools for this intent.
    pub candidates: Vec<(String, f64)>,
}

impl ToolCall {
    /// One line for logs and permission prompts.
    pub fn describe(&self) -> String {
        match &self.footprint.command {
            Some(c) => format!("{} · {}", self.tool, c),
            None => format!("{} {}", self.tool, self.args),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ToolOutput {
    pub tool: String,
    pub body: String,
    pub access: Access,
    pub ok: bool,
    pub paths: Vec<PathBuf>,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn id(&self) -> &'static str;
    /// Tier 1: one line, always cheap.
    fn snippet(&self) -> &'static str;
    fn access(&self) -> Access;
    /// Tier 2: the JSON schema for the arguments, loaded on demand.
    fn schema(&self) -> Value;
    /// Tier 3: the manual.
    fn docs(&self) -> &'static str;
    fn footprint(&self, args: &Value, root: &Path) -> Footprint;
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput>;
}

#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        Self { tools }
    }

    pub fn builtin() -> Self {
        Self::new(builtin::all())
    }

    /// Only read-only tools, for sub-agents that must never contend for a lease.
    pub fn read_only(&self) -> Self {
        Self { tools: self.tools.iter().filter(|t| t.access() == Access::Read && t.id() != "delegate").cloned().collect() }
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.id() == id).cloned()
    }

    pub fn snippets(&self) -> Vec<(String, String)> {
        self.tools.iter().map(|t| (t.id().to_string(), t.snippet().to_string())).collect()
    }

    /// Route an intent to a validated, typed call.
    pub async fn route(&self, intent: &str, jev: &Jev, builder: &dyn ArgBuilder, root: &Path) -> Result<ToolCall> {
        let snippets = self.snippets();
        let (qid, q) = questions::tool_pick(&snippets);
        let answer = jev.ask_one(json!({ "intent": intent }), &qid, q).await?;
        let ranked = answer.ranked();
        let candidates: Vec<(String, f64)> = ranked.iter().take(3).cloned().collect();
        let mut last_err = None;
        for (tool_id, _) in &candidates {
            let Some(tool) = self.get(tool_id) else { continue };
            let schema = tool.schema();
            let docs = tool.docs();
            let mut prev_err: Option<String> = None;
            for _attempt in 0..2 {
                let built = builder.build(intent, tool_id, &schema, docs, prev_err.as_deref()).await;
                match built {
                    Ok(args) => match args::validate(&schema, &args) {
                        Ok(()) => {
                            let footprint = tool.footprint(&args, root);
                            return Ok(ToolCall { tool: tool_id.clone(), args, intent: intent.to_string(), access: tool.access(), footprint, candidates });
                        }
                        Err(e) => prev_err = Some(e),
                    },
                    Err(e) => prev_err = Some(e.to_string()),
                }
            }
            last_err = prev_err.map(|e| format!("{tool_id}: {e}"));
        }
        Err(anyhow!("no tool accepted the intent {intent:?}: {}", last_err.unwrap_or_else(|| "no candidates".into())))
    }

    pub async fn run(&self, call: &ToolCall, cx: &ToolCx) -> Result<ToolOutput> {
        let tool = self.get(&call.tool).ok_or_else(|| anyhow!("unknown tool {}", call.tool))?;
        tool.run(call.args.clone(), cx).await
    }
}

/// Resolve `p` against `root` and say whether it stays inside.
pub fn resolve(root: &Path, p: &str) -> (PathBuf, bool) {
    let expanded = if let Some(rest) = p.strip_prefix("~/") {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(rest)
    } else {
        PathBuf::from(p)
    };
    let joined = if expanded.is_absolute() { expanded } else { root.join(expanded) };
    let norm = normalize(&joined);
    let root_n = normalize(root);
    let inside = norm.starts_with(&root_n);
    (norm, inside)
}

fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}
