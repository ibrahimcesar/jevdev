//! Harness configuration, loaded from `jevdev.toml` in the repository root.

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CONFIG_FILE: &str = "jevdev.toml";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub jev: JevConfig,
    pub llm: LlmConfig,
    pub budget: BudgetConfig,
    pub models: Vec<ModelSpec>,
    pub security: SecurityConfig,
    pub policy: PolicyConfig,
    pub session: SessionConfig,
}

/// How the harness reaches Jev.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct JevConfig {
    /// `auto` uses HTTP when `TYPESAFE_API_KEY` is set and the local heuristic
    /// transport otherwise. `http` and `local` force one.
    pub transport: String,
    pub endpoint: String,
    pub model: String,
    /// Price per million input tokens, used only for the cost line.
    pub price_per_mtok: f64,
}

impl Default for JevConfig {
    fn default() -> Self {
        Self {
            transport: "auto".into(),
            endpoint: "https://api.typesafe.ai".into(),
            model: "jev-latest".into(),
            price_per_mtok: 0.042,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// `anthropic` or `scripted`.
    pub provider: String,
    pub frontier: String,
    pub worker: String,
    pub cheap: String,
    pub max_tokens: u32,
    /// `low` | `medium` | `high` | `xhigh` | `max`
    pub effort: String,
    /// Send the server-side refusal fallback parameter on frontier calls.
    pub fallbacks: bool,
    /// Let the router move a turn off the frontier model when Jev and the cost
    /// model agree. When false, every main-loop call stays on `frontier`.
    pub routing: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "anthropic".into(),
            frontier: "claude-opus-5".into(),
            worker: "claude-sonnet-5".into(),
            cheap: "claude-haiku-4-5".into(),
            max_tokens: 16000,
            effort: "high".into(),
            fallbacks: true,
            routing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BudgetConfig {
    /// Token budget for the assembled context sent to the model each turn.
    pub context_tokens: u32,
    /// Chunks from this many most recent turns always reach Jev for scoring.
    pub recent_turns: u32,
    /// Older chunks that survive the deterministic prefilter and reach Jev.
    pub candidates: usize,
    /// Characters of each chunk shown to Jev when scoring visibility.
    pub preview_chars: usize,
    /// Approximate token ceiling for one Jev request's state.
    pub jev_state_tokens: u32,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            context_tokens: 60_000,
            recent_turns: 3,
            candidates: 120,
            preview_chars: 600,
            jev_state_tokens: 20_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Trust {
    /// First-party frontier API. Eligible for restricted data.
    FirstParty,
    /// Vetted provider. Eligible for application code.
    Vetted,
    /// Any provider, cheapest first. Public data only.
    Open,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    Frontier,
    Worker,
    Cheap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub provider: String,
    /// USD per million tokens.
    pub input: f64,
    pub cached: f64,
    pub output: f64,
    pub trust: Trust,
    pub tier: Tier,
    pub description: String,
}

impl ModelSpec {
    pub fn price(&self) -> crate::router::Price {
        crate::router::Price { input: self.input, cached: self.cached, output: self.output }
    }
}

pub fn default_models() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            id: "claude-opus-5".into(),
            provider: "anthropic".into(),
            input: 5.0,
            cached: 0.5,
            output: 25.0,
            trust: Trust::FirstParty,
            tier: Tier::Frontier,
            description: "Most capable. Plans, reviews, hard debugging, anything touching restricted files.".into(),
        },
        ModelSpec {
            id: "claude-sonnet-5".into(),
            provider: "anthropic".into(),
            input: 2.0,
            cached: 0.2,
            output: 10.0,
            trust: Trust::FirstParty,
            tier: Tier::Worker,
            description: "Fast worker for scoped subtasks with a small, purpose-built context.".into(),
        },
        ModelSpec {
            id: "claude-haiku-4-5".into(),
            provider: "anthropic".into(),
            input: 1.0,
            cached: 0.1,
            output: 5.0,
            trust: Trust::FirstParty,
            tier: Tier::Cheap,
            description: "Argument filling, summaries, and classification. Not for reasoning.".into(),
        },
    ]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    /// Globs for public data: docs, open-source deps. Any model.
    pub open: Vec<String>,
    /// Globs for secrets, env, infra config. First-party frontier only.
    pub restricted: Vec<String>,
    /// Globs for proprietary research code. Excludes `custom_excludes` vendors.
    pub custom: Vec<String>,
    pub custom_excludes: Vec<String>,
    /// Paths a command may never touch, regardless of policy.
    pub sensitive_paths: Vec<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            open: vec!["docs/**".into(), "**/*.md".into(), "LICENSE*".into()],
            restricted: vec![
                "**/.env*".into(),
                "**/secrets/**".into(),
                "infra/**".into(),
                "**/*.pem".into(),
                "**/*.key".into(),
                "**/id_rsa*".into(),
            ],
            custom: vec![],
            custom_excludes: vec![],
            sensitive_paths: vec!["~/.ssh".into(), ".env".into(), "~/.aws".into(), "~/.cargo/credentials".into()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicyConfig {
    /// Cedar policy file. Written by `jevdev init`; the built-in default is used when absent.
    pub file: String,
    /// Programs (or program + first argument) considered read-only.
    pub read_only: Vec<String>,
    /// When Cedar neither permits nor forbids, Jev's `allow` needs at least this
    /// confidence to run without asking.
    pub jev_auto_allow: f64,
    /// Value of `context.task_scope` seen by policies.
    pub task_scope: String,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            file: ".jevdev/exec.cedar".into(),
            read_only: [
                "ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "pwd", "echo", "which", "tree",
                "git status", "git log", "git diff", "git show", "git branch",
                "cargo check", "cargo build", "cargo test", "cargo fmt", "cargo clippy",
                "npm test", "pytest", "go test",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            jev_auto_allow: 0.9,
            task_scope: "develop".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionConfig {
    pub dir: String,
    pub max_turns: u32,
    /// Sub-agent turn cap.
    pub subagent_turns: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { dir: ".jevdev".into(), max_turns: 40, subagent_turns: 8 }
    }
}

impl Config {
    /// Load `jevdev.toml` from `root`, falling back to defaults.
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join(CONFIG_FILE);
        let mut cfg: Config = if path.exists() {
            let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
        } else {
            Config::default()
        };
        if cfg.models.is_empty() {
            cfg.models = default_models();
        }
        Ok(cfg)
    }

    /// The default configuration, rendered as TOML for `jevdev init`.
    pub fn default_toml() -> String {
        let mut cfg = Config::default();
        cfg.models = default_models();
        toml::to_string_pretty(&cfg).expect("default config serialises")
    }

    pub fn model(&self, id: &str) -> Option<&ModelSpec> {
        self.models.iter().find(|m| m.id == id)
    }
}
