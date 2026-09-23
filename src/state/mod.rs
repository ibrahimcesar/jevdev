//! Explicit, typed, content-addressed state.
//!
//! Everything the model has ever seen is a [`Chunk`]. Chunks are immutable and
//! append-only; how visible each one is on a given turn is a per-query decision
//! made by Jev, never a mutation of the chunk.

pub mod store;
pub mod tokens;

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

pub use store::{ChunkStore, Snapshot};

/// Content address: blake3 over the kind tag, the turn (where it matters), and the body.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId(pub [u8; 32]);

impl ChunkId {
    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }
    /// First eight hex characters, for labels.
    pub fn short(&self) -> String {
        hex::encode(&self.0[..4])
    }
    pub fn parse(s: &str) -> Option<Self> {
        let v = hex::decode(s.trim()).ok()?;
        if v.len() != 32 {
            return None;
        }
        let mut a = [0u8; 32];
        a.copy_from_slice(&v);
        Some(Self(a))
    }
}

impl fmt::Display for ChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.short())
    }
}
impl fmt::Debug for ChunkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChunkId({})", self.short())
    }
}
impl Serialize for ChunkId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.hex())
    }
}
impl<'de> Deserialize<'de> for ChunkId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        ChunkId::parse(&s).ok_or_else(|| serde::de::Error::custom("bad chunk id"))
    }
}

/// Whether an operation only reads state or also writes it. Read-only work
/// never contends for a lease, which is what makes parallel sub-agents cheap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Access {
    Read,
    Write,
}

impl fmt::Display for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Access::Read => "read",
            Access::Write => "write",
        })
    }
}

/// The visibility ladder. One chunk can sit on a different rung for every query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    Hide,
    Short,
    Long,
    Full,
}

impl Visibility {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "hide" => Some(Self::Hide),
            "short" => Some(Self::Short),
            "long" => Some(Self::Long),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hide => "hide",
            Self::Short => "short",
            Self::Long => "long",
            Self::Full => "full",
        }
    }
}

impl fmt::Display for Visibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Data-sensitivity tier, from Table IV of the design notes. Ordered so the
/// most restrictive tier across a set of paths wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sensitivity {
    /// Public docs, open-source deps: any model, cheapest first.
    Open,
    /// Application code: vetted providers.
    Standard,
    /// Secrets, env, infra config: first-party frontier only.
    Restricted,
    /// Proprietary research code: excludes named vendors.
    Custom,
}

impl fmt::Display for Sensitivity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Open => "open",
            Self::Standard => "standard",
            Self::Restricted => "restricted",
            Self::Custom => "custom",
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Kind {
    UserTurn,
    Assistant,
    Reasoning,
    ToolCall { tool: String, args: serde_json::Value, intent: String },
    ToolOutput { call: ChunkId, tool: String, access: Access, ok: bool },
    File { path: PathBuf, rev: String },
    Diff { path: PathBuf },
    /// A conditional instruction fragment. Pinned fragments are always shown in
    /// full while their condition holds; they are reloaded from state each turn,
    /// so nothing can summarise them away.
    Instruction { condition: String, pinned: bool },
    /// A derived summary of another chunk at one visibility level. Generated
    /// once, kept forever, never scored on its own.
    Summary { of: ChunkId, level: Visibility },
    SubagentResult { goal: String, model: String },
}

impl Kind {
    pub fn tag(&self) -> &'static str {
        match self {
            Kind::UserTurn => "user",
            Kind::Assistant => "assistant",
            Kind::Reasoning => "reasoning",
            Kind::ToolCall { .. } => "tool_call",
            Kind::ToolOutput { .. } => "tool_output",
            Kind::File { .. } => "file",
            Kind::Diff { .. } => "diff",
            Kind::Instruction { .. } => "instruction",
            Kind::Summary { .. } => "summary",
            Kind::SubagentResult { .. } => "subagent",
        }
    }
    /// Instruction and file chunks are addressed by content alone, so re-adding
    /// them is a no-op. Everything else folds the turn in.
    fn turn_sensitive(&self) -> bool {
        !matches!(self, Kind::Instruction { .. } | Kind::File { .. } | Kind::Summary { .. })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub kind: Kind,
    pub body: String,
    pub tokens: u32,
    pub turn: u32,
    pub session: String,
    pub sensitivity: Sensitivity,
    /// Repository paths this chunk references, for lease and sensitivity checks.
    pub paths: Vec<PathBuf>,
    pub created: i64,
}

impl Chunk {
    pub fn new(kind: Kind, body: impl Into<String>, turn: u32, session: &str) -> Self {
        let body = body.into();
        let mut h = blake3::Hasher::new();
        h.update(kind.tag().as_bytes());
        h.update(b"\0");
        if kind.turn_sensitive() {
            h.update(&turn.to_le_bytes());
            h.update(session.as_bytes());
        }
        if let Kind::Summary { of, level } = &kind {
            h.update(&of.0);
            h.update(level.as_str().as_bytes());
        }
        if let Kind::Instruction { condition, .. } = &kind {
            h.update(condition.as_bytes());
        }
        h.update(b"\0");
        h.update(body.as_bytes());
        let tokens = tokens::count(&body);
        Self {
            id: ChunkId(*h.finalize().as_bytes()),
            kind,
            body,
            tokens,
            turn,
            session: session.to_string(),
            sensitivity: Sensitivity::Standard,
            paths: Vec::new(),
            created: chrono::Utc::now().timestamp(),
        }
    }

    pub fn with_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.paths = paths;
        self
    }

    pub fn with_sensitivity(mut self, s: Sensitivity) -> Self {
        self.sensitivity = s;
        self
    }

    pub fn pinned(&self) -> bool {
        matches!(self.kind, Kind::Instruction { pinned: true, .. })
    }

    /// The first `n` characters, single-spaced, for Jev previews and UI rows.
    pub fn preview(&self, n: usize) -> String {
        let mut out = String::with_capacity(n.min(self.body.len()) + 1);
        let mut last_ws = false;
        for ch in self.body.chars() {
            if out.len() >= n {
                out.push('…');
                break;
            }
            if ch.is_whitespace() {
                if !last_ws {
                    out.push(' ');
                }
                last_ws = true;
            } else {
                out.push(ch);
                last_ws = false;
            }
        }
        out
    }

    /// A short human label: `tool_output grep · turn 4`.
    pub fn label(&self) -> String {
        let what = match &self.kind {
            Kind::UserTurn => "user turn".to_string(),
            Kind::Assistant => "assistant".to_string(),
            Kind::Reasoning => "reasoning".to_string(),
            Kind::ToolCall { tool, .. } => format!("call {tool}"),
            Kind::ToolOutput { tool, access, .. } => format!("{tool} output · {access}"),
            Kind::File { path, .. } => format!("file {}", path.display()),
            Kind::Diff { path } => format!("diff {}", path.display()),
            Kind::Instruction { condition, .. } => format!("instruction [{condition}]"),
            Kind::Summary { of, level } => format!("{level} summary of {of}"),
            Kind::SubagentResult { goal, .. } => format!("sub-agent: {}", truncate(goal, 40)),
        };
        format!("{what} · turn {}", self.turn)
    }
}

pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}
