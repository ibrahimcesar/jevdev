//! # jevdev
//!
//! A coding-agent harness built around [Jev](https://docs.typesafe.ai), TypeSafe's
//! System One decision model.
//!
//! The harness keeps all agent state as explicit, typed, content-addressed chunks
//! ([`state`]) and asks Jev the per-turn questions a coding agent normally leaves
//! to defaults ([`jev::questions`]):
//!
//! | Decision point | Question | Typed answer |
//! |---|---|---|
//! | Context | How visible should this chunk be for this query? | hide / short / long / full |
//! | Cache | Reuse the cached prefix or rebuild? | noul + probability |
//! | Routing | Can this subtask leave the frontier model? | choice + cost estimate |
//! | Tools | Which tool fits this intent? | ranked choice, top-k |
//! | Permissions | Should this command run? | allow / ask / deny |
//! | Security | Which files will this task touch? | sensitivity score |
//!
//! Frontier models ([`llm`]), tools ([`tools`]), and deterministic code do the
//! actual work. Context is assembled per query ([`context`]), never appended;
//! routing is priced per context rebuild ([`router`]); commands pass a Cedar
//! policy plus Jev deep inspection ([`policy`]); and the turn loop lives in
//! [`runtime`], with a terminal UI in [`tui`].

pub mod config;
pub mod context;
pub mod jev;
pub mod llm;
pub mod policy;
pub mod router;
pub mod runtime;
pub mod state;
pub mod tools;
pub mod tui;

/// Crate version, as compiled.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
