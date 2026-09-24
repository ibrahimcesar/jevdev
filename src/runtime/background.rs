//! Background processing on explicit state.
//!
//! After every write the runtime runs one retrieval pass, finding the files,
//! chunks and symbols related to the change, and publishes it on a watch
//! channel. Every read-only background task consumes the same pass instead of
//! repeating it. The bundled task keeps `.jevdev/progress.md` current so a long
//! run can be checked from anywhere.

use crate::state::{ChunkId, Snapshot};
use chrono::Utc;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::watch;

#[derive(Clone, Debug, Default)]
pub struct Retrieval {
    pub session: String,
    pub turn: u32,
    pub goal: String,
    pub changed: Vec<PathBuf>,
    /// Chunks that mention the changed files, newest first.
    pub related: Vec<(ChunkId, String)>,
    pub last_action: String,
}

/// The shared retrieval pass: done once per write, read by every subscriber.
pub fn retrieve(snap: &Snapshot, session: &str, turn: u32, goal: &str, changed: Vec<PathBuf>, last_action: &str) -> Retrieval {
    let names: Vec<String> = changed.iter().filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string())).collect();
    let mut related: Vec<(ChunkId, String)> = snap
        .iter()
        .rev()
        .filter(|c| c.paths.iter().any(|p| changed.iter().any(|q| p.ends_with(q) || q.ends_with(p))) || names.iter().any(|n| !n.is_empty() && c.body.contains(n.as_str())))
        .take(12)
        .map(|c| (c.id, c.label()))
        .collect();
    related.dedup();
    Retrieval { session: session.into(), turn, goal: goal.into(), changed, related, last_action: last_action.into() }
}

pub fn channel() -> (watch::Sender<Arc<Retrieval>>, watch::Receiver<Arc<Retrieval>>) {
    watch::channel(Arc::new(Retrieval::default()))
}

/// Writes a small progress page after each retrieval pass.
pub fn spawn_progress_writer(mut rx: watch::Receiver<Arc<Retrieval>>, path: PathBuf) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while rx.changed().await.is_ok() {
            let r = rx.borrow().clone();
            if r.turn == 0 {
                continue;
            }
            let mut s = String::new();
            s.push_str(&format!("# jevdev progress\n\nupdated {} · session `{}` · turn {}\n\n", Utc::now().to_rfc3339(), r.session, r.turn));
            s.push_str(&format!("## Goal\n\n{}\n\n## Last action\n\n{}\n\n", r.goal, r.last_action));
            s.push_str("## Changed files\n\n");
            for p in &r.changed {
                s.push_str(&format!("- `{}`\n", p.display()));
            }
            s.push_str("\n## Related chunks\n\n");
            for (id, label) in &r.related {
                s.push_str(&format!("- `{}` {label}\n", id.short()));
            }
            if let Some(parent) = Path::new(&path).parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let _ = tokio::fs::write(&path, s).await;
        }
    })
}
