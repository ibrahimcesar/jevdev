//! Sub-goal deduplication and per-path leases for parallel work.

use crate::jev::{questions, Jev};
use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub enum Status {
    Running,
    Done(String),
}

#[derive(Clone, Debug)]
pub struct Subgoal {
    pub id: usize,
    pub goal: String,
    pub status: Status,
}

/// Every sub-goal is registered before launch and checked against all earlier
/// ones, so work already done or in flight is never launched twice.
#[derive(Default)]
pub struct SubgoalRegistry {
    goals: Vec<Subgoal>,
}

impl SubgoalRegistry {
    pub fn all(&self) -> &[Subgoal] {
        &self.goals
    }

    /// An existing sub-goal that would be duplicated by `goal`, if any.
    pub async fn duplicate_of(&self, goal: &str, jev: &Jev) -> Result<Option<Subgoal>> {
        let mut best: Option<(f64, &Subgoal)> = None;
        for g in &self.goals {
            let s = crate::jev::local::overlap(goal, &g.goal);
            if best.map(|(b, _)| s > b).unwrap_or(true) {
                best = Some((s, g));
            }
        }
        let Some((s, g)) = best else { return Ok(None) };
        if s >= 0.85 {
            return Ok(Some(g.clone()));
        }
        if s >= 0.3 {
            let (id, q) = questions::duplicate();
            let a = jev.ask_one(json!({ "new": goal, "existing": g.goal, "existing_status": format!("{:?}", g.status) }), &id, q).await?;
            if a.as_noul().unwrap_or(0.0) >= 0.7 {
                return Ok(Some(g.clone()));
            }
        }
        Ok(None)
    }

    pub fn register(&mut self, goal: &str) -> usize {
        let id = self.goals.len();
        self.goals.push(Subgoal { id, goal: goal.to_string(), status: Status::Running });
        id
    }

    pub fn finish(&mut self, id: usize, result: &str) {
        if let Some(g) = self.goals.get_mut(id) {
            g.status = Status::Done(result.to_string());
        }
    }
}

/// Per-path write leases. Read-only work never takes one.
#[derive(Default)]
pub struct LeaseTable {
    map: Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>,
}

impl LeaseTable {
    pub fn lease(&self, p: &Path) -> Arc<tokio::sync::Mutex<()>> {
        self.map.lock().unwrap().entry(p.to_path_buf()).or_default().clone()
    }

    /// Acquire leases on all `paths` in sorted order (no deadlocks between writers).
    pub async fn acquire(&self, paths: &[PathBuf]) -> Vec<tokio::sync::OwnedMutexGuard<()>> {
        let mut sorted: Vec<PathBuf> = paths.to_vec();
        sorted.sort();
        sorted.dedup();
        let mut guards = Vec::new();
        for p in sorted {
            guards.push(self.lease(&p).lock_owned().await);
        }
        guards
    }
}
