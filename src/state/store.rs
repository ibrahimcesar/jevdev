//! Append-only chunk log with cheap snapshots.
//!
//! The store keeps every chunk in an on-disk `redb` table keyed by sequence
//! number and mirrors it in persistent (`im`) maps, so a [`Snapshot`] is an O(1)
//! clone that read-only tasks can hold without blocking writers. Restarting a
//! session is a new session id over the same log: old chunks stay addressable
//! and are reloaded only when Jev scores them relevant.

use super::{Chunk, ChunkId, Kind, Visibility};
use anyhow::{Context as _, Result};
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;

const CHUNKS: TableDefinition<u64, &[u8]> = TableDefinition::new("chunks");

pub struct ChunkStore {
    db: Option<Database>,
    seq: u64,
    by_seq: im::OrdMap<u64, Chunk>,
    by_id: im::HashMap<ChunkId, u64>,
}

/// A point-in-time view of the store. Cloning is O(1).
#[derive(Clone, Default)]
pub struct Snapshot {
    by_seq: im::OrdMap<u64, Chunk>,
    by_id: im::HashMap<ChunkId, u64>,
}

impl ChunkStore {
    /// In-memory store, for tests and dry runs.
    pub fn ephemeral() -> Self {
        Self { db: None, seq: 0, by_seq: im::OrdMap::new(), by_id: im::HashMap::new() }
    }

    /// Open or create the on-disk log and load it into memory.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let db = Database::create(path).with_context(|| format!("opening {}", path.display()))?;
        {
            let tx = db.begin_write()?;
            tx.open_table(CHUNKS)?;
            tx.commit()?;
        }
        let mut store = Self { db: None, seq: 0, by_seq: im::OrdMap::new(), by_id: im::HashMap::new() };
        {
            let tx = db.begin_read()?;
            let table = tx.open_table(CHUNKS)?;
            for row in table.iter()? {
                let (k, v) = row?;
                let chunk: Chunk = serde_json::from_slice(v.value())?;
                let seq = k.value();
                store.by_id.insert(chunk.id, seq);
                store.by_seq.insert(seq, chunk);
                store.seq = store.seq.max(seq);
            }
        }
        store.db = Some(db);
        Ok(store)
    }

    /// Append a chunk. Returns its id. Appending an identical chunk is a no-op.
    pub fn append(&mut self, chunk: Chunk) -> Result<ChunkId> {
        if self.by_id.contains_key(&chunk.id) {
            return Ok(chunk.id);
        }
        self.seq += 1;
        let seq = self.seq;
        if let Some(db) = &self.db {
            let bytes = serde_json::to_vec(&chunk)?;
            let tx = db.begin_write()?;
            {
                let mut table = tx.open_table(CHUNKS)?;
                table.insert(seq, bytes.as_slice())?;
            }
            tx.commit()?;
        }
        let id = chunk.id;
        self.by_id.insert(id, seq);
        self.by_seq.insert(seq, chunk);
        Ok(id)
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot { by_seq: self.by_seq.clone(), by_id: self.by_id.clone() }
    }

    pub fn len(&self) -> usize {
        self.by_seq.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_seq.is_empty()
    }
    pub fn get(&self, id: &ChunkId) -> Option<&Chunk> {
        self.by_id.get(id).and_then(|s| self.by_seq.get(s))
    }
}

impl Snapshot {
    /// Chunks in append order.
    pub fn iter(&self) -> impl Iterator<Item = &Chunk> {
        self.by_seq.values()
    }
    pub fn get(&self, id: &ChunkId) -> Option<&Chunk> {
        self.by_id.get(id).and_then(|s| self.by_seq.get(s))
    }
    pub fn len(&self) -> usize {
        self.by_seq.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_seq.is_empty()
    }
    pub fn last_turn(&self) -> u32 {
        self.by_seq.values().map(|c| c.turn).max().unwrap_or(0)
    }
    /// The stored summary of `of` at `level`, if one was generated before.
    pub fn summary(&self, of: &ChunkId, level: Visibility) -> Option<&Chunk> {
        self.by_seq.values().rev().find(|c| matches!(&c.kind, Kind::Summary { of: o, level: l } if o == of && *l == level))
    }
    /// Chunks of one session, in order.
    pub fn session<'a>(&'a self, session: &'a str) -> impl Iterator<Item = &'a Chunk> + 'a {
        self.by_seq.values().filter(move |c| c.session == session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_is_idempotent_and_ordered() {
        let mut s = ChunkStore::ephemeral();
        let a = Chunk::new(Kind::UserTurn, "hello", 1, "t");
        let b = Chunk::new(Kind::UserTurn, "world", 2, "t");
        s.append(a.clone()).unwrap();
        s.append(b.clone()).unwrap();
        s.append(a.clone()).unwrap();
        assert_eq!(s.len(), 2);
        let ids: Vec<_> = s.snapshot().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![a.id, b.id]);
    }

    #[test]
    fn persists_across_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.redb");
        {
            let mut s = ChunkStore::open(&path).unwrap();
            s.append(Chunk::new(Kind::UserTurn, "persist me", 1, "t")).unwrap();
        }
        let s = ChunkStore::open(&path).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.snapshot().iter().next().unwrap().body, "persist me");
    }
}
