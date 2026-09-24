//! Conditional instructions.
//!
//! `AGENTS.md` is split into sections. A section preceded by, or opening with,
//! a marker such as `<!-- when: glob **/*.rs -->` attaches to that condition
//! and is loaded only while it holds. Sections without a marker always apply.
//! Fragments are re-evaluated from task state every turn and pinned in full,
//! so a long session can never summarise them away.
//!
//! Conditions: `glob <pattern>` (a recently touched path matches),
//! `dir <path>` (a touched path is under the directory),
//! `fuzzy <question>` (Jev answers a noul against the current task).

use crate::jev::{questions, Jev};
use crate::state::{Chunk, Kind};
use anyhow::Result;
use globset::{Glob, GlobMatcher};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub enum Condition {
    Always,
    Glob(GlobMatcher, String),
    Dir(PathBuf, String),
    Fuzzy(String),
}

impl Condition {
    pub fn describe(&self) -> String {
        match self {
            Condition::Always => "always".into(),
            Condition::Glob(_, g) => format!("touches {g}"),
            Condition::Dir(_, d) => format!("inside {d}"),
            Condition::Fuzzy(q) => format!("when: {q}"),
        }
    }
    fn parse(spec: &str) -> Option<Self> {
        let spec = spec.trim();
        let (kind, rest) = spec.split_once(char::is_whitespace).unwrap_or((spec, ""));
        match kind {
            "glob" => Glob::new(rest.trim()).ok().map(|g| Condition::Glob(g.compile_matcher(), rest.trim().to_string())),
            "dir" => Some(Condition::Dir(PathBuf::from(rest.trim()), rest.trim().to_string())),
            "fuzzy" => Some(Condition::Fuzzy(rest.trim().to_string())),
            "always" | "" => Some(Condition::Always),
            _ => None,
        }
    }
}

pub struct Fragment {
    pub title: String,
    pub body: String,
    pub condition: Condition,
    pub source: String,
}

#[derive(Default)]
pub struct Instructions {
    fragments: Vec<Fragment>,
}

fn marker(line: &str) -> Option<&str> {
    let l = line.trim();
    l.strip_prefix("<!--")?.strip_suffix("-->")?.trim().strip_prefix("when:")
}

impl Instructions {
    pub fn load(root: &Path) -> Result<Self> {
        let mut me = Self::default();
        for name in ["AGENTS.md", "CLAUDE.md"] {
            let p = root.join(name);
            if p.exists() {
                me.parse(&std::fs::read_to_string(&p)?, name);
                break;
            }
        }
        let dir = root.join(".jevdev").join("instructions");
        if dir.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(&dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().map(|e| e == "md").unwrap_or(false)).collect();
            entries.sort();
            for p in entries {
                let name = format!(".jevdev/instructions/{}", p.file_name().unwrap_or_default().to_string_lossy());
                me.parse(&std::fs::read_to_string(&p)?, &name);
            }
        }
        Ok(me)
    }

    pub fn parse(&mut self, text: &str, source: &str) {
        let mut sections: Vec<(String, Vec<String>)> = vec![(String::new(), Vec::new())];
        for line in text.lines() {
            if let Some(h) = line.strip_prefix("## ") {
                sections.push((h.trim().to_string(), Vec::new()));
            } else {
                sections.last_mut().unwrap().1.push(line.to_string());
            }
        }
        let mut pending: Option<Condition> = None;
        for (title, lines) in sections {
            let mut cond = pending.take().unwrap_or(Condition::Always);
            let mut body = Vec::new();
            for l in &lines {
                if let Some(spec) = marker(l) {
                    if let Some(c) = Condition::parse(spec) {
                        if body.iter().all(|b: &String| b.trim().is_empty()) {
                            cond = c;
                        } else {
                            pending = Some(c);
                        }
                        continue;
                    }
                }
                body.push(l.clone());
            }
            let body = body.join("\n").trim().to_string();
            if body.is_empty() {
                continue;
            }
            self.fragments.push(Fragment { title: if title.is_empty() { "Preamble".to_string() } else { title }, body, condition: cond, source: source.to_string() });
        }
    }

    pub fn len(&self) -> usize {
        self.fragments.len()
    }
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }
    pub fn fragments(&self) -> &[Fragment] {
        &self.fragments
    }

    /// The fragments whose conditions hold, as pinned instruction chunks.
    pub async fn active(&self, query: &str, paths: &[PathBuf], root: &Path, jev: &Jev, session: &str, turn: u32) -> Result<Vec<Chunk>> {
        let rel: Vec<PathBuf> = paths.iter().map(|p| p.strip_prefix(root).map(Path::to_path_buf).unwrap_or_else(|_| p.clone())).collect();
        let mut fuzzy: BTreeMap<String, crate::jev::Question> = BTreeMap::new();
        for (i, f) in self.fragments.iter().enumerate() {
            if let Condition::Fuzzy(q) = &f.condition {
                let (id, question) = questions::condition(i, q);
                fuzzy.insert(id, question);
            }
        }
        let fuzzy_answers = if fuzzy.is_empty() {
            None
        } else {
            let state = json!({ "query": query, "paths": rel.iter().map(|p| p.display().to_string()).collect::<Vec<_>>() });
            Some(jev.ask(state, fuzzy).await?)
        };
        let mut out = Vec::new();
        for (i, f) in self.fragments.iter().enumerate() {
            let holds = match &f.condition {
                Condition::Always => true,
                Condition::Glob(g, _) => rel.iter().any(|p| g.is_match(p)),
                Condition::Dir(d, _) => rel.iter().any(|p| p.starts_with(d)),
                Condition::Fuzzy(_) => fuzzy_answers.as_ref().and_then(|r| r.answers.get(&format!("cond:{i}"))).and_then(|a| a.as_noul()).map(|p| p >= 0.5).unwrap_or(false),
            };
            if holds {
                let body = format!("# {} ({})\n{}", f.title, f.source, f.body);
                out.push(Chunk::new(Kind::Instruction { condition: f.condition.describe(), pinned: true }, body, turn, session));
            }
        }
        Ok(out)
    }
}

pub const SAMPLE_AGENTS_MD: &str = r#"# Project instructions

Sections without a marker always apply. A section that opens with a
`<!-- when: ... -->` marker is loaded only while the condition holds, and is
pinned in full while it does. Conditions: `glob <pattern>`, `dir <path>`,
`fuzzy <question for Jev>`.

## Working agreements

- Read and search before editing. Make the smallest exact edit that works.
- Run the test suite after any change to source files.

## Rust style
<!-- when: glob **/*.rs -->
- Prefer `anyhow::Result` in binaries and typed errors in libraries.
- No `unwrap()` outside tests. Keep functions under about 60 lines.

## Billing footguns
<!-- when: dir src/billing -->
- Amounts are integer minor units. Never use floats for money.

## Writing for people
<!-- when: fuzzy the task is writing documentation or user-facing prose -->
- Short sentences. Lead with the outcome. No marketing language.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_markers() {
        let mut i = Instructions::default();
        i.parse(SAMPLE_AGENTS_MD, "AGENTS.md");
        let conds: Vec<String> = i.fragments().iter().map(|f| f.condition.describe()).collect();
        assert!(conds.contains(&"always".to_string()));
        assert!(conds.contains(&"touches **/*.rs".to_string()));
        assert!(conds.contains(&"inside src/billing".to_string()));
        assert!(conds.iter().any(|c| c.starts_with("when:")));
    }
}
