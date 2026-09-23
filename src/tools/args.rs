//! Argument construction and validation for tier-2 schemas.

use crate::llm::LlmClient;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

#[async_trait]
pub trait ArgBuilder: Send + Sync {
    async fn build(&self, intent: &str, tool: &str, schema: &Value, docs: &str, previous_error: Option<&str>) -> Result<Value>;
}

/// Validate `args` against a JSON schema; the error text goes back to the builder.
pub fn validate(schema: &Value, args: &Value) -> Result<(), String> {
    let v = jsonschema::validator_for(schema).map_err(|e| format!("bad schema: {e}"))?;
    let errors: Vec<String> = v.iter_errors(args).map(|e| format!("{} at {}", e, e.instance_path())).collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

/// Extract the first JSON object embedded in free text.
pub fn extract_json(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str(&text[start..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Offline builder: uses an embedded JSON object when the intent carries one,
/// otherwise derives arguments from quoted strings, paths, and keywords.
pub struct HeuristicArgBuilder;

fn quoted(intent: &str) -> Vec<String> {
    let mut out = Vec::new();
    for q in ['`', '"', '\''] {
        let mut parts = intent.split(q);
        parts.next();
        while let (Some(inner), Some(_)) = (parts.next(), parts.next()) {
            if !inner.trim().is_empty() {
                out.push(inner.to_string());
            }
        }
    }
    out
}

fn pathlike(intent: &str) -> Vec<String> {
    intent
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '(' || c == ')')
        .map(|t| t.trim_matches(|c: char| c == '`' || c == '"' || c == '\'' || c == ':' || c == '.'))
        .filter(|t| (t.contains('/') || t.contains('.')) && !t.starts_with("http") && t.len() > 2 && !t.ends_with('.'))
        .map(String::from)
        .collect()
}

#[async_trait]
impl ArgBuilder for HeuristicArgBuilder {
    async fn build(&self, intent: &str, tool: &str, _schema: &Value, _docs: &str, _prev: Option<&str>) -> Result<Value> {
        if let Some(v) = extract_json(intent) {
            return Ok(v);
        }
        let q = quoted(intent);
        let paths = pathlike(intent);
        let first_path = paths.first().cloned();
        Ok(match tool {
            "read_file" => json!({ "path": first_path.ok_or_else(|| anyhow!("no path in intent"))? }),
            "list_files" => {
                let dir = paths.iter().find(|p| !p.contains('*')).cloned().unwrap_or_else(|| ".".into());
                let glob = paths.iter().find(|p| p.contains('*')).cloned();
                match glob {
                    Some(g) => json!({ "dir": dir, "glob": g }),
                    None => json!({ "dir": dir }),
                }
            }
            "grep" => {
                let pattern = q.first().cloned().or_else(|| {
                    let lower = intent.to_ascii_lowercase();
                    ["for ", "search ", "grep ", "find "].iter().find_map(|k| lower.find(k).map(|i| intent[i + k.len()..].split_whitespace().next().unwrap_or("").to_string()))
                });
                let pattern = pattern.filter(|p| !p.is_empty()).ok_or_else(|| anyhow!("no pattern in intent"))?;
                let path = paths.iter().find(|p| p != &&pattern).cloned();
                match path {
                    Some(p) => json!({ "pattern": pattern, "path": p }),
                    None => json!({ "pattern": pattern }),
                }
            }
            "write_file" => json!({ "path": first_path.ok_or_else(|| anyhow!("no path in intent"))?, "content": q.first().cloned().unwrap_or_default() }),
            "str_replace" => {
                if q.len() < 2 {
                    return Err(anyhow!("str_replace needs the old and new text quoted"));
                }
                json!({ "path": first_path.ok_or_else(|| anyhow!("no path in intent"))?, "old": q[0], "new": q[1] })
            }
            "shell" => {
                let cmd = q.first().cloned().or_else(|| {
                    let lower = intent.to_ascii_lowercase();
                    ["run ", "execute ", "exec "].iter().find_map(|k| lower.find(k).map(|i| intent[i + k.len()..].trim().to_string()))
                });
                json!({ "command": cmd.ok_or_else(|| anyhow!("no command in intent"))? })
            }
            "delegate" => json!({ "goal": intent }),
            _ => json!({}),
        })
    }
}

/// The cheap model fills the arguments against the schema; the heuristic
/// builder is the fallback when the model output is not valid JSON.
pub struct LlmArgBuilder {
    llm: Arc<dyn LlmClient>,
    model: String,
}

impl LlmArgBuilder {
    pub fn new(llm: Arc<dyn LlmClient>, model: &str) -> Self {
        Self { llm, model: model.to_string() }
    }
}

#[async_trait]
impl ArgBuilder for LlmArgBuilder {
    async fn build(&self, intent: &str, tool: &str, schema: &Value, docs: &str, prev: Option<&str>) -> Result<Value> {
        if let Some(v) = extract_json(intent) {
            if validate(schema, &v).is_ok() {
                return Ok(v);
            }
        }
        let system = "You turn a coding agent's stated intent into the arguments of one tool. Reply with a single JSON object that validates against the schema. No prose, no code fences.";
        let mut user = format!("Tool: {tool}\n\nManual:\n{docs}\n\nJSON schema:\n{}\n\nIntent:\n{intent}\n", serde_json::to_string_pretty(schema)?);
        if let Some(e) = prev {
            user.push_str(&format!("\nYour previous attempt failed validation: {e}\nFix it.\n"));
        }
        match self.llm.small(&self.model, system, &user, 1024).await {
            Ok(text) => extract_json(&text).ok_or_else(|| anyhow!("model returned no JSON: {}", crate::state::truncate(&text, 200))),
            Err(_) => HeuristicArgBuilder.build(intent, tool, schema, docs, prev).await,
        }
    }
}
