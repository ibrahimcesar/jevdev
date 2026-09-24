//! Programmable permissions.
//!
//! Every command is judged by a Cedar policy over attributes the harness
//! computes first (what it touches, whether it is read-only, whether it is
//! destructive), plus one attribute Jev fills by deep inspection: whether a
//! script the command would execute reaches the network. Cedar's outcomes map
//! to the design notes' three: a forbid is `Deny`, a permit is `Allow`, and no
//! match is `Ask`, where Jev gets the first opinion and a human the last word.

use crate::config::Config;
use crate::jev::{questions, Jev};
use crate::state::Access;
use crate::tools::ToolCall;
use anyhow::{anyhow, Context as _, Result};
use cedar_policy::{Authorizer, Context, Decision, Entities, EntityUid, PolicySet, Request};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub const DEFAULT_POLICY: &str = include_str!("../../policies/exec.cedar");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Allow,
    Ask,
    Deny,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Verdict::Allow => "allow",
            Verdict::Ask => "ask",
            Verdict::Deny => "deny",
        })
    }
}

#[derive(Clone, Debug)]
pub struct PolicyDecision {
    pub verdict: Verdict,
    /// Cedar policy ids that fired, and Jev's opinion when consulted.
    pub reasons: Vec<String>,
    pub attrs: Value,
}

pub struct Policy {
    set: PolicySet,
    /// Cedar's parser assigns `policy0`, `policy1`, … when reading a policy
    /// set from text, so each rule's `@id("…")` annotation is looked up here
    /// to report the name the file gives it.
    names: std::collections::HashMap<String, String>,
    root: PathBuf,
    read_only: Vec<String>,
    sensitive: Vec<PathBuf>,
    task_scope: String,
    jev_auto_allow: f64,
    source: String,
}

const SCRIPT_RUNNERS: &[&str] = &["python", "python3", "sh", "bash", "zsh", "node", "ruby", "perl", "deno", "bun", "php"];

impl Policy {
    pub fn load(root: &Path, cfg: &Config) -> Result<Self> {
        let path = root.join(&cfg.policy.file);
        let src = if path.exists() { std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))? } else { DEFAULT_POLICY.to_string() };
        Self::from_source(&src, root, cfg)
    }

    pub fn from_source(src: &str, root: &Path, cfg: &Config) -> Result<Self> {
        let set = PolicySet::from_str(src).map_err(|e| anyhow!("cedar policy parse error: {e}"))?;
        let names = set
            .policies()
            .filter_map(|p| p.annotation("id").map(|a| (p.id().to_string(), a.to_string())))
            .collect();
        let home = std::env::var("HOME").unwrap_or_default();
        let sensitive = cfg
            .security
            .sensitive_paths
            .iter()
            .map(|p| PathBuf::from(p.replacen("~", &home, 1)))
            .collect();
        Ok(Self {
            set,
            names,
            root: root.to_path_buf(),
            read_only: cfg.policy.read_only.clone(),
            sensitive,
            task_scope: cfg.policy.task_scope.clone(),
            jev_auto_allow: cfg.policy.jev_auto_allow,
            source: src.to_string(),
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    fn is_read_only(&self, command: &str) -> bool {
        let words: Vec<String> = shell_words::split(command).unwrap_or_default();
        if words.is_empty() {
            return false;
        }
        if command.contains('>') || command.contains("| tee") || command.contains("&&") || command.contains(';') {
            return false;
        }
        let one = words[0].as_str();
        let two = if words.len() > 1 { format!("{} {}", words[0], words[1]) } else { String::new() };
        self.read_only.iter().any(|r| r == one || r == &two)
    }

    fn is_destructive(command: &str) -> bool {
        let l = command.to_ascii_lowercase();
        let pats = ["rm -rf", "rm -fr", "git push --force", "git push -f", "git reset --hard", "git clean -f", "sudo ", "mkfs", "dd if=", "> /dev/", "chmod -r 777", ":(){", "shutdown", "reboot", "kill -9 -1"];
        pats.iter().any(|p| l.contains(p))
    }

    fn touches_sensitive(&self, paths: &[PathBuf], command: &str) -> bool {
        let l = command.to_ascii_lowercase();
        if l.contains(".env") || l.contains("~/.ssh") || l.contains(".ssh/") || l.contains("id_rsa") || l.contains("credentials") {
            return true;
        }
        paths.iter().any(|p| {
            let s = p.to_string_lossy().to_ascii_lowercase();
            self.sensitive.iter().any(|sp| p.starts_with(sp)) || s.contains("/.env") || s.ends_with(".env") || s.contains("/.ssh/") || s.contains("id_rsa")
        })
    }

    fn is_test(&self, command: &str, paths: &[PathBuf]) -> bool {
        let l = command.trim_start().to_ascii_lowercase();
        let runners = ["cargo test", "npm test", "pnpm test", "yarn test", "pytest", "go test", "bun test", "mix test", "make test"];
        runners.iter().any(|r| l.starts_with(r)) || (!paths.is_empty() && paths.iter().all(|p| p.components().any(|c| c.as_os_str() == "tests" || c.as_os_str() == "test")))
    }

    /// The script a command would execute, if any and if it lives in the repo.
    fn script_path(&self, command: &str) -> Option<PathBuf> {
        let words = shell_words::split(command).ok()?;
        let first = words.first()?;
        let prog = Path::new(first).file_name()?.to_string_lossy().to_string();
        let candidate = if SCRIPT_RUNNERS.contains(&prog.as_str()) {
            words.iter().skip(1).find(|w| !w.starts_with('-'))?.clone()
        } else if first.starts_with("./") || first.ends_with(".sh") || first.ends_with(".py") {
            first.clone()
        } else {
            return None;
        };
        let (p, inside) = crate::tools::resolve(&self.root, &candidate);
        if inside && p.is_file() {
            Some(p)
        } else {
            None
        }
    }

    fn attrs(&self, call: &ToolCall, script_egress: bool) -> Value {
        let command = call.footprint.command.clone().unwrap_or_else(|| call.describe());
        let paths = &call.footprint.paths;
        let outside_root = paths.iter().any(|p| !p.starts_with(&self.root));
        let program = shell_words::split(&command).ok().and_then(|w| w.first().cloned()).unwrap_or_default();
        json!({
            "command": command,
            "program": program,
            "tool": call.tool,
            "access": call.access.to_string(),
            "touches": paths.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "touches_sensitive": self.touches_sensitive(paths, &command),
            "outside_root": outside_root,
            "read_only": call.access == Access::Read || self.is_read_only(&command),
            "is_test": self.is_test(&command, paths),
            "destructive": Self::is_destructive(&command),
            "script_egress": script_egress,
        })
    }

    fn action_for(call: &ToolCall) -> &'static str {
        if call.footprint.command.is_some() {
            "exec"
        } else if call.access == Access::Write {
            "write"
        } else {
            "read"
        }
    }

    /// Evaluate Cedar over the computed attributes. Pure, synchronous, no Jev.
    pub fn evaluate(&self, call: &ToolCall, script_egress: bool, goal: &str) -> Result<PolicyDecision> {
        let attrs = self.attrs(call, script_egress);
        let action = Self::action_for(call);
        let id = blake3::hash(call.describe().as_bytes()).to_hex()[..16].to_string();
        let entities = Entities::from_json_value(
            json!([
                { "uid": { "type": "Agent", "id": "main" }, "attrs": {}, "parents": [] },
                { "uid": { "type": "Cmd", "id": id }, "attrs": attrs, "parents": [] }
            ]),
            None,
        )
        .map_err(|e| anyhow!("cedar entities: {e}"))?;
        let context = Context::from_json_value(json!({ "task_scope": self.task_scope, "goal": goal }), None).map_err(|e| anyhow!("cedar context: {e}"))?;
        let principal = EntityUid::from_str(r#"Agent::"main""#).map_err(|e| anyhow!("{e}"))?;
        let action_uid = EntityUid::from_str(&format!(r#"Action::"{action}""#)).map_err(|e| anyhow!("{e}"))?;
        let resource = EntityUid::from_str(&format!(r#"Cmd::"{id}""#)).map_err(|e| anyhow!("{e}"))?;
        let req = Request::new(principal, action_uid, resource, context, None).map_err(|e| anyhow!("cedar request: {e}"))?;
        let resp = Authorizer::new().is_authorized(&req, &self.set, &entities);
        let mut reasons: Vec<String> = resp.diagnostics().reason().map(|p| self.names.get(&p.to_string()).cloned().unwrap_or_else(|| p.to_string())).collect();
        for e in resp.diagnostics().errors() {
            reasons.push(format!("cedar error: {e}"));
        }
        let verdict = match resp.decision() {
            Decision::Allow => Verdict::Allow,
            Decision::Deny if !reasons.is_empty() => Verdict::Deny,
            Decision::Deny => Verdict::Ask,
        };
        Ok(PolicyDecision { verdict, reasons, attrs })
    }

    /// Full check: Jev deep inspection of any script, Cedar, then Jev's opinion
    /// when Cedar has none.
    pub async fn check(&self, call: &ToolCall, goal: &str, jev: &Jev) -> Result<PolicyDecision> {
        let mut egress = false;
        let mut extra = Vec::new();
        if let Some(cmd) = &call.footprint.command {
            if let Some(script) = self.script_path(cmd) {
                if let Ok(text) = tokio::fs::read_to_string(&script).await {
                    let (qid, q) = questions::egress();
                    let a = jev.ask_one(json!({ "script": crate::state::truncate(&text, 12_000), "path": script.display().to_string() }), &qid, q).await?;
                    let p = a.as_noul().unwrap_or(0.0);
                    egress = p >= 0.5;
                    extra.push(format!("jev egress check on {}: p={p:.2}", script.file_name().unwrap_or_default().to_string_lossy()));
                }
            }
        }
        let mut d = self.evaluate(call, egress, goal)?;
        d.reasons.extend(extra);
        if d.verdict == Verdict::Ask {
            let (qid, q) = questions::permit();
            let state = json!({
                "command": call.footprint.command.clone().unwrap_or_else(|| call.describe()),
                "access": call.access.to_string(),
                "root": self.root.display().to_string(),
                "goal": goal,
                "touches": d.attrs.get("touches").cloned().unwrap_or(Value::Null),
            });
            let a = jev.ask_one(state, &qid, q).await?;
            if let Some((choice, p)) = a.as_choice() {
                d.reasons.push(format!("jev says {choice} (p={p:.2})"));
                if choice == "allow" && p >= self.jev_auto_allow {
                    d.verdict = Verdict::Allow;
                } else if choice == "deny" && p >= 0.8 {
                    d.verdict = Verdict::Deny;
                }
            }
        }
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Footprint;
    use serde_json::json;

    fn call(cmd: &str, access: Access, paths: Vec<PathBuf>) -> ToolCall {
        ToolCall {
            tool: "shell".into(),
            args: json!({ "command": cmd }),
            intent: cmd.into(),
            access,
            footprint: Footprint { paths, command: Some(cmd.into()) },
            candidates: vec![],
        }
    }

    fn policy() -> Policy {
        let cfg = Config { models: crate::config::default_models(), ..Default::default() };
        Policy::from_source(DEFAULT_POLICY, Path::new("/repo"), &cfg).unwrap()
    }

    #[test]
    fn read_only_allowed_secrets_denied_rest_asks() {
        let p = policy();
        assert_eq!(p.evaluate(&call("git status", Access::Write, vec![]), false, "g").unwrap().verdict, Verdict::Allow);
        let d = p.evaluate(&call("cat ~/.ssh/id_rsa", Access::Write, vec![PathBuf::from("/Users/x/.ssh/id_rsa")]), false, "g").unwrap();
        assert_eq!(d.verdict, Verdict::Deny);
        assert!(d.reasons.iter().any(|r| r.contains("deny_sensitive_paths")));
        assert_eq!(p.evaluate(&call("rm -rf target", Access::Write, vec![]), false, "g").unwrap().verdict, Verdict::Deny);
        assert_eq!(p.evaluate(&call("cargo test", Access::Write, vec![]), false, "g").unwrap().verdict, Verdict::Allow);
        assert_eq!(p.evaluate(&call("python deploy.py", Access::Write, vec![]), true, "g").unwrap().verdict, Verdict::Deny);
        assert_eq!(p.evaluate(&call("make release", Access::Write, vec![]), false, "g").unwrap().verdict, Verdict::Ask);
    }
}
