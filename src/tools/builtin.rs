//! The batteries: file, search, edit, shell, and delegation tools.

use super::{resolve, Footprint, Tool, ToolCx, ToolOutput};
use crate::state::Access;
use anyhow::{anyhow, Context as _, Result};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub fn all() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ReadFile),
        Arc::new(ListFiles),
        Arc::new(Grep),
        Arc::new(WriteFile),
        Arc::new(StrReplace),
        Arc::new(Shell),
        Arc::new(Delegate),
    ]
}

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).expect("schema serialises")
}

fn parse<T: for<'de> Deserialize<'de>>(args: Value) -> Result<T> {
    serde_json::from_value(args).map_err(|e| anyhow!("bad arguments: {e}"))
}

fn cap(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let head = &s[..max * 2 / 3];
    let tail = &s[s.len() - max / 3..];
    format!("{head}\n[… {} bytes elided …]\n{tail}", s.len() - max)
}

fn out(tool: &str, body: String, access: Access, ok: bool, paths: Vec<PathBuf>) -> ToolOutput {
    ToolOutput { tool: tool.into(), body, access, ok, paths }
}

fn rel(root: &Path, p: &Path) -> PathBuf {
    p.strip_prefix(root).map(Path::to_path_buf).unwrap_or_else(|_| p.to_path_buf())
}

// ---------------------------------------------------------------- read_file

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadFileArgs {
    /// Path relative to the repository root.
    pub path: String,
    /// First line to show, 1-based.
    #[serde(default)]
    pub start: Option<u32>,
    /// Last line to show, inclusive.
    #[serde(default)]
    pub end: Option<u32>,
}

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn id(&self) -> &'static str {
        "read_file"
    }
    fn snippet(&self) -> &'static str {
        "read_file: show the contents of one file, optionally a line range"
    }
    fn access(&self) -> Access {
        Access::Read
    }
    fn schema(&self) -> Value {
        schema::<ReadFileArgs>()
    }
    fn docs(&self) -> &'static str {
        "Reads a text file inside the repository. Use start/end to read a slice of a large file. Lines are numbered in the output."
    }
    fn footprint(&self, args: &Value, root: &Path) -> Footprint {
        let p = args.get("path").and_then(Value::as_str).unwrap_or("");
        Footprint { paths: vec![resolve(root, p).0], command: None }
    }
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput> {
        let a: ReadFileArgs = parse(args)?;
        let (path, _) = resolve(&cx.root, &a.path);
        let text = tokio::fs::read_to_string(&path).await.with_context(|| format!("reading {}", path.display()))?;
        let start = a.start.unwrap_or(1).max(1) as usize;
        let end = a.end.map(|e| e as usize).unwrap_or(usize::MAX);
        let body: String = text
            .lines()
            .enumerate()
            .filter(|(i, _)| *i + 1 >= start && *i + 1 <= end)
            .map(|(i, l)| format!("{:>5}  {l}\n", i + 1))
            .collect();
        Ok(out(self.id(), cap(body, 60_000), Access::Read, true, vec![rel(&cx.root, &path)]))
    }
}

// ---------------------------------------------------------------- list_files

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListFilesArgs {
    /// Directory relative to the repository root. Defaults to the root.
    #[serde(default)]
    pub dir: Option<String>,
    /// Optional glob such as `**/*.rs`.
    #[serde(default)]
    pub glob: Option<String>,
    /// Maximum entries, default 300.
    #[serde(default)]
    pub max: Option<u32>,
}

pub struct ListFiles;

#[async_trait]
impl Tool for ListFiles {
    fn id(&self) -> &'static str {
        "list_files"
    }
    fn snippet(&self) -> &'static str {
        "list_files: list files under a directory, honouring .gitignore, with an optional glob"
    }
    fn access(&self) -> Access {
        Access::Read
    }
    fn schema(&self) -> Value {
        schema::<ListFilesArgs>()
    }
    fn docs(&self) -> &'static str {
        "Walks a directory like `git ls-files`. Ignored and hidden files are skipped. Pass a glob to filter, for example `src/**/*.ts`."
    }
    fn footprint(&self, args: &Value, root: &Path) -> Footprint {
        let d = args.get("dir").and_then(Value::as_str).unwrap_or(".");
        Footprint { paths: vec![resolve(root, d).0], command: None }
    }
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput> {
        let a: ListFilesArgs = parse(args)?;
        let (dir, _) = resolve(&cx.root, a.dir.as_deref().unwrap_or("."));
        let max = a.max.unwrap_or(300) as usize;
        let glob = a.glob.as_deref().map(globset::Glob::new).transpose()?.map(|g| g.compile_matcher());
        let root = cx.root.clone();
        let body = tokio::task::spawn_blocking(move || {
            let mut lines = Vec::new();
            let mut total = 0usize;
            for entry in ignore::WalkBuilder::new(&dir).hidden(true).build().flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let r = rel(&root, entry.path());
                if let Some(g) = &glob {
                    if !g.is_match(&r) {
                        continue;
                    }
                }
                total += 1;
                if lines.len() < max {
                    lines.push(r.display().to_string());
                }
            }
            lines.sort();
            let mut s = lines.join("\n");
            if total > max {
                s.push_str(&format!("\n[… {} more not shown …]", total - max));
            }
            s
        })
        .await?;
        Ok(out(self.id(), body, Access::Read, true, vec![]))
    }
}

// ---------------------------------------------------------------- grep

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GrepArgs {
    /// Rust-flavoured regular expression.
    pub pattern: String,
    /// File or directory to search, relative to the root. Defaults to the root.
    #[serde(default)]
    pub path: Option<String>,
    /// Maximum matching lines, default 200.
    #[serde(default)]
    pub max: Option<u32>,
}

pub struct Grep;

#[async_trait]
impl Tool for Grep {
    fn id(&self) -> &'static str {
        "grep"
    }
    fn snippet(&self) -> &'static str {
        "grep: search file contents with a regex, returning file:line matches"
    }
    fn access(&self) -> Access {
        Access::Read
    }
    fn schema(&self) -> Value {
        schema::<GrepArgs>()
    }
    fn docs(&self) -> &'static str {
        "Searches text files under a path (ignored and binary files skipped) for a regular expression. Output is `path:line: text`. Narrow with `path` when the repository is large."
    }
    fn footprint(&self, args: &Value, root: &Path) -> Footprint {
        let p = args.get("path").and_then(Value::as_str).unwrap_or(".");
        Footprint { paths: vec![resolve(root, p).0], command: None }
    }
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput> {
        let a: GrepArgs = parse(args)?;
        let re = regex::Regex::new(&a.pattern).map_err(|e| anyhow!("bad regex: {e}"))?;
        let (start, _) = resolve(&cx.root, a.path.as_deref().unwrap_or("."));
        let max = a.max.unwrap_or(200) as usize;
        let root = cx.root.clone();
        let (body, hits, files) = tokio::task::spawn_blocking(move || {
            let mut lines = Vec::new();
            let mut hits = 0usize;
            let mut files = Vec::new();
            for entry in ignore::WalkBuilder::new(&start).hidden(true).build().flatten() {
                if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    continue;
                }
                let Ok(bytes) = std::fs::read(entry.path()) else { continue };
                if bytes.iter().take(1024).any(|&b| b == 0) {
                    continue;
                }
                let text = String::from_utf8_lossy(&bytes);
                let r = rel(&root, entry.path());
                let mut any = false;
                for (i, line) in text.lines().enumerate() {
                    if re.is_match(line) {
                        hits += 1;
                        any = true;
                        if lines.len() < max {
                            lines.push(format!("{}:{}: {}", r.display(), i + 1, crate::state::truncate(line.trim(), 240)));
                        }
                    }
                }
                if any {
                    files.push(r);
                }
            }
            let mut s = lines.join("\n");
            if hits > max {
                s.push_str(&format!("\n[… {} more matches not shown …]", hits - max));
            }
            if hits == 0 {
                s = "no matches".into();
            }
            (s, hits, files)
        })
        .await?;
        Ok(out(self.id(), format!("{hits} matches\n{body}"), Access::Read, true, files))
    }
}

// ---------------------------------------------------------------- write_file

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteFileArgs {
    /// Path relative to the repository root. Parent directories are created.
    pub path: String,
    /// The complete new contents of the file.
    pub content: String,
}

pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn id(&self) -> &'static str {
        "write_file"
    }
    fn snippet(&self) -> &'static str {
        "write_file: create or overwrite a whole file with given contents"
    }
    fn access(&self) -> Access {
        Access::Write
    }
    fn schema(&self) -> Value {
        schema::<WriteFileArgs>()
    }
    fn docs(&self) -> &'static str {
        "Writes the full contents of a file. Prefer str_replace for edits to existing files so the diff stays small."
    }
    fn footprint(&self, args: &Value, root: &Path) -> Footprint {
        let p = args.get("path").and_then(Value::as_str).unwrap_or("");
        Footprint { paths: vec![resolve(root, p).0], command: None }
    }
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput> {
        let a: WriteFileArgs = parse(args)?;
        let (path, inside) = resolve(&cx.root, &a.path);
        if !inside {
            return Err(anyhow!("refusing to write outside the repository: {}", path.display()));
        }
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, &a.content).await.with_context(|| format!("writing {}", path.display()))?;
        let r = rel(&cx.root, &path);
        Ok(out(self.id(), format!("wrote {} ({} bytes)", r.display(), a.content.len()), Access::Write, true, vec![r]))
    }
}

// ---------------------------------------------------------------- str_replace

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StrReplaceArgs {
    /// Path relative to the repository root.
    pub path: String,
    /// Exact text to replace. Must occur exactly once.
    pub old: String,
    /// Replacement text.
    pub new: String,
}

pub struct StrReplace;

#[async_trait]
impl Tool for StrReplace {
    fn id(&self) -> &'static str {
        "str_replace"
    }
    fn snippet(&self) -> &'static str {
        "str_replace: replace one exact occurrence of a string in a file with new text"
    }
    fn access(&self) -> Access {
        Access::Write
    }
    fn schema(&self) -> Value {
        schema::<StrReplaceArgs>()
    }
    fn docs(&self) -> &'static str {
        "Edits a file by exact string replacement. `old` must match exactly once, including whitespace; include enough surrounding lines to make it unique."
    }
    fn footprint(&self, args: &Value, root: &Path) -> Footprint {
        let p = args.get("path").and_then(Value::as_str).unwrap_or("");
        Footprint { paths: vec![resolve(root, p).0], command: None }
    }
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput> {
        let a: StrReplaceArgs = parse(args)?;
        let (path, inside) = resolve(&cx.root, &a.path);
        if !inside {
            return Err(anyhow!("refusing to edit outside the repository: {}", path.display()));
        }
        let text = tokio::fs::read_to_string(&path).await.with_context(|| format!("reading {}", path.display()))?;
        let n = text.matches(&a.old).count();
        if n != 1 {
            return Ok(out(self.id(), format!("old text occurs {n} times in {}; it must occur exactly once", a.path), Access::Write, false, vec![]));
        }
        let new_text = text.replacen(&a.old, &a.new, 1);
        tokio::fs::write(&path, &new_text).await?;
        let r = rel(&cx.root, &path);
        let diff = format!("--- {0}\n+++ {0}\n-{1}\n+{2}", r.display(), a.old.replace('\n', "\n-"), a.new.replace('\n', "\n+"));
        Ok(out(self.id(), cap(diff, 8000), Access::Write, true, vec![r]))
    }
}

// ---------------------------------------------------------------- shell

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShellArgs {
    /// The command line, run with `sh -c` in the repository root.
    pub command: String,
    /// Timeout in seconds, default 120.
    #[serde(default)]
    pub timeout_s: Option<u32>,
}

pub struct Shell;

#[async_trait]
impl Tool for Shell {
    fn id(&self) -> &'static str {
        "shell"
    }
    fn snippet(&self) -> &'static str {
        "shell: run a shell command in the repository root and capture its output"
    }
    fn access(&self) -> Access {
        Access::Write
    }
    fn schema(&self) -> Value {
        schema::<ShellArgs>()
    }
    fn docs(&self) -> &'static str {
        "Runs `sh -c <command>` with the repository as the working directory. Every command passes the execution policy first: read-only programs and the test suite run freely, destructive commands and anything touching credentials are refused, the rest asks."
    }
    fn footprint(&self, args: &Value, root: &Path) -> Footprint {
        let cmd = args.get("command").and_then(Value::as_str).unwrap_or("").to_string();
        let paths = shell_words::split(&cmd)
            .unwrap_or_default()
            .into_iter()
            .filter(|t| t.contains('/') || t.contains('.') || t.starts_with('~'))
            .map(|t| resolve(root, &t).0)
            .collect();
        Footprint { paths, command: Some(cmd) }
    }
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput> {
        let a: ShellArgs = parse(args)?;
        let timeout = Duration::from_secs(a.timeout_s.unwrap_or(120) as u64);
        let child = tokio::process::Command::new("sh").arg("-c").arg(&a.command).current_dir(&cx.root).output();
        let result = tokio::time::timeout(timeout, child).await;
        let body = match result {
            Ok(Ok(o)) => {
                let mut s = String::new();
                s.push_str(&String::from_utf8_lossy(&o.stdout));
                if !o.stderr.is_empty() {
                    s.push_str("\n[stderr]\n");
                    s.push_str(&String::from_utf8_lossy(&o.stderr));
                }
                let code = o.status.code().unwrap_or(-1);
                let ok = o.status.success();
                let body = cap(s, 20_000);
                return Ok(out(self.id(), format!("$ {}\n{body}\n[exit {code}]", a.command), Access::Write, ok, vec![]));
            }
            Ok(Err(e)) => format!("$ {}\nfailed to start: {e}", a.command),
            Err(_) => format!("$ {}\ntimed out after {}s", a.command, timeout.as_secs()),
        };
        Ok(out(self.id(), body, Access::Write, false, vec![]))
    }
}

// ---------------------------------------------------------------- delegate

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DelegateArgs {
    /// A self-contained, read-only sub-task, phrased as you would to a colleague.
    pub goal: String,
}

pub struct Delegate;

#[async_trait]
impl Tool for Delegate {
    fn id(&self) -> &'static str {
        "delegate"
    }
    fn snippet(&self) -> &'static str {
        "delegate: hand a read-only research sub-task to a helper agent with its own small context"
    }
    fn access(&self) -> Access {
        Access::Read
    }
    fn schema(&self) -> Value {
        schema::<DelegateArgs>()
    }
    fn docs(&self) -> &'static str {
        "Spawns a sub-agent on a cheaper model with a purpose-built context. It can read, list, and search but never write. Duplicate sub-goals are deduplicated against earlier ones. The result comes back as one scored chunk."
    }
    fn footprint(&self, _args: &Value, _root: &Path) -> Footprint {
        Footprint::default()
    }
    async fn run(&self, args: Value, cx: &ToolCx) -> Result<ToolOutput> {
        let a: DelegateArgs = parse(args)?;
        let Some(d) = &cx.delegator else { return Err(anyhow!("delegation is not available in this context")) };
        let body = d.delegate(&a.goal).await?;
        Ok(out(self.id(), body, Access::Read, true, vec![]))
    }
}
