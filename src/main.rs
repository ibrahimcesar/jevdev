//! `jevdev`: a coding-agent harness built around Jev.

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use jevdev::config::{Config, CONFIG_FILE};
use jevdev::router::{cost_table, Price};
use jevdev::runtime::{self, print_plain, AutoPermissioner, Event, Permissioner, Session, StdinPermissioner};
use jevdev::state::{tokens::fmt_k, Chunk, ChunkId};
use jevdev::tools::ToolRegistry;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "jevdev", version, about = "A coding-agent harness built around Jev, TypeSafe's System One decision model.", long_about = None)]
struct Cli {
    /// Repository root. Defaults to the current directory.
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Write jevdev.toml, the default Cedar policy, and a sample AGENTS.md.
    Init {
        /// Overwrite existing files.
        #[arg(long)]
        force: bool,
    },
    /// Interactive terminal session (the default).
    Tui {
        /// A goal to start with.
        goal: Option<String>,
    },
    /// Run one goal with plain output.
    Run {
        goal: String,
        /// Answer every permission prompt with yes.
        #[arg(long)]
        yes: bool,
        /// Answer every permission prompt with no (unattended).
        #[arg(long)]
        no: bool,
    },
    /// Assemble the context for a query and show the visibility ladder, without calling a model.
    Context { query: String },
    /// Tier 1: list tools. With an id, tier 2 (schema) or tier 3 (docs).
    Tools {
        id: Option<String>,
        #[arg(long)]
        docs: bool,
    },
    /// Evaluate the execution policy for a shell command.
    Policy { command: String },
    /// The routing arithmetic from the design notes, in millions of tokens.
    Cost {
        #[arg(long, default_value_t = 0.65)]
        x: f64,
        #[arg(long, default_value_t = 0.12)]
        y: f64,
        #[arg(long, default_value_t = 0.23)]
        z: f64,
    },
    /// List stored chunks, or show one.
    State {
        /// Show one chunk in full by its id prefix.
        show: Option<String>,
        /// Only this session.
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value_t = 60)]
        limit: usize,
    },
    /// Configured models, prices, and trust tiers.
    Models,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).init();
    let cli = Cli::parse();
    let root = runtime::ensure_root(&cli.root.unwrap_or(std::env::current_dir()?))?;
    let cfg = Config::load(&root)?;
    match cli.command.unwrap_or(Command::Tui { goal: None }) {
        Command::Init { force } => init(&root, force),
        Command::Tui { goal } => jevdev::tui::run(cfg, &root, goal).await,
        Command::Run { goal, yes, no } => run(cfg, &root, &goal, yes, no).await,
        Command::Context { query } => context(cfg, &root, &query).await,
        Command::Tools { id, docs } => tools(id, docs),
        Command::Policy { command } => policy(cfg, &root, &command).await,
        Command::Cost { x, y, z } => {
            let f = cfg.model(&cfg.llm.frontier).map(|m| m.price()).unwrap_or(Price { input: 5.0, cached: 0.5, output: 25.0 });
            let w = cfg.model(&cfg.llm.worker).map(|m| m.price()).unwrap_or(Price { input: 3.0, cached: 0.3, output: 15.0 });
            println!("frontier {} (${}/{} per Mtok) · worker {} (${}/{} per Mtok)\n", cfg.llm.frontier, f.input, f.output, cfg.llm.worker, w.input, w.output);
            print!("{}", cost_table(x, y, z, &f, &w));
            Ok(())
        }
        Command::State { show, session, limit } => state(cfg, &root, show, session, limit).await,
        Command::Models => {
            println!("{:<22} {:<10} {:<9} {:>7} {:>7} {:>7}  description", "model", "tier", "trust", "in", "cached", "out");
            for m in &cfg.models {
                println!("{:<22} {:<10} {:<9} {:>7.2} {:>7.2} {:>7.2}  {}", m.id, format!("{:?}", m.tier).to_lowercase(), format!("{:?}", m.trust).to_lowercase(), m.input, m.cached, m.output, m.description);
            }
            println!("\nfrontier: {} · worker: {} · cheap: {} · routing: {}", cfg.llm.frontier, cfg.llm.worker, cfg.llm.cheap, cfg.llm.routing);
            Ok(())
        }
    }
}

fn init(root: &std::path::Path, force: bool) -> Result<()> {
    let write = |rel: &str, content: &str| -> Result<()> {
        let p = root.join(rel);
        if p.exists() && !force {
            println!("kept    {rel}");
            return Ok(());
        }
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&p, content).with_context(|| format!("writing {rel}"))?;
        println!("wrote   {rel}");
        Ok(())
    };
    write(CONFIG_FILE, &Config::default_toml())?;
    write(".jevdev/exec.cedar", jevdev::policy::DEFAULT_POLICY)?;
    write("AGENTS.md", runtime::instructions::SAMPLE_AGENTS_MD)?;
    let gi = root.join(".gitignore");
    let has = std::fs::read_to_string(&gi).map(|s| s.lines().any(|l| l.trim() == ".jevdev/")).unwrap_or(false);
    if !has {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&gi)?;
        writeln!(f, "\n# jevdev session state\n.jevdev/")?;
        println!("updated .gitignore");
    }
    println!("\nnext: export TYPESAFE_API_KEY for Jev and ANTHROPIC_API_KEY for the model, then `jevdev`.");
    Ok(())
}

async fn run(cfg: Config, root: &std::path::Path, goal: &str, yes: bool, no: bool) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let printer = tokio::spawn(async move {
        while let Some(e) = rx.recv().await {
            print_plain(&e);
        }
    });
    let permissioner: Arc<dyn Permissioner> = if yes {
        Arc::new(AutoPermissioner(true))
    } else if no {
        Arc::new(AutoPermissioner(false))
    } else {
        Arc::new(StdinPermissioner)
    };
    let mut session = Session::open(cfg, root, tx, permissioner).await?;
    let _ = session.run_goal(goal).await?;
    drop(session);
    let _ = printer.await;
    Ok(())
}

async fn context(cfg: Config, root: &std::path::Path, query: &str) -> Result<()> {
    let (mut session, mut rx) = runtime::open_quiet(cfg, root).await?;
    session.turn = session.snapshot().await.last_turn() + 1;
    let ctx = session.assemble(query).await?;
    while let Ok(e) = rx.try_recv() {
        if matches!(e, Event::Info(_) | Event::Decision { .. } | Event::Context { .. }) {
            print_plain(&e);
        }
    }
    println!("\n--- rendered ({} tok) ---\n{}", fmt_k(ctx.tokens), ctx.render());
    Ok(())
}

fn tools(id: Option<String>, docs: bool) -> Result<()> {
    let reg = ToolRegistry::builtin();
    match id {
        None => {
            println!("tier 1 · {} snippets, the only thing the model ever sees about tools:\n", reg.len());
            for (id, s) in reg.snippets() {
                let t = reg.get(&id).unwrap();
                println!("  {:<12} {:<6} {s}", id, t.access().to_string());
            }
        }
        Some(id) => {
            let t = reg.get(&id).ok_or_else(|| anyhow::anyhow!("no tool {id}"))?;
            if docs {
                println!("tier 3 · {id}\n\n{}", t.docs());
            } else {
                println!("tier 2 · {id} schema, loaded only when Jev picks it:\n\n{}", serde_json::to_string_pretty(&t.schema())?);
            }
        }
    }
    Ok(())
}

async fn policy(cfg: Config, root: &std::path::Path, command: &str) -> Result<()> {
    let p = jevdev::policy::Policy::load(root, &cfg)?;
    let jev = jevdev::jev::Jev::from_config(&cfg.jev)?;
    let shell = ToolRegistry::builtin().get("shell").unwrap();
    let args = serde_json::json!({ "command": command });
    let call = jevdev::tools::ToolCall { tool: "shell".into(), args: args.clone(), intent: command.into(), access: shell.access(), footprint: shell.footprint(&args, root), candidates: vec![] };
    let d = p.check(&call, "evaluate policy", &jev).await?;
    println!("{}  {}", d.verdict, command);
    println!("reasons: {}", if d.reasons.is_empty() { "none (no policy matched)".into() } else { d.reasons.join("; ") });
    println!("attributes: {}", serde_json::to_string_pretty(&d.attrs)?);
    Ok(())
}

async fn state(cfg: Config, root: &std::path::Path, show: Option<String>, session: Option<String>, limit: usize) -> Result<()> {
    let path = root.join(&cfg.session.dir).join("state.redb");
    let store = jevdev::state::ChunkStore::open(&path)?;
    let snap = store.snapshot();
    if let Some(prefix) = show {
        let found: Vec<&Chunk> = snap.iter().filter(|c| c.id.hex().starts_with(&prefix)).collect();
        match found.as_slice() {
            [c] => {
                println!("{} · {} · session {} · {} tok · {}\n\n{}", c.id.hex(), c.label(), c.session, c.tokens, c.sensitivity, c.body);
            }
            [] => println!("no chunk starts with {prefix}"),
            many => println!("{} chunks match; be more specific", many.len()),
        }
        return Ok(());
    }
    let all: Vec<&Chunk> = snap.iter().filter(|c| session.as_ref().map(|s| &c.session == s).unwrap_or(true)).collect();
    println!("{} chunks in {}{}\n", all.len(), path.display(), session.map(|s| format!(" (session {s})")).unwrap_or_default());
    let skip = all.len().saturating_sub(limit);
    for c in all.iter().skip(skip) {
        println!("{}  {:<8} t{:<3} {:>6} tok  {:<10} {}", c.id.short(), c.session, c.turn, c.tokens, c.sensitivity.to_string(), jevdev::state::truncate(&c.label(), 60));
    }
    let _ = ChunkId::parse("");
    Ok(())
}
