//! The whole loop, offline: scripted model, local Jev heuristics, real tools,
//! real policy, real store.

use jevdev::config::{default_models, Config};
use jevdev::llm::scripted::ScriptedClient;
use jevdev::runtime::{AutoPermissioner, Event, Session, SessionOptions};
use jevdev::state::{ChunkStore, Kind};
use std::sync::Arc;
use std::time::Duration;

fn config() -> Config {
    let mut cfg = Config { models: default_models(), ..Default::default() };
    cfg.jev.transport = "local".into();
    cfg.llm.provider = "scripted".into();
    cfg
}

#[tokio::test]
async fn offline_loop_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("README.md"), "# Demo\n\nLogin sessions expire early on Safari.\n").unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() { println!(\"hi\"); }\n").unwrap();
    std::fs::write(root.join("AGENTS.md"), jevdev::runtime::instructions::SAMPLE_AGENTS_MD).unwrap();

    let replies = [
        "Let me look around first.\n<act>list files under the repository root</act>",
        "<act>search for \"expire\" in README.md</act>",
        "<act>write the file notes/plan.md with content \"# Plan\n\n1. Reproduce the early expiry.\n2. Fix it.\"</act>",
        "<act>run `cat ~/.ssh/id_rsa`</act>",
        "<done>Done: listed the repo, found the expiry note, wrote notes/plan.md; the credential read was refused.</done>",
    ];
    let llm = Arc::new(ScriptedClient::new(replies.iter().map(|s| s.to_string()).collect()));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let mut session = Session::open_with(config(), root, tx, Arc::new(AutoPermissioner(false)), SessionOptions { llm: Some(llm), ..Default::default() }).await.unwrap();

    let answer = session.run_goal("understand the login expiry bug and write a plan").await.unwrap();
    assert!(answer.starts_with("Done:"), "unexpected answer: {answer}");
    let session_id = session.id.clone();
    drop(session);

    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }

    // Jev routed each intent to the right tool, in order.
    let tools: Vec<String> = events.iter().filter_map(|e| if let Event::ToolPicked { tool, .. } = e { Some(tool.clone()) } else { None }).collect();
    assert_eq!(tools, ["list_files", "grep", "write_file", "shell"]);

    // The policy allowed the reads and the in-repo write, and denied the credential read.
    let verdicts: Vec<(String, String)> = events.iter().filter_map(|e| if let Event::Permit { verdict, call, .. } = e { Some((verdict.clone(), call.clone())) } else { None }).collect();
    assert_eq!(verdicts.iter().map(|(v, _)| v.as_str()).collect::<Vec<_>>(), ["allow", "allow", "allow", "deny"]);
    assert!(verdicts[3].1.contains("id_rsa"));
    assert!(events.iter().any(|e| matches!(e, Event::Permit { reasons, .. } if reasons.iter().any(|r| r == "deny_sensitive_paths"))));

    // The write happened, the denied command did not run, and the tool output says so.
    let plan = std::fs::read_to_string(root.join("notes/plan.md")).unwrap();
    assert!(plan.starts_with("# Plan"));
    let ran: Vec<(String, bool)> = events.iter().filter_map(|e| if let Event::ToolRan { tool, ok, .. } = e { Some((tool.clone(), *ok)) } else { None }).collect();
    assert_eq!(ran, [("list_files".to_string(), true), ("grep".into(), true), ("write_file".into(), true), ("shell".into(), false)]);

    // The grep found the line.
    assert!(events.iter().any(|e| matches!(e, Event::ToolRan { tool, preview, .. } if tool == "grep" && preview.contains("README.md:3"))));

    // Instructions were pinned; the cache decision fired from turn two; the security tier came from paths once a file was touched.
    assert!(events.iter().any(|e| matches!(e, Event::Context { rows, .. } if rows.iter().any(|r| r.pinned))));
    assert!(events.iter().filter(|e| matches!(e, Event::Decision { point: "cache", .. })).count() >= 3);
    assert!(events.iter().any(|e| matches!(e, Event::Decision { point: "security", detail, .. } if detail.contains("touched paths"))));

    // The background retrieval pass produced the progress page after the write.
    let progress = root.join(".jevdev/progress.md");
    for _ in 0..40 {
        if progress.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let page = std::fs::read_to_string(&progress).expect("progress.md written by the background task");
    assert!(page.contains("notes/plan.md"));

    // Everything is in the store, addressable, and survives reopening.
    let store = ChunkStore::open(&root.join(".jevdev/state.redb")).unwrap();
    let snap = store.snapshot();
    let outputs = snap.session(&session_id).filter(|c| matches!(c.kind, Kind::ToolOutput { .. })).count();
    assert_eq!(outputs, 4);
    assert!(snap.session(&session_id).any(|c| matches!(c.kind, Kind::Reasoning)));
    assert!(snap.session(&session_id).any(|c| matches!(c.kind, Kind::Assistant)));
    assert!(snap.session(&session_id).any(|c| matches!(&c.kind, Kind::ToolOutput { tool, .. } if tool == "write_file") && c.paths.iter().any(|p| p.ends_with("notes/plan.md"))));
}

#[tokio::test]
async fn unknown_intent_becomes_a_failed_output_not_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(root.join("README.md"), "x\n").unwrap();
    let llm = Arc::new(ScriptedClient::new(vec!["<act>replace in README.md</act>".into(), "<done>gave up</done>".into()]));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let mut session = Session::open_with(config(), root, tx, Arc::new(AutoPermissioner(false)), SessionOptions { llm: Some(llm), ..Default::default() }).await.unwrap();
    let answer = session.run_goal("do something vague").await.unwrap();
    assert_eq!(answer, "gave up");
    drop(session);
    let mut saw_failed = false;
    while let Some(e) = rx.recv().await {
        if let Event::ToolRan { ok: false, .. } = e {
            saw_failed = true;
        }
    }
    assert!(saw_failed);
}
