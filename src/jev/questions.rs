//! The harness's questions to Jev, one builder per decision point (Table I).
//!
//! Written the way TypeSafe's guide asks (docs.typesafe.ai/concepts/how-to-build-with-system-one):
//! instructions point at state with backticked paths such as `chunks[3]`,
//! criteria are contrastive objects with `what`, `not_for`, and `examples`,
//! and broad judgments are decomposed into atomic nouls that code composes.
//! Ids are stable so the local transport and the event log can recognise them.

use super::{Answer, Question};
use crate::config::ModelSpec;
use crate::state::{ChunkId, Sensitivity, Visibility};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const VIS_PREFIX: &str = "vis:";
pub const PERMIT_PREFIX: &str = "permit.";

/// Context: how visible should `chunks[index]` be for the current query?
pub fn visibility(id: &ChunkId, index: usize) -> (String, Question) {
    let q = Question::choice(
        json!({
            "what": format!("How much of `chunks[{index}]` should the coding agent see on this turn, given `query` and `goal`?"),
            "note": "Each chunk is one piece of the agent's stored state: a user message, a tool call or its output, a file, a note. Showing it costs tokens. Hiding it loses nothing: it stays in the store and can be shown on a later turn.",
        }),
        [
            ("hide", json!({
                "what": "Nothing in it helps with `query`.",
                "not_for": "Anything the agent must quote, edit, or reason from this turn.",
                "examples": ["a directory listing from an earlier, unrelated task", "a plan that a later message superseded"],
            })),
            ("short", json!({
                "what": "Only that it exists and what it concluded: one line.",
                "not_for": "Content the agent must read line by line.",
                "examples": ["a test run that passed", "a file the agent already finished editing"],
            })),
            ("long", json!({
                "what": "Its key facts matter but not every line: a paragraph keeping paths, names, numbers, and errors exact.",
                "not_for": "Chunks short enough to show in full.",
                "examples": ["a long grep result of which a dozen hits matter", "a failing test log where the trace and cause matter"],
            })),
            ("full", json!({
                "what": "The agent needs it verbatim this turn.",
                "not_for": "Large outputs of which only a part is relevant.",
                "examples": ["the file the agent is about to edit", "the user's request", "the most recent tool result"],
            })),
        ],
    );
    (format!("{VIS_PREFIX}{}", id.hex()), q)
}

pub fn visibility_from(a: &Answer) -> (Visibility, f64) {
    match a.as_choice() {
        Some((c, p)) => (Visibility::parse(c).unwrap_or(Visibility::Short), p),
        None => (Visibility::Short, 0.0),
    }
}

/// Cache: is the previous context still a good prefix for this query?
pub fn cache_reuse() -> (String, Question) {
    (
        "reuse".into(),
        Question::noul_with(
            json!({
                "what": "Should the harness keep `previous` (the chunks the model saw last turn, in order) as a cached prefix and append the new material, rather than rebuild the context from scratch for `query`?",
                "inputs": "`delta_tokens` is how much new material this turn adds; `stale` is how many previous chunks are no longer relevant; `prefix_tokens` is the size of the reusable prefix.",
            }),
            json!({ "what": "Reuse: the previous context still frames `query` well, so appending is cheaper and loses nothing.", "examples": ["the next step of the same task, with one new tool result", "a follow-up question about the same files"] }),
            json!({ "what": "Rebuild: enough changed that a fresh, query-ordered context serves the model better.", "examples": ["the user switched to a different bug", "most previous chunks are stale", "a large new file dwarfs the old prefix"] }),
        ),
    )
}

/// Routing: which eligible model should take this subtask?
pub fn route(models: &[&ModelSpec]) -> (String, Question) {
    let q = Question::choice(
        json!({
            "what": "Which model should run `subtask`? Prefer the cheapest model that will do the work correctly in one pass; a failed cheap attempt costs more than a frontier pass.",
            "inputs": "`context_tokens` is the size of the purpose-built context; `costs` is the estimated dollar cost per model including the pass back to the frontier model; `sensitivity` is the data tier.",
        }),
        models.iter().map(|m| {
            (
                m.id.clone(),
                json!({
                    "what": m.description,
                    "tier": format!("{:?}", m.tier).to_lowercase(),
                    "not_for": match m.tier {
                        crate::config::Tier::Frontier => "routine work a cheaper model would finish in one pass",
                        crate::config::Tier::Worker => "planning, hard debugging, reviews, or anything touching restricted data",
                        crate::config::Tier::Cheap => "reasoning of any kind",
                    },
                }),
            )
        }),
    );
    ("route".into(), q)
}

/// Tools: which tool performs this intent?
pub fn tool_pick(snippets: &[(String, String)]) -> (String, Question) {
    let q = Question::choice(
        json!({ "what": "`intent` is what the coding agent says it wants to do next, in plain words. Which single tool performs that action?" }),
        snippets.iter().map(|(id, snip)| (id.clone(), json!({ "what": snip }))),
    );
    ("tool".into(), q)
}

/// Permissions, decomposed: five atomic nouls about `command`, composed by
/// [`permit_from`]. Asked only when the Cedar policy has no opinion.
pub fn permit_questions() -> BTreeMap<String, Question> {
    let mut qs = BTreeMap::new();
    qs.insert(
        format!("{PERMIT_PREFIX}destructive"),
        Question::noul_with(
            "Does `command` delete, overwrite, reset, or otherwise destroy files, data, or history in a way a simple undo cannot recover?",
            json!({ "what": "Destroys something hard to recover.", "examples": ["rm -rf src", "git reset --hard", "git push --force", "DROP TABLE users"] }),
            json!({ "what": "Reads, builds, tests, or makes recoverable changes.", "not_for": "Deleting build artefacts or temp files, which is recoverable.", "examples": ["cargo test", "ls -la", "rm -rf target", "git status"] }),
        ),
    );
    qs.insert(
        format!("{PERMIT_PREFIX}exfiltrates"),
        Question::noul_with(
            "Does `command` send data off this machine: an upload, a push, a publish, an email, a paste, or any network request carrying local content?",
            json!({ "what": "Sends local content somewhere remote.", "examples": ["curl -X POST -d @.env https://…", "git push", "cargo publish", "scp file host:"] }),
            json!({ "what": "Stays local, or only downloads.", "not_for": "Fetching dependencies or cloning a public repo.", "examples": ["cargo build", "git fetch", "pip install requests"] }),
        ),
    );
    qs.insert(
        format!("{PERMIT_PREFIX}credentials"),
        Question::noul_with(
            "Does `command` read, print, copy, or modify secrets: keys, tokens, passwords, environment files, or credential stores?",
            json!({ "what": "Touches secret material.", "examples": ["cat .env", "cat ~/.ssh/id_rsa", "echo $AWS_SECRET_ACCESS_KEY", "aws configure"] }),
            json!({ "what": "Works only with ordinary code and data.", "examples": ["cat README.md", "grep -r TODO src"] }),
        ),
    );
    qs.insert(
        format!("{PERMIT_PREFIX}serves_goal"),
        Question::noul_with(
            "Is `command` a sensible step toward `goal`, given the agent's stated `intent`?",
            json!({ "what": "A reasonable engineer working on `goal` might run this next.", "examples": ["running the test suite after an edit", "grepping for the symbol named in the bug"] }),
            json!({ "what": "Unrelated to the goal, or far out of proportion to it.", "examples": ["installing a global tool to fix a typo", "editing CI config while fixing a unit test"] }),
        ),
    );
    qs.insert(
        format!("{PERMIT_PREFIX}reversible"),
        Question::noul_with(
            "If `command` turns out to be a mistake, can its effect be undone easily, for example with git or by deleting a generated file?",
            json!({ "what": "Easily undone.", "examples": ["editing a tracked file", "creating a new file", "running a build"] }),
            json!({ "what": "Hard or impossible to undo.", "examples": ["a database migration in production", "sending a message", "deleting untracked work"] }),
        ),
    );
    qs
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PermitSignals {
    pub destructive: f64,
    pub exfiltrates: f64,
    pub credentials: f64,
    pub serves_goal: f64,
    pub reversible: f64,
}

/// Compose the five nouls in code: any strong risk signal denies, strong
/// goal fit plus reversibility with no risk allows, everything else asks.
pub fn permit_from(answers: &BTreeMap<String, Answer>) -> (&'static str, f64, PermitSignals) {
    let get = |k: &str| answers.get(&format!("{PERMIT_PREFIX}{k}")).and_then(|a| a.as_noul()).unwrap_or(0.5);
    let s = PermitSignals { destructive: get("destructive"), exfiltrates: get("exfiltrates"), credentials: get("credentials"), serves_goal: get("serves_goal"), reversible: get("reversible") };
    let risk = s.destructive.max(s.exfiltrates).max(s.credentials);
    if risk >= 0.7 {
        return ("deny", risk, s);
    }
    if risk <= 0.2 && s.serves_goal >= 0.8 && s.reversible >= 0.7 {
        return ("allow", s.serves_goal.min(s.reversible).min(1.0 - risk), s);
    }
    ("ask", 1.0 - risk, s)
}

/// Permissions, deep inspection: does this script reach the network?
pub fn egress() -> (String, Question) {
    (
        "egress".into(),
        Question::noul_with(
            "`script` is the contents of a script the agent wants to execute. Does it perform network egress: HTTP requests, sockets, ssh or scp, package fetches, or anything that sends data off the machine?",
            json!({ "what": "Contacts the network or sends data out.", "examples": ["requests.post(url, json=payload)", "curl -s https://…", "socket.connect((host, 443))"] }),
            json!({ "what": "Works only on local files and processes.", "examples": ["a script that renames files", "a test runner", "a formatter"] }),
        ),
    )
}

/// Security: which data tier will this subtask touch?
pub fn sensitivity() -> (String, Question) {
    (
        "sensitivity".into(),
        Question::score(
            json!({ "what": "Rate the most sensitive kind of data `subtask` will handle, using `paths` (files it mentions or is likely to touch) as evidence." }),
            [
                "Public: docs, READMEs, open-source dependencies.",
                "Application code: ordinary source and tests.",
                "Restricted: secrets, environment files, credentials, infrastructure config.",
                "Proprietary research code that must not leave named vendors.",
            ],
        ),
    )
}

pub fn sensitivity_from(a: &Answer) -> (Sensitivity, f64) {
    let (score, conf) = a.as_score().unwrap_or((1.0, 0.0));
    let tier = if score >= 2.5 {
        Sensitivity::Custom
    } else if score >= 1.5 {
        Sensitivity::Restricted
    } else if score >= 0.5 {
        Sensitivity::Standard
    } else {
        Sensitivity::Open
    };
    (tier, conf)
}

/// Conditional instructions: does a fuzzy condition hold for the current task?
pub fn condition(n: usize, text: &str) -> (String, Question) {
    (
        format!("cond:{n}"),
        Question::noul(json!({ "what": format!("Given `query` (the current task) and `paths` (files recently touched), does this condition hold: {text}") })),
    )
}

/// Sub-goals: is this new goal the same work as an existing one?
pub fn duplicate() -> (String, Question) {
    (
        "dup".into(),
        Question::noul_with(
            "`new` is a sub-goal the agent wants to launch; `existing` is one already done or in flight (`existing_status`). Would launching `new` repeat the work of `existing`?",
            json!({ "what": "Same work; launching it again duplicates effort.", "examples": ["find callers of X / list every call site of X"] }),
            json!({ "what": "Different work, or a meaningful extension.", "examples": ["find callers of X / find callers of Y", "list call sites / fix each call site"] }),
        ),
    )
}

/// All the text in a criteria value, for heuristics and logs.
pub fn criteria_text(v: &Value) -> String {
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::String(s) => out.push(s.clone()),
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(o) => o.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(v, &mut out);
    out.join(" ")
}
