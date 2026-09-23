//! The harness's questions to Jev, one builder per decision point (Table I).
//!
//! Each builder returns a question id and a [`Question`]; ids are stable so the
//! local transport and the event log can recognise them.

use super::{Answer, Question};
use crate::config::ModelSpec;
use crate::state::{ChunkId, Sensitivity, Visibility};

pub const VIS_PREFIX: &str = "vis:";

/// Context: how visible should this chunk be for the current query?
pub fn visibility(id: &ChunkId) -> (String, Question) {
    let q = Question::choice(
        format!(
            "Chunk {} is one piece of the agent's stored state, previewed in state.chunks. \
             Given state.goal and state.query, how much of it should the model see on this turn?",
            id.hex()
        ),
        [
            ("hide", "Irrelevant to the query; showing it would only spend tokens."),
            ("short", "Worth a one-line reminder that it exists and what it concluded."),
            ("long", "Worth a paragraph: the key facts, errors, or hits, but not every line."),
            ("full", "Directly needed; show it verbatim."),
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
            "state.previous lists the chunks the model saw last turn, in order. state.delta_tokens is how much new \
             material this turn adds and state.stale how many previous chunks are no longer relevant. Should the \
             harness keep the previous ordering as a cached prefix and append the new material, rather than \
             rebuilding the context from scratch for state.query?",
            "Reuse: the previous context still frames this query well; appending is cheaper and loses nothing.",
            "Rebuild: enough has changed that a fresh, query-ordered context will serve the model better.",
        ),
    )
}

/// Routing: which eligible model should take this subtask?
pub fn route(models: &[&ModelSpec]) -> (String, Question) {
    let q = Question::choice(
        "state.subtask describes the work, state.context_tokens the size of the purpose-built context, \
         state.costs the estimated dollar cost per model including the reprocessing pass back to the frontier \
         model, and state.sensitivity the data tier. Which model should run it? Prefer the cheapest model that \
         will do the work correctly in one pass; a failed cheap attempt costs more than a frontier pass.",
        models.iter().map(|m| (m.id.clone(), format!("{:?} tier. {}", m.tier, m.description))),
    );
    ("route".into(), q)
}

/// Tools: which tool matches this intent?
pub fn tool_pick(snippets: &[(String, String)]) -> (String, Question) {
    let q = Question::choice(
        "state.intent is what the model says it wants to do next, in plain words. Which single tool best \
         performs that action?",
        snippets.iter().map(|(id, snip)| (id.clone(), snip.clone())),
    );
    ("tool".into(), q)
}

/// Permissions: should this command run?
pub fn permit() -> (String, Question) {
    let q = Question::choice(
        "state.command is a shell command (or a file operation) the agent wants to run inside state.root as part \
         of state.goal, with declared access state.access. Should it run?",
        [
            ("allow", "Safe and clearly in service of the goal; run it without asking."),
            ("ask", "Plausible but consequential or unusual; a human should confirm."),
            ("deny", "Destructive, exfiltrating, out of scope, or touching credentials; do not run."),
        ],
    );
    ("permit".into(), q)
}

/// Permissions, deep inspection: does this script reach the network?
pub fn egress() -> (String, Question) {
    (
        "egress".into(),
        Question::noul_with(
            "state.script is the contents of a script the agent wants to execute. Does it perform network egress \
             (HTTP requests, sockets, ssh/scp, package fetches, or anything that sends data off the machine)?",
            "Yes, it contacts the network or sends data out.",
            "No, it works only on local files and processes.",
        ),
    )
}

/// Security: which data tier will this subtask touch?
pub fn sensitivity() -> (String, Question) {
    (
        "sensitivity".into(),
        Question::score(
            "state.subtask describes the work and state.paths lists files it mentions or is likely to touch. \
             Rate the most sensitive kind of data it will handle.",
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
        Question::noul(format!(
            "state.query is the current task and state.paths the files recently touched. Does this condition hold: {text}"
        )),
    )
}

/// Sub-goals: is this new goal the same work as an existing one?
pub fn duplicate() -> (String, Question) {
    (
        "dup".into(),
        Question::noul_with(
            "state.new is a sub-goal the agent wants to launch; state.existing is one already done or in flight. \
             Would launching state.new repeat the work of state.existing?",
            "Same work: launching it again would duplicate effort.",
            "Different work, or a meaningful extension of it.",
        ),
    )
}
