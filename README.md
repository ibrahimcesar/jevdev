# jevdev

A coding-agent harness built around [Jev](https://docs.typesafe.ai), TypeSafe's System One decision model.

[![crates.io](https://img.shields.io/crates/v/jevdev.svg)](https://crates.io/crates/jevdev)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Coding agents are a loop around a model with a few tools, and the loop is not where the leverage is. The leverage is in what the harness puts in front of the model on every turn. Today that decision is made by default, by an append-only transcript shaped around KV-cache economics. `jevdev` makes it on purpose: all agent state lives as explicit, typed, content-addressed chunks, and Jev answers the per-turn questions that current agents leave to defaults.

| Decision point | Question to Jev | Typed answer |
|---|---|---|
| Context | How visible should this chunk be for this query? | choice: hide / short / long / full |
| Cache | Reuse the cached prefix or rebuild? | noul + probability |
| Routing | Can this subtask leave the frontier model? | choice + cost estimate |
| Tools | Which tool fits this intent? | ranked choice, top-k |
| Permissions | Should this command run? | allow / ask / deny |
| Security | Which files will this task touch? | sensitivity score |

Jev is not the model that writes the code. Frontier models, sub-agents, tools, and deterministic code do the work. Jev returns typed choices, scores, and noul decisions with probabilities, so the harness validates, thresholds, and branches on them without parsing prose.

## Install

```sh
cargo install jevdev
```

Or from source:

```sh
git clone https://github.com/ibrahimcesar/jevdev && cd jevdev && cargo install --path .
```

## Quick start

```sh
cd your-repo
jevdev init                 # writes jevdev.toml, .jevdev/exec.cedar, AGENTS.md
export TYPESAFE_API_KEY=…   # Jev (https://docs.typesafe.ai)
export ANTHROPIC_API_KEY=…  # or `ant auth login`
jevdev                      # the terminal UI
```

Without either key the harness still runs end to end: Jev's questions are answered by a local heuristic transport, and a scripted demo model stands in for the frontier model. That mode exists for development and tests, not for real work.

```sh
jevdev run "make the failing test in auth pass" --yes   # plain output, no prompts
jevdev context "why does login expire early"            # dry-run the context assembly
jevdev policy "python3 scripts/deploy.py"               # what the execution policy says
jevdev tools grep                                       # tier 2: a tool's schema
jevdev cost --x 0.65 --y 0.12 --z 0.23                  # the routing arithmetic
jevdev state                                            # every chunk in the store
```

## The terminal UI

```
 jevdev  session 5b9221 · turn 3 · claude-sonnet-5 · anthropic · ~/src/app
┌ transcript ───────────────────────────────┐┌ jev decisions ───────────────────┐
│ you › fix the login expiry bug            ││ turn 3                           │
│ act › search for "validateSession"        ││ cache     reuse prefix · kept 5  │
│       → grep {"pattern":"validateSession"}││ security  standard · 2 paths     │
│ tool › grep · read · 340 tok              ││ route     claude-sonnet-5 $0.031 │
│       auth/session.ts:12: validateSession ││ tool      grep  [grep 0.91 …]    │
│       auth/session.ts:40: expire(login)   ││ permit    allow  allow_reads     │
│ act › read auth/session.ts lines 1 to 80  │└──────────────────────────────────┘
│ …                                         │┌ context 4.2k / 60k tok ─────────┐
│                                           ││ ████ full   120  1.00 📌 Rust st…│
│                                           ││ ███  long   380  0.60 grep outp… │
│                                           ││ ██   short   60  0.55 file auth… │
└───────────────────────────────────────────┘└──────────────────────────────────┘
 llm in 18k · cached 12k · out 2k · $0.11   jev 21 calls · 96 questions · 31k tok · $0.0013
┌ input ─────────────────────────────────────────────────────────────────────────┐
│ ▏                                                                              │
└────────────────────────────────────────────────────────────────────────────────┘
```

Enter sends a goal. When the policy says `ask`, the input line becomes the permission prompt and `y` or `n` answers it. Esc quits. Up, Down, PageUp, PageDown, End scroll the transcript.

## How it works

### Explicit state

Everything the model has ever seen is a `Chunk`: user turns, model notes, tool calls, tool outputs, files, diffs, instruction fragments, summaries, sub-agent results. Chunks are immutable, addressed by a blake3 hash of their content, and appended to a `redb` log under `.jevdev/state.redb`. A snapshot of the store is an O(1) clone of persistent maps, so read-only tasks hold one without blocking writers. A restart is a new session id over the same log: old chunks stay addressable and come back only when Jev scores them relevant.

### Questions, the TypeSafe way

Every question follows [TypeSafe's guide](https://docs.typesafe.ai/concepts/how-to-build-with-system-one): instructions point at state with backticked paths such as `chunks[3]`, choice and noul criteria are contrastive objects with `what`, `not_for`, and `examples`, and broad judgments are decomposed into atomic nouls that code composes. Permissions, for instance, are five nouls (destructive, exfiltrates, credentials, serves the goal, reversible) combined by thresholds in `permit_from`, never one "should this run?" question. All questions about one state go in one request, so Jev evaluates them in parallel.

### Context as a decision

Each turn the assembler prefilters candidates deterministically (recent turns always, older chunks by keyword overlap), then asks Jev one batched System One request with a `choice` question per chunk: hide, short, long, or full. Short and long views are summaries generated once and stored as chunks. The result is packed to the token budget, pinned instructions first, then by probability per token.

Before ordering, Jev answers a `noul`: is the previous prefix still a good frame for this query? If so, the previous order is kept and new material appended, so the provider cache hits. If not, the context is rebuilt chronologically. Both costs are shown.

### Routing priced per context rebuild

The router filters models by data sensitivity (Table IV of the design notes: open, standard, restricted, custom), prices each candidate with the two formulas from the notes, and lets Jev choose. The routed path is priced with a purpose-built context and a scored summary on the way back, which is what makes it cheaper than pure frontier:

```
$ jevdev cost
Path 1  pure frontier                                    4.15
Path 2  frontier → worker → frontier (full transcript)   4.71
Path 3  frontier → worker → frontier (jev harness)       1.94
```

### Tools in tiers

The model never sees a schema. It writes one `<act>` block in plain words. Jev ranks the tier-1 snippets (one line per tool), the harness loads the tier-2 schema for the top candidates only, the cheap model fills the arguments, `jsonschema` validates them, and the typed call runs. Nothing from tiers two and three is ever written into a chunk.

Built in: `read_file`, `list_files`, `grep`, `write_file`, `str_replace`, `shell`, `delegate`.

### Programmable permissions

Every call is judged by a [Cedar](https://www.cedarpolicy.com) policy in `.jevdev/exec.cedar` over attributes the harness computes first: what it touches, whether it is read-only, whether it is destructive, whether it leaves the repository. One attribute is filled by Jev deep inspection: whether a script the command would execute reaches the network. A forbid is `deny`, a permit is `allow`, no match is `ask`, where Jev gets the first opinion and a human the last word.

```cedar
@id("deny_script_egress_outside_deploy")
forbid (principal, action == Action::"exec", resource)
when { resource.script_egress && context.task_scope != "deploy" };
```

### Conditional instructions

`AGENTS.md` sections can carry a marker:

```markdown
## Rust style
<!-- when: glob **/*.rs -->
- No `unwrap()` outside tests.
```

Conditions are `glob <pattern>`, `dir <path>`, or `fuzzy <question for Jev>`. Fragments are re-evaluated from task state each turn and pinned in full while they hold, so no summary can lose them.

### Sub-agents and background work

`delegate` spawns a read-only sub-agent on the worker model with its own small context. Sub-goals are deduplicated against earlier ones (keyword overlap, then a Jev `noul`). Writes take per-path leases; reads never contend. After each write, one retrieval pass finds the related files and chunks and publishes them on a watch channel; every background task reads that pass instead of repeating it. The bundled task keeps `.jevdev/progress.md` current.

## Configuration

`jevdev init` writes a commented `jevdev.toml`. The parts you will touch:

```toml
[jev]
transport = "auto"        # http when TYPESAFE_API_KEY is set, else local heuristics
model = "jev-latest"

[llm]
provider = "anthropic"    # or "scripted"
frontier = "claude-opus-5"
worker = "claude-sonnet-5"
cheap = "claude-haiku-4-5"
effort = "high"
routing = true            # false pins the main loop to the frontier model

[budget]
context_tokens = 60000

[security]
restricted = ["**/.env*", "**/secrets/**", "infra/**"]

[policy]
file = ".jevdev/exec.cedar"
jev_auto_allow = 0.9
```

Models are listed under `[[models]]` with prices and a trust tier (`first-party`, `vetted`, `open`) that the sensitivity policy uses to decide eligibility.

## Layout

```
src/
  state/     chunk types, redb-backed store, snapshots
  jev/       System One client, HTTP and local transports, the six questions
  context/   assembler, visibility ladder, cache decision, summaries
  router/    prices, cost model, sensitivity policy, model choice
  tools/     tool trait, tiered registry, arg builders, built-ins
  policy/    Cedar engine plus Jev deep inspection
  llm/       Anthropic Messages API over HTTP, scripted client, step parsing
  runtime/   turn loop, events, conditional instructions, sub-goals, background
  tui/       ratatui interface
policies/exec.cedar   the default policy
```

## Status

Early. The design is from a synthesis of design notes on building a coding agent around Jev; the code implements each part of it end to end, with an offline mode for development. Things that are deliberately simple for now: token counts are a byte heuristic, summaries fall back to truncation without a model, and the local Jev transport is keyword heuristics rather than Jev.

## License

Apache-2.0
