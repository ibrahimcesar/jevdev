# jevdev · roadmap

Ordered by value. Checked items are shipped on `main`.

## Now

- [x] **Per-chunk Jev memo.** Key visibility scores on (chunk, goal) instead of the whole batch, so unchanged older chunks are never re-scored. Rescore recent chunks every turn and everything after a "rebuild" cache decision.
- [x] **`jevdev doctor`.** Check keys and credential sources, round-trip Jev and the cheap model with latency, parse the policy and instructions, open the store.

## Next

- [x] **Integration test + `run --json` + `replay`.** The offline loop is a `cargo test`; `run --json` emits events as JSON lines; `replay` re-assembles a recorded session turn by turn without a model. `RUST_LOG=jev=debug` prints every System One request.
- [x] **Questions per TypeSafe's guide.** Backticked state paths, contrastive `what`/`not_for`/`examples` criteria, permissions decomposed into five nouls composed in code.
- [ ] **Heatmap filtering of big outputs.** Score lines or hunks of a large tool output with a Jev fan-out and show only the relevant ones at the `long` level.
- [ ] **Parallel sub-agents.** Run several read-only sub-agents concurrently in a `JoinSet`; leases and snapshots already permit it.
- [ ] **Streaming frontier calls.** SSE parsing so the TUI shows text as it arrives and long turns cannot time out.
- [ ] **File and diff chunks.** `Kind::File` with a revision on reads, `Kind::Diff` after writes, git history via git2.
- [ ] **Cross-model background review.** A second retrieval subscriber that asks another model to review each change and writes findings back as chunks.

## Later

- [ ] **Real token counts.** Anthropic `count_tokens` or a local tokenizer in place of the byte heuristic.
- [ ] **Batteries from the shortlist.** ast-grep structural search; a headroom-style noul asking Jev whether a summary kept the facts the query needs; MCP servers registered as tier-1 snippets.
- [ ] **CLI polish.** Shell completions (clap_complete), `jevdev sessions` to list and resume, `jevdev config` to print the effective config.
- [ ] **Open-weight routes.** An OpenAI-compatible provider so sensitivity-based routing has real cheaper alternatives to route to.
- [ ] **TUI tests.** Snapshot tests with ratatui's `TestBackend`.
