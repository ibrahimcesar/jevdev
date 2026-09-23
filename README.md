# jevdev

A coding-agent harness built around [Jev](https://docs.typesafe.ai), TypeSafe's System One decision model.

`jevdev` treats the context window as something assembled on purpose rather than something that accumulates by accident. All agent state lives as explicit, typed, content-addressed chunks, and Jev answers the per-turn questions that current agents leave to defaults:

| Decision point | Question to Jev | Typed answer |
|---|---|---|
| Context | How visible should this chunk be for this query? | choice: hide / short / long / full |
| Cache | Reuse the cached prefix or rebuild? | noul + probability |
| Routing | Can this subtask leave the frontier model? | choice + cost estimate |
| Tools | Which tool fits this intent? | ranked choice, top-k |
| Permissions | Should this command run? | allow / ask / deny |
| Security | Which files will this task touch? | sensitivity score |

## Status

`0.0.1` reserves the crate name. The harness (chunk store, context assembler, router, tiered tool registry, Cedar permissions, Anthropic client, and a terminal UI) lands in `0.1.0`.

## License

Apache-2.0
