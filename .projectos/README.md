# .projectos - project memory (ProjectOS)

Shared, agent-agnostic memory for this project (Claude Code, Codex, Antigravity, humans).
Read `PROTOCOL.md` for the rules. Files:

| file | role | who edits |
|---|---|---|
| PURPOSE.md | why / done-when / constraints / non-goals | user (agents propose) |
| MAP.md | architecture map so nobody re-reads the codebase | cheap mapper model |
| STATE.md | live focus, next steps, blockers | every agent, continuously |
| ATTEMPTS.md | append-only log of approaches tried and their outcome | every agent |
| DECISIONS.md | append-only design decisions (ADR-lite) | every agent |
| checkpoints/ | immutable snapshots (manual + auto) | `pos checkpoint` / hooks |
| scan.md | tier-0 mechanical scan (no LLM) | `pos scan` |
| .runtime/ | machine-local caches (gitignored) | hooks |
