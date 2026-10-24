# ProjectOS Protocol v1

One protocol for every agent (Claude Code, Codex, Antigravity/Gemini, others) and for humans.
Goal: any agent can pick up any project cold, in under 2k tokens, without re-reading the codebase,
without repeating failed approaches, without losing work when usage runs out, and without
stepping on another agent working on the same project.

`.projectos/` is the single source of truth for project state. Other memory tools (claude-mem,
cortex, openviking, auto-memory) are optional caches; if they disagree with `.projectos/`, `.projectos/` wins.

## 1. Files

| file | content | rule |
|---|---|---|
| `PURPOSE.md` | why, definition of done, measurable success criteria, constraints, non-goals | Owned by the user. Agents draft; changes need user confirmation + a DECISIONS entry. |
| `MAP.md` | architecture: how to run, modules, data flow, entry points, gotchas | Written by a cheap model from `scan.md`. Kept incremental. |
| `STATE.md` | focus, done recently, in progress, next steps, blockers, notes for next agent | Live. Every agent keeps it current. Must let a cold agent continue. |
| `ATTEMPTS.md` | append-only log of approaches/experiments and outcomes | Never delete. Update status. |
| `DECISIONS.md` | append-only design decisions (ADR-lite) | Supersede, never rewrite. |
| `checkpoints/` | immutable snapshots (manual + automatic) | Written by `pos checkpoint` and hooks. |
| `scan.md` | tier-0 mechanical scan (layout, deps, entry points, symbol index) | Regenerate with `pos scan`. Free. |
| `map.lock.json` | fingerprints of mapped file groups (staleness detection) | Written by `pos map-done`. |
| `.runtime/` | machine-local caches (sessions, scan cache) | gitignored; never edit. |

## 2. Session lifecycle

1. **Orient** (budget ~2k tokens). Claude Code/Codex: the SessionStart hook injects a brief. Other agents: run `pos brief`
   (or read STATE.md + PURPOSE.md + the tail of ATTEMPTS.md). Do not scan the repo before reading the brief.
2. **Locate** code through `MAP.md` -> `scan.md` -> graphify report (`graphify-out/GRAPH_REPORT.md`, if present)
   -> targeted Grep -> targeted reads. Never read data/output dirs wholesale.
3. **Before choosing an approach**: `pos attempt list --grep "<keywords>"`. If a similar attempt FAILED or was
   ABANDONED, either state explicitly what is different this time or choose another path.
4. **Work** in small verifiable units. Announce parallel work with `pos claim "<task>"` when other agents are active.
5. **After each unit** (feature done, experiment finished, bug fixed, before risky operations):
   log attempts/decisions -> update STATE.md -> `pos checkpoint -m "<one-line summary>"`.
6. **End** of session: STATE.md "Next steps" + "Notes for the next agent" must be sufficient to continue.
   Checkpoint with `--reason end`.

## 3. Attempts (memory of what was tried)

Log every approach whose outcome would matter to a future agent: experiments, model/hyperparameter trials,
debugging hypotheses that were tested, refactors that were reverted, tools/libraries evaluated.

```
pos attempt add -t "GAT 8 heads on grn500" -s failed --goal "beat GCN AUPRC" \
  --approach "heads=8, lr=1e-3, 200 ep, commit abc123" --result "AUPRC 0.41 vs GCN 0.47" \
  --evidence "OUT/run_17/metrics.json" --lesson "attention overfits at n<1k edges; try edge dropout" --tags gat,grn500
```

Status: RUNNING (started, outcome pending) | SUCCESS | PARTIAL | FAILED | ABANDONED | INCONCLUSIVE.
Long jobs: log RUNNING at launch, `pos attempt update A-007 -s success --result "..."` when done.
A lesson is mandatory for FAILED/ABANDONED: it is what prevents the next agent from repeating it.

## 4. Decisions

Record choices that constrain future work (architecture, library, data split, metric, scope):
`pos decide -t "Use PyG over DGL" --decision "..." --why "..." --alternatives "..." [--supersedes D-002]`.
Use `--proposed` when the user has not confirmed. PURPOSE changes are always decisions.

## 5. Purpose alignment

Before starting a sizeable unit, name the PURPOSE success criterion it advances. If none: flag it to the user
(scope creep) instead of silently doing it. Checkpoints take `--advances "<criterion>"`.
If PURPOSE.md is still a template, ask the user 2-3 short questions at a natural moment and fill it.

## 6. Checkpoints

Content = one-line summary + STATE.md snapshot + mechanical delta (edited files, git status, usage).
Triggers, in order of reliability:
- **Agent**: after each unit of work, before risky ops, at session end (protocol above).
- **Usage** (hooks): at `warn_pct` (75%) a reminder; at `force_pct` (88%) and +7% a forced checkpoint request
  (context window or 5h/7d rate limits, whichever is highest). Obey immediately, emergency mode (below).
- **Periodic** (Stop hook): files edited and no checkpoint for 60 min.
- **Automatic** (no agent involvement): before context compaction and at session end the hooks write an auto
  checkpoint with edited files + recent prompts, so even a hard stop leaves a trace.

**Emergency mode** (usage forced, <2 min): edit STATE.md sections Focus / In progress / Next steps / Notes
(bullets, no prose) -> `pos checkpoint -m "..." --reason usage`. Skip everything else.

Agents without hooks (Antigravity, others): checkpoint after every unit of work; if you can observe your own
usage, apply the same thresholds. Codex/Claude can query `pos usage`.

## 7. Multi-agent coordination (no overlap)

- The brief lists other active sessions on the same project (any agent, any machine) and their claims.
- Before editing, check claims; do not modify files another active agent is working on. Split by file/module.
- `pos claim "<task>"` when starting a distinct task; `pos release` when done.
- Handoff between agents = STATE.md + checkpoint. For agents without file access: `pos export --out HANDOFF.md`.
- Instructions live only here + in the thin entry blocks of AGENTS.md / CLAUDE.md / `.agent/rules/projectos.md`.
  Never copy protocol text elsewhere; point to this file.

## 8. Model economy (never waste usage)

| tier | Claude | Codex / others | use for |
|---|---|---|---|
| T0 | script (`pos`) | script (`pos`) | scanning, fingerprints, digests, checkpoints: zero LLM tokens |
| T1 | haiku | smallest/mini model | mapping, summarising, extraction, formatting, log condensing |
| T2 | sonnet | standard model | synthesis of large maps, code generation, review, debugging |
| T3 | opus | top model | novel reasoning, architecture, critical decisions only |

Mapping is never done on T3 and never by the main agent reading the whole repo. Delegate to `pos-mapper`
(T1). Only size class L gets a T2 `pos-architect` synthesis pass, and it reads the T1 parts, not the code.
Incremental remaps touch only the groups listed as stale.

## 9. Without the `pos` CLI

Everything is plain markdown. Read the files directly; append entries using the exact heading formats in
ATTEMPTS.md / DECISIONS.md; write checkpoints as `checkpoints/YYYY-MM-DD_HHMMSS_<agent>_<slug>.md` with a
`---` frontmatter (time, agent, kind, reason, summary) followed by a STATE.md snapshot.
