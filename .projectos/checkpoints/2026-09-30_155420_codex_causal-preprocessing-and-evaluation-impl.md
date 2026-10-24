---
time: 2026-09-30T15:54:20+02:00
agent: codex
kind: manual
reason: milestone
summary: Causal preprocessing and evaluation implemented; graph/model modules in progress
usage: ctx 17% | 5h 59% | 7d 60%
machine: DESKTOP-0HEPV91
---
# Checkpoint - Causal preprocessing and evaluation implemented; graph/model modules in progress

## STATE.md snapshot

#### State - Temporal GNN
Updated: 2026-09-30T15:49:51+02:00 by codex

#### Focus
Implement the user-requested isolated tgl-grn-continuous prototype on original pancreas/gastrulation aggregate GRNs. Compare direct Temporal GNN and conditional flow matching without copying original code.

#### Done recently
-

#### In progress
- Parent: causal preprocessing, evaluation, training/CLI integration. graph_data agent: edges.py/graph.py. dynamics agent: models.py. map_state agent: MAP.md only.

#### Next steps
1. Verify tests, then run pancreas and gastrulation end-to-end with baseline/control comparisons.

#### Blockers / open questions
- No real benchmark result yet. Source data are aggregate gene summaries, not raw cells; rebuilding biological bins or recovering cell distributions is outside what these files support.

#### Notes for the next agent
- Python: C:\Users\02gab\AppData\Local\Programs\Python\Python312\python.exe. Set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and PYTHONUTF8=1. rtk unavailable. Original data: tgl-grn-1CCD/GRN. 6.51 GB RAM available, CPU only at start of 2026-09-30 work.

