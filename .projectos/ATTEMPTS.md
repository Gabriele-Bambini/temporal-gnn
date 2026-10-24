# Attempts - Temporal GNN
Append-only. One entry per approach/experiment. Never delete; update status instead.
Format (keep the heading exactly, `pos attempt` parses it):

    ### A-001 | YYYY-MM-DD | agent | STATUS | short title
    - goal: what it should achieve
    - approach: what was done (config, params, commit)
    - result: outcome + metric
    - evidence: log/file/commit path
    - lesson: what to do or avoid next time
    - tags: kw1, kw2

STATUS: RUNNING | SUCCESS | PARTIAL | FAILED | ABANDONED | INCONCLUSIVE


### A-001 | 2026-09-30 | codex | SUCCESS | Causal context and change evaluation
- approach: Train-prefix scaler, support-weighted past context, change metrics and conservative verdict
- result: 16 tests passed in 12.57 seconds including node loader
- evidence: tgl-grn-continuous/tests
- lesson: Software unit checks only; no real benchmark evidence yet

### A-002 | 2026-09-30 | codex | SUCCESS | Directed temporal encoder and flow objective
- goal: Implement matched-capacity direct Temporal GNN and aggregate-state Flow Matching
- approach: Pure Torch directed weighted means, per-node GRU with BPTT, midpoint tau integration and displacement loss
- result: 13 targeted tests passed in 9.52 seconds after missing-module RED
- evidence: tgl-grn-continuous/tests/test_models.py
- lesson: Unit tests validate mechanics only; real-data prediction remains untested
- tags: dynamics,flow-matching,tgnn,unit-tests

### A-003 | 2026-09-30 | codex | SUCCESS | Bounded streaming GRN top-k reader and directed statistics
- goal: Enable original-data Direct TGNN versus FM experiment without loading multi-GB CSVs in RAM
- approach: Pandas chunk streaming, per-source top-k, metadata validation, NPZ fingerprint cache, directed weighted strengths and dangling PageRank
- result: 16 focused tests passed in 2.76 seconds; no full-data run performed
- evidence: tgl-grn-continuous/tests/test_edge_cache.py; tgl-grn-continuous/tests/test_graph.py
- lesson: Top-k is computational sparsification, not regulatory confidence; graph stats describe retained graph only
- tags: streaming,graph-data,prototype

### A-004 | 2026-09-30 | codex | RUNNING | Original-dataset end-to-end quick benchmark
- goal: Compare direct Temporal GNN and graph flow on held-out changes
- approach: Streaming top8 per regulator, original pancreas/gastrulation, 40 epochs, seed0, six neural variants plus three baselines
- lesson: Unlike prior full-table scan, chunks bound edge memory; one seed remains exploratory
