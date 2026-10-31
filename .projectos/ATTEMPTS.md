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

### A-005 | 2026-10-30 | claude | SUCCEEDED | OT coupling vs kNN on clonal truth, independent LARRY replication
- goal: Confirm on independent barcoding data that, at equal spread, OT couplings recover sister cells better than kNN and than the kernel without mass constraints
- approach: LARRY cytokine perturbation GSM4185644, clone-label integrity check first, prereg tack/prereg/2026-10-30-larry-cytokine-prereg.md, 9 units, frozen larry_vivo_confirm unit()/rule()
- lesson: H1 9/9 and H2 9/9 pass, best-vs-best 5/9 fails (sharp kNN K 1-3 wins in G-csf / M-csf); LARRY in vivo GEO clone labels are broken (sisters carry no fate signal inside mice), always run the integrity check before using clone labels

### A-006 | 2026-10-31 | claude | FAILED (primary) | WFR (Hellinger-Kantorovich) interpolant vs 50/50 mix
- goal: Test whether letting far mass fade (WFR) while transporting near pairs beats the mix (interpolation) and persistence (forecast)
- approach: tack/hpc/wfr_confirm.py, prereg tack/prereg/2026-10-31-wfr-prereg.md, atlas + zebrafish + LARRY products already on HPC; differs from failed uot (A in decisions 2026-10-25) by fading far mass and nesting the mix as delta=0
- lesson: H1 vs mix fails (atlas 4/7, zebrafish 5/9 collapsing to delta 0); WFR beats OT pairs everywhere; forecast beats persistence on atlas only (6/6, 4/6 after a same-subsample control), not zebrafish (3/8); delta=0 and persistence must share the same subsample

### A-007 | 2026-10-31 | claude | FAILED | Mellon time-continuous density vs 50/50 mix
- goal: Test whether a density-over-time model (not transport) beats the mix at reconstructing a missing time point
- approach: tack/hpc/mellon_confirm.py (Mellon 1.7.1 in ~/envs/mellon, own Palantir-style diffusion map), prereg tack/prereg/2026-10-31-mellon-prereg.md; differs from failed dynmix (logit shares, k-means) by a GP density over continuous state and time, reweighting the neighbours' own cells
- lesson: H1 fails (atlas 0/7, zebrafish 1/9); log-density weights collapse (ESS 4-9 % atlas), corr(log w, one-sided gap) -0.44..-0.99; log-geometry kills one-sided states
