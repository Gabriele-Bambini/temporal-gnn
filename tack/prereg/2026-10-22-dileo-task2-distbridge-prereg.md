# Pre-registration: DistBridge vs Dileo & Sottoriva, Task 2, on their terrain (2026-10-22)

Frozen before any DistBridge metric on real test genes is computed. Exploratory results already seen (and labelled
as such): the strict ObsLevel rule in their loop (job 712742: gastrulation-human obslevel6 0.600 / 0.398 / 0.530,
pancreas-human obslevel4 0.474 / 0.292 / 0.417) and the workflow prototypes listed in
tack/dileo_bench/distbridge_spec.md section 7.

## Terrain (not ours to change)
- Code: the original repository https://anonymous.4open.science/r/tgl-grn-1CCD, untouched in ~/tgl-grn-orig
  (md5 dataset.py 96bcd18e..., train_models_gene.py 98034848..., models_gene.py 556897b5..., train_gene.py af755820...).
  Anchor (job 712735): their Linear reproduces the paper row on both builds.
- Data: their GRN/ files, builds gastrulation-human (29 transitions, N = 21430) and pancreas-human (14, N = 18561).
- Our runner tack/hpc/dileo_harness.py imports their load_dataset, train_val_test_split,
  build_snapshot_split_future_node_delta_ranking (targets and RandomNodeSplit 25 % test genes) and _delta_metrics;
  seeding as train_gene.py; seeds 1234, 42, 7, 0, 1. Score = their nanmean over transitions, then mean over seeds.

## Models (code hashes in the decisions.md row that records this file's sha256)
- distbridge_strict (headline): tack/hpc/dist_bridge.py. Per-gene cell-count law from their node stats; pool in
  {Box k = 2, 4, 6, 8; Kalman theta = 0.1, 0.2, 0.35}; OT / Schroedinger-bridge step (own log-domain Sinkhorn, Brownian
  cost on the Anscombe scale, eps = 0.25 L, L = min(3, i)) with extrapolation delta in {0, 0.5, 1}; count observation
  model with n = cells of the current bin; 21 configurations chosen per transition by rolling origin over the past
  <= 8 transitions (j >= 1) with J = (Pearson + tie-aware expected P@K up + down) / 3, K = round(200 / round(0.25 N) * N);
  ties within 1e-3 prefer delta = 0, then the earlier configuration; default (Kalman 0.2, delta 0) for i <= 2;
  own bin at i = 0. Strict: reads only raw node stats of snapshots <= t; no training-gene labels.
- distbridge_strict_nosb (ablation): identical, delta fixed to 0 (no OT step).
- obslevel_rollk (comparator): Box k only, delta 0, same selection.
- References already defined: level4, level6, obslevel4, obslevel6 (post-hoc k, exploratory).

## Gates
- G1 (beats the paper): for each dataset and metric (PCC, P@200 up, P@200 down), mean - 2.132 sd / sqrt(5) over the
  5 seeds exceeds the paper's best: gastrulation-human 0.400 / 0.110 / 0.387, pancreas-human 0.434 / 0.178 / 0.354.
  "Beats the paper" requires all 6 cells.
- G3: no NaN Pearson transition in our runs (i = 0 is scored).
- G6 (the OT step contributes), per dataset, paired on identical masks, Delta = distbridge_strict - distbridge_strict_nosb:
  5-seed mean Delta >= +0.01 on at least one metric and >= -0.005 on the other two; one-sided Wilcoxon on the
  per-transition seed-averaged Delta of J, p < 0.05; delta > 0 selected in >= 25 % of transitions. If G6 fails the
  report says: the OT/SB step was selected off or did not contribute; the margin comes from the noise-aware pooling
  and the finite-cell observation model.
- Reported alongside: MAE, the 28 / 13 averages without the last transition, the count of test genes tied at +/-5,
  selected configurations per transition, and the comparator obslevel_rollk (a tie with it is disclosed).

## Not in this registration (later, separately labelled)
- Same-information tier (fitting on the 75 % training genes of the same transition, as their models do).
- Protocol corrections (strict reruns of their models, pseudocount target, paper's x(t+1) - x(t) target), each
  reported next to the original protocol, per the user's rule.
