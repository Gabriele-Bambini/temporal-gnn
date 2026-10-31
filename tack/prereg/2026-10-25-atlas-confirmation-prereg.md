# Pre-registration: confirmation of the erythroid OT signals on the gastrulation atlas (2026-10-25)

Written and frozen before any result on the atlas was computed (the atlas products exist, only their cell counts
were read). The two signals were found post hoc on the erythroid lineage (decisions.md, 2026-10-24;
tack/results/ery/report.md): (a) OT interpolation of a held-out stage beats the single-stage and 50/50 baselines in
PCA space; (b) forecasting by reweighting the current cells with unbalanced-OT growth (uot_rw) beats persistence and
mean shift. This study tests them once on independent cells, with the methods frozen as they were.

## Data
- Pijuan-Sala et al. 2019 mouse gastrulation atlas, already on the Bocconi HPC (~/rafm_data), 9 real stages
  E6.5 .. E8.5, 0.25 days apart.
- Kept: annotated cells, not doublets, not stripped, and NOT the blood lineage (Haematoendothelial progenitors, Blood
  progenitors 1/2, Erythroid1/2/3), so no cell of the erythroid study is reused. At most 3000 cells per stage,
  drawn proportionally to each pool (seed 0). Cell types are used only for this selection and descriptively.
  Cells per stage: 2999, 2072, 3000, 3000, 3000, 3001, 3000, 3000, 3000; pools per stage: 3, 1, 6, 3, 6, 4, 4, 3, 4.
- Products: tack/hpc/atlas_prep.py (HPC job 718013, ~/tack/out/atlas). Every gene filter, normalisation, HVG set
  (2000), PCA (30) and proliferation score is fitted on the fit cells only (origins: stages <= s; interpolation:
  stages k-1 and k+1).
- Known limitation, stated in advance: as in the erythroid data, stage and sequencing pool are partly confounded
  (E6.75 has a single pool); a win can partly reflect pool structure, so per-pool ED_U is reported.

## Methods (frozen)
- Interpolation: tack/hpc/interp_bench_frozen.py, a byte copy of the erythroid
  tack/results/ery/attempt_interp/interp_bench.py, used through tack/hpc/atlas_confirm.py: prev, next, union (50/50),
  indep, ot_pairs (McCann pairs, eps_rel 0.05), sb_half, uot (tau 1, beta 2); 3 repetitions, same seeds as the
  erythroid run; predictions min(n_held, 1500) points.
- Forecasting: atlas_confirm.py, the erythroid definitions of persist, meanshift, uot_rw (grid: lambda 0, or
  lambda in {0.5, 1, 2} x tau in {0.1, 1} x beta in {0, 2}) and tilt_rw (kNN density ratio, k 30, clipped to
  [0.05, 20], lambda in {0, 0.25, 0.5, 1, 1.5, 2}); hyperparameters by rolling origin (each grid point forecasts
  stage s from origin s-1, argmin of ED_U), no other tuning.
- Parity check, passed before freezing: on the erythroid products atlas_confirm.py --parity reproduces the erythroid
  judge's interpolation numbers (k = 2..5): ot_pairs 1.908 / 1.287 / 0.942 / 1.198, uot 1.762 / 1.396 / 0.857 / 0.948,
  union 2.206 / 1.041 / 1.154 / 1.473, prev 4.192 / 3.323 / 1.985 / 3.264, next 3.975 / 1.724 / 2.802 / 6.428
  (tack/results/atlas/parity_erythroid.json).

## Frozen code (sha256 of the HPC copies, LF line endings)
- atlas_prep.py          305b47f8afa41afc51391a94e5e32ec494906a0c1eea46948bbab375256db3a0
- atlas_confirm.py       0594c9d11f9d0d6a64e86e1557e9c30b945494bd41627b2a67550301e80e503f
- interp_bench_frozen.py 5231d262a19ebc40f56a4f65fad55aadf5c1c952cf0febb335fdd4968efa6e16
- ery_forecast.py        64351c57f109b3f86213d7cd06ecc526108a169a5e5fb92674f1b1512ce94a9d (kernel_regress only)

## Metric
- Primary: ED_U, the energy distance with self-pairs excluded (weighted U-statistic), in the PCA space of each product,
  against the true stage (subsample <= 1500, seed 1).
- Secondary (reported, not used for the verdict): ED_U in the 2000-HVG gene space, ED_U per embryo pool (>= 20 cells),
  gene-level PCC of the mean change (forecasting), effective sample size of the reweighting.

## Hypotheses and decision rules
Units: interpolation held stages k = 1..7 (E6.75 .. E8.25), 7 units; forecasting origins s = 2..7 (forecast E7.25 ..
E8.5), 6 units. "A beats B" = A has a strictly lower ED_U in at least ceil(2/3 of the units) (5 of 7; 4 of 6) AND a
lower mean ED_U over the units, against every listed B.
- H1 (primary): ot_pairs beats prev, next and union.
- H1b: uot beats prev, next and union.
- H1c: ot_pairs beats indep (the OT pairing matters).
- H2 (primary): uot_rw beats persist and meanshift.
- H2b: uot_rw beats tilt_rw (the reweighting needs OT, not just a density ratio).
All five verdicts are reported whatever they are. A pass of H1 / H2 with a failed H1c / H2b is reported as such
(the gain would not be specific to OT). No method, grid, metric or unit is changed after the first atlas result; any
later analysis is labelled exploratory.
