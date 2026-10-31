# Pre-registration: dynamic mixture model (reweighting, not transport) vs the simple baselines (2026-10-29)

Written and frozen before any result of this model on real data was computed. Motivation: OT lost to the 50/50 mix and
to persistence on the mouse atlas (2026-10-25) and on zebrafish (2026-10-27, pre-registered); the diagnoses showed that
held snapshots are mixtures of states already present, with non-uniform pace (decisions.md). The proposed alternative is
a Fisher-Rao type model: fixed cell states whose shares change smoothly in time.

## Data (already on the HPC, same cells as before)
- zf: Saunders 2023 GAP14 controls, 11 timepoints 18..38 hpf (zf_prep.load); unit = embryo.
- atlas: Pijuan-Sala 2019 without blood, 9 stages E6.5..E8.5 (atlas_prep.load); unit = biological pool (meta `pool`).
- Note: both datasets were already used for the OT tests and the exploratory diagnoses that motivated this model;
  this is a pre-registered test of a new model on known data, not an independent replication.

## Model (tack/hpc/dynmix.py, frozen) and rivals
- Products per unit with fit-only preprocessing (atlas_prep.transform): interpolation of k fitted on timepoints
  k-2, k-1, k+1, k+2 that exist; forecast from s fitted on timepoints <= s.
- 100 k-means states (seed 0) on the fit cells; multinomial logistic regression of state on tau = t - t_target
  (tau, tau^2 when >= 3 timepoints), C = 1, timepoints weighted equally, embryos / pools weighted equally within a
  timepoint. Prediction = real reference cells reweighted by the predicted state shares.
- Interpolation: dynmix (4 timepoints) vs union (50/50 of k-1 and k+1). Forecast: dynmix with damped trend
  (PHI = 0.5 on the log-share change, fitted on s-2..s) vs persist (stage s cells).
- Synthetic checks passed before freezing (tack/tests/test_dynmix.py, 3/3).

## Metric
ED_U (interp_bench_frozen.ed_uv) in the PCA space of each product vs the target subsample (<= 1500, seed 1).
Secondary: per-unit (embryo / pool) ED_U; dm_lin2 (same model on k-1, k+1 only) vs union.

## Hypotheses and decision rules
"A beats B" = strictly lower ED_U in >= ceil(2/3 of units) AND lower mean ED_U.
- H1z: dynmix beats union, zebrafish, k = 1..9 (need 6 of 9).
- H1a: dynmix beats union, atlas, k = 1..7 (need 5 of 7).
- H2z: dynmix beats persist, zebrafish forecast, s = 2..9 (need 6 of 8).
- H2a: dynmix beats persist, atlas forecast, s = 2..7 (need 4 of 6).
All four reported whatever they are; nothing changed after the first result; later analyses exploratory.

## Frozen code: sha256 of the HPC copies recorded in the decisions.md row with this file's sha256.
