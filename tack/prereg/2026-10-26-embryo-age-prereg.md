# Pre-registration: embryo age instead of nominal stage (2026-10-26)

Written and frozen before any result of this test was computed. Motivation: the exploratory diagnosis of the failed OT
confirmation (tack/results/atlas/diag/diagnosis.md, decisions.md 2026-10-25) found that on the Pijuan-Sala atlas the
stage label is a noisy clock (biological pools of one nominal stage differ by up to 1.5 stages in estimated age; the
estimated pool age tracked the best mixture weight, Spearman 0.79 over 14 pools, oracle). That analysis looked at the
same cells, so this test is a confirmation on the same data with a method fixed in advance, not an independent replication.

## Data
- Same cells as the atlas confirmation (atlas_prep.load: no blood lineage, <= 3000 cells per stage, seed 0).
- Unit of biology = biological embryo pool (metadata column `pool`; samples of one pool are technical replicates).
- H1 products: for each held stage k = 1..7, fit-only preprocessing (atlas_prep.transform) on ALL other stages; stage k
  only transformed. H2 products: the existing ~/tack/out/atlas/origin<s>.npz (fit on stages <= s), s = 2..7.

## Method (tack/hpc/age_confirm.py, frozen)
- Age of a reference pool: leave-one-pool-out multinomial logistic regression (C 0.1, standardised 30 PCs, <= 600 training
  cells per pool, seed fixed) of the nominal stage on the cells of the other reference pools; age = mean expected stage
  over the pool's cells. The held / future stage is never used for training, calibration or weights.
- Calibration: linear fit age = alpha + beta * nominal over the reference pools; sigma = residual sd (floor 0.05).
  Target age t = alpha + beta * k (H1) or alpha + beta * (s + 1) (H2).
- age: mixture of real reference cells, pool mass = exp(-(age_p - t)^2 / 2 sigma^2) x cells of the pool.
- Rivals: union (H1; 50/50 of stages k-1 and k+1, cells uniform within stage), persist (H2; stage s cells uniform),
  stagek (same kernel on alpha + beta * nominal_p), shufw (age with ages permuted among pools of the same nominal stage,
  20 permutations, mean ED_U). 3 repetitions of the per-pool cell subsample (<= 300 cells per pool); ED_U averaged.
- No graph, no transport, no hyperparameter search.
- Synthetic checks passed before freezing (tack/tests/test_age_confirm.py: 3/3; age recovered, wins under asynchrony,
  no gain when the label is the true age).

## Metric
ED_U (interp_bench_frozen.ed_uv) in the PCA space of the product against the held / future stage (subsample <= 1500,
seed 1). Secondary, not used for the verdict: ED_U per biological pool of the target stage (>= 20 cells), ages, sigma,
effective number of pools.

## Hypotheses and decision rules
"A beats B" = strictly lower ED_U in at least ceil(2/3 of the units) (5 of 7 for H1; 4 of 6 for H2) AND lower mean ED_U.
- H1 (primary): age beats union (interpolation, k = 1..7).
- H1b: age beats stagek (the gain comes from age, not from using more pools).
- H1c: age beats shufw (the within-stage age differences carry the signal).
- H2 (primary): age beats persist (forecasting, s = 2..7).
- H2b: age beats shufw in forecasting.
All verdicts are reported whatever they are. A pass of H1 / H2 with failed H1b-c / H2b is reported as not specific to
age. Nothing is changed after the first result; later analyses are exploratory.

## Frozen code (sha256 of the HPC copies, LF): see the decisions.md row recording this file's sha256.
