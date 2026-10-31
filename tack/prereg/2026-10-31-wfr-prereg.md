# Pre-registration: does a Wasserstein-Fisher-Rao interpolant beat the 50/50 mix? (2026-10-31)

Written and frozen before any WFR number was computed on these data. The data, units, seeds, targets and frozen rivals
are those of the atlas confirmation (2026-10-25), the zebrafish replication (2026-10-27) and LARRY in vitro
(2026-10-29); their frozen results are known. This is therefore a pre-registered test of a new model on data already
used, not an independent replication.

## Context and what differs from the failed `uot`
On every time course tested, the 50/50 mixture of the two neighbouring samples ("mix") beat every OT interpolant
(OT pairs 0/7, 1/9, 0/4), while OT pairs always beat random pairs and the OT coupling recovers sister cells (LARRY).
The Wasserstein-Fisher-Rao (Hellinger-Kantorovich) geodesic moves mass only along pairs closer than pi * delta and
lets the rest fade out at the source and fade in at the target. Its normalised midpoint is the mix when nothing is
transported (delta = 0) and McCann's OT midpoint when everything is (large delta). The frozen `uot` differs in three
ways: there every coupled pair moves (no fading), masses grow geometrically along a balanced-style path (not along
the cone), and the mix is not a limit of the family. A literature check (2026-10-31: TIGON, DeepRUOT, VGFM, WFR-FM,
WFR-MFM, USB) found no WFR model compared against the mix or persistence.

## Method (frozen: tack/hpc/wfr_confirm.py; synthetic tests tack/tests/test_wfr_confirm.py, 6/6 on the cluster)
- theta_ij = |x_i - y_j| / (2 delta), delta = delta_rel * sqrt(median squared distance between the two samples).
  Cost c_ij = -2 log cos(theta_ij) if theta_ij < pi/2, otherwise no transport.
- Plan: entropic KL-unbalanced Sinkhorn (interp_bench_frozen.sinkhorn), KL marginal weight 1 (fixed by the theory),
  eps = 0.05 * median of the finite costs, 2,000 iterations.
- Semi-couplings a_ij = gamma_ij mu0_i / gamma0_i, b_ij = gamma_ij mu1_j / gamma1_j. Each pair follows the straight line
  in the cone, giving its mass |z(t)|^2 and its position fraction phi(t) / theta along x_i -> y_j. Rows and columns with
  no transportable partner fade: (1 - t)^2 mu0_i at x_i and t^2 mu1_j at y_j.
- Interpolation at t = 1/2.
- Forecast at t = 2 from stages (s-1, s). This is a construction, not a geodesic:
  - point reflection z = 2 z1 - z0;
  - pairs with 2 sqrt(b) cos theta <= sqrt(a) are dropped;
  - faded source mass becomes 0, faded target mass becomes 4 mu1_j.
  delta = 0 gives exactly persistence.
- Pair atoms: 20 n pairs drawn proportionally to mass, with duplicates merged into weights; then the frozen subsample
  and scoring path (ED_U in the PCA space of the product). Forecast uses at most 1,500 cells per side (fixed seeds).
- Grid delta_rel = 0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2.
- **Selection, never using the target:**
  - Interpolation of held stage k: the delta with the best mean rank of ED_U over the files k' with |k' - k| >= 2 (none
    contains stage k cells); ties go to the smaller delta.
  - Forecast: rolling origin, i.e. the argmin of ED_U when forecasting stage s from origin s-1; ties go to the smaller
    delta.
  - LARRY (one triple per population): delta chosen on the other population. Descriptive only.
- **Parity gate**, a condition for reading the verdicts. The frozen rivals are recomputed with the frozen seeds:
  prev / next / union / indep / ot_pairs for interpolation, persist / meanshift for forecast. They must reproduce
  confirm.json and confirm_zf.json with a maximum absolute ED_U difference of 1e-9.

## Hypotheses and decision rule
"A beats B" = lower ED_U in >= ceil(2n/3) units AND lower mean ED_U.
- **H1 (primary, interpolation): WFR beats the mix.**
  - Atlas: 7 held stages, needs 5.
  - Zebrafish: 9 held times, needs 6.
  - Reading: both pass = confirmed; exactly one passes = data-set dependent; none passes = negative.
- **H2 (secondary): WFR beats OT pairs**, same rule. This asks whether letting mass fade helps over pure transport.
- **H3 (secondary, forecast): WFR beats persistence AND mean shift.**
  - Atlas: 6 origins, needs 4.
  - Zebrafish: 8 origins, needs 6.
- **Descriptive:**
  - LARRY day 4 (4 wells, delta chosen on the other population) against the mix and OT pairs;
  - chosen delta and transported mass fraction per unit;
  - ED_U in gene space;
  - ED_U for every delta on the final problems (oracle-labelled, never used for a verdict).

## Pre-stated readings
- If delta = 0 is chosen in most units, the reading is "WFR collapses to the mix", whatever the win count: a tie with
  the mix fails the strict rule by design, and that is the expected worst case, not a bug.
- H1 passes with intermediate deltas chosen: the useful part of OT can be added to the mix by letting distant mass fade
  instead of moving, and the WFR family is the generative model to build on.
- H1 fails with intermediate deltas chosen and losing: moving the nearby mass hurts even when far mass is left alone;
  for these data the population at an unobserved time is a mixture, and no transport helps.
- H3 has no directional expectation (persistence has never been beaten on these data).

## Caveats stated in advance
- These are not new data. The atlas held stages are known to be mixtures of their neighbours (exploratory diagnosis,
  2026-10-25), and that is part of the reason for this test.
- With 2,000 iterations the KL-unbalanced Sinkhorn converges slowly when eps is much smaller than 1 (the largest
  deltas), so plans there are approximate. The semi-coupling construction reproduces both endpoints exactly whatever
  the plan.
- The forecast rule is a pre-stated construction; WFR geodesics are defined only between two observed measures.
- Nothing will be changed after the first result; any later analysis is labelled exploratory.

## Frozen code: sha256 of the cluster copies in the decisions.md row with this file's sha256.
