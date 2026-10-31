# Pre-registration: does a time-continuous cell-state density model (Mellon) beat the 50/50 mix? (2026-10-31)

Written and frozen before any Mellon number was computed on these data. The data, held-out units, seeds, targets and
frozen rivals are those of the atlas confirmation (2026-10-25) and the zebrafish replication (2026-10-27), whose
results are known. This is a pre-registered test of a new model on data already used, not an independent replication.

## Why this model
Every transport model lost to the mix of the neighbouring samples:
- OT pairs, unbalanced OT and the Schrödinger bridge;
- WFR (2026-10-31: 4/7 on the atlas, 5/9 on zebrafish, where it collapsed to the mix);
- an estimated embryo age, and a dynamic logit mixture.

The diagnosis says that between neighbouring times cells barely move; what changes is how many cells sit in each
state, and embryos progress at different paces. Mellon (Otto, Jordan, Dury, Dien, Setty, bioRxiv 2023) is built for
this signal. It models the log density of cell states as a Gaussian process over state and time, and its authors
interpolated held-out stages of this same atlas, reporting no baseline. Mellon is an external method used through an
adapter, which the project's original-code rule allows for methods under test; it is installed in a separate
environment (~/envs/mellon, mellon 1.7.1, jax 0.10.2) with the user's permission.

## Method (frozen: tack/hpc/mellon_confirm.py; tests tack/tests/test_mellon_confirm.py, 4/4 on the cluster)
For each held stage k (stage k never enters any fit):
1. **Cells.** The frozen products' own cells, loaded with atlas_prep.load / zf_prep.load (seed 0, same order).
   Preprocessing is atlas_prep.transform with fit = every stage except k (gene filter, normalisation, 2,000 HVG, 30 PCs).
2. **Embedding.**
   - Primary "dm25": a Palantir-style multiscale diffusion map with 25 components, as in the Mellon paper's analysis of
     this atlas. Own implementation, because Palantir is not installed: kNN 30, adaptive Gaussian kernel with sigma =
     distance to the 10th neighbour, symmetrised. Eigenvalue-1 vectors (trivial or disconnected components) are
     dropped, and the components are scaled by lambda / (1 - lambda).
   - Descriptive "pc30": the 30 PCs directly.
   - No setting is tuned: every candidate validation fit would contain stage k.
3. **Density fit.** mellon.TimeSensitiveDensityEstimator(normalize_per_time_point=True), other settings at Mellon's
   defaults, fitted on all cells of every stage except k with real times (atlas: days 6.5 + 0.25 i; zebrafish:
   hpf 18 + 2 i).
4. **Weights.** w(x) = rho(x, t_k) / (rho(x, t_{k-1}) / 2 + rho(x, t_{k+1}) / 2) at the cells of stages k-1 and k+1,
   divided by its mean under the mix and clipped to [0.05, 20] (the frozen tilt_rw clip). The synthetic test checks
   that per-time densities are on one scale when cell counts differ.
5. **Scoring.** Frozen interpolation loop (atlas_confirm.run_interp seeds and subsamples, interp<k>.npz PCA space, ED_U):
   - "mellon_dm25" / "mellon_pc30": subsampled k-1 and k+1 cells with weights (1/2) w / n_side;
   - control "mix_full": the same cells with w = 1.

**Parity gate**, a condition for reading the verdict:
- interp<k>.npz fit cells equal the loaded stage k-1 / k+1 cells (same samples, same order);
- the frozen rivals reproduce confirm.json and confirm_zf.json (maximum |ED_U difference| <= 1e-9).

## Hypotheses and decision rule
"A beats B" = lower ED_U in >= ceil(2n/3) units AND lower mean ED_U.
- **H1 (primary): mellon_dm25 beats the mix (union).**
  - Atlas: 7 held stages, needs 5.
  - Zebrafish: 9 held times, needs 6.
  - Reading: both pass = confirmed; exactly one passes = data-set dependent; none passes = negative.
- **H1b: mellon_dm25 beats mix_full.** This is the same comparison on an identical pipeline, isolating the weights.
- **Descriptive:**
  - mellon_pc30;
  - mix_full vs union (pipeline effect alone);
  - per unit: effective sample fraction, clipped fraction, weight share on stage k-1 (Mellon's implied pace);
  - correlation between log w and |log rho_{k-1} - log rho_{k+1}| (one-sided states);
  - Mellon's length scales;
  - ED_U in gene space and per pool.

## Pre-stated readings
- H1 passes: a time-continuous density model is the tool for this signal, and the generative recipe (reweight the
  neighbours' cells by the predicted density) is the building block.
- H1 fails and the log w vs gap correlation is strongly negative: Mellon interpolates in log density, which
  down-weights states present on one side only (the dm_lin2 lesson). The failure is the geometry, not the density idea.
- H1 fails with weights near 1 (effective sample fraction near 1): Mellon adds nothing beyond the mix.
- Expected worst case: a tie with the mix, which fails the strict rule.

## Caveats stated in advance
- These are not new data; LARRY is excluded (holding out day 4 leaves two time points to fit).
- Mellon's published setup used batch-corrected PCs and Palantir; ours uses uncorrected PCs and an own diffusion map.
- Forecasting (t+1) is not tested: from snapshots alone it needs other measurements (metabolic labelling).
- Nothing will be changed after the first result. A crash fix that does not change the design is recorded as an
  amendment. Any later analysis is labelled exploratory.

## Frozen code: sha256 of the cluster copies in the decisions.md row with this file's sha256.
