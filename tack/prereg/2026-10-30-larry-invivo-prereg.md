# Pre-registration: does the OT coupling advantage seen in vitro replicate in vivo? (LARRY in vivo, 2026-10-30)

Written and frozen before any coupling, plan or enrichment was computed on these data. Inspected beforehand, and only
these: the design table (time points 2 / 9 / 16, cohorts A / B, recipient mice, cell-type counts per time point), the
numbers of clones shared between each day-2 cohort sample and each mouse (tack/hpc/larry_vivo_design.py) and the unit
sizes written by tack/hpc/larry_vivo_prep.py (meta.json). No expression-based quantity was looked at.

## Context
In vitro (pre-registered 2026-10-29): OT coupling (eps_rel 0.05) beat kNN (K = 10) in clonal enrichment in 3/4 units.
Exploratory robustness (2026-10-30, larry_robust / larry_sharp): that comparison was spread-mismatched (OT spread 133-367
effective partners per cell vs 10); at MATCHED spread OT beat kNN 4/4 and beat the same kernel without mass constraints
4/4; tuned to the best setting on the truth, OT vs kNN was 2-2. This test re-asks the matched-spread questions on
independent data.

## Data
Weinreb et al. 2020 (Science 367), GEO GSM4185643 (in vivo LARRY), downloaded 2026-10-30 with the user's permission
(~/rafm_data/weinreb2020_invivo). Barcoded Lin-Kit+Sca1hi cells; two cohorts (A, B), each with a day-2 in vitro sample
and recipient mice sampled at day 9 or day 16. The Lin-Kit population has no clones shared across time and is not used.

## Units (rule fixed before any result)
Transition day-2 cohort sample -> recipient mouse of the same cohort with >= 50 shared clones (single-clone cells).
By the design counts this gives 8 units: A1, A2, A3, A5, B1, B2, B3, B5 (6 at day 9, 2 at day 16); A4 (14 shared) and
B4 (23) are excluded. Source = all day-2 single-clone cells of the cohort (8231 / 7940); target = <= 6000 single-clone
cells of the mouse (proportional to libraries, seed 0); PCA fitted on source + target without clone information.

## Methods (frozen: tack/hpc/larry_vivo_confirm.py)
Couplings, clonal enrichment and spread exactly as in larry_robust: ot = balanced entropic OT plan
(interp_bench_frozen.plan), eps_rel 0.005-0.1, Sinkhorn marginal error recorded; soft = row-normalised Gaussian kernel
(OT without mass constraints), eps_rel 0.005-0.5; knn = uniform over the K nearest targets, K 1-2000. Spread =
exp(mean row entropy) = effective number of partners per source cell. Matched values by log-log interpolation along each
grid at OT's spread.

## Hypotheses and decision rule
"OT beats X" = ratio > 1 in >= 6 of the 8 units (ceil(2n/3)) AND geometric mean of the 8 ratios > 1.
- H1 (primary): OT at eps_rel 0.05 beats kNN at the same spread.
- H2 (primary): OT at eps_rel 0.05 beats soft at the same spread (the mass constraint helps).
- H3 (secondary): best converged OT on its grid (marginal error <= 1e-3) beats best kNN on its grid (both tuned on the
  clonal truth); a unit with no converged OT plan counts as a loss.
- Descriptive: OT eps 0.05 vs kNN K = 10 (the in vitro comparison); H1 / H2 ratios on 3 random 80 % subsamples; whether
  OT's spread lies inside the kNN / soft grids; results per cohort.

## Pre-stated reading
- H1 and H2 pass: the in vitro matched-spread advantage replicates on independent data and across an in vitro -> in vivo
  transition; the claim "at equal spread the OT coupling recovers sister pairs better than local similarity, and the
  mass constraint is what does it" can be written.
- H1 passes, H2 fails: OT beats hard kNN but not the soft kernel; the gain is soft weighting, not transport.
- H1 fails: the in vitro advantage does not replicate; the OT coupling claim is downgraded to "in vitro only".
- H3: no directional expectation (in vitro 2-2); it says whether tuned OT is the best estimator.

## Caveats stated in advance
- Units within a cohort share the same day-2 source cells, so the 8 units are 2 cohorts x mice, not 8 independent
  experiments; results are also reported per cohort.
- Only ~1.5-2.5 % of source cells have a sister in a given mouse; enrichment is driven by them.
- The step crosses a culture -> transplant shift; all day-2 cells are annotated "Undifferentiated", so the cell-type
  coupling is uninformative and is not run.
- With 8 units a pass is moderate evidence and a fail is not proof of no effect. Nothing will be changed after the first
  result; any later analysis is labelled exploratory.

## Frozen code: sha256 of the HPC copies in the decisions.md row with this file's sha256.
