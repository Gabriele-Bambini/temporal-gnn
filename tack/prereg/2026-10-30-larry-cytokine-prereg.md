# Pre-registration: does the OT coupling advantage seen in vitro replicate on an independent LARRY experiment? (cytokine perturbation, 2026-10-30)

Written and frozen before any coupling, plan or enrichment was computed on these data. Inspected beforehand, and only
these (tack/hpc/larry_cyto_design.py, job 729828; metadata + clone matrix, no expression): the design table (time points
2 / 4 / 6 x cytokine condition x library, cell-type counts per time point), clones shared between (time point,
condition) groups, and the clone-label integrity check below.

## Context
In vitro (pre-registered 2026-10-29, GSM4185642): OT coupling (eps_rel 0.05) beat kNN (K = 10) in clonal enrichment in
3/4 units. Exploratory robustness (2026-10-30, larry_robust / larry_sharp): at MATCHED spread OT beat kNN 4/4 and beat the
same kernel without mass constraints 4/4; tuned to the best setting on the truth, OT vs kNN was 2-2. The in vivo
replication (GSM4185643, prereg 2026-10-30-larry-invivo-prereg.md) is NOT EVALUABLE: its clone labels carry no
cell-level information inside mice (GEO and Klein-lab files identical). This test re-asks the same matched-spread
questions on a third, independent barcoding experiment.

## Data
Weinreb et al. 2020 (Science 367), GEO GSM4185644 (cytokine perturbation), downloaded 2026-10-30 with the user's
permission (~/rafm_data/weinreb2020_cyto). 65,075 cells, 25,749 with exactly one clone barcode (5,257 clones). Lin-Kit+
cells barcoded in the full cocktail, sampled at day 2 (libraries d2_1-6), then split into Epo / G-csf / M-csf / SCF /
full cocktail and sampled at day 4 and day 6 (SCF day 4 only).

## Integrity check (passed before freezing)
Same-type sister enrichment inside each (time point, condition) group (each cell uniform over the other cells of its
annotated type), vs a type-permutation null: 1.53-4.94 vs null 0.29-0.89 in all 10 groups (in vitro reference 3.2-4.3;
in vivo 1.00 = null). The null is below 1 by construction (most clones are singletons inside a group), so values are
read against the printed null, not against 1. Per-library values were printed without a null and are not used.

## Units (rule fixed before any result)
Consecutive transitions within a condition, >= 50 shared clones (single-clone cells; threshold inherited from the in vivo
prereg): day 2 (common full-cocktail sample) -> day 4 of each condition, and day 4 -> day 6 of the same condition.
Day 2 -> 6 and cross-condition pairs are excluded (they overlap with the consecutive units, and different conditions are
different wells). By the design counts this gives 9 units (source -> target single-clone cells, shared clones):
Epo_24 3776 -> 1128 (195), Gcsf_24 3776 -> 2121 (387), Mcsf_24 3776 -> 2109 (396), SCF_24 3776 -> 2432 (403),
full_24 3776 -> 282 (101), Epo_46 1128 -> 1202 (143), Gcsf_46 2121 -> 1852 (260), Mcsf_46 2109 -> 5131 (671),
full_46 282 -> 5716 (163). All under the caps (source <= 9000, target <= 6000), so no subsampling. PCA fitted on source +
target without clone information (atlas_prep.transform, cell-cycle genes removed), per unit (tack/hpc/larry_cyto_prep.py).
If prep yields a different unit set, the confirm step is not run until the difference is explained in decisions.md.

## Methods (frozen: tack/hpc/larry_cyto_confirm.py, which imports unit() and rule() unchanged from larry_vivo_confirm.py)
Couplings, clonal enrichment and spread exactly as in larry_robust / the in vivo prereg: ot = balanced entropic OT plan
(interp_bench_frozen.plan), eps_rel 0.005-0.1, Sinkhorn marginal error recorded; soft = row-normalised Gaussian kernel
(OT without mass constraints), eps_rel 0.005-0.5; knn = uniform over the K nearest targets, K 1-2000 (K < target size).
Spread = exp(mean row entropy) = effective number of partners per source cell. Matched values by log-log interpolation
along each grid at OT's spread.

## Hypotheses and decision rule
"OT beats X" = ratio > 1 in >= 6 of the 9 units (ceil(2n/3)) AND geometric mean of the 9 ratios > 1.
- H1 (primary): OT at eps_rel 0.05 beats kNN at the same spread.
- H2 (primary): OT at eps_rel 0.05 beats soft at the same spread (the mass constraint helps).
- H3 (secondary): best converged OT on its grid (marginal error <= 1e-3) beats best kNN on its grid (both tuned on the
  clonal truth); a unit with no converged OT plan counts as a loss.
- Descriptive: OT eps 0.05 vs kNN K = 10; H1 / H2 ratios on 3 random 80 % subsamples; whether OT's spread lies inside
  the kNN / soft grids; every rule split by transition type (day 2 -> 4 vs day 4 -> 6).

## Pre-stated reading
- H1 and H2 pass: the in vitro matched-spread advantage replicates on an independent barcoding experiment; the claim "at
  equal spread the OT coupling recovers sister pairs better than local similarity, and the mass constraint is what does
  it" can be written (in vitro haematopoiesis, two experiments).
- H1 passes, H2 fails: OT beats hard kNN but not the soft kernel; the gain is soft weighting, not transport.
- H1 fails: the in vitro advantage does not replicate; the OT coupling claim is downgraded to "one experiment only".
- H3: no directional expectation (in vitro 2-2); it says whether tuned OT is the best estimator.

## Caveats stated in advance
- The five day 2 -> 4 units share the same source sample (only targets differ), so they are correlated and 6/9 is less
  stringent than 9 independent units; results are also reported per transition type.
- This regime is harder than the main in vitro experiment: four of the five targets are cytokine-shifted relative to the
  full-cocktail source, so a weaker H1 / H2 than in vitro is not by itself a contradiction.
- full_24 is the thinnest unit (282 target cells, 101 shared clones); its K grid is truncated at K < 282 (K <= 200, also on
  the 80 % subsamples), so inside_grid may be false there.
- Most day-2 cells are "Undifferentiated" (11,141 / 15,728), so the cell-type coupling is weakly informative and is not run.
- With 9 units a pass is moderate evidence and a fail is not proof of no effect. Nothing will be changed after the first
  result; any later analysis is labelled exploratory.

## Frozen code: sha256 of the HPC copies in the decisions.md row with this file's sha256.
