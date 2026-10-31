# Pre-registration: does OT win where cells really move and the true couplings are known? (LARRY, 2026-10-29)

Written and frozen before any result on these data was computed (only the design table was inspected: days 2/4/6,
two starting populations, wells 0 at day 2 and 1, 2 at days 4 and 6, 5864 clones). Context: OT lost to the 50/50 mix on
the mouse atlas and on zebrafish (pre-registered), where consecutive snapshots barely differ (separation 0.1-0.5). The
argued condition for OT to help is data that contain real movement and a way to check couplings.

## Data
Weinreb et al. 2020 (Science 367), GEO GSM4185642 (in vitro LARRY), downloaded 2026-10-29 with the user's permission.
Products tack/hpc/larry_prep.py: per starting population (Lin-Kit+Sca1+, Lin-Kit+Sca1-), fit-only preprocessing
(atlas_prep.transform) on the GEO normed counts; <= 3000 cells per day (proportional to wells, seed 0); coupling sets
use cells with exactly one clone barcode.

## Methods (frozen)
- H1: atlas_confirm.run_interp unchanged (prev, next, union, indep, ot_pairs, sb_half, uot; 3 reps), day 4 held out.
- H3: tack/hpc/larry_confirm.py: entropic OT plan (interp_bench_frozen.plan, eps_rel 0.05) vs kNN coupling (10 nearest
  targets) vs cell-type coupling (uniform over targets of the same annotated type).

## Metrics and units
- H1: ED_U per day-4 culture well (atlas_confirm pool_mean), 4 units (2 populations x 2 wells).
- H3: clonal enrichment = share of plan mass on same-clone pairs / the same share under independence; units 2
  populations x 2 transitions (2->4, 4->6) = 4.
- Descriptive: separation |mean(day 6) - mean(day 2)| / RMS radius(day 4); population-level ED_U.

## Hypotheses and decision rules ("A beats B": better in >= 3 of 4 units AND better mean)
- H1 (primary): ot_pairs beats prev, next and union.
- H1b: uot beats prev, next and union.
- H1c: sb_half (OT pairs + bridge noise; the variant that beat the mix at 18-24 h gaps on the atlas, exploratory)
  beats prev, next and union.
- H3 (primary): OT coupling has higher clonal enrichment than kNN coupling.
- H3b: OT coupling has higher clonal enrichment than cell-type coupling.
Pre-stated reading: H1 passing here (large steps) but not on the embryos supports "OT helps only when the data contain
movement"; H1 failing here as well means OT interpolation does not beat the mix even then. H3 says whether OT recovers
true lineage better than the simplest alternatives. With 4 units, a pass is weak evidence and a fail is not proof of
no effect; both are reported as such. Nothing changed after the first result.

## Frozen code: sha256 of the HPC copies in the decisions.md row with this file's sha256.
