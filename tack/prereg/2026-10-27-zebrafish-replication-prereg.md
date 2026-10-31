# Pre-registration: replication of the OT interpolation / reweighting test on zebrafish with a true clock (2026-10-27)

Written and frozen before any result on these data was computed (only the design table and data products were
inspected). Context: on the Pijuan-Sala mouse atlas the pre-registered test failed (decisions.md 2026-10-25: the 50/50 mix
beat OT 7/7; uot_rw 1/6) and the exploratory diagnosis blamed the data (stage label != embryo age, 1-5 pools per stage,
batch tied to stage). This test repeats the SAME frozen methods on data without those defects, to decide between
"data problem" and "architecture problem".

## Data
- Saunders et al. 2023 (Nature 623:782), GEO GSE202639 reference, downloaded 2026-10-26 with the user's permission.
- Kept: experiment GAP14 only (one sequencing experiment), uninjected control embryos, 11 timepoints 18..38 hpf every
  2 h, 48 individually hashed embryos per timepoint, 28 C. Fertilisation-synchronised clutches: the timepoint label is
  the clock. At most 3000 cells per timepoint, proportional to embryos (seed 0).
- Products: tack/hpc/zf_prep.py, fit-only preprocessing identical to the atlas (atlas_prep.transform).

## Methods (frozen, unchanged from the atlas confirmation)
tack/hpc/atlas_confirm.py (sha256 0594c9d1...) with interp_bench_frozen.py and ery_forecast.py, run with
--ks 1..9 --origins 2..9. The fixed time constants inside the frozen code (DT 0.25, DT_INT 0.5) only scale the
proliferation prior of the growth terms; they are kept as frozen and not rescaled to 2-h steps.

## Metric
ED_U in the PCA space of each product (primary); secondary: gene-space ED_U, per-embryo ED_U, forecast PCC and ESS.

## Hypotheses and decision rules
Units: held timepoints k = 1..9 (20..36 hpf), 9 units, need 6; forecast origins s = 2..9 (forecast 24..38 hpf),
8 units, need 6. "A beats B" = strictly lower ED_U in >= ceil(2/3 of units) AND lower mean, against every listed B.
- H1 (primary): ot_pairs beats prev, next and union.
- H1b: uot beats prev, next and union.
- H1c: ot_pairs beats indep.
- H2 (primary): uot_rw beats persist and meanshift.
- H2b: uot_rw beats tilt_rw.
Pre-stated reading: if H1 / H2 fail here as on the atlas, the failure is not explained by the atlas data defects and is
attributed to the methods (architecture) for this kind of task; if they pass here, the atlas failure is attributed to
the data. All verdicts reported; nothing changed after the first result; later analyses exploratory.

## Frozen code (sha256 of HPC copies, LF): recorded in the decisions.md row with this file's sha256.
