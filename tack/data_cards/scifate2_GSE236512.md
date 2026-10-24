# Data card: sci-FATE2 (GSE236512), metadata only

Built 2026-10-02 from the three processed h5ad files in backed mode (`tack/hpc/datacard_scifate2.py`, full output
`tack/hpc/out/datacard_scifate2.txt`). No expression value was read: the pre-registration for the decisive test
must be frozen before any expression-level look.

- Source: Maizels, Snell, Briscoe, "Deep dynamical modelling of developmental trajectories with temporal
  transcriptomics" (bioRxiv 10.1101/2023.07.06.547989); GEO GSE236512 (pilot GSE236517 not downloaded).
- Files on the HPC (`~/tack/data/scifate2/`, sha256 in `~/tack/logs/download_sha256.txt`):
  - `GSE236512_processed_data_counting.h5ad` (8.4 GB): layers `new`, `old`, `total`
  - `GSE236512_processed_data_estimate.h5ad` (8.0 GB): layers `new_estimated`, `old_estimated`, `total`
  - `GSE236512_processed_data_splicing.h5ad` (4.5 GB): layers `spliced`, `unspliced`
- Shape: 68,866 cells x 24,967 genes in all three files (the paper reports 44,713 after its own QC; the filter
  that maps one to the other is still to be found in the methods).
- Per-cell metadata: `sample` (1,534 values), `timepoint`, `rep`, labelling quantities `p_c_TC` (per-cell T>C
  conversion rate, median 0.0131, range 0.00015-0.189), `p_e` (background error, 0.000142 for all cells),
  `*_TC_alpha` (per-layer estimates; `X_TC_alpha` has extreme outliers up to 4.8e11).
- No `var` annotation beyond gene symbols; no embeddings, no cell types.

## Design (cells per timepoint x replicate)

| timepoint | r1 | r2 | r3 | r4 |
|---|---|---|---|---|
| 05h | 0 | 0 | 0 | 4,695 |
| 10h | 0 | 0 | 0 | 4,217 |
| 15h | 0 | 0 | 0 | 4,750 |
| 20h | 0 | 0 | 0 | 4,147 |
| D3 | 3,208 | 2,191 | 1,224 | 0 |
| D4 | 2,439 | 2,374 | 2,609 | 0 |
| D5 | 4,109 | 4,065 | 3,446 | 0 |
| D6 | 2,365 | 2,046 | 2,618 | 0 |
| D7 | 2,621 | 4,566 | 4,578 | 0 |
| D8 | 2,132 | 2,063 | 2,403 | 0 |

- Replicates r1-r3: full time courses D3-D8 (6 days). Replicate r4: four samples at 5-h spacing, which per the paper
  lie between day 3 and day 4; it has NO day-3 sample of its own (the TACK plan assumed one as onset anchor; that
  anchor must be replaced, e.g. by the r1-r3 day-3 data plus a per-replicate onset shift).
- Per the paper: all timepoints prepared and sequenced together (time not confounded with batch); 4sU 500 uM for
  2 h before fixation.

## What it can support
- Held-out labelled RNA (`new` given `total`) with per-cell conversion efficiency `p_c_TC` and error `p_e`: the
  decisive CLOCK test.
- Generation at intermediate real times: D4 (r1-r3) and the r4 5-h samples.
- Spliced/unspliced for a two-species variant.

## Resolved from the paper's methods (bioRxiv full text, read 2026-10-02)
- Protocol: day 0-2 N2B27 + bFGF; day 2: bFGF + 5 uM CHIR99021 for 20 h; **from day 3, and every 24 h after,
  N2B27 + 100 nM RA + 500 nM SAG**. Day 3 (T = 72 h) is therefore a known protocol onset for the neural programme
  (the "onset edge" case of P1'), with medium renewals every 24 h (candidate declared steps).
- `05h`..`20h` = D3.2, D3.4, D3.6, D3.8, i.e. day 3 + 5, 10, 15, 20 h.
- QC giving 44,713 cells: 10th-90th percentile on genes and UMIs (about 4,000-25,000 UMIs, 2,000-7,000 genes per
  cell), labelling rate above the 10th percentile (about 10 % labelled), mitochondrial fraction below the 90th
  percentile (about 5 %), scrublet doublet score below 0.3.
- `new`/`old`: read-level U-to-C mutations, dynast `count` method (raw assignment). `*_estimated`: dynast
  statistical correction. Use the raw `new`/`old` with `p_c_TC`/`p_e` for scoring, never the estimated layers.
- Cell types in the paper (not in the files): NMP, early neural, neural, pMN, MN, p3, V3, floor plate, other.
  Declared topology to pre-register: NMP -> early neural -> neural -> {pMN -> MN, p3 -> V3, floor plate}.

## Still open
- Our own cell-type annotation (markers decided before looking at held-out samples).
