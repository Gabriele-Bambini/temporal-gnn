# Data card — Pijuan-Sala et al. (2019), mouse gastrulation atlas E6.5–E8.5

**Why this dataset.** It is the second fresh data set of pre-registration v0.2
(`docs/superpowers/prereg/2026-09-25-prereg-v02.md`). It covers the same biology as the Argelaguet series (mouse
gastrulation), on a finer and longer clock: **nine stages a quarter day apart**, E6.5 to E8.5, about 109 000
wild-type cells. Its E8.0 stage predicted from E7.75 is a direct replication test of the Argelaguet E8.0
negative, on embryos, a protocol and a time grid the diagnosis has never seen.

**Which step of the project it serves.** Step 1 (reconstruction of a hidden time point), through pre-registration
Steps 5 (parsing), 6 (fresh-fold selection, folds P1–P234) and 7 (primary test: **E8.0 predicted from E7.75**;
secondary: the same with E8.25 also removed, the Argelaguet bracket E7.75 → E8.5).

## Source and files

| item | value |
|---|---|
| article | Pijuan-Sala B. et al., "A single-cell molecular map of mouse gastrulation and early organogenesis", *Nature* 566, 490–495 (2019) |
| repository | ArrayExpress `E-MTAB-6967`; files taken from the authors' processed share `https://content.cruk.cam.ac.uk/jmlab/atlas_data/` (the "atlas" of the MouseGastrulationData package) |
| terms | those of the public deposit ArrayExpress E-MTAB-6967; cite the article |
| downloaded | 2026-09-25, on the HPC, by `scripts_dev/hpc_download_v02.sh` (each file checked against the byte size recorded before approval) |
| where | on the HPC only: `bocconi:~/rafm_data/pijuan_sala2019/` |
| processed cache | `bocconi:~/rafm_data/processed/pijuan_sala2019_atlas.npz` + `pijuan_sala2019_atlas.qc.json` (written by `scripts_dev/pijuan_sala_prepare.py`) |

| file | bytes | sha256 (pre-registration) |
|---|---|---|
| `raw_counts.mtx.gz` | 1 527 653 430 | `a032d82e4f948214e69bda04ac954fa0a322b5bc0c3928a7ef61a8ffdb89c809` |
| `genes.tsv.gz` | 228 898 | `e1dbb4662b224fd77f6c3ac52c5882da2abd1013daf869cb2fb99a004044f85f` |
| `meta.tab.gz` | 4 941 002 | `1b5e09b6c16803278f9b2d427dff0888f20d41889de3369eebbae840cc886988` |

The parser recomputes all sizes and hashes and stops if any differs.

**Contents.**
- `raw_counts.mtx.gz`: MatrixMarket coordinate, integer, **genes × cells** (29 452 × 139 331), 483 512 215
  entries, 1-based indices.
- `genes.tsv.gz`: no header; Ensembl id, symbol; one row per matrix row.
- `meta.tab.gz`: tab-separated with header `cell, barcode, sample, pool, stage, sequencing.batch, theiler,
  doub.density, doublet, cluster, cluster.sub, cluster.stage, cluster.theiler, stripped, celltype, colour, umapX,
  umapY`; **its rows are the matrix columns, in order** (there is no barcode file).

## Design

Cells kept after the pre-registered filter, recorded before approval and checked by code:

| stage | time (quarter days from E6.5) | kept cells |
|---|---|---|
| E6.5 | 0 | 3 484 |
| E6.75 | 1 | 2 075 |
| E7.0 | 2 | 14 749 |
| E7.25 | 3 | 13 537 |
| E7.5 | 4 | 10 994 |
| E7.75 | 5 | 14 493 |
| **E8.0 (test)** | **6** | **16 681** |
| E8.25 | 7 | 15 935 |
| E8.5 | 8 | 16 909 |
| **total** | | **108 857** |

- **Filter** (the authors' flags in `meta.tab`): drop `doublet == TRUE`, `stripped == TRUE` (stripped nuclei) and
  `stage == mixed_gastrulation` (pooled embryos without a single stage). No gene filter.
- **Time** = quarter days from E6.5 (E6.5 → 0, …, E8.5 → 8); the grid is even, so g = 0.25 day.
- **Replicate** = `all`: a **documented deviation** from the pre-registration's Step 5 "replicate = pool", logged
  in `docs/superpowers/decisions.md` (2026-09-26 afternoon entry, which reclassifies the
  "clarification" of the earlier 2026-09-26 entry). Reason: every embryo pool exists at one stage only, so with replicate = pool no stage
  transition stays inside a replicate and the training set would be empty (no model and no transition control
  could be trained on this data set). Applied: training, coupling and scoring pool the pools of a stage, as
  Argelaguet's `wild_type_course` does, and the cap of 3 000 cells applies per stage as pre-registered. Unchanged:
  the pool is kept for the replicate floor and the per-pool ROBUST check at E8.0. The cache's
  `qc["replicate"]` string records the same deviation.
- **Pool** (`pool`), 10x **sample** (`sample`) and **sequencing batch** (`batch`) are kept per cell. The pool is
  used **only** for the replicate floor at the test stage and for Step 7's ROBUST check against each E8.0 pool
  separately (`freshfold_measure.ROBUST_KEY`).
- **Condition** = `WT` (constant); the cache also keeps the `stage` label.
- **Test stage**: E8.0 (time 6). Only its cell count is printed by the parser; its expression is not inspected
  before Step 7, and Job D removes it before anything else.

## Parsing and checks (`scripts_dev/pijuan_sala_prepare.py`, its own HPC job)

The matrix is streamed in chunks of 2·10⁷ entries; only the entries of kept cells are held (int32 cell and gene
indices, float32 counts; about 4.5 GB per copy for the expected ~3.8·10⁸ kept entries, far below 96 GB) and then
turned into CSR (cells × genes). Measured locally on a synthetic file, the streaming runs at about 1.5–1.6·10⁶
entries per second, i.e. 5–6 minutes for the full matrix; SHA-256 of the 1.5 GB input and the compressed save of
a CSR of about 3 GB come on top. The two parsers together can therefore exceed the pre-registration's "about
30 min" estimate for Step 5: each parser is submitted as its own job with `--time=02:00:00`. The cache is written
under a temporary name (`*.partial.npz`) and renamed only when complete, so a killed job leaves no partial cache
under the final name. All checks are applied by code:

- input sizes and SHA-256 against the pre-registration (stop);
- matrix rows = `genes.tsv` rows; **matrix columns = `meta.tab` rows** (stop: matrix and metadata not aligned);
- required metadata columns, unique cell names, every stage known and on the quarter-day grid, `doublet` and
  `stripped` exactly TRUE/FALSE (stop);
- every entry within the declared dimensions, as many entries as the header declares, no duplicated (gene, cell)
  entry among kept cells (stop);
- reported: non-integer counts, counted outside E8.0 only (the test stage's expression is not inspected before
  Step 7; the kept-entry count also leaves E8.0 out); duplicated or empty gene symbols (read literally, so a gene named `NA` stays
  `NA`; the loader makes names unique as `symbol|ensembl`, an empty symbol becoming its Ensembl id; the cache keeps
  the deposit's symbols and all Ensembl ids); whether the `cell` names run `cell_1 … cell_139331` in column order
  (an extra alignment witness); cells per stage and per (stage, pool); the sequencing batches of every (stage,
  pool); the stages whose pools span several batches;
- kept cells per stage against the table above: a difference makes the job exit with an error **after** writing
  the cache and its report, and `load_pijuan_sala` refuses such a cache (unless `strict=False`, which needs a
  logged decision).

Printed output (everything the parser prints; no expression statistic that includes an E8.0 cell): the size and
SHA-256 of each input and whether it matches the pre-registration; the matrix dimensions and declared entries from
its header; the kept cells and the filter counts (doublet, stripped, mixed_gastrulation); the number of entries
streamed; the gene count; the non-integer entries outside E8.0; whether the `cell` names run `cell_1 … cell_N` in
column order; cells per stage; the stages whose pools span several sequencing batches; any mismatch against the
expected counts; the cache size and run time.

**Observed values** (fill from `pijuan_sala2019_atlas.qc.json` after the Step-5 job): cells flagged doublet,
stripped and mixed_gastrulation; cells kept; entries kept outside E8.0 (`matrix.entries_kept_excl_test`);
non-integer entries outside E8.0 (`non_integer_entries_excl_test`); duplicated and empty symbols;
`meta_cell_names_sequential`; `per_stage_pool`; `stage_pool_batches`; `expected_counts.ok`.

## Loader and preprocessing for the benchmark

`pijuan_sala_prepare.load_pijuan_sala(path)` (path defaults to the parser's `--out` default,
`~/rafm_data/processed/pijuan_sala2019_atlas.npz`, one constant `DEFAULT_OUT`, so the fold scripts also work with
neither `--data` nor `--cache`) returns (X CSR raw counts, obs[cell, stage, time, replicate,
condition, sample, pool, batch], unique gene symbols, qc) after the pre-registered QC gate (a stage enters only
with ≥ 500 cells; a test stage that fails it is dropped and reported, never replaced). The fold scripts then
remove the test stage (Job D), cap at 3 000 cells per stage, normalise per cell to 10⁴ UMIs and log1p, choose
genes **on the fold's fit cells only** (≥ 1 % detection, 1 000 most dispersed, union the mouse CollecTRI
regulators) and standardise as for the other data sets.

## Prior

`~/rafm_data/priors/collectri_mouse.tsv` (CollecTRI via OmniPath), the same file as for Argelaguet.

## Known confounds and limitations

- **Sequencing batch is partly tied to stage.** Batch and stage are partly collinear, so a batch shift can look
  like a time step. The parser reports the batches of every (stage, pool).
- **The E8.0 pools are in different sequencing batches.** The replicate floor at E8.0 (distance between its pools)
  therefore mixes biological and batch differences; Step 7 flags this (`freshfold_measure.FLOOR_NOTE`), and the
  per-pool ROBUST check inherits it.
- **Pools do not link stages.** There is no biological unit followed across time; couplings are between pooled
  populations, and confidence intervals are over training seeds.
- **Small early stages.** E6.5 (3 484) and E6.75 (2 075) pass the 500-cell gate but give the first folds fewer
  distinct source cells.
- **Expected E8.5 count.** The per-stage counts were read, before approval, from the UCSC Cell Browser summary of
  the *extended* atlas (E6.5–E9.5), whose E6.5–E8.25 counts are those of the 2019 atlas but whose E8.5 also holds
  later samples. The 2019 E8.5 value (16 909) was therefore obtained by subtraction: 116 312 cells in the 2019
  atlas, minus the eight other stages and the 7 455 mixed_gastrulation cells. If any count differs, the parser
  stops loudly and the difference needs a logged decision before Job D.
- **Mixed lineages.** Gastrulation splits into many lineages within these stages; a population-level distance
  mixes their errors.
