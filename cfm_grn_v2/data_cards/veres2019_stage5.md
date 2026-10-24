# Data card — Veres et al. (2019), human in-vitro β-cell differentiation, Stage 5 daily time course

**Why this dataset.** It is one of the two fresh data sets of pre-registration v0.2
(`docs/superpowers/prereg/2026-09-25-prereg-v02.md`): no earlier step of the project has looked at it, so the
fold selection of Step 6 and the single measurement of Step 7 are not shaped by earlier diagnosis. It has what
the method needs and the embryoid-body series lacks: a **daily** clock (8 time points, one day apart) and **two
independent differentiations** followed across the same days, so a biological replicate exists at every stage.

**Which step of the project it serves.** Step 1 (reconstruction of a hidden time point), through pre-registration
Steps 5 (parsing), 6 (fresh-fold selection, folds V1–V56) and 7 (primary test: **day 4 predicted from day 3**).

## Source and files

| item | value |
|---|---|
| article | Veres A. et al., "Charting cellular identity during human in vitro β-cell differentiation", *Nature* 569, 368–373 (2019) |
| repository | GEO `GSE114412`, supplementary files of the series |
| terms | GEO terms: NCBI places no restriction on use; cite the article and the accession |
| downloaded | 2026-09-25, on the HPC, by `scripts_dev/hpc_download_v02.sh` (each file checked against the byte size recorded before approval) |
| where | on the HPC only: `bocconi:~/rafm_data/veres2019/` |
| processed cache | `bocconi:~/rafm_data/processed/veres2019_stage5.npz` + `veres2019_stage5.qc.json` (written by `scripts_dev/veres_prepare.py`) |

| file | bytes | sha256 (pre-registration) |
|---|---|---|
| `GSE114412_Stage_5.all.processed_counts.tsv.gz` | 108 394 226 | `9304c5ed767f72af3108e303176c8487737b0ede25965be66e2455b708cfba15` |
| `GSE114412_Stage_5.all.cell_metadata.tsv.gz` | 1 764 292 | `5006773f740191732bc1b4a6c3f086cc63d00a601cdeedc7cfcb4fa126facda7` |

The parser recomputes both sizes and hashes and stops if either differs.

**Contents.** The counts file is a **dense** tab-separated table, cells × genes: a header row whose first field is
`library.barcode` (possibly written with a leading `#`), followed by the gene symbols, then one row per cell (cell
id, then UMI counts); 51 274 cells are expected (table below). The layout was not opened before approval: the
header form, the gene count and whether all counts are whole numbers are observed values, filled below from the
qc report. The metadata table has one row per cell: `library.barcode`, `Assigned_cluster`, `Assigned_subcluster`, `tSNE_dim1`, `tSNE_dim2`, `Differentiation`
(1 or 2), `CellWeek` (despite the name, the **day within Stage 5**, 0–7), `Lib_prep_batch` (e.g.
`stg5diff1_S5d0_b1`), `Indrops_barcode_sequence`. The deposit holds the authors' processed cells (inDrops).

## Design

Stage 5 of the differentiation was sampled every day, in two independent differentiations. Cells per
(differentiation, day), recorded before approval and checked by code:

| day | 0 | 1 | 2 | 3 | 4 (test) | 5 | 6 | 7 | total |
|---|---|---|---|---|---|---|---|---|---|
| differentiation 1 | 4 563 | 3 406 | 1 942 | 2 901 | 3 337 | 3 346 | 2 448 | 2 141 | 24 084 |
| differentiation 2 | 1 627 | 2 771 | 1 620 | 9 576 | 3 143 | 3 058 | 3 565 | 1 830 | 27 190 |
| **both** | 6 190 | 6 177 | 3 562 | 12 477 | 6 480 | 6 404 | 6 013 | 3 971 | **51 274** |

- **Time** = the Stage-5 day (0–7). The grid is even, so the median gap g is 1 day and h_rel is in days.
- **Replicate** = the differentiation (`diff1`, `diff2`). It is a biological unit followed across days, so the
  training couples cells only within one differentiation (pre-registration Step 5: "couplings stay within a
  group"), and Step 7's ROBUST check scores each differentiation's day-4 cells separately.
- **Library** (`b1`, `b2` in `Lib_prep_batch`): two technical library preparations of the same day's sample. Kept
  as the `library` field (the loader also copies it into `pool`, only because the fold scripts' generic cache
  carries a pool column). It is never a unit of analysis.
- **Condition** = `stage5` (constant); **sample** = `diff<d>_d<day>`.
- **Test stage**: day 4. Only its cell count is printed by the parser; its expression is not inspected before
  Step 7, and Job D removes it before anything else.

## Parsing and checks (`scripts_dev/veres_prepare.py`, its own HPC job)

The dense table is read in chunks of 2 000 rows and stored as sparse CSR (cells × genes, float32 counts), then
joined to the metadata on `library.barcode`. **No cell or gene filter**: genes are chosen later on each fold's fit
cells, so a filter here would let the test stage shape the gene space. Parsing a full-width synthetic table (16 224
gene columns, about 14 % non-zero) took 19–54 s per 4 000 rows in local runs, i.e. up to about 12 min for the
51 274 rows, plus hashing and the compressed save. The two parsers together can therefore exceed the
pre-registration's "about 30 min" estimate for Step 5: each parser is submitted as its own job with
`--time=02:00:00`. The cache is written under a temporary name (`*.partial.npz`) and renamed only when complete,
so a killed job leaves no partial cache under the final name. All checks are applied by code:

- input sizes and SHA-256 against the pre-registration (stop);
- the header's first field must be `library.barcode`, with or without a leading `#`, which fixes the orientation
  (cells in rows), and no gene name may be empty (stop); the form found is recorded as `header_first_field`;
- every row complete, no duplicate barcode, required metadata columns present, integer `Differentiation` and
  `CellWeek` (stop);
- count rows without metadata and metadata rows without counts: reported (the unmatched rows are dropped);
- `Lib_prep_batch` parsed as `stg5diff<d>_S5d<day>_b<k>` and compared with (`Differentiation`, `CellWeek`):
  disagreements reported;
- non-integer counts, counted outside day 4 only (the test stage's expression is not inspected before Step 7), and
  duplicated gene symbols: reported (the loader makes symbols unique, a repeat becoming `symbol|k`, since
  the deposit has no separate gene ids; the cache keeps the deposit's symbols);
- cells per (differentiation, day) against the table above: a difference makes the job exit with an error
  **after** writing the cache and its report, and `load_veres` refuses such a cache (unless `strict=False`, which
  needs a logged decision).

Printed output (everything the parser prints; no expression statistic that includes a day-4 cell): the size and
SHA-256 of each input and whether it matches the pre-registration; the table's dimensions (rows × genes); the gene
count and the cells kept; the count rows without metadata (with up to 10 examples) and the metadata rows without
counts; the library-label disagreements; the non-integer entries outside day 4; cells per (differentiation, day)
and per day; any mismatch against the expected counts; the cache size and run time.

**Observed values** (fill from `veres2019_stage5.qc.json` after the Step-5 job): header form
(`header_first_field`), gene count (`genes`), cells kept, count rows without metadata, metadata rows without
counts, library-label disagreements, non-integer entries outside day 4 (`non_integer_entries_excl_test`; 0 means
all counts outside the test day are whole numbers), duplicated gene symbols, `expected_counts.ok`.

## Loader and preprocessing for the benchmark

`veres_prepare.load_veres(path)` (path defaults to the parser's `--out` default,
`~/rafm_data/processed/veres2019_stage5.npz`, one constant `DEFAULT_OUT`, so the fold scripts also work with
neither `--data` nor `--cache`) returns (X CSR raw counts, obs[cell, time, replicate, condition, sample,
library, pool], unique gene symbols, qc) after the pre-registered QC gate (a stage enters only with ≥ 500 cells; a
test stage that fails it is dropped and reported, never replaced). `scripts_dev/freshfold_select.py` (Job D) and
`freshfold_measure.py` (Job E) then: remove the test stage (Job D); cap at 1 500 cells per (day, differentiation)
and drop groups under 200 cells; normalise per cell to 10⁴ UMIs and log1p; choose genes **on the fold's fit cells
only** (among genes detected in ≥ 1 % of them, the 1 000 most dispersed, union the human CollecTRI regulators);
standardise on fit cells for training and on observed cells for evaluation.

## Prior

`~/rafm_data/priors/collectri_human.tsv` (CollecTRI via OmniPath). Coverage in each fold's gene space is written
into that run's `summary.json`.

## Known confounds and limitations

- **Libraries b1/b2 are technical.** They split one day's sample and are nested inside (differentiation, day), so
  they cannot confound time; a library effect only adds within-stage spread.
- **Differentiation = biological replicate = culture batch.** Each differentiation is a separate culture with its
  own libraries, so the day-4 replicate floor (distance between the two differentiations) mixes biological and
  batch differences.
- **Two replicates only.** The ROBUST check of Step 7 has two targets.
- **Unbalanced day 3.** Differentiation 2 has 9 576 cells at day 3, three times differentiation 1; the cap of 1 500
  cells per (day, differentiation) equalises what the training sees. Day 3 is the source of the primary test.
- **In-vitro mixture.** Stage 5 contains endocrine and non-endocrine populations (the authors' clusters); a
  population-level distance mixes their errors.
- **Column name.** `CellWeek` holds the Stage-5 day, not a week; the parser checks it against the day written in
  `Lib_prep_batch`.
