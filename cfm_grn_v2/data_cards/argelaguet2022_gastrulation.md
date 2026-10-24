# Data card — Argelaguet et al. (2022), mouse gastrulation multiome time course

**Why this dataset.** It is the real series the project was designed around: a time course of mouse embryos
from gastrulation to early organogenesis, profiled with single-cell multiome (RNA and chromatin accessibility
from the same cell), with **biological replicates** at most stages and a Brachyury (T) knockout. Step 1 (predict a
hidden stage) uses the RNA of the wild-type course; the chromatin half is kept for step 2 (do the inferred active
links agree with accessibility?) and the knockout for step 3.

**Which step of the project it serves.** Step 1 now; steps 2 and 3 later from the same files.

## Source and files

| item | value |
|---|---|
| repository | GEO `GSE205117` (per-sample 10x matrices) + the authors' processed-data share (cell metadata) |
| files downloaded | 33 GEO files `GSM62054xx_<sample>_GEX_{matrix.mtx,barcodes.tsv,features.tsv}.gz`, 5.07 GB in total, on 2026-09-24, each checked against the size GEO declares (`geo_download.log`: `DONE ok=33 bad=0`); `metadata.txt.gz` (1 760 514 bytes) from the authors' share, `data/processed/rna/` |
| where | on the HPC only: `bocconi:~/rafm_data/argelaguet2022/{geo,meta}/` |
| processed cache | `bocconi:~/rafm_data/processed/argelaguet2022_rna.npz` (760 MB, sparse) |
| contents | each matrix holds **both** modalities: 32 285 "Gene Expression" rows and 199 885 "Peaks" rows; the barcodes are the cells the authors kept after their own quality control and match the metadata table one to one |

The Dropbox copy of the authors' AnnData objects was throttled to 5–20 KB/s, so the GEO route was used, on the
HPC, by the user's choice.

## Design

| stage | biological time (quarter days from E7.5) | samples | cells |
|---|---|---|---|
| E7.5 | 0 | rep1, rep2 | 22 117 |
| E7.75 | 1 | rep1 | 5 257 |
| E8.0 | 2 | rep1, rep2 | 12 206 |
| E8.5 | 4 | rep1, rep2 | 22 322 |
| E8.75 | 5 | rep1, rep2 | 9 274 |
| E8.5 CRISPR | — | T knockout, matched wild type | 7 359 + 7 604 (not used in step 1) |

Stages are unevenly spaced (a quarter day, then half a day), and the unequal spacing is kept: the duration clock
integrates over the true biological horizon. Replicate labels do not link embryos across stages, so the
replicates of a stage are pooled into one population (`replicate = pool`). What the replicates DO give is a
biological yardstick: the energy distance between the two replicates of a stage, in the evaluation space.

## Quality control (`scripts_dev/argelaguet_prepare.py`)

No cell filter beyond the authors' (the barcodes are already their QC cells). Gene rows only; a gene is kept when
it is detected in ≥ 20 cells. 32 285 → 24 672 genes, 86 139 cells in total. **Nothing in the filter uses the stage.**

## Preprocessing for the benchmark (`scripts_dev/argelaguet_benchmark.py`)

1. Wild-type course only (`wild_type_course`): the two CRISPR samples are a perturbation experiment and are kept
   out of this task.
2. Per-cell normalisation to 10⁴ UMIs and log1p (sparse).
3. Gene selection **on the fit cells only** (neither the held-out stage nor the validation stage): among genes
   detected in at least 1 % of the fit cells, the 1 000 with the highest binned normalised dispersion, union the
   mouse CollecTRI regulators. The 1 % detection filter (SCENIC default) was added on 2026-09-24, after the
   first run: without it 504 of the 1 544 selected genes (33 %) were detected in fewer than 1 fit cell in 100,
   and after standardisation each was a spike of one cell at |z| 77–95 (one gene, Ptf1a, had zero variance on the
   observed stages and reached |z| 263 821 in E8.0). `--min-detect 0` reproduces the first run.
4. At most 3 000 cells per stage (a compute decision); standardisation fitted on fit cells for training and on
   observed cells for evaluation.

## Prior coverage

`~/rafm_data/priors/collectri_mouse.tsv` (CollecTRI via OmniPath). In the first run's gene space: 1 974 prior
edges and 407 regulators present. Every run writes its own coverage into `summary.json`.

## Known limitations

- **One replicate at E7.75.** That stage is the source of the E8.0 prediction, so the source population is a
  single embryo pool.
- **Pooled replicates.** Confidence intervals are over training seeds; the replicate distance is reported as a
  yardstick, not used as a unit of analysis.
- **Short, uneven horizon.** E8.0 is a quarter day from its source and the bracketing gap E7.75 → E8.5 is three
  quarter days, so persistence is a strong baseline and interpolation of the stage means is stronger still.
- **Mixed lineages.** Gastrulation splits into many lineages within these stages; a single population-level
  distance mixes their errors.
