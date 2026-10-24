# Data card — Moon et al. (2019), human embryoid-body differentiation

**Why this dataset.** It is the time course of the held-out-timepoint benchmark introduced with TrajectoryNet
(Tong et al. 2020) and reused by later flow-matching methods (OT-CFM, Tong et al. 2024, and others). They score it
in 5 whitened principal components with the Wasserstein-1 distance, taken from TrajectoryNet's processed file
`eb_velocity_v5.npz` (which holds the first 100 components), holding out each of the three inner windows in turn
and averaging. MIOFlow (Huguet et al. 2022) uses a variant: 200 principal components computed by its authors from
the same raw deposit, scored with W1 and MMD. So our gene-space energy distances are not comparable to their
numbers; a like-for-like run in their space is still to be done. It is also the smallest real series that
has the structure the method needs: several snapshots of one differentiating population, unpaired, with a real
biological clock.

**Which step of the project it serves.** Step 1 only (reconstruction of a hidden time point). It has no
chromatin and no perturbation, so it cannot serve steps 2 and 3; those need the multiome series.

## Source and files

| item | value |
|---|---|
| repository | Mendeley Data, `10.17632/v6n743h5ng.1`, "Embryoid Body data for PHATE" |
| licence | CC BY 4.0 |
| file downloaded | `scRNAseq.zip`, 161 681 597 bytes (161.7 MB), on 2026-09-20 |
| local raw copy | `C:\Users\Gabriele\data\rafm\raw\moon2019\scRNAseq\` |
| local processed cache | `C:\Users\Gabriele\data\rafm\processed\moon2019_eb.npz` (101 MB) |
| contents | five 10x Genomics samples in MatrixMarket form (`matrix.mtx`, `genes.tsv`, `barcodes.tsv`), identical gene list (33 694 genes, Ensembl + symbol) in all five |

The related article is the PHATE paper (Moon et al., *Nature Biotechnology* 2019); the deposit is the single-cell
half of it. Bulk RNA-seq (`bulkRNAseq.zip`, 2.2 MB) was not downloaded: it is not needed for this task.

## Design

One embryoid-body differentiation followed for 27 days and sampled in five windows. There is **one sample per
window and no biological replicate**, so the time course is a single unit.

| sample | window | cells (raw) | cells (after QC) |
|---|---|---|---|
| T0_1A | days 0–3 | 4 649 | 4 636 |
| T2_3B | days 6–9 | 7 377 | 7 375 |
| T4_5C | days 12–15 | 6 245 | 6 241 |
| T6_7D | days 18–21 | 6 563 | 6 555 |
| T8_9E | days 24–27 | 6 327 | 6 247 |
| **total** | | **31 161** | **31 054** |

Biological time is the window index 0–4; the windows are equally spaced six days apart, so one time unit is six
days. Condition = `EB`, replicate = `eb1` (constant): the five samples are the same culture at five times, not
five independent units.

## Quality control (`scripts_dev/eb_prepare.py`)

A cell is kept when it has ≥ 1 000 UMIs, ≥ 500 detected genes and a mitochondrial fraction < 0.20. A gene is
kept when it is detected in ≥ 20 of the kept cells. **Nothing in the filter uses the time label**, so it cannot
favour one window over another.

- cells: 31 161 → 31 054 (0.3 % removed; the deposit is already lightly filtered)
- genes: 33 694 → 17 619
- median UMIs per cell 4 152, median genes per cell 1 513

## Preprocessing for the benchmark (`scripts_dev/eb_benchmark.py`)

1. Per-cell library-size normalisation to 10⁴ UMIs followed by log1p. This is a per-cell operation, so it cannot
   move information between cells or between windows; the matrix stays sparse.
2. Gene selection **on the fit cells only**: the 1 000 genes with the highest binned normalised dispersion,
   union the CollecTRI transcription factors. The fit cells here are the windows that are neither held out nor
   used for early stopping, so neither the held-out window nor the validation window influences the gene space.
   Since 2026-09-24 only genes detected in at least 1 % of the fit cells are candidates (SCENIC default). The
   first runs (`outputs/eb_interp`, `outputs/eb_interp_pca50`) used no such filter: 524 of their 1 272 genes
   (41 %) were detected in fewer than 1 fit cell in 100, each a one-cell spike after standardisation (|z| up to
   77). `--min-detect 0` reproduces them.
3. Subsampling to at most 2 000 cells per (condition, replicate, window) — a compute decision, declared here and
   in the report; groups with fewer than 200 cells would be dropped (none are).
4. Standardisation is fitted inside the training routine on fit cells only, and the evaluation space is
   standardised on the observed cells, as in the synthetic benchmark.

No PCA in the primary task: the regulatory phase needs genes. A PCA-50 run is kept as a robustness control.

## Prior coverage

`C:\Users\Gabriele\data\rafm\priors\collectri_human.tsv` (CollecTRI via OmniPath, signed TF → target). Coverage
within the selected gene space is written into `outputs/eb_interp/summary.json` at every run
(`prior_edges_within_selected_genes`, `prior_regulators_present`).

## Known limitations

- **No biological replicate.** Confidence intervals are over training seeds only; a replicate-level control is
  not possible on this dataset. The multiome series has replicates and will supply it.
- **Coarse time.** Each snapshot pools three days of differentiation, so within-window heterogeneity is part of
  the "snapshot" and the effective clock is coarse.
- **One lineage mixture, no ground truth.** The dataset cannot tell us which regulatory links are active; it
  tests step 1 only.
- **Shallow sequencing.** A median of 4 152 UMIs per cell means a noisy per-gene signal; this is one reason the
  published analyses work in a reduced space.
