The cell-level data exist on the HPC for gastrulation, but not for pancreas. Dileo's "gastrulation" set is the scVelo `gastrulation_erythroid` dataset, and it is at `/home/3393519/tack/data/erythroid/erythroid_lineage.h5ad`. Dileo's "pancreas" set is the scVelo `pancreas` dataset (Bastidas-Ponce 2019, 3696 cells), and I did not find it locally or on the HPC. No file anywhere stores DPT pseudotime or per-cell bin labels. However, Dileo's own files record how many cells fell in each bin, so a rebuild can be checked exactly.

## How I matched the datasets
Dileo's `*_node.csv` files hold a mean and a sum per gene for each bin. Sum divided by mean gives the number of cells in that bin, and it comes out identical for every gene.
- **Pancreas:** 15 bins (t000â€“t014). Cells per bin: 111, 612, 361, 116, 50, 53, 68, 79, 191, 218, 132, 286, 698, 616, 105. Total **3696**, the exact size of scVelo pancreas.
- **Gastrulation:** 35 bins (t000â€“t034). Cells per bin: 1, 4, 5, 63, 569, 775, 922, 810, â€¦, 103, 33, 7. Total **9815**, the exact size of `erythroid_lineage.h5ad`. All 6345/6345 of Dileo's gastrulation genes are in that file's gene list.
- **Snapshots that have edges:** I counted the `ts` column of the edge files.
  - Pancreas has edges only for ts 1â€“12, so **12 snapshots** with about 0.89â€“1.0M edges each. Bins 0, 13 and 14 (111, 616 and 105 cells) have features but no edges.
  - Gastrulation has edges for ts 2â€“34, so **33 snapshots**, not the 32 in the brief. Bins 0 and 1 (1 and 4 cells) have no edges; bin 2, with 5 cells, has 446,866.
- **Gene counts:** 7197 pancreas genes and 6345 gastrulation genes, which fits the ~7k / ~6k in the brief. I did not re-measure the ~660 / ~550 regulators.
- **How the bins were built (my inference):** every `bin_mid` falls inside [k/15, (k+1)/15] for pancreas and [k/35, (k+1)/35] for gastrulation, with k = ts. That points to equal-width bins on a pseudotime scaled to [0,1]. `bin_mid` looks like the mean pseudotime of the bin's cells, not the bin centre.
- **Expression used:** each edge row's `feat` JSON says `"expression_layer": "Ms"`, which is scVelo's smoothed expression.
- **What is not recorded:** the pseudotime method (DPT or velocity pseudotime) and the root cell. `tgl-grn-1CCD\README.md:58` says the code that built the networks was not released.

The bin geometry is also written in two local files:
- `tgl-grn-1CCD\AUTOPSY.md:248-257` (cells per bin, 50â†’698 for pancreas and 5â†’922 for gastrulation).
- `tgl-grn-1CCD\logs-baseline\centrality_persistence_and_bins.json:99-216` (`bin_mid` and `n_cells_bin`, edge bins only).

## Candidate sources on the HPC (`/home/3393519/...`)

| Path | Size | Contents | Match? |
|---|---|---|---|
| `tack/data/erythroid/erythroid_lineage.h5ad` | 485.8 MB | 9815 cells Ã— 53801 genes, sparse counts. Layers `spliced` and `unspliced`. Cell info: celltype (5 types), stage (7: E7.0 to E8.5, with 43 / 65 / 471 / 1709 / 2811 / 1695 / 3021 cells), sample, sequencing.batch, theiler. Gene info includes `MURK_gene` and `Î”m`. Embeddings `X_pca` and `X_umap`. Old AnnData format. **No** `Ms`, neighbour graph, pseudotime or bin labels. | **Yes: Dileo gastrulation** |
| `rafm_data/processed/pijuan_sala2019_atlas.npz` | 769.0 MB | 108857 Ã— 29452 raw counts. Cell info: stage (9, E6.5â€“E8.5), sample, pool, batch, time. No cell type, no spliced/unspliced, no pseudotime. Raw files are in `rafm_data/pijuan_sala2019/`. | Parent atlas only, not a substitute |
| `rafm_data/processed/veres2019_stage5.npz` | 154.4 MB | Human stem-cell-derived islets, 51274 Ã— 16224, days 0â€“7 | No |
| `rafm_data/processed/moon2019_eb.npz` | 100.9 MB | Human embryoid bodies, 31054 Ã— 17619 | No |
| `rafm_data/processed/argelaguet2022_rna.npz` | 760.4 MB | Mouse, 86139 Ã— 24672, E7.5â€“E8.75, includes a T knockout | No |
| `tgl-grn/GRN/` | â€” | Only Dileo's derived files: `pancreas.csv` 1.66 GB, `gastrulation.csv` 3.81 GB, `*_node.csv`, `*_y_*.csv`, `*.pt` embeddings. `data.zip` (2.53 GB) holds the same 23 files plus a README. | No cell data |
| `tack/data/scifate2*/*.h5ad` (4.5â€“8.4 GB) | â€” | sci-fate2 data | No |
| `gb_runs/`, `grnbench_*` | â€” | Cellxgene files, including a human pancreas cel-seq2 set | No |
| `cfm_grn_v2/` | â€” | Code, data cards and outputs only | No cell data |

## Local
- The project folder has no `*.h5ad`, `*.loom` or `*.h5` files. Its only `.npz` files are synthetic `grn_truth.npz` outputs and two files under `tack/results/scifate2/clock_probe/`.
- `tgl-grn-1CCD\GRN\` holds local copies of Dileo's derived CSVs and `data.zip`, the same files as on the HPC, with no cell data.
- `C:\Users\Gabriele\data\rafm\processed\moon2019_eb.npz` and `C:\Users\Gabriele\data\rafm\raw\argelaguet2022\rna\anndata.h5ad` are not Dileo's datasets. I did not open the Argelaguet file.
- Downloads and Documents have no `h5ad` or `loom` files.

## Pancreas (Bastidas-Ponce, `scvelo.datasets.pancreas`)
I found it nowhere: not locally, not in `rafm_data`, `tack/data`, `tgl-grn` or `cfm_grn_v2`. Two searches were incomplete, so this is not a guaranteed "absent":
- The whole-home `find` runs on the HPC were cut off by output limits.
- The last, narrower `find` had printed nothing when I stopped and had not finished.
- A local search across all of `C:\Users\Gabriele` timed out.

Rebuilding pancreas would need a download, which needs your permission and should run on the HPC.

## What a rebuild would involve
- **Gastrulation:** this can be done from the HPC file. It means recomputing scVelo preprocessing and the `Ms` layer, a pseudotime scaled to [0,1], and 35 equal-width bins. A correct rebuild must reproduce the 35 cell counts above and the per-gene means in `gastrulation_node.csv`.
- **Pancreas:** the same steps with 15 bins and the 15 counts above, once the data are downloaded.
- **Main unknowns:** the pseudotime method and the root cell are recorded nowhere, so they must be found by trial against those counts.

The three read-only scripts I ran on the HPC are in `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\`: `inspect_meta.py`, `inspect_genes.py` and `inspect_bins.py`.
