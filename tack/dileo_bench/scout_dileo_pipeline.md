I rebuilt the benchmark and metrics from your replication code and the files on the HPC. Your OT model can only be scored directly on Task 2, and only after the per-bin cell groups are rebuilt, because the cell-to-bin assignments were never saved. Before you compare any numbers, two points matter:

- **No unmodified copy of the Dileo & Sottoriva code exists** in any of the three places. Local `tgl-grn-1CCD`, `tgl-grn-1CCD-fix/backup_local_20260917` and HPC `backup_20260917_ns` all contain your fixes. The original behaviour comes only from `AUTOPSY.md` and the `TGL_LEGACY_EARLYSTOP` switch.
- **`results/results.md` holds your own HPC reproduction (with fixes), not the paper's table.** The only numbers credited to the paper are in `NEGATIVE_SAMPLING_AUDIT_2026-09-17.md:3,35`: GCRN-GRU 0.94â€“0.95 and EdgeBank 0.79 on pancreas.

Paths below are relative to L = `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tgl-grn-1CCD` and H = `bocconi:~/tgl-grn`. Line numbers are local L files unless prefixed H.

## 1. Datasets, bins and GRNs

**Format.** The data comes from HF `manuel-dileo/temporal-grn` (`README.md:56`).
- Edge file columns are `u,i,ts,label,feat` (`dataset.py:209-235`). `label` is the GRNBoost2 importance and becomes the edge weight, because `feat` is a JSON string, not a number (`dataset.py:240-248`).
- Each `feat` row carries `{"method":"grnboost2","bin_mid",...,"n_cells_bin",...,"expression_layer":"Ms"}`. Mouse uses layer `Ms`; `pancreas-human` uses layer `X` (from the CSV heads on H).

**Snapshots (from the target CSV headers on H):**

| Dataset | GRN timestamps | Bins in `_node.csv` | Transitions evaluated |
|---|---|---|---|
| pancreas | 1â€“12 (12) | 15 | 11 |
| gastrulation | 2â€“34 (33) | 35 | 31 (ts 34 never used) |
| pancreas-human | 0â€“14 (15) | 15 | 14 |
| gastrulation-human | 2â€“31 (30) | 32 | 29 |

- `train_split` is 31, 11, 29 and 14 respectively (`train_models.py:18-28`, `train_models_gene.py:17-37`).
- In mouse pancreas, bins 0, 13 and 14 have expression statistics but no GRN. Pancreas-human keeps bin 0 (111 cells). The files don't say why the filtering differs.

**Pancreas, measured on H (one pass over the edge file):**
- N = 7197 nodes; 662â€“663 source nodes per snapshot; 895kâ€“1.0M edges per snapshot, no pruning.
- `n_cells_bin` per ts 1..12: 612, 361, 116, 50, 53, 68, 79, 191, 218, 132, 286, 698. Bins are not equal-size.
- Gastrulation ts 2 has only 5 cells (`AUTOPSY.md:251-256`).

**Binning (my inference, not a stated procedure).** All 12 pancreas `bin_mid` values fall inside [k/15, (k+1)/15], and gastrulation ts 2 (0.0769) falls inside [2/35, 3/35]. This fits equal-width bins on a pseudotime scaled to [0,1], with `bin_mid` being the mean pseudotime of the cells in each bin.

**Not determinable from these files:**
- Which pseudotime was used (DPT, latent time, other) and which root.
- The TF list and the GRNBoost2 settings.
- That the sources are Bastidas-Ponce or Pijuan-Sala. `Ms` points to scVelo-preprocessed data, and the 2,864 pancreas cells in bins 1â€“12 are consistent with scVelo pancreas (3,696 cells), but this is not proof.

**Human variants.** The `gene` column mixes numeric IDs (17685, 17686, â€¦) with human symbols (â€¦ ZZZ3), 18,561 genes in total. Whether they are ortholog-mapped is not determinable.

**Per-gene node statistics.** Each bin has columns `t{bin}_s00_mean, s01_median, s02_std, s03_frac_nonzero, s04_sum`, ordered by the `s` index (`dataset.py:279-310`).

**Cell-level preprocessing code (AnnData to bins) is absent everywhere.** `README.md:58` says it "will be made available in the next iterations". My greps for h5ad, grnboost, dpt, scanpy and arboreto over H `*.py` and `*.ipynb` found nothing relevant. **Cell-to-bin assignments are not saved**; only `bin_mid` and `n_cells_bin` per edge row survive. In the upgraded loader they are kept, with an optional `min_cells_bin` filter (H `dataset.py:15-25, 268-299, 599-626`).

## 2. Task 1: link prediction

**Live-update loop** (`train_models.py:423-441`). For each transition i, using `snapshots = dataset[:train_split]`:
1. Snapshot i is split with `RandomLinkSplit(num_val=0.0, num_test=0.25)` (`:290-291`).
   - The 75% part is used for message passing and supervision. PyG's default training negatives apply: 1:1, uniform over all N nodes, independent of `--ns`.
   - The held-out 25% is used as validation for early stopping on validation AUPRC (`:875-905`).
2. The test positives are all edges of snapshot i+1 (`:294-299, 321`).
3. The encoder at test time sees only the 75% edges of snapshot i (`:919-926, 1014-1020`).
4. Training then continues on i+1. The val/test entries of `dataset_split_point` are never used for scoring.

**Negatives for the t+1 test set** (count = number of positives, `:326-373`):
- `active` (the paper's choice, `README.md:18`, code `:81-103`): u and v uniform over every node with an edge at t+1, rejecting (u,v) and (v,u). About 92.4% of these negatives have a non-regulator source (audit md:22).
- `random`: PyG `negative_sampling` on t+1 (`:328-333`).
- `historical`: edges from snapshots 0..i-1 that are absent at t+1. History is updated after `build_snapshot_split` (`:443` then `:490-493`). At i=0 it falls back to random (`:344-349`).
- Your audit variants (H `train_models.py`):
  - `--restrict_neg_source` (`:97-120`, `train_lp.py:38-77`).
  - `degree`: u proportional to out-degree at t, v proportional to in-degree at t, true t+1 edges rejected (`:355-380`).
  - `tfuniform` (`:383-406`).
  - Branches at `:502, :509`.

**AUPRC.** sklearn `average_precision_score` on sigmoid scores, plus AUC, computed per transition (`:1022-1027`). The reported value is the plain mean over transitions (`:541`). The per-snapshot list is written to `logs/live_{ns}_{model}_grn_{name}.json` (`train_lp.py:12-20,106`).

**EdgeBank.** Binary score 1 if (u,v) appears in the union of snapshots 0..i, with history updated before scoring. AUPRC is taken on the binary scores and averaged over transitions (`:1170-1241`).

**Training settings.**
- Epochs: 1000 now (`train_models.py:797-803`, H `:967`); 100 originally (`AUTOPSY.md:64-65`).
- Early stopping: patience 20 now (`:806-931`); the original tolerance-0.05 rule is preserved at `:887-894`.
- Seed 1234 (`train_lp.py:39`).

## 3. Task 2: expression forecasting

**Target.** `log(mean_{t+1} + 1e-8) - log(mean_t + 1e-8)`, using raw `s00_mean` (`mean_feat_idx=0`), clipped to Â±5 (`train_models_gene.py:165-168, 332-338`; default clip at `:402`).
- Genes that switch on or off hit Â±5, and these dominate the MAE.
- All nodes are used.
- The human variants are scored only on their own files (`results.md:53`).

**Input.** The 5 statistics at t, z-scored across genes within each snapshot (`:171-186, 330`), plus the GRN at t with importance weights. This requires `--gene_features` (`README.md:30`); without it x is `randn(N,5)` (`dataset.py:338-344`).

**Split.** Within each transition, a random 75/25 split over genes via `torch.randperm` on the global RNG (`:345-373`). The original used `RandomNodeSplit(num_train_per_class=20, key="x_future_delta")` (`AUTOPSY.md:86-91`). The split is not paired across models, because model initialisation and training consume the RNG first.
- `val_ratio=0`, so there is no early stopping: a fixed 100 epochs (`:65-67, 460`).
- The model and optimizer carry over across transitions (`:416-418`).

**Metrics.** Per transition, on the 25% test genes: MAE (`:208-215`), Pearson pooled over those genes (`:218-235`), and P@200 up/down (`:238-262`). The reported values are the `nanmean` over transitions (`:565-566`).

**Baselines in the repo:** `linear` and `mlp` (`train_gene.py:77-78`, `train_models_gene.py:656-667`). The no-change (Å·=0) and momentum baselines come only from your audit (`AUTOPSY.md:236-246`). I could not find their script among the `logs-baseline/*.py` files.

## 4. Task 3: out-degree centrality

**Target.** `nx.out_degree_centrality` of the t+1 GRN, i.e. out-degree / (Nâˆ’1), shifted one step; the last column is filled with 0.0 (`grn_centrality.py:7, 26-27, 61-93`).
- I checked this on the data: node 6416 has `y_outdeg[t001]` = 0.478321 = 3442/7196, its out-degree at ts 2. `[t011]` = 0.124653 (ts 12) and `[t012]` = 0.0.
- About 91% of targets are exactly 0: 6535/7197 pancreas and 5764/6345 gastrulation at the first column.
- PageRank is `nx.pagerank(G)` on the unweighted graph (`:72`).

**Input.** By default, random 5-dimensional features: the README command has no `--gene_features` (`README.md:41`, `dataset.py:338-344`).

**Split.** A random 75/25 split over nodes (`train_future_node_centrality.py:410-438`); the original was `RandomNodeSplit(num_train_per_class=20)` (`AUTOPSY.md:89-90`). 100 epochs, no validation (`:60-61`).

**Metrics.** Per transition on test nodes: MSE, MAE, and Spearman with average ranks for ties (`:283-320, 343-356`), plus P@10/50/200 (`:323-340`). The reported values are the `nanmean` over transitions (`:620-624`).

**Persistence (audit only).** Å· = out-degree at t, over all nodes and the training snapshots (`AUTOPSY.md:222-232`). Because of the block of zeros, Spearman here mostly rewards tying the non-regulators, which persistence does trivially.

## 5. Numbers

**Your HPC reproduction, `results.md`:**
- LP (`:38-46`):
  - Gastrulation: GCRN-GRU 0.945, ChebNet 0.945, ROLAND 0.933, EdgeBank 0.877.
  - Pancreas: GCRN-GRU 0.886, ChebNet 0.939, ROLAND 0.931, EdgeBank 0.791.
- Gene MAE / Pearson (`:53-62`): best Pearson 0.414 (GCRN-GRU, gastrulation-human) and 0.433 (MLP, pancreas-human). Best MAE is ROLAND: 0.549 and 0.666.
- Centrality MAE / Spearman (`:69-78`): the best GNNs reach Spearman â‰ˆ0.49â€“0.50.

**Audit corrections:**
- **Negatives:** degree-matched negatives give GCRN-GRU 0.504 / ChebNet 0.507 on pancreas and 0.504 / 0.500 on gastrulation, with EdgeBank 0.503 (`hpc_ns_audit_summary.md:5-39`). Source out-degree alone, with no learning, scores 0.940 / 0.944 under the paper's negatives (audit md:31,83). A learned logistic regression under degree-matched negatives gets 0.552 (`:52`).
- **Seed variance:** pancreas GCRN-GRU is 0.938 Â± 0.001 over 3 seeds (`AUTOPSY.md:337-343`).
- **Persistence on centrality:** Spearman 0.9957 (pancreas) and 0.9984 (gastrulation) for out-degree; 0.117 and 0.232 for PageRank (`AUTOPSY.md:225-230`).
- **No-change MAE:** 0.1207 (gastrulation), 0.2880 (pancreas), 0.5408 (gastrulation-human), 0.6638 (pancreas-human) (`AUTOPSY.md:239-244`). This is below every model in `results.md`. The node set it was computed on is unclear.

**Count mismatch.** Audit Â§3d/Â§4.4 used 32 gastrulation transitions; the repo evaluates 31.

## 6. What this means for scoring your OT model

- **Task 2 is the only one an expression model can be scored on directly.** It needs:
  1. Rebuilding the bins from the h5ad: 15 equal-width bins for pancreas, 35 for gastrulation. Validate by matching `n_cells_bin` per bin exactly.
  2. Predicting the per-gene mean at t+1 in the same layer (`Ms`, or `X` for human).
  3. Applying the log-delta with eps 1e-8 and the Â±5 clip.
  4. Scoring on a saved, fixed 25% gene mask per transition, then the nanmean over transitions.
  5. Always reporting the no-change baseline next to it.
- **Tasks 1 and 3 need a GRN.** They only apply if GRNBoost2 is re-run on the predicted cells.

Scratch script used (read-only on the HPC): `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\inspect_hpc.sh`
