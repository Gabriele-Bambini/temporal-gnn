**Bottom line:** The OT model can be scored with Dileo's own metric code on gastrulation, but first the per-bin cell groups have to be rebuilt, because the cell-to-bin assignments were never saved. Pancreas also needs a dataset download, which needs the user's permission. Task 2 can be scored directly. Tasks 1 and 3 need GRNBoost2 to be re-run on the predicted cells, and only the non-paper negatives and companion controls make their numbers mean anything.

One correction to the brief: Dileo's protocol has no held-out "test snapshot" score. `roland_test_streaming_delta_ranking` (`train_models_gene.py:1084`) is never called, and the val/test slices each hold one snapshot. Every reported number is a plain mean over the live-update transitions inside `dataset[:train_split]`. That is the split to copy:
- gastrulation: ts 2â†’3 â€¦ 32â†’33, 31 transitions (ts 34 is never used);
- pancreas: ts 1â†’2 â€¦ 11â†’12, 11 transitions.

Dump the exact (t, t+1) list from Dileo's loader rather than re-deriving it.

## 1. Feasibility

| Task / dataset | Verdict | Reason and what must be rebuilt |
|---|---|---|
| Task 2 (MAE, Pearson, P@200), gastrulation | **PARTLY, YES if the bin rebuild passes its check** | The cells are on the HPC (`/home/3393519/tack/data/erythroid/erythroid_lineage.h5ad`, 9815 cells, the same total as Dileo's file). Rebuild: scVelo preprocessing and the `Ms` layer; a pseudotime scaled to [0,1] (method and root are recorded nowhere, so search DPT, velocity pseudotime and latent time over root candidates); 35 equal-width bins. |
| Task 2, pancreas | **PARTLY, blocked** | scVelo pancreas (3696 cells) is not found locally or on the HPC. Downloading `endocrinogenesis_day15.h5ad` (about 50 MB, from memory) onto the HPC needs the user's permission. After that the steps are the same, with 15 bins. |
| Task 1 (link-prediction AUPRC) | **PARTLY** | Needs the rebuilt bins plus GRNBoost2 on the predicted cells. The TF list can be recovered as the union of edge sources (about 662 pancreas, about 550 gastrulation). The GRNBoost2 settings and seed are unknown, so a ceiling control is mandatory. Under the paper's "active" negatives the result is uninformative (section 3). |
| Task 3 (out-degree MAE, Spearman) | **PARTLY** | Same GRNBoost2 dependency. Prediction = number of non-zero-importance edges / (Nâˆ’1). The metric is saturated: about 91% of targets are 0 and persistence already scores Spearman 0.996. |
| Human variants | **NO for now** | Gene IDs are mixed (numeric and symbols), orthology is unknown, the layer is `X`, and the 32-bin gastrulation-human geometry matches no known source. Pancreas-human is probably the same 3696 cells (its bin 0 also has 111 cells), but that is unverified. Defer. |

**Gate before any scoring (fail means stop).** All three checks must pass:
- the rebuilt bins reproduce the 35 / 15 `n_cells_bin` vectors exactly;
- the per-gene `s00_mean` values match `*_node.csv` to within numerical tolerance;
- `bin_mid` equals the mean rebuilt pseudotime per bin.

The gene filter is a clue in itself: find which scVelo filter turns 53,801 genes into 6345.

## 2. Most faithful protocol: Dileo's targets and metric code unchanged, only the predictor swapped

**Model training (rolling origin; the honest forecast):**
- For each origin t: a nested fit on bins < t, validated on tâˆ’1â†’t; then a refit on bins â‰¤ t at the best iteration found; then predict t+1 with `FlowModel.predict(cells_t, "ctrl", t, t+1)`.
- The train-on-everything version is not usable for Task 2. The path's centre is pinned to the snapshot means, so the t+1 mean is matched by construction. Report it only as a leaky upper bound, or drop it.
- **Data space:** train directly on the rebuilt `Ms` matrix, using log1p(Ms) and skipping `normalize_log1p`. Then predicted cells are in Dileo's space. Clip predicted Ms at â‰¥0. A negative predicted mean becomes log(1e-8) and hits the âˆ’5 clip.
- **Gene set:** primary run on Dileo's full node set (6345 / 7197). This has never been tested at that size, so pilot one origin first and add a floor on the per-gene standard deviation. Fallback: the model's usual ~1.4k genes, fixed once, with every predictor and companion restricted to the same genes and labelled "subset".
- **Small bins:**
  - train only on bins with â‰¥50 cells (Dileo's own warning threshold);
  - set `min_cells=50` and remove the 500-cell freshfold gate;
  - patch `chain_ot` so the smallest bin does not shrink every chain, or keep bins under ~100 cells out of the training points only. The field is autonomous, so any source bin's cells can still be pushed forward.
- **Transitions the model can score:**
  - gastrulation ts 4â†’5 â€¦ 32â†’33, 29 of 31 (ts 2 and 3 have no usable history);
  - pancreas all 11 (bin 0, 111 cells, has no GRN but can be used for training).

**(i) Task 2 target.**
- Predicted Î¼Ì‚â‚œâ‚Šâ‚ = mean over predicted cells of Ms (computed in Ms space, not log space).
- Å· = clip(log(Î¼Ì‚â‚œâ‚Šâ‚+1e-8) âˆ’ log(meanâ‚œ+1e-8), Â±5), with meanâ‚œ and the true target read from Dileo's `*_node.csv` unchanged.
- Generate the 25% test-gene mask once per transition with a fixed seed, save it, and score every predictor on it with Dileo's `_mae_np`, `_pearson_corr_np` and `_precision_at_k_top_up_down`. Average with `nanmean`.
- `results.md` numbers are not comparable, because their masks were not paired. Re-run Dileo's GNN, linear and MLP models with the saved mask.
- Label the asymmetry: Dileo's models are trained on 75% of the genes' t+1 changes in the same transition. That is filling in missing genes, not forecasting. The OT model sees nothing from t+1.

**(ii) Task 1 edge scores.**
- Score for (u,v) = GRNBoost2 importance on the predicted t+1 cells, 0 if the edge is absent.
- Use the recovered TF list, arboreto defaults and no pruning, matching Dileo's ~0.45â€“1M edges per snapshot.
- Edge count depends on how many cells GRNBoost2 sees, so resample the predicted cells to nâ‚œâ‚Šâ‚. Apply the same resampling to every GRNBoost2-based control, and also report the native-size run.
- Positives = all edges of Dileo's t+1 graph. Score with `average_precision_score`, take the plain mean over transitions.
- Secondary only: Jacobian edge scores, the mean over bin-t cells of |dváµ¥/dxáµ¤|, TF columns only, rescaled by Ïƒáµ¥/Ïƒáµ¤. This has no code yet, rank â‰¤128, and is a different object from a GRNBoost2 graph. Do not call it "the same metric".

**(iii) Task 3.**
- Predicted out-degree = number of non-zero GRNBoost2 edges from u on the predicted cells / (Nâˆ’1).
- Target = Dileo's `y_outdeg` file unchanged.
- MSE, MAE, Spearman with average ranks, and P@10/50/200 on a saved 25% node mask; `nanmean` over transitions.

**Common rule:** every predictor and companion is scored on exactly the same transitions, genes or nodes, and negative pairs. Report per-transition values as well as means, because `nanmean` silently drops missing transitions.

## 3. Mandatory companions (same code, same masks, same pairs)

- **Task 2:**
  - no change (Å·=0), which beats every model in `results.md`;
  - momentum (last step's change);
  - Dileo's linear and MLP models on the saved masks;
  - OT model in-sample (leaky upper bound, labelled).
- **Task 1:**
  - **Negatives:** generate each type once per transition with a fixed seed, save it, and share it across all predictors. The types are:
    - the paper's `active` negatives (report them, but flag that any TF-aware scorer gets about 0.94 because about 92% of these negatives have non-TF sources);
    - `degree` (the audit's degree-matched negatives);
    - `tfuniform`;
    - `restrict_neg_source`;
    - a new-edge subset: t+1 edges absent at t, against matched negatives.
  - **Baselines:**
    - EdgeBank;
    - persistence (importance at t);
    - source out-degree at t;
    - **GRNBoost2 on real bin-t cells**, which is persistence through the same pipeline;
    - **GRNBoost2 on real t+1 cells**, which is the ceiling and measures how reproducible Dileo's own target is;
    - a GRNBoost2 rerun-noise floor on about 5 bins;
    - the audit references (GNNs about 0.50 and logistic regression 0.552 under degree-matched negatives).
  - The real forecasting signal is the share of the gap between persistence-through-pipeline and the ceiling that the OT model closes.
- **Task 3:**
  - persistence (out-degree at t);
  - GRNBoost2 on real bin-t cells and on real t+1 cells (ceiling);
  - an extra, labelled TF-only Spearman, because the score over all nodes mostly rewards ranking the block of zeros.
- **Seeds:** 3, or 5 if the budget allows. Report the number of diverged seeds and per-transition medians.

## 4. Compute and risks (Bocconi: 1 running and 2 submitted jobs, no arrays, 8 CPUs, A100 MIG, 24 h limit)

Estimates to confirm with a pilot job: one origin at the full gene set, plus one GRNBoost2 run on about 800 cells Ã— 6345 genes Ã— 550 TFs.

| Step | Estimate |
|---|---|
| Bin rebuild, gastrulation (CPU, pseudotime grid search) | 2â€“6 h, one job. Same for pancreas after the download. |
| OT rolling run, gastrulation: 29 origins Ã— 3 seeds Ã— (nested fit + refit) | ~15â€“30 min per origin per seed (859 s measured at 1.3k genes; 6k genes untested) â†’ about 25â€“45 h, split by origin range into 2 chained jobs |
| OT rolling run, pancreas: 11 Ã— 3 | about 8â€“12 h, one job |
| GRNBoost2 | About 90 runs: gastrulation 32 real + 29 predicted, pancreas 12 + 11, plus noise reruns. At ~10â€“40 min per bin on 8 CPUs, 15â€“60 h. |
| Jacobian (optional) | Forward-mode products over ~550 TF columns, minutes per bin on the GPU |

**Wall clock:** about 4â€“7 days of chained jobs. A first Task 2 gastrulation result is about 2 days after the gate passes.

**Main risks:**
1. Pseudotime cannot be reproduced exactly (method, root and scVelo version are all unknown), and the gate fails.
2. Rolling-origin extrapolation is unstable: two of five seeds diverged before, and the path centre has no anchor beyond t.
3. The model has never run at 6â€“7k genes.
4. GRNBoost2 is stochastic with unknown settings, so the ceiling may be low.
5. Out-degree depends on the cell count fed to GRNBoost2, and resampling to nâ‚œâ‚Šâ‚ uses target-side information (applied equally to all GRNBoost2-based predictors).
6. `Ms` smoothing averages each cell with neighbours from other bins, including t+1. This affects every predictor; an optional check is to recompute `Ms` from bins â‰¤ t only.
7. Binning on pseudotime is partly circular, so persistence is strong by construction.
8. Pancreas cannot start without the user approving the download.
9. Task 3 cannot tell models apart.

## 5. If the bins cannot be rebuilt exactly

- **Closest legitimate alternative:** use the same raw cells with a pseudotime I choose and document, and the same 35 / 15 equal-width bins. Regenerate the GRNBoost2 graphs per bin (TF list from Dileo's sources, the same settings). Rebuild the three targets with Dileo's own target code (the log change in `train_models_gene.py`, `grn_centrality.py`). Then re-run Dileo's GNNs, all companions and the OT model with identical metric code. Label it "Dileo's metrics on a rebuilt benchmark", never "Dileo's numbers".
- **Weaker fallback:** use the real-time stage labels instead of pseudotime bins (7 stages, E7.0â€“E8.5, in the erythroid file) with the same metric code. This fits better with the cfm_grn_v2 pre-registration.
- **Not acceptable:** approximate bins scored against Dileo's targets, unless the per-gene means match almost exactly.

Code referenced: `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tgl-grn-1CCD\train_models_gene.py` (lines 143-152, 303-343, 435, 1084-1128). The scout scripts are in `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\`.
