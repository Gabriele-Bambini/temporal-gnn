1. **Inputs and the largest gene space it has run on**
- **Data object.** It takes a `SnapshotSet`: a dense cells x genes float32 matrix plus a biological time, condition and replicate for every cell (`src/cfm_grn_v2/data/real.py:105-141`). Time can be any float, so a pseudotime bin index works.
- **Grouping.** Groups with fewer than `min_cells` cells (default 200) are dropped. Groups larger than `max_cells_per_group` are subsampled (`real.py:118-126`).
- **Normalisation.** Per cell: library size to 1e4, then log1p (`real.py:18-36`).
- **Gene selection.** Done on the fit cells only: the top HVG by dispersion (`real.py:39-68`) union the CollecTRI regulators. A gene must be detected in at least 1 % of fit cells (`select_genes`, `real.py:79-96`). The benchmarks use `n_hvg=1000` and `min_detect=0.01`. The filter matters: without it, rare-gene spikes distorted the first real-data runs (`DELIVERABLE_2026-09-19_OT_flow_matching.md:70`).
- **Standardisation.** Per-gene z-score fitted on the fit cells only (`data/preprocessing.py:21-22,49`; `training/trainer.py:418-419`). `predict` maps back to log1p space with `inverse_transform` (`trainer.py:261-264`).
- **Largest gene space run:** 1428 genes (`~/cfm_grn_v2/outputs/argelaguet_v0110/summary.json`). The others were 1197 (EB), 1216 (Veres) and 1312 (Pijuan-Sala) (`summary.json` â†’ `tests.*.n_genes`). PCA-50 was run only as a control (`scripts_dev/eb_benchmark.py:100-109`).
- **Gene-count mismatch.** Dileo's graphs have about 6345 nodes (`~/tgl-grn/GRN/gastrulation_node.csv`, 6346 lines) and 7197 (pancreas). The model has never run at that size.
- **Frozen v0.1.10 config.** `freshfold_select.measured_config()` (`scripts_dev/freshfold_select.py:113-115`) is `eb_benchmark.BASE.replace(**FT).replace(path_center="linear")` (`eb_benchmark.py:39-42`). That means:
  - hidden 128, depth 3, duration clock, OT-chained spline path;
  - stage 1 of up to 8000 iterations;
  - stage 2 of 1000 marginal-matching iterations with `marginal_window=1`.

2. **Forecasting the next timepoint**
- **Yes, it can.** `FlowModel.predict(x0_raw, cond, bio_t0, bio_t1)` (`trainer.py:261-264`) calls `predict_standardized` (`trainer.py:234-259`).
  - It runs an RK4 ODE of the autonomous field over the horizon `bio_t1 - bio_t0`, which must be > 0 (`trainer.py:251-256`).
  - It is deterministic and returns one predicted cell per source cell, so the predicted population is the size of the bin-t population after caps.
- **Existing machinery.** The forecast benchmark is `run_forecast_benchmark` (`training/search.py:203-265`). It has a nested fit early-stopped on `val_task`, an optional refit, and the controls persistence, train_mean, mean_shift, linear, mlp and freefm_replay. It returns metrics only, not the predicted cells.
- **Two protocols. Only (b) counts as a forecast, and (b) is the unstable one.**
  - (a) Train once on all bins, then predict tâ†’t+1. This is in-sample: the spline passes through the chained cells of bin t+1 (`flows/path.py:63`), and stage 2 directly matches the tâ†’t+1 marginal (`trainer.py:351-361`). It is a reconstruction. Use it only as an upper bound.
  - (b) Rolling origin: train on bins â‰¤ t, predict t+1. This is the honest one, and the repo documents it as unstable:
    - The terminal forecast is "pure extrapolation" (`DELIVERABLE:76`).
    - The best flow models scored 0.40â€“0.59 against 0.56 for the linear map, but the refit at the transferred iteration count gave two divergent seeds out of five, mean 5.67 (`DELIVERABLE:148-150`).
    - A refit on the forecast task produced energy distances above 12 (`DELIVERABLE:258`).
  - **I recommend (b) for Dileo-equivalence.** Report (a) only as an upper bound.
- **Training without a validation task.** Without `val_task`, stage 1 runs all `n_iters` and stage 2 runs 1000 iterations with no guard (`trainer.py:329-336,402,503-505`). The measured `best_iter` was 1000â€“1750 (`DELIVERABLE:330`). In a refit, pass that as `n_iters`.

3. **Velocity field and gene-gene Jacobian**
- **The field exists; no Jacobian code does.** "jacobian" appears nowhere in `src/` or `scripts_dev/`.
- **Field.** `model.vector_field` is `MLPVectorField` (`models/vector_field.py:21-47`). In v0.1.10 `use_time=False` (`trainer.py:153-154`), so it is the autonomous v(x, condition). Call it as `vf(x, torch.zeros(b), model._context("ctrl", 0.0, 1.0, b))`, the same way the trainer does (`trainer.py:217-219,453`; zero time as in `flows/path.py:117`).
- **Recipe.** Use `torch.autograd.functional.jacobian`, or `torch.func.vmap(jacrev)`, with respect to x. This gives a G x G matrix per cell, where J[i,j] = dv_i/dx_j is the signed edge jâ†’i.
  - Per-bin edge scores: average over the cells of bin t (or |J|, or only the TF rows/cols).
  - Because the field is autonomous, edges vary over time only through the state, i.e. which cells you evaluate at.
- **Caveats.**
  - J is computed in standardised space. In log1p space, J_x[i,j] = (Ïƒ_i/Ïƒ_j)Â·J_z[i,j], with Ïƒ = `standardizer.std_` (`preprocessing.py:22`).
  - A dense 3-layer width-128 MLP gives a Jacobian of rank â‰¤ 128 per cell.
  - There is no prior or sparsity in the drift; that is still planned (`DELIVERABLE:272`).
  - J is a linearisation of the dynamics. Dileo's targets are GRNBoost2 co-expression graphs, so the two are different objects.
- **Alternative route.** Run GRNBoost2 on the predicted cells at t+1 and score that edge set against Dileo's t+1 graph.

4. **Runtime on the HPC**

Settings: 8 CPUs per task, 24 h time limit (`scripts_dev/hpc_jobD_*.sbatch`, `hpc_jobE_*.sbatch` lines 7 and 9). Env: `source /software/miniconda3/etc/profile.d/conda.sh; conda activate sheaf`.

| Dataset | One training, nested fit + refit, one seed (mean `train_seconds`) | Whole benchmark (`summary.json` `seconds`) |
|---|---|---|
| EB | 326 s (shuffled 503 s) | 8101 s |
| Argelaguet | 409 s (shuffled 337 s) | 8469 s |
| Veres | 630 s | 11220 s |
| Pijuan-Sala | 859 s | 19012 s |

- Whole benchmarks are 5 seeds with controls and nulls. `eb_published` took 12566 s.
- Rough estimate for a rolling forecast at about 10 min per training:
  - 12 bins: about 10 origins Ã— 5 seeds â‰ˆ 8 h.
  - 32 bins: about 30 Ã— 5 â‰ˆ 25 h. That is over the 24 h limit, and QoS allows 1 running job, 2 submitted and no arrays. Split it by origin ranges or use 3 seeds.

5. **Simplest adapter for 12 or 32 bins**
- **Data prerequisite (open).** You need the cell-level counts plus each cell's bin label.
  - `~/tgl-grn/GRN/*_node.csv` holds only per-bin per-gene summaries (`t000_s00_mean, median, std, frac_nonzero, sum`), not cells. `pancreas.csv` and `gastrulation.csv` appear to be the per-bin edge lists.
  - Cell caches exist in `~/rafm_data/processed`: `pijuan_sala2019_atlas.npz` and `veres2019_stage5.npz`. Whether these are the same cells Dileo binned is unverified.
  - Dileo's code warns about bins under 50 cells (`~/tgl-grn/dataset.py:291`).
- **Sketch** (new script, run from `cfm_grn_v2/`):
```python
sys.path[:0] = ["src", "scripts_dev"]; from freshfold_select import measured_config
obs = pd.DataFrame({"time": bin_idx.astype(float), "replicate": "all", "condition": "ctrl"})
Xn = normalize_log1p(X_counts, keep_sparse=True)
sel = select_genes(Xn[early_cells], symbols, n_hvg=1000, must_keep=regs, min_detect=0.01)  # fix ONCE for all t
s = snapshot_set_from_arrays(Xn, obs, symbols, "time", "replicate", "condition", gene_idx=sel,
                             max_cells_per_group=3000, min_cells=50)
cfg = measured_config()
for t in bins[3:-1]:                                    # rolling origin
    m, _ = train_flow(s, s.mask(times=bins[bins < t]), bins[bins < t], cfg, val_task=time_task(s, t - 1, t))
    m, _ = train_flow(s, s.mask(times=bins[bins <= t]), bins[bins <= t], cfg.replace(n_iters=max(m.best_iter, cfg.eval_every)))
    pred = m.predict(s.x[s.mask(times=[t])], "ctrl", t, t + 1)     # predicted cells at t+1, log1p space
    # optional: Jacobian edge scores at bin-t cells (item 3); controls via evaluation/controls.py
```
- This follows the same pattern as `search.py:237-245`. The validation target must not overlap the fit cells (`trainer.py:416-417`).
- **Gotchas:**
  - The default `min_cells=200` silently drops small bins (`real.py:122-123`); the freshfold scripts also apply a 500-cell gate (`freshfold_select.py:78`).
  - `chain_ot` subsamples every bin to n = min over all bins, capped at 1000 (`flows/coupling.py:51`; `trainer.py:158`). One small bin shrinks every pseudo-trajectory, so with 32 bins merge or drop small ones.
  - Couplings stay within (condition, replicate), so keep both constant, as `argelaguet_benchmark.py:61-62` does.
  - Fix the gene space once so edge sets are comparable across t.
  - Checkpoints go through `FlowModel.save/load`, and `load` refuses a checkpoint from a different code version (`trainer.py:274-299`).

6. **Known limitations relevant here**
- **Extrapolation is unstable.** See item 2 (`DELIVERABLE:76,148-150,258`). Absolute time in the context diverged outside the training range, which is why the field was made autonomous (`DELIVERABLE:254`).
- **Centroid pinned.** With `path_center="linear"`, the path centre is straight segments between the knot means (`flows/path.py:65-72,82-89`). The mean of the spline path is fixed by the snapshot means for any coupling (`DELIVERABLE:16,71`). In a rolling forecast there is no next knot, so the centre is unanchored beyond t.
- **Representation dependence.**
  - EB in gene space beats mean interpolation (âˆ’0.104); in PCA-50 it only ties (+0.002) (`DELIVERABLE:18,266`).
  - Gastrulation (Argelaguet) is negative: it loses to mean interpolation (+0.52), and the path over-disperses E8.0 (var ratio 1.54) (`DELIVERABLE:18,224,243`).
  - The trained field loses 0.09â€“0.22 of its own path on most folds (`DELIVERABLE:20,242`).
  - Results depend on the rare-gene filter (`DELIVERABLE:70`).
- **Scale.** At most about 1.4k genes have been run, against 6â€“7k nodes in Dileo's graphs. Dileo edges outside the model's genes cannot be scored, so restrict the evaluation to the model's gene set, or run at 6k genes (untested).
- **Jacobian limits.** No code yet, rank â‰¤ 128, no sparsity or prior, time variation only through the state (item 3).
- **No noise in predictions.** They are deterministic ODE pushforwards, so the spread of the predicted cells comes only from the source cells.
- **Dileo's own metric is flawed.** The paper's AUPRC of 0.94 is a negative-sampling artefact. Under degree-matched negatives the models score 0.50, and below chance on new edges (memory note `dileo-lp-protocol-audit.md`, `tgl-grn-1CCD/logs-baseline/NEGATIVE_SAMPLING_AUDIT_2026-09-17.md`). Report the OT model under the paper's negatives and also under degree-matched / same-source regulator negatives. Otherwise a scorer that only uses persistence and hub degree can match it.
