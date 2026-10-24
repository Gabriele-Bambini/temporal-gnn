# Plan: run corrected Task-2 protocols for Dileo & Sottoriva's models and ours, without editing their files

A copy of this plan is saved at C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\plan_dileo_protocols.md

**Files that must not change**
- **Their code:** `~/tgl-grn-orig` on the HPC stays untouched.
- **Frozen by the 2026-10-22 pre-registration** (sha256 in docs/superpowers/decisions.md): `dist_bridge.py`, `sinkhorn.py`, `dileo_harness.py`, `task2_forecasters.py`, `tests/test_dist_bridge.py`, `dileo_harness.sbatch`.
- **Kept only for cross-checks:** `dileo_corrected.py` and `dileo_protocol.py` are superseded but stay as they are. Their target names clash with the new ones:
  - their `log1p` is log1p of the raw per-cell mean, with the ±5 clip (not target L below);
  - their `linear` is P0.

## 1. Runner: `tack/hpc/dileo_protocols.py`

This is a new file with three subcommands.

```
python dileo_protocols.py run   --orig ~/tgl-grn-orig --work ~/tack/work/dileo_runs --out ~/tack/out/dileo_protocols \
    --names pancreas-human,gastrulation-human --models linear,mlp,gcn,gat,chebnet,evolvegcn,gcrngru,roland,tgnnmem \
    --seeds 1234,42,7,0,1 --protocol {orig,preqA,preqB}   (strict = alias of preqB) \
    --target {orig,p0,p1,l1,l01,log1p_raw} --inputs {zscore,log1p,zlog1p} --masks {theirs,shared} \
    --variant {none,roland_fix,gcrn_carry,evolve_pin,tgnn_memfix,readme_hp} --device {auto,cpu,cuda} \
    [--max-transitions K] [--deterministic 1]
python dileo_protocols.py run   --models distbridge_strict,obslevel4,obslevel6,distbridge_l1,... --target ...   # ours, once per (dataset, target)
python dileo_protocols.py gate  --stage {g1..g6}
python dileo_protocols.py score --out ~/tack/out/dileo_protocols    # every metric correction, offline, numpy only
```

**Setup** (copied from `dileo_corrected.py:main`):
- `sys.path.insert(0, orig)`.
- Work directory with a symlink `GRN -> ~/tgl-grn-orig/GRN`, then `chdir` there, so their checkpoints and logs land in the work directory.
- Import `dataset`, `models_gene as mg`, `train_models_gene as tm`.
- One dataset per process; `del` it before loading the next.

**Patches.** All replace module globals that their code looks up at call time. They are installed per run and undone after it.

| patch | what it does | why it is safe |
|---|---|---|
| `tm.load_dataset` → cache | First call runs `dataset.load_dataset('grn', name, True)`; later calls return the cached list. | No RNG with gene_features=True. The builder deep-copies `snapshots[i]` and `.to()` only touches copies. Assert at the end of each run that the sha of every cached `x` / `edge_index` is unchanged. |
| `tm.build_snapshot_split_future_node_delta_ranking` → wrapper | Calls the original first, which keeps their randperm and the global RNG stream. Then three optional steps, listed below this table. | All callers pass keywords (:410, :860, :977). In PyG 2.7.0, train_rest only calls `torch.randperm(num_nodes)`. |
| `tm._normalize_node_features_per_snapshot` | `zscore` = theirs. `log1p` = `lambda x, eps=1e-8, normalize=True: torch.log1p(torch.clamp(x, min=0))`. `zlog1p` = their z-score applied to log1p(x). | Called by its global name at :330 and :1278. The target is built from the raw snapshots (:332). |
| determinism | `torch.use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initialises. | Expected to be a no-op on CPU; G1 checks it. |

The build wrapper's three steps:
- **(a) target ≠ orig:** recompute y from the raw `snapshots[i].x[:,0]` and `future.x[:,0]`, with no clip. `future` is `snapshots[i+1]`, or `dataset[train_split]` for the last i (as at :308-313). Set `.x_future_delta` on all four returned objects, because ROLAND and TGNN-mem train on the `train_data` copy (:749-750).
- **(b) masks = shared:** overwrite `train_mask`, `val_mask` and `test_mask` on all four objects. Use `g = torch.Generator().manual_seed(1_000_003*seed + i)`, `perm = torch.randperm(N, generator=g)`, test = `perm[:round(0.25*N)]` (5358 / 4640), train = the rest, val all False.
- **(c) always:** record the test indices, m_t, m_t+1 and n_cells for t and t+1.

**Model loops**

`--protocol orig`
- Seed exactly as train_gene.py:50-56, then call their wrapper unchanged: `train_linear`, `gnca`, `evolvegcn`, `gcrngru`, `roland`, `tgnn_mem` `_future_delta_ranking`. This is exactly what train_gene.py does.

`--protocol preqA|preqB`, live models (linear, mlp, gcn, gat, chebnet, evolvegcn, gcrngru)
- Copy of `train_live_model_future_delta_ranking`, keeping their RNG order: `mg.load_model_node` → `tm.load_dataset` → `tm.train_val_test_split` → `Adam(lr=tm.lrs['grn'])` → `reset_parameters()` → `.to(dev)`.
- For each i:
  1. Build the snapshot and move it to the device.
  2. **PRE:** `model.eval()` under `no_grad`; forward on `(snap.x, snap.edge_index, edge_attr, **tm._build_fm_forward_kwargs(snap))`. Score with `tm._delta_metrics(y[test], tm._reshape_prediction_like_target(pred[test], y[test]))` and keep the prediction for all N genes.
  3. Their training block :434-453 verbatim (MSE, 100 epochs). The loss uses `train_mask` for A and all genes for B.
  4. **POST:** their eval block :485-500 verbatim.
- Eval forwards draw no random numbers (the only randomness is nn.Dropout, off in eval mode), so the masks equal the orig run. That makes preqA's POST the original-protocol score, paired and at no extra cost.
- preqB's POST has been fitted on the test genes, so it is never reported.
- EvolveGCN: the extra PRE forward evolves its weight once more, so its POST differs from an untouched run. Always run `--protocol orig` for it as well.

`--protocol preqA|preqB`, ROLAND / TGNN-mem
- Copy of their drivers (:836-934 and :953-1051), keeping their order: `load_model_node` → `load_dataset` → `.to(dev)` → `Adam` → `reset_parameters`. On GPU the reset draws from the CUDA generator.
- Loop over `range(len(snapshots))`. This fixes their off-by-one. Transitions 0..T-2 are identical to theirs, so one run gives both the 28/13 and the 29/14 means.
- For each i:
  1. `pre = tm.roland_test_live_delta_ranking(model, train_data=tr, eval_data=te, full_data=snap, isnap=i, last_embeddings=last_emb, device=dev)`.
  2. Then `model, opt, post, last_emb = tm.roland_train_single_snapshot_delta_ranking(...)`. Its 4th return value is carried to i+1.
- preqB: set `tr.train_mask[:] = True` on the `train_data` copy only.
- Device: `NodeROLAND.device` is fixed at construction from CUDA availability (models_gene.py:251) and used at :290. Set `model.device = dev` after construction; otherwise a forced-CPU run on a GPU node crashes.
- TGNN-mem: its forward loops in Python over all N genes (:468-469). Run it on CPU.

Ours (registries from `task2_forecasters`, `dist_bridge` and the new `dist_bridge_targets`)
- Call `fc.predict(hist[:i+1], ts[:i+1], {})` over all N genes and save once per (dataset, target).
- Strict forecasters never read `ctx`, and their RNG is keyed by (dataset, transition, config). Their predictions therefore do not depend on masks or seeds, so scoring on any mask set is done offline. Check once that a second seed gives identical predictions.

**Model variants.** These change the model, are always labelled "MODEL CHANGE", and run in prequential order.
- **`roland_fix`:** subclass `NodeROLAND` so the update is `h = self.updates[z](h, self.previous_embeddings[z].detach())`, with no numpy round trip. Use a runner copy of `roland_train_single_snapshot_delta_ranking` with MSE, a fixed 100 epochs, no early stopping, and the final epoch's embeddings returned detached.
- **`gcrn_carry`:** a forward hook on `model.evolve` stores H (detached) from the last training epoch of transition i-1. A `register_forward_pre_hook(with_kwargs=True)` injects `kwargs['H'] = H_prev` (None at i=0) into every forward of transition i, PRE included.
- **`evolve_pin`:** a pre-hook sets `module.weight = W_prev` (detached, saved after the last epoch of i-1) before every forward. Write it only after dumping `inspect.getsource(EvolveGCNO)` on the HPC, because attribute names differ between PyG-Temporal versions.
- **`tgnn_memfix`:** subclass with `ChebConv(K=2)`. After the PRE score of transition i, call `update_memory(train_idx, y[train_idx], state_id)` once (all genes under preqB). Never call it inside the epoch loop: that would feed transition i's labels back in as inputs.
- **`readme_hp`:** `tm.epochs['grn'] = 1000`, and replace `torch.optim.Adam` with `functools.partial(Adam, weight_decay=5e-3)` for the call.

**Outputs**
- Per (dataset, target): `targets.npz` with y, m_t, m_t+1 and n_cells for every i.
- Per (model, seed, configuration):
  - `seed<s>.json`: their per-transition dicts plus `pre`, `post` and `test_idx_hash`.
  - `seed<s>.npz` (compressed): test indices and the PRE and POST predictions over all N.
- Per run, `env.json`:
  - pip freeze and the versions of torch, PyG, PyG-Temporal and numpy;
  - GPU name and MIG profile, and the determinism flags;
  - sha256 of their 5 files and of ours;
  - wall time per model.
- Layout: `<out>/<name>/<protocol>_<target>_<inputs>_<masks>[_<variant>]/<model>/`.

**Validation gate.** Stop at the first failure.

| id | check | pass condition |
|---|---|---|
| G1 | Linear, seed 1234, `--device cpu`, `--protocol orig --target orig --inputs zscore --masks theirs`. In the same job, run untouched `python ~/tgl-grn-orig/train_gene.py --dataset grn --name X --model linear --gene_features --seed 1234` from the work directory. | The per-transition dicts are identical to their logs-gene JSON (max abs diff 0.0, all 4 keys, all i). The means match the CPU anchors of job 712735 to 4 dp: pancreas-human 0.1068 / 0.0618 / 0.0561, gastrulation-human 0.0619 / 0.0414 / 0.0069. On GPU only PCC, MAE and P@200 down must match; P@200 up for pancreas is 0.0579 there (job 713005, a tie-order effect). |
| G2 | preqA, linear and mlp, seed 1234, CPU | POST equals the orig run for every transition (diff 0.0) and the test-mask hashes are identical. PRE for i≥1 matches the local mirror within 0.005 (allowing for torch version): linear PCC 0.0505 gastr. / 0.0597 panc., MLP 0.216 / 0.266. For the GNNs on GPU: mask hashes identical, metrics within 2e-3 (scatter is nondeterministic). |
| G3 | ROLAND and TGNN-mem, seed 1234, CPU: runner orig path vs their driver | Transitions 0..T-2 identical; preqA POST equals orig for i ≤ T-2. |
| G4 | target and mask wrapper | With `--target orig`, x_future_delta is bitwise equal to the unwrapped builder. With `l1`, masks equal the orig run and y matches a numpy recomputation from the node CSV (max diff < 1e-5). With shared masks, every model gets the same indices at a given (seed, i). |
| G5 | ours | Saved predictions scored on replayed harness masks (seed, then builder, no model init) reproduce jobs 712958, 712959 and 712742 exactly, seed by seed; for example distbridge_strict on pancreas 0.488 / 0.301 / 0.442. |
| G6 | scorer | Their keys recomputed offline equal the online JSON. P_ta of an exact copy of y equals P_max. A random predictor gives P_ta ≈ k/n_test. The noisy oracle at gastrulation 2→3 gives about 200/n_tied. |

Optional cross-check: existing `dileo_corrected.py --protocol strict` outputs must equal preqB PRE for every model except EvolveGCN.

## 2. Corrected definitions for each confirmed error, and what combines

T = 29 / 14 transitions; the common prequential set is i = 1..T-1 (28 / 13). Every metric correction is computed by `score` from the saved arrays, for every run, at no compute cost.

| finding | original | corrected definition | switch |
|---|---|---|---|
| same-transition-leak | Fit 100 epochs on 75% of transition i's labels, then score the other 25%. | **preqB (primary strict):** score transition i with weights fitted only on transitions < i (all genes of the past transitions), then refine on all genes of i. **preqA (minimal change):** same, but refine on train_mask only. Report the i≥1 mean, and transition 0 (untrained model) on its own. | `--protocol` |
| target-paper-vs-code | clip(log(m1+1e-8) − log(m0+1e-8), ±5) on raw s00_mean | **P0:** m1 − m0 (raw per-cell UMI mean; the paper's literal formula). **P1:** c1 − c0 with c_t = 1e4·m_t / Σ_g m_t (CP10k). No clip. | `--target p0\|p1` |
| target-eps-clip-atoms | three-atom target driven by bin size | **L1 (primary corrected target):** log(c1+1) − log(c0+1), no clip. **L0.1:** α = 0.1 as sensitivity. Extra rows drop transitions with min n_cells < 50 (gastr. 2→3, 29→30, 30→31) and ≤ 50 (adds panc. 3→4 and 4→5). n_cells = median(s04/s00) over expressed genes. | `--target l1\|l01`; scorer |
| pearson-on-atoms | pooled PCC over test genes | Add: Spearman (average ranks); PCC and Spearman on test genes with m_t > 0; AUROC of the on-event (m_t+1 > 0) among test genes with m_t = 0. Average over transitions both arithmetically and via Fisher-z. Compute under every target. | scorer |
| p200-ties | `set(np.argsort(...)[:200])` with the default sort | **P_ta (primary):** let τ be the 200th largest y. Truth credit c_j = 1 if y_j > τ, (k − n_above)/n_tied if y_j = τ, 0 otherwise; inclusion s_j is defined the same way on y_pred; P_ta = Σ s_j·c_j / k, and "down" negates both arrays. Also report P_max = (n_above + (k − n_above)²/n_tied)/k and P_ta/P_max. Secondary P_mc ("coupled tie-breaking", 100 joint permutations seeded by (seed, i)); optional tie-inclusive version with its own chance level. Their keys are kept. | scorer |
| plusminus-not-seed-sd | ± is the SD over transitions, labelled as SD over 5 seeds | Mean over the 5 listed seeds of the per-seed transition means, ± SD over seeds (ddof 1). SD over transitions goes in its own column. First check job 712735's JSON with np.std over transitions against the paper's ±. | scorer |
| table2-build-mislabel | caption says "mouse"; Table 1 lists mouse-build sizes | Label rows gastrulation-human / pancreas-human, with N 21430 / 18561, 30 / 15 snapshots, 29 / 14 transitions, n_test 5358 / 4640, layer X (raw counts). | report |
| mouse-ms-smoothing | Ms (kNN-smoothed) statistics | Score Task 2 only on the X builds. Mouse builds appear only in a side block marked "contaminated". | `--names` |
| no-chance-baselines | none reported | On every mask set: chance k/n_test (0.0373 / 0.0431, and the matching chance level for each tie variant), y_hat = 0 (MAE only), and the indicator 1[m_t == 0]. P@200 also reported as lift over chance. | scorer |
| roland-tgnn-fewer-transitions | 28 / 13 transitions vs 29 / 14 for every other model | Loop over all transitions; compare all models on the same transition set. | always |
| roland-recipe-frozen-encoder | encoder frozen after transition 0, GRU never trained, SmoothL1, early stop on train loss, best_model kept by reference | The faithful column stays as is, with the recipe disclosed; corrected model is `roland_fix`. | `--variant` |
| recurrent-stateless | GConvGRU called without H | Label the row "no hidden-state carry-over"; corrected model is `gcrn_carry`. | `--variant` |
| evolvegcn-version | weight evolves on every forward; behaviour depends on the library version | Probe first (does it crash? what is the source?). The as-is row is labelled "weight evolved per optimizer step"; corrected model is `evolve_pin`. | `--variant` |
| fm-rows-unverified | the loader never sets the FM flags | `tm.load_dataset = lambda ds, n, gf: dataset.load_grn(n, gf, use_scgpt=m=='scgpt', use_scfoundation=m=='scfoundation')['graphs']`. Mark rows "provenance unverified", with coverage 77% / 87%. | `--models scgpt,scfoundation` |
| readme-hparams | README says 1000 epochs and weight decay 5e-3; code runs 100 and 0 | Document the code configuration as the reproduction; `readme_hp` as sensitivity. | `--variant` |
| tgnnmem-dead-memory | memory never written; K=1 ignores edges | Label the row "graph-free, memory-free"; `tgnn_memfix` variant, prequential order only. | `--variant` |
| nanmean-nan-drop | NaN PCC silently dropped | Report n_valid per metric. The headline PCC scores a NaN from a constant prediction as 0, with the nanmean beside it. Assert y is never constant within a transition. | scorer |
| nondeterminism-env | no determinism settings, no environment lock | Determinism flags, env.json, and the multi-seed SD. | runner |

**Design choices, not errors.** These are kept as extra axes:
- `--inputs` (zscore-inputs);
- `--masks shared` (shared-masks);
- Fisher-z mean as an extra average (transition-weighting);
- temporal hold-out on the last 25% of transitions (gastr. i ≥ 22, panc. i ≥ 10), as a scorer subset;
- a fixed gene split across transitions, optional.

**Combinability.** Protocol, target, inputs and masks are independent axes, and every combination is valid. All scorer corrections apply to every run. Caveats:
- preqB's POST is never reported.
- Under `--masks shared` the training genes change too, so preqA's POST is no longer the published score.
- EvolveGCN needs its own orig run.
- MAE can only be compared within one target, because the scales differ.
- The indicator baseline is meaningful only under target O.
- P_ta matters mostly under O; under L1 the ties at the cut drop to ≤ 18 / 36 genes.
- `gcrn_carry` and `tgnn_memfix` are leak-free only in prequential order.
- `readme_hp` costs 10 times the compute.

**"All corrections combined" row:** preqB + L1 + their better input variant + all transitions, i ≥ 1 + P_ta + NaN → 0 + SD over seeds.

## 3. What changes in our forecaster for each correction

| correction | change to ours |
|---|---|
| leak, masks, ROLAND transition set, NaN, chance rows, seed SD, n_cells filter, hold-out | None; ours is already strict. Rescore the saved predictions on each model's saved masks (paired), on shared masks and on harness masks. |
| P_ta, Spearman, m_t > 0 subset, AUROC on-event | None. Under target O, the prediction for genes with m_t = 0 is 5·(1 − P0), which rises with the probability of switching on. It is therefore already the AUROC score and the ranking that maximises P_ta. DistBridge already selects its configuration on tie-aware expected P@200. |
| inputs | None; ours reads the raw history. |
| P0 | Already exists: `H[:,0] − m_t` (`distbridge_strict_linear`). |
| P1 | New: `1e4·H[:,0]/Σ_g H[:,0] − c0`, plugging in the forecast library size (error of order 1/N). |
| L1 / L0.1 | New: `expected_value(H, m_t, n_t, rng, tfn)` with the closure `tfn(m0, s) = log(1e4·s/L̂ + α) − log(1e4·m0/L_t + α)`, where L̂ = Σ_g H[:,0] and L_t = Σ_g m_t. The selection target `_y` is recomputed on past bins, each with its own library size. |

**Implementation**
- New `tack/hpc/dileo_targets.py`: torch and numpy formulas for orig, p0, p1, l1, l01, plus `log1p_raw`, the old `dileo_protocol.log1p` kept for continuity.
- New `tack/hpc/dist_bridge_targets.py`: `DistBridgeT(DistBridge)`.
  - `_forecast` re-implements the H-building part verbatim and replaces only the observation step.
  - Registry entries `distbridge_p1`, `distbridge_l1`, `distbridge_l01` and their `_nosb` versions, on the existing corrected-target grid: CORR_POOLS, CORR_DELTAS, lags (1, 3), Anscombe and count scales.
- Tests:
  - `DistBridgeT(target='orig')` is bitwise equal to `DistBridge`;
  - the L1 expected value matches a cell-by-cell simulation;
  - no future information leaks.
- These configurations come after the pre-registration. Before the HPC run, freeze an amendment in decisions.md (registry names plus sha256 of the new files) and label the results exploratory.

## 4. Priority order and compute

**Inputs to the estimate.** Nothing in the repo records time per epoch, so the per-model figures below are order-of-magnitude guesses that P0 replaces. Known quantities:
- 0.39 to 1.02 M edges per snapshot (scratchpad sk2/edges_scan.json);
- 100 epochs × 29 + 14 transitions = 4300 steps per (model, seed) across both datasets;
- linear took about 1 min per dataset on CPU (job 712735).

**HPC limits.** One running and two submitted jobs, no arrays. Keep `--mem=120G` because the edge CSVs are several GB, and load one dataset per process.

| order | block | estimate |
|---|---|---|
| P0 | **Probe and gate job.** G1-G6 on CPU (linear, mlp, roland, tgnnmem), and every model with `--max-transitions 2`. Record time per step, peak GPU and RAM, load time, PyG-Temporal version, `getsource(EvolveGCNO)`, and whether unmodified EvolveGCN raises "backward through the graph a second time". | 1-2 h |
| P1 | **Leak correction.** Target O, zscore inputs, their masks; preqA (gives orig and A) and preqB; 9 models × 5 seeds × 2 datasets; EvolveGCN also under orig. Ours and the baselines are scored offline. | Per (model, seed): linear/mlp 1-2 min CPU; gcn/evolvegcn 0.5-1 min; chebnet/roland 1-2 min; gat 2-4 min; gcrngru 2-6 min (GPU); tgnnmem 7-20 min CPU. That is 1.5-3.5 h per configuration and about 3-7 h for P1, one job. |
| P2 | log1p and zlog1p inputs under preqB, for fairness to their models (ours is unaffected) | 3-7 h |
| P3 | Target L1: ours retargeted (CPU job, once per dataset); theirs under preqA and preqB × {zscore, log1p} | 6-14 h over 2 jobs, plus 1-3 h CPU for ours |
| P4 | P0, P1 and L0.1 × {preqA, preqB} on linear, mlp, chebnet, gcrngru, gat; 3 seeds | 4-8 h |
| P5 | `roland_fix`, `gcrn_carry`, `evolve_pin`, `tgnn_memfix` under preqB, targets O and L1, 3 seeds | 2-4 h |
| P6 | Optional: shared masks (for their close calls), scGPT/scFoundation, `readme_hp` (1 seed), mouse side block | 5-15 h |

**Totals.** P0-P5 is about 20-45 h of job time, or 3-5 days with one job running at a time. Saved arrays are about 5 MB per (model, seed, configuration), roughly 1 GB for P1; check the home quota.

**Before P0, locally:** unit tests for the scorer, the target wrapper and the mask overwrite. G2 has already been reproduced locally for linear and mlp on the stubbed mirror (scratchpad sk2/mirror.py, preq.py).

## 5. Reporting next to the original protocol (user rule)

**Decision rule.** Fix it in decisions.md before the P1 results arrive.
- Decide per correction as a whole, never cell by cell.
- **Cells:** 6 = 2 datasets × PCC / P@200 up / P@200 down, using P_ta when the tie correction is on.
- **Margin** = ours − the best of their rows under the same protocol and target:
  - paired per transition on that model's masks, averaged over seeds;
  - 95% bootstrap confidence interval over transitions, 10,000 resamples, with a moving-block bootstrap (length 3) as sensitivity.
- **"Hurts"** means a cell that ours won under the original (CI lower bound > 0) is lost.
- **Outcome per correction:**
  - **KEEP:** the correction helps or is neutral.
  - **FIX-OURS:** it hurts, so retarget ours (section 3) and rerun.
  - **KEEP-THEIRS:** it still hurts after the fix, so the headline keeps their option and the corrected numbers are printed beside it.

**Main table, one per dataset**
- **Rows:** their 9 models, ours (distbridge_strict, obslevel), and the baselines (chance, y = 0, indicator).
- **Column blocks:**
  - (a) The paper value, with "paper ±, definition unclear".
  - (b) Our reproduction under the original protocol on their masks. ROLAND and TGNN-mem appear on both 28/13 and 29/14 transitions.
  - (c) One correction at a time, each beside (b): preqB, preqA, L1, P0, P1, log1p inputs (under preqB), P_ta, the common transition set, NaN → 0.
  - (d) All corrections combined.
- **Each cell:** mean over seeds ± SD over seeds, with SD over transitions in a separate column, plus n_valid.

**For each correction:** one line with the margin and its CI, the rule outcome, and the reason in plain words.

**Appendix**
- Per-transition table for each model: every metric, n_cells at t and t+1, test genes at +5 and −5, n_tied at the cut, P_max, and the largest group of tied predictions.
- Transition 0 on its own.
- Sensitivity rows: min n_cells < 50 and ≤ 50, temporal hold-out, Fisher-z mean.
- Model variants labelled "MODEL CHANGE"; FM rows marked "provenance unverified"; the mouse block marked "contaminated".
- A summary of env.json and the list of seeds.
- Rows are always labelled with the builds actually scored: gastrulation-human and pancreas-human (X layer, raw counts).