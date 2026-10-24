# DistBridge: final specification for Dileo & Sottoriva Task 2 (gene-expression change forecasting)

## 1. Goal and tiers

**Goal.** Forecast their target on their two '-human' builds:

y_g(t->t+1) = clip(log(m_g,t+1 + 1e-8) - log(m_g,t + 1e-8), -5, 5)

- gastrulation-human: snapshots ts 2..31, 29 transitions, N = 21430.
- pancreas-human: snapshots ts 0..14, 14 transitions, N = 18561.

The aim is to beat the paper's best per-metric means (Table 2) on PCC, P@200 up and P@200 down, on both datasets, over 5 seeds.

Their original code computes everything except the prediction: dataset, targets, the 25% test genes (RandomNodeSplit) and the metrics (_delta_metrics, nanmean over transitions). We supply only the length-N prediction vector.

**Model.** DistBridge takes the skeleton of Design 1 (ZINB-Bridge) and adds the grafts the judges agreed on.
- Core: a noise-aware filter over per-gene cell-count laws; a per-gene entropic-OT / Schroedinger-bridge (SB) law extrapolation computed by our own Sinkhorn; a finite-cell observation model; rolling-origin selection.
- Grafts: long-memory prior (D3); strict ridge head (D3) and kNN residual head (D2); same-information ridge head with 5-fold cross-fitting (D3); own-bin rule at i=0 (D2); nested rolling origin (D2); paired reruns of their models and harness hygiene (D2/D3).
- Excluded, and reported as negative ablations: transport across gene identity, graph heat-kernel smoothing, graph prior, balanced Sinkhorn kernel head.

**Expected outcome, stated up front.**
- The margin over the paper is expected to come from the noise-aware filter and the finite-cell observation model, not from OT/SB.
- For per-gene 1-D laws, the mean displacement of any coupling equals E_B[z] - E_A[z] (section 3.3). So the SB step is a law-level trend extrapolator with a zero boundary, and exploratory prototypes selected it off or found it harmful.
- The SB step is implemented, checked on synthetic known drift, and chosen by rolling origin. It gets credit only if gate G6 passes (section 8).
- The headline wording is "their files, split, targets and metrics; our sampling-aware prediction". The OT/SB label describes the architecture, not the gain, unless G6 passes.

**Corrections to the brief.**
1. The harness is `tgl-grn-orig`, the anonymous.4open.science original. Job 712735 reproduces the paper's Linear row with it: pancreas 0.1068/0.0618/0.0561, gastrulation 0.0619/0.0414/0.0069. MAP 1's claims that `tgl-grn-1CCD` is the original and that no pristine upstream exists are out of date: 1CCD's split takes the tail of the permutation, the original takes the head.
2. The '-human' node stats are **raw UMI counts**, not log-normalized: every s04_sum is an integer, every median a half-integer, single-cell values are 1, and s02_std is the population sd. The observation model is therefore a count model, and this is asserted at load.

**Tiers** (each is judged separately):

| Tier | Inputs at transition i (t -> t+1) | Meaning |
|---|---|---|
| STRICT | raw node stats of loaded snapshots j <= i only | no information from bin t+1 |
| SAME-INFO | STRICT plus `train_idx` and `y_train` of transition i (the 75% training genes their loss is fitted on) | same labels as their models |

SAME-INFO matches their models on labels, not on features. Our raw scale, n_t and multi-bin history cannot be recovered from their z-scored forward() input. MAP 2 defines this tier differently; we use the definition above and say so in the report.

There are two separate claims (section 8): **A**, "beats the paper" (per tier), and **B**, "OT/SB contributes" (ablation).

## 2. Data and exact leakage rules

We use only their repo data, and only through their loader. The loader call `tgl-grn-orig/dataset.load_dataset('grn', name, True)` runs inside our harness and reads:
- `GRN/<name>_node.csv`: s00_mean, s01_median, s02_std, s03_frac_nonzero, s04_sum per gene and bin.
- `GRN/<name>.csv`: the edges. Their loader uses it to define the snapshots and N; our core never reads it.

| | Rule |
|---|---|
| A. STRICT may read | `hist = [dataset[j].x as float64 numpy, j <= i]` (the 5 raw columns) and `ts[:i+1]`. Derived: n_b, l_b and triplets for b <= t; past targets y_j for j <= i-1, recomputed from the raw means in hist (all genes); constants N, i, eps, C, q. |
| B. SAME-INFO adds | `train_idx` and `y_train = x_future_delta[train_mask]` of transition i only, as numpy |
| C. Never, any tier | `dataset[j]` for j > i (including `dataset[train_split]`, which only their target code reads); x_future_delta of non-training genes; anything indexed by test_mask; y_outdeg and y_pagerank (centralities of graph t+1); n_cells, bin_mid and the feat JSON of any bin; edges; gene names; FM embeddings; torch's global RNG; Data objects (the forecaster gets numpy only); any choice made on the test-gene metrics of scored transitions. The post-hoc k=4/6 picks of level/obslevel are references only. |
| D. Excluded by default (labelled sensitivity only) | gastrulation-human node bins t000/t001: 2 and 3 cells, no edges, not in their snapshot list |

Enforcement:
- The harness passes a numpy-only ctx: `{}` for strict, `{train_idx, y_train}` for same-info, with a type assert.
- The forecast for (snapshot j, config c) is computed only when `len(hist) == j+1`, and cached.
- The leak test is stage S8.

## 3. Model math

**3.0 Notation.**
- Transition i maps bin t = ts_i to t+1. For each bin b <= t and gene g: m_gb (mean), s_gb (population sd), f_gb (fraction nonzero), S_gb (sum).
- n_b = round(median over {g: m_gb > 0} of S_gb/m_gb).
- l_b = sum_g m_gb.
- e_gb = s_gb^2 + m_gb^2, so the triplet is x_gb = (m, f, e).
- eps = 1e-8, C = 5.

**3.1 Per-cell law of a triplet.**
- P(c=0) = 1 - f. For f > 0: mu+ = m/f and v+ = max(e/f - mu+^2, 0).
- c | c>0 = 1 + E, with E chosen as follows:
  - E ~ NB(mean mu+ - 1, var v+) if v+ > mu+ - 1 > 0 (moments then match (m, f, e) exactly);
  - E ~ Poisson(mu+ - 1) if 0 < v+ <= mu+ - 1;
  - E = 0 if mu+ <= 1 + 1e-9.
- Closure: a convex combination sum_b w_b x_b is the triplet of the mixture law. Every pooled state is therefore a valid law (0 <= f <= 1, e f >= m^2).

**3.2 Latent population at t (pool family).**
- (a) Box(k): H_t = sum_{b in W} n_b x_b / sum_{b in W} n_b, where W = the last k_eff = min(k, i+1) loaded snapshots <= t.
- (b) Kalman(theta), run causally over j = 0..i. Initialise H = x_0 and P = R_0. Then, at each new bin b:
  - mbar_g = cell-weighted mean of m_g over bins <= b;
  - R_gb = (max(s_gb^2, mbar_g) + c_R/n_b) / n_b;
  - q_g = theta^2 (mbar_g^2 + c_q mbar_g);
  - P- = P + q_g;
  - K = P- / (P- + R_gb);
  - H <- (1-K) H + K x_gb, on all three channels with the common gain;
  - P <- (1-K) P-.
  - H_b and P_b are stored for every b.
  - On the mean channel this is the exact local-level Kalman filter with known noise. Formally it is the time-t marginal of the KL projection of a Brownian-reference path law onto soft observation constraints (a soft-marginal SB). We call it a filter because that is what it computes.
  - R is on the count scale. If a log-scale variance is ever needed, it is Var(log m_hat) = (kappa + 1 - f)/(n f), with kappa = v+/mu+^2. This fixes D2's slip, which dropped the 1 - f term.
- (c) lib = on: for transition i, rescale each past bin to bin t before pooling: m_b and s_b are multiplied by l_t/l_b, e_b by (l_t/l_b)^2, and f_b is unchanged.
- (d) Long-memory prior gamma: for genes with H_f,t = 0 that had m > 0 in some bin <= t, set H_t <- gamma * x_all, where x_all is the cell-weighted triplet over all bins <= t. This is a mixture with a point mass at 0, so it remains a valid law.

**3.3 OT/SB step (per gene; i >= 1; L_i = min(L, i); extrapolation delta).**

The OT couples cell populations of the same gene at two bins. Cells are destructively sampled, so their identity across bins is genuinely unknown; gene identities are never transported.

Setup:
- Source A = H at bin t - L_i, from the same pool configuration. Target B = H_t, taken before gamma is applied.
- Skip the step (H' = B) if f_A < 1e-6 or f_B < 1e-6.
- Anscombe coordinate: z(c) = 2 sqrt(c + 3/8).
- Grid: the integers c = 1..c_max if c_max <= 48; otherwise 48 points uniform in z on [z(1), z(c_max)]. Here c_max is the 0.999-quantile of the heavier of A+ and B+.
- p_A and p_B are the normalised positive-part pmfs on that grid.

Static SB with a Brownian reference of variance sigma_SB^2 per bin:

pi = argmin sum_jk pi_jk (z_j - z_k)^2 / 2 + eps_SB KL(pi | p_A (x) p_B), with marginals p_A and p_B and eps_SB = sigma_SB^2 L_i.

Our own batched Sinkhorn:
- K = exp(-C/eps_SB); iterate u <- p_A/(K v) and v <- p_B/(K^T u).
- Arithmetic is float64 in the scaling domain, falling back to the log domain where u or v overflow.
- Stop when max |v (K^T u) - p_B| < 1e-9, or after 1000 iterations. The eps_SB floor is 1e-3, and a finite-plan assert is applied.

Forecast law:
- Velocity on the target grid: v_k = (z_k - sum_j pi_jk z_j / p_B,k) / L_i.
- Mass trend: g = log(f_B / f_A) / L_i.
- Move the cells: z'_k = z_k + delta v_k.
- Boundary: cells with z'_k < z(1/2) leave the positive part (switch-off). Cells with z(1/2) <= z'_k < z(1) snap to c = 1.
- f' = min(1, f_B e^{delta g}) * sum over {k: z'_k >= z(1/2)} of p_B,k.
- c'_k = max(1, (z'_k/2)^2 - 3/8). mu+' and v+' are the mean and variance of c' over the surviving p_B mass.
- H' = (f' mu+', f', f' (v+' + mu+'^2)).
- delta = 0 skips the step, so H' = B bitwise.

**Identity (stated in the write-up and tested).** sum_k p_B,k v_k = (E_B[z] - E_A[z]) / L_i holds for every coupling with these marginals.
- The mean drift on the z scale is therefore coupling-independent.
- The coupling (through eps_SB) acts only via the spread of v: which cells cross the zero boundary, and the dispersion of the positive part.
- The mass channel g is a log-linear trend, not OT.

**Forecast step.** An SB with no terminal constraint is its reference. On the population mean that is the Kalman predict step (P- = P + q). The cell law is not diffused again, because finite-cell noise is handled in 3.4.

**3.4 Observation model.** The next bin is a sample of n' cells. In STRICT, n' = n_t and lambda = 1.
- K ~ Binomial(n', f').
- S | K = K + NB(mean K(mu+' - 1), var K v+'). NB and Poisson are closed under iid sums; in the degenerate case S = K.
- M = lambda S / n'.
- y_hat_g = E[clip(log(M + eps) - log(m_gt + eps), -C, C)], where m_gt is the raw observed mean of bin t.

Analytic parts, with P0 = (1 - f')^{n'}:
- If m_gt = 0: y_hat = C (1 - P0) exactly, since S >= 1 gives a log-ratio >= log(1e8/n') > 5.
- If m_gt > 0: y_hat = -C P0 + (1 - P0) E[clip(...) | K >= 1].

The conditional term uses D = 256 Monte Carlo draws: K from the zero-truncated binomial by inverse CDF, then S | K. The generator is `np.random.default_rng([ds_id, i, cfg_id])`, so draws are deterministic and never touch torch.

Two alternative modes:
- CLT mode (ablation and nesting only): exactly `task2_forecasters.expected_target`, i.e. `default_rng(0 + len(hist))`, first `binomial(n, f[:,None], size=(N,D))`, then `standard_normal((N,D))`, with `n = int(round(n_cells(hist[-1])))`.
- Gauss-Hermite mode (test cross-check, and the differentiable form for Route B): 8 Gauss-Hermite nodes on the normal approximation of M | K >= 1.

**3.5 Rule at i = 0** (fixes D1's EB + large-n' failure).
- Own bin only: k_eff = 1, H = x_0, delta = 0, no head.
- n' = n_t in **both** tiers; n_hat and c_hat are not estimated.
- The output is non-constant, so Pearson is never NaN and the transition is never dropped.
- The EB Beta-prior variant is a sensitivity analysis only.

**3.6 Strict heads (past transitions only, nested).**
- ridge_past: y = clip(w0 + w . phi, +/-C). Features phi_g:
  - y_hat;
  - y_hat_inf (the n' -> infinity limit);
  - Box level deltas for k = 2, 4 and 6;
  - the last change y_{i-1};
  - 1[m_t = 0], f_t, log(m_t + eps)/18.42;
  - P0 and P0 * 1[m_t > 0];
  - log P_t (0 for Box).
  The head is fitted on all genes of the transitions j in W(i), with ridge penalty 1e-3 * N_fit and features standardised on the fit set.
- knn_past (the tau -> 0 limit of unbalanced transport, i.e. Nadaraya-Watson): pool residuals rho = y_j - y_hat_j over j in W(i), subsampling 3000 genes per j with a private RNG. For each gene, r_g = mean rho over its 100 nearest pool points in standardised (y_hat, log(m_t + eps)/18.42, f_t, P0). Then y = clip(y_hat + r_g).
- A gene's own past residuals are never pooled by identity; prototypes found this harmful because residuals anti-persist.

**3.7 SAME-INFO calibration (i >= 1; Tr = train_idx).**

n_hat = argmax over 140 log-spaced integers n in [1, 3000] of

  sum over {g in Tr, m_t > 0} of [z- n log(1 - f') + (1 - z-) log(1 - (1 - f')^n)]
  + sum over {g in Tr, m_t = 0, 0 < f' < 1} of [z+ log(1 - (1 - f')^n) + (1 - z+) n log(1 - f')],

where z- = 1[y <= -C + 1e-6] and z+ = 1[y >= C - 1e-6].

- c_hat = median of (y - y_hat(n_hat)) over {g in Tr: m_t > 0, |y| < C - 1e-6}. Set lambda = e^{c_hat} and recompute y_hat(n_hat, lambda).
- ridge_same head: the ridge_past features plus y_hat(n_hat, lambda) and P0(n_hat), fitted on Tr.
- Mandatory 5-fold cross-fitting: folds come from a private RNG seeded by (seed, i). For training genes in fold F, every feature that uses n_hat or c_hat is rebuilt with estimates from Tr \ F. All other genes use the estimates from the full Tr.
- The head is switched on or off by 5-fold CV on Tr with criterion J (section 4); ties go to off.
- Output: the head's prediction if on, otherwise y_hat(n_hat, lambda).
- At i = 0: n_hat = n_t and lambda = 1.

**3.8 Nesting (tested in stage S5).** Take Box(k) with delta = 0, gamma = 0, lib off and head off. Then:
- n' -> infinity (analytically y_hat = clip(log(m_pool + eps) - log(m_t + eps))) gives `task2_forecasters.Level(k)` for i >= 1;
- unit bin weights instead of n_b give `dileo_task2.level(k)`;
- CLT mode with n' = n_t gives `task2_forecasters.ObsLevel(k)` bitwise for i >= 1;
- Kalman theta -> infinity gives Box(1).

## 4. Hyperparameters and leakage-free tuning

**Fixed constants, frozen in the pre-registration:**
- c_R = 1, c_q = 1/50
- L = 3, sigma_SB^2 = 0.25 per bin on the z scale (Poisson noise on z is about 1)
- grid 48 points; Sinkhorn tol 1e-9 and 1000 iterations; eps_SB floor 1e-3
- D = 256 MC draws
- window 8; kNN 100 neighbours with a pool of 3000 per transition; ridge 1e-3
- n_hat grid of 140 points; 5 folds
- eps = 1e-8, C = 5
- q = 200/round(0.25N), i.e. 200/5358 for gastrulation and 200/4640 for pancreas
- default config c0 = (Kalman theta = 0.2, delta = 0, gamma = 0, lib off, head off), used for i <= 2

Disclosure: c_R, c_q and the theta grid were set by D1 while it was looking at exploratory results on these transitions. They are frozen, and their sensitivity at x0.5 and x2 is reported but never used as the headline.

**Strict grid, stage 1 (base configs, head off), 200 configs:**

| Pool | delta | gamma | lib | Count |
|---|---|---|---|---|
| Box k in {1,2,3,4,6,8,12} | {0} | {0, 0.1, 0.3, 1} | {off, on} | 56 |
| Kalman theta in {0.05,0.1,0.2,0.35,0.6,1.0} | {0, 0.5, 1} | {0, 0.1, 0.3, 1} | {off, on} | 144 |

**Stage 2:** head in {off, ridge_past, knn_past}, evaluated on the stage-1 winner only.

**Criterion J** is metric-matched and tie-robust. It resolves the judges' disagreement: MSE cost pancreas P@200 up 0.296 -> 0.261, and raw P@200 depends on how np.argsort breaks ties.
- For a past transition j, use all genes G. Let K_q = round(q |G|) and let v be the K_q-th largest value.
- Inclusion probability under uniform random tie-breaking:
  - pi_g(y) = 1 if y_g > v;
  - pi_g(y) = (K_q - #{y > v}) / #{y = v} if y_g = v;
  - pi_g(y) = 0 otherwise.
- EP_up = sum_g pi_g(y) pi_g(p) / K_q. EP_down is the same with -y and -p.
- J_j = [Pearson(y, p) + EP_up + EP_down] / 3.

**Rolling origin, nested:**
- For i >= 3: c*(i) = argmax over c of the mean of J_j over W(i) = {max(1, i-8), ..., i-1}. j = 0 is excluded because it does not depend on the config.
- Each forecast for j uses only hist[:j+1], so the score is a true forecast.
- A head scored on transition j is fitted on W(j). Heads are scored only on j >= 3, so they stay off for i <= 4.
- Tie rule: if |Delta J| < 1e-3, prefer delta = 0, then head off, then gamma = 0, then lib off, then the lower grid index.

**SAME-INFO:** no extra grid. It uses c*(i), then n_hat and c_hat, then the head gate (CV on Tr with J).

**Seeds:** {1234, 42, 7, 0, 1}. Strict forecasts do not depend on the seed; the masks do.

**Sensitivities** (reported, never the headline):
- MSE selector;
- W(i) = all past transitions;
- c_R and c_q at x0.5 and x2;
- EB rule at i = 0;
- CLT observation mode;
- t000/t001 bins included (gastrulation).

**Compute:**
- Strict cache: 200 configs x T transitions x ~0.3 s, plus 12 couplings per transition (6 theta x 2 lib; the coupling does not depend on delta or gamma) at ~10 s each. That is about 1.5 h for gastrulation and 0.6 h for pancreas.
- The cache is module-level, keyed by (sha1(hist[0]), j, cfg_id), so it is shared across seeds and forecaster instances.
- SAME-INFO adds about 2 min per seed per dataset. RAM is under 2 GB for the forecaster.
- Pre-registered fallback: if building the gastrulation strict cache exceeds 3 h of wall time, couplings are computed only for theta in {0.2, 0.6}.

## 5. Harness integration

**Harness.** `tgl-grn-orig` is never edited. It lives locally at `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tgl-grn-orig` and on the HPC at `~/tgl-grn-orig`, where GRN/ is a symlink to the same data. Asserted md5 hashes:

| File | md5 |
|---|---|
| dataset.py | 96bcd18e129df0973bf84a26239800fb |
| train_models_gene.py | 980348482cffda98652535161d69fc54 |
| models_gene.py | 556897b595a715be27485caa7da73150 |
| train_gene.py | af7558203abae43f38b01cb62aeb6493 |

**Route A (primary, both tiers): `tack/hpc/dileo_harness.py` (ours).**

What it already does:
- imports their `load_dataset`, `train_val_test_split`, `build_snapshot_split_future_node_delta_ranking` and `_delta_metrics`;
- seeds as `train_gene.main` does;
- loads each dataset once and reuses each forecaster object across seeds;
- writes per-seed JSON in their logs-gene format, plus `summary.json`.

Edits, to our file only:
1. Assert the md5 hashes and the snapshot lists (gastrulation ts 2..31 with N = 21430; pancreas ts 0..14 with N = 18561). Assert the raw-count premise (s04_sum integer).
2. Make the same-info ctx numpy `{train_idx, y_train}` only, dropping the torch edge tensors. The strict ctx stays `{}`. Assert numpy-only types.
3. Assert that `torch.get_rng_state()` is unchanged across each `predict`.
4. Extend the summary with:
   - the 28/13 nanmean (drop `snapshot_idx == len(snapshots)-1`);
   - the NaN-Pearson count;
   - per-transition counts of test genes tied at +5 and at -5;
   - an EP@200 diagnostic.
5. Optional `--mask_ref <linear|chebnet|...>`: replay the reference model's CPU-RNG use (`models_gene.load_model_node(ref,'grn')`, then `.reset_parameters()`, then discard) before the transition loop. It is valid only if the stage S8 randperm-hash test passes.

Label for Route A results: "their loader, split, targets, metrics; our prediction (closed-form fit, no Adam)".

**New files (ours).**

Code:
- `tack/hpc/sinkhorn.py`
- `tack/hpc/dist_bridge.py`: reconstruction, pools, SB step, observation model, i=0 rule, heads, n_hat and c_hat, J, rolling selector, cache.
- registry entries in `tack/hpc/task2_forecasters.py`.

Registry names:
- headline: `distbridge_strict`, `distbridge_same`;
- ablations: `distbridge_strict_nosb` (delta forced to 0), `_box`, `_nohead`, `_nogamma`, `_clt`, `distbridge_same_nonhat`, `distbridge_same_nohead`;
- honest comparators: `obslevel_rollk` (Box only, delta 0, gamma 0, lib off, CLT, selected by the same J) and `level_rollk`;
- post-hoc references: the existing `obslevel4/6` and `level4/6`.

Tests: `tack/tests/test_sinkhorn.py`, `test_dist_bridge.py`, `test_synthetic_drift.py`, `test_nesting.py`, `test_harness_guards.py`.

Jobs: `tack/hpc/ref_reruns.sh` with `ref_reruns.sbatch`, and `tack/hpc/collect_refs.py`.

Pre-registration: `tack/prereg/2026-10-22-dileo-task2-distbridge-prereg.md`, with its sha256 recorded in a new row of `docs/superpowers/decisions.md`.

Results: `tack/results/dileo_task2/harness/distbridge_*`.

**Reference reruns** (their code, untouched). Models: linear (anchor), mlp, chebnet, gcrngru, roland. For each dataset, model and seed:
- `mkdir runs/<name>/<model>/seed<s>`;
- `ln -s ~/tgl-grn-orig/GRN GRN`;
- `python ~/tgl-grn-orig/train_gene.py --dataset grn --name <name> --model <model> --gene_features --seed <s>`.

Output lands in `logs-gene/gene_regr_<model>_grn_<name>.json` in each workdir.
- ROLAND produces 28/13 entries keyed by `state_id`, so it is compared on 28/13.
- scGPT and scFoundation cannot run as published (AttributeError; the README requires a hand edit), so they are compared on the paper number only.

**Route B (optional; only after the Route A verdict): `tack/hpc/train_gene_distbridge.py`.**
- It calls their `train_live_model_future_delta_ranking(model=DistBridgeModule, ...)` with their defaults (Adam 0.01, MSE, 100 epochs, 75% training genes), then their `save_results`, in per-seed workdirs.
- Module contract:
  - find the transition by `torch.equal` against `_normalize_node_features_per_snapshot(dataset[j].x)`;
  - read raw history through a guarded view (j <= i, x only; raises otherwise);
  - keep strict features in non-persistent buffers;
  - re-initialise in place (Adam is built before `reset_parameters`);
  - build the module inside `torch.random.fork_rng`;
  - return an [N,1] output that is never constant.
- Gate: their LinearModel run through this runner must match `train_gene.py --model linear` per transition (seed 1234, CPU).
- Results are labelled "in-harness".
- Equal-information control: their MLP class (imported, not copied) fed [x_t, y_hat].

## 6. Staged implementation plan (each stage has a pass test)

| Stage | Deliverable | Test (pass condition) |
|---|---|---|
| S0 | Local preflight | md5 values as in section 5. Both node CSVs pass the raw-count checks. The stored job-712742 summaries read back as 0.474/0.292/0.417 (pancreas) and 0.600/0.398/0.530 (gastrulation). |
| S1 | `sinkhorn.py` | Marginal error < 1e-9. On a 400-point grid, Gaussian laws give the cross-covariance c01 = (sqrt(4 v0 v1 + eps^2) - eps)/2 within 1e-2 relative error. At eps = 1e-3, E[z1\|z0] equals the quantile map within one grid step. No NaN on one-point measures. 20k genes x 48 points runs in < 20 s. |
| S2 | Reconstruction, pools, filter | Law moments equal (m, f, e) within 1e-9 (NB case). Every pooled or updated triplet is valid. Kalman theta = 1e9 equals Box(1). H_t is unchanged when bins > t are permuted. |
| S3 | SB step | delta = 0 gives a bitwise identity. The identity sum p_B v = Delta E[z]/L holds within 1e-10 for eps in {0.01, 0.25, 4}. **Synthetic known drift** (setup below): pass criteria (a)-(d). |
| S4 | Observation model | Analytic plus MC agrees with brute-force MC (1e5 draws) within 3 SE on 200 genes. Gauss-Hermite agrees with MC within 0.01. NB sums are closed. Output at i = 0 is non-constant on real bin 0 of both datasets. |
| S5 | **Level rule as a special case** | On the real node files (their snapshot list and N), for k in {2,4,6} and i >= 1: the n' -> infinity form equals `Level(k)` within 1e-6; unit weights equal `dileo_task2.level(k)`; CLT mode equals `ObsLevel(k)` bitwise. |
| S6 | J, selector, strict heads | EP@K agrees with the average over 2000 random tie-breaks within 3 SE. Without ties, EP@K equals their `_precision_at_k_top_up_down`, imported from tgl-grn-orig. Overwriting y_j for j >= i with garbage leaves c*(i) and the heads unchanged. |
| S7 | SAME-INFO | n_hat recovers a known n = 300 on synthetic bins within 15%. Changing y of a training gene leaves that gene's cross-fitted features unchanged. Setting all non-Tr labels to NaN leaves predictions bitwise unchanged. i = 0 uses n_t. |
| S8 | Harness guards | torch RNG state is unchanged. Test-mask hashes are identical for `distbridge_*` and `obslevel4` at each seed. Leak test: on a deepcopy with x and edges of dataset[j > i] permuted, predictions for transitions <= i are bitwise identical. Asserts fire on a tampered file. For `--mask_ref`, randperm hashes match their train_gene.py run on CPU. |
| S9 | Freeze the pre-registration | Grid, constants, J, seeds, gates G1-G6, ablation list and fallback trigger are written down, the sha256 is recorded in decisions.md, and `pos attempt add` is logged. Before this point, no full-model test-gene metric is computed on real data. |

**Synthetic setup for S3.**
- G = 3000 genes, 14 bins, n_b ~ U{50..600}.
- logit f_b = a_g + s_g b; mu+_b - 1 = exp(c_g + r_g b); NB size 2.
- Drift case: s and r ~ N(0, 0.15^2). Null case: s = r = 0.
- Sample cells, then compute the 5 node stats exactly (mean, median, population sd, frac_nonzero, sum).

**Pass criteria for S3:**
- (a) Over genes with f > 0.2: corr(d_hat, d_true) > 0.9 and |median bias| < 0.1 sd, where d is the per-bin Delta of E[z+].
- (b) corr(g_hat, g_true) > 0.9.
- (c) On drift data, delta = 1 beats delta = 0 on both J and MSE, averaged over i >= 4.
- (d) On null data, rolling J selects delta = 0 in at least 80% of transitions with i >= 4.

## 7. Run order

The HPC steps are for whoever has access; this planning task did no ssh. Before the first run, sync `tack/hpc/*.py` to `~/tack/code`. QoS stud allows 1 running and 2 submitted jobs, with no arrays.

1. **Local:** stages S0-S8 (synthetic data plus node CSVs; edge files not needed), then S9 (freeze).
2. **HPC preflight (read-only):** md5 of `~/tgl-grn-orig`, the snapshot-list assert, the raw-count assert.
3. **Job A, pancreas-human** (smaller, fails fast):
   ```
   MODELS=distbridge_strict,distbridge_same,distbridge_strict_nosb,distbridge_strict_box,distbridge_strict_nohead,distbridge_strict_nogamma,distbridge_strict_clt,distbridge_same_nonhat,distbridge_same_nohead,obslevel_rollk,level_rollk,obslevel4,obslevel6,level4,level6 NAMES=pancreas-human sbatch tack/hpc/dileo_harness.sbatch
   ```
   This is one process (load once), 5 seeds, on 4 CPUs with 120G and 6 h.
4. **Job B, gastrulation-human:** the same command with `NAMES=gastrulation-human`.
5. **Jobs C and D:** reference reruns, one job per dataset, looping models x seeds. Split a job if its estimate exceeds 6 h, using the job-712735 timing.
6. **Collect and check:** run `collect_refs.py`, check the anchors (G5), then build the report:
   - 29/14 and 28/13 averages;
   - both i=0 conventions: A (scored) and B (NaN-drop, for baselines that emit constants);
   - paired per-transition differences against `obslevel_rollk` and against the reruns;
   - per-transition selected configs (pool, theta, delta, gamma, head) and n_hat trajectories;
   - clip-tie counts and MAE;
   - the exploratory negative ablations, labelled as exploratory:
     - identity-aware residual pooling: pancreas PCC 0.459 -> 0.307, gastrulation 0.584 -> 0.473;
     - balanced Sinkhorn head: pancreas 0.448;
     - heat smoothing at eta = 0.05: pancreas 0.459 -> 0.269;
     - graph prior: pancreas P@200 up 0.287 -> 0.220;
   - the graph diagnostics: 0% of +5 genes have an edge in graph t, and the edge-residual correlation is 0.0095.
   Then log `pos attempt add` / `pos decide`, update STATE.md, and record the run in the shared acts ledger.
7. **Optional:** Route B, and `--mask_ref` paired runs.

## 8. Pass/fail criteria

For tier T, dataset d and metric k in {PCC, P@200 up, P@200 down}: S_s is their nanmean over all transitions for seed s, and mu and sd are taken over the 5 seeds.

Paper best values:

| Dataset | PCC | P@200 up | P@200 down |
|---|---|---|---|
| gastrulation-human | 0.400 (GCRN-GRU) | 0.110 (ROLAND) | 0.387 (ChebNet) |
| pancreas-human | 0.434 (MLP) | 0.178 (scFoundation) | 0.354 (ChebNet) |

Gates:
- **G1 (beats the paper means):** mu - 2.132 * sd / sqrt(5) > paper best, for all 6 cells (one-sided 95% t lower bound).
- **G2:** G1 also holds on the 28/13 subset for all 6 cells (the like-for-like set for ROLAND/TGNN-mem).
- **G3:** zero NaN-Pearson transitions in our runs.
- **G4 (paired reruns):** in each cell, our per-seed score exceeds their per-seed rerun of that cell's best model in at least 4 of 5 seeds, and the 5-seed mean difference is > 0. The pancreas P@200-up cell (scFoundation) cannot be rerun, so it is paper-only and flagged.
- **G5 (validity):** their linear rerun at seed 1234 reproduces job 712735 to 3 decimals. `obslevel4` and `obslevel6` reproduce job 712742 within +/-0.002. If G5 fails, the run is invalid.

Verdict:
- "Beats the paper (tier T)" requires G1-G5 on all 6 cells.
- If G1-G3 pass but G4 fails somewhere, the wording is "beats the paper's means, within paper seed noise".
- STRICT and SAME-INFO are reported separately; a STRICT pass is the stronger claim.

**G6 (claim B: OT/SB contributes), per dataset.** Masks are identical, so the comparison is exactly paired: Delta = `distbridge_T` - `distbridge_T_nosb`. G6 passes only if all of these hold:
- the 5-seed mean Delta is >= +0.01 on at least one metric and >= -0.005 on the other two;
- a one-sided Wilcoxon test on the per-transition, seed-averaged Delta of J gives p < 0.05;
- delta > 0 was selected in at least 25% of transitions.

If G6 fails, the write-up must say: "the OT/SB step was selected off or did not contribute; the margin comes from the noise-aware filter and the finite-cell observation model."

`obslevel_rollk` is always reported next to our results, and a tie with it is disclosed. Any change after the freeze requires a new pre-registration version, and the earlier result stays reported.

## 9. Risks

- **Goal risk:** OT/SB is unlikely to carry the margin (by the identity in 3.3 and the prototypes). Mitigation: G6 is a separate claim, and section 1 frames the result this way up front.
- **Inherited margin:** the margin largely comes from ObsLevel, our own earlier baseline. Gastrulation STRICT may only tie `obslevel_rollk`.
- **Noise:** the pancreas STRICT PCC margin (about +0.03 to +0.06) sits inside the paper's seed sd (about 0.1). Only G4 can support it.
- **Unknown paper seeds:** the paper's Table 2 may effectively come from one seed of this lineage. Their reruns use different masks per model unless `--mask_ref` passes the hash test, which is only valid for references with no CPU RNG during training.
- **P@200 ties:** P@200 is tie-dominated (more than 200 test genes tied at +5 in 17/29 gastrulation and 7/14 pancreas transitions). Differences under 0.005 are noise. We never tune on raw P@200; EP is reported.
- **i=0 conventions:** the baselines NaN-drop i=0 while ours is scored, which makes them look about 0.015 better on gastrulation PCC. Both conventions are reported.
- **Premise and harness drift:** the raw-count premise or the HPC files could differ from local. The asserts stop the run.
- **Constants:** c_R, c_q and the theta grid were set while looking at the data. They are frozen and their sensitivities reported.
- **RNG:** any use of torch's global RNG would shift their masks. Asserted.
- **Compute:** compute figures are estimates. The fallback trigger is pre-registered. Their loader needs large RAM for 3.5 GB of edges; the job requests 120G.
- **MAE and n_hat:** heads and n_hat can raise MAE (e.g. pancreas 0.666 -> 0.713). n_hat is an effective sample size, not a cell count. Both are reported.
- **Unequal information:** their models never see n_t, raw scale or multi-bin history. The equal-information control (their MLP given y_hat, Route B) addresses whether the gain is information or architecture.