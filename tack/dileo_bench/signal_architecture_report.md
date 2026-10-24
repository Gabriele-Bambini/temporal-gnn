# Why OT does not win, what in the signal to exploit, and which architecture fits

Scope: Dileo & Sottoriva Task 2 on their `-human` builds: gastrulation-human (29 transitions) and pancreas-human (14). Every check is strict and rolling-origin. Each triple is PCC / P@200 up / P@200 down.

Averaging conventions: `all` is the mean over all transitions, as in the CONTEXT DistBridge numbers. `i>=k` means transitions k..T-1.

Provenance tags:
- **[J]**: re-run or re-derived with independent code by a judge.
- **[A2]**: the same quantity measured independently by two signal analysts; a range is given.
- **[A]**: one analyst's script only, not re-run by a judge.

## 1. Why our OT did not win

On these bins the predictable part of a change is mostly three things:
- (a) finite-cell sampling events, i.e. genes crossing zero;
- (b) the reversal of today's sampling noise;
- (c) a weak one-bin momentum.

Per-gene 1-D OT cannot do (a) or (b), and it dilutes (c).

- **What per-gene OT can express.** For one gene, the mean displacement of any coupling between two 1-D laws equals the difference of their means. So OT can only extend the gene's own mean trend and reshape its spread, i.e. how many cells end up at zero. It uses no other gene and no other cell.
  - [J] I regressed the saved OT increment on quantities any filter already has:
    - l1: the pooled 3-bin mean trend alone gives R² 0.71-0.74.
    - orig: the trend alone gives R² 0.04-0.13. Adding the P(all-zero) shift from the nonzero-fraction trend raises it to 0.68-0.80.
    - The remaining 20-30 % has no value. Its partial correlation with the truth, given the NASS stacker, is ≤ +0.008 on orig and -0.02 on l1.
- **orig (the paper's target) is a zero-crossing counting problem.**
  - 91-95 % of the target's sum of squares sits in the 8-9 % of genes that cross zero [A2]. Observed switch-offs are 86-90 % of what pure sampling, (1-f)^n, predicts [A].
  - Changes undo themselves: lag-1 correlation is -0.47 (gastr) and -0.40 (pancr), and lags 2-5 are about 0 [A2]. Any trend therefore hurts: trend PCC is -0.47 / -0.39 [J].
  - DistBridge wins because of its finite-cell observation head, worth +0.15 / +0.18 PCC [A]. A denoised level plus that head, with no OT and no trend, reproduces it: gastr 0.587-0.588 vs 0.585, pancr 0.478-0.489 vs 0.488 [A2].
  - Pancreas, same masks [J], `all`: DistBridge with OT 0.492/0.304/0.439, without OT 0.495/0.305/0.441. The OT step was chosen in 2 of 14 transitions.
  - The OT push alone, on top of the box-4 level [J]: gastr -0.018 PCC (better in 1 transition), pancr -0.002.
- **l1 (the corrected target) is mostly sampling noise, and its largest predictable piece enters with a minus sign.**
  - Noise budget [J, matching A2]. Shares of noise in bin t / noise in bin t+1 / signal:
    - gastr: 0.37 / 0.42 / 0.21;
    - pancr: 0.20 / 0.24 / 0.57.
  - Noise in bin t can be predicted by denoising today's level. Noise in bin t+1 cannot be predicted by anything.
  - The true change keeps its direction for about one bin: noise-corrected lag-1 is 0.5-0.7 across two estimators, and about 0 beyond [A2]. DistBridge couples over 3 pooled bins, which averages away a persistence that lasts one bin.
  - Pancreas l1:
    - DistBridge without OT: -0.035, paired [J].
    - DistBridge with OT: 0.10 (CONTEXT, unpaired). OT only puts the trend back.
    - The plain last change scores 0.14-0.16, above both.
  - The OT push alone ends at 0.16 (gastr) and -0.12 (pancr) [J]. The noise-aware stack reaches 0.30-0.35.
- **linear** is decided by about 10-30 highly expressed genes and the drift in library size [A2]. There is no transport content to exploit.
- **The richer OT variants fail for the same structural reason.** Both prototypes were reproduced exactly by both judges.
  - Population OT on NMF archetypes: the gene forecast is linear in the change of proportions, so OT minus linear extrapolation of proportions is about 0 (|Δ| ≤ 0.03, not significant). The static SB kernel scores -0.10 against linear extrapolation (p 0.01-0.02). Even the true next composition gives only 0.141 on pancreas l1, below trend at 0.160.
  - Gaussian latent SB reduces to a Kalman predict step with a drift. The factor scores are almost noise-free (noise/signal 5e-4 to 1.4e-3), so the coupling carries nothing.

## 2. Signal peculiarities, ranked by predictable variance

| Target / dataset | Rank | What to exploit | Size | How |
|---|---|---|---|---|
| **orig**, both | 1 | Zero-crossings, P(all-zero among n cells) = (1-f)^n | 91-95 % of SS [A2]. Turn-on AUC from the gene's own detection history: 0.86 / 0.81 [A] | Finite-cell observation head |
| | 2 | Reversal of bin-t noise | lag-1 -0.47 / -0.40 [A2] | Denoised level, no trend |
| | 3 | Level-dependent calibration of the head; forecasting n_{t+1} | about +0.05 on pancreas; n_{t+1} +0.007 / +0.04 even when known (oracle) [A2] | Small |
| **l1 gastr** | 1 | Reversal of bin-t noise | 0.37-0.39 of variance | Denoise today with the known noise s²/n |
| | 2 | True change: a weak, persistent, low-rank, stationary drift | 0.18-0.21. 53 % of it lies in the top-3 subspace of past bins [A] | Long-window or shrinkage estimation. Cross-gene SVD adds only +0.005 (n.s.) once a known-noise per-gene filter is present (NASS ablation). They capture the same noise reversal. |
| | (unpredictable) | Noise of bin t+1 | 0.41-0.43. Bins with 46 and 7 cells are almost pure noise | Down-weight them |
| **l1 pancr** | 1 | True change: non-smooth, keeps opening new directions, real in highly expressed genes | 0.54-0.57. Second-difference ratio 1.3 on l1 and 3.8 on linear [A2]. Only 9-10 % lies in the past subspace [A]. Lag-1 by expression quintile: +0.32 top, -0.23 bottom [A2] | Per-gene momentum gated by SNR |
| | 2 | Bin-t noise | 0.20-0.23 | Denoise low-SNR genes only. In highly expressed genes today's deviations persist (negative denoise coefficient in classes 4-8) [A] |
| | 3 | GRN modules | Partial correlation 0.10-0.11 vs a level-matched null of about 0 [A2]. PCC +0.019, n.s. (p=0.20) | Light module prior at most |
| **linear**, both | 1 | Top 1 % of genes | 78-85 % of SS. 12-32 effective genes; mask sd of PCC 0.05-0.10 [A2] | Low precision |
| | 2 | Library scalar | Knowing the true direction gives PCC 0.39-0.44; strict 0.09-0.13. The sign persists only 57 % / 69 % of the time [A2] | Separate head. Do not choose an architecture on this target; its noise model is uncalibrated (slope 17-38) [A] |

Things that carry about nothing:
- GRN edges on orig: 0.610 vs 0.611. The true top-200 up list is 88 % / 75 % genes that were at zero, and those have no edges by construction [A].
- Degree and hub features: Spearman with expression level 0.87-0.93 [A].
- Global L and n on orig and l1 [A].

Cross-cutting findings:
- The best gain drifts at almost every transition. Rolling-origin selection agrees with the hindsight best in only 7-35 % of transitions [A]. The gain should therefore come from the known noise rather than from grid tuning.
- Signal variance must be gene-specific, set by expression class [A].
- Pancreas orig: transitions with bins of 50-79 cells hold 39-49 % of DistBridge's score sums [A]. Report every result with and without bins under 100 cells.

## 3. Architecture that fits: NASS (noise-aware stacked state-space)

NASS has three parts:
- A per-gene state with known observation noise R = s²/(n-1). It combines a box pool and an empirical-Bayes local-level Kalman, plus a denoised one-bin momentum term and SNR interactions.
- A finite-cell head for orig: expected clipped target, P(all-zero) and a zero-today flag.
- A recency-weighted ridge stacker, gated by 5 expression classes, refitted at each transition on past transitions only.

Status:
- Re-run by both judges: all tables within 0.0022 / 0.0025.
- Truncation test bit-exact (max |d| = 0.0) for both judges.
- Re-implemented independently by both judges with other masks (seeds 4242, 777, 99).

**l1 (primary)**

| Method | gastr, i>=2 | pancr, i>=2 | pancr, `all` |
|---|---|---|---|
| trend (repeat last change) | -0.179/0.101/0.092 | 0.135/0.220/0.216 | 0.141/0.211/0.215 |
| **NASS full** (pre-specified rolling-origin grid) | **0.335/0.276/0.290** | **0.308/0.221/0.283** | 0.290/0.212/0.272 |
| Judge 1, independent code | 0.324 (RO); 0.349 (h=0.5) | 0.297 | |
| Judge 2, independent code (h=1 fixed) | 0.321 | 0.310 | |
| Paired NASS - trend | +0.514 (25/2, p<1e-4) | +0.173 (10/2, p=0.002) | |
| Recency-weighted per-gene filter, i>=3 | 0.350 vs NASS 0.353: **tie** (+0.003, p=0.33; judge 1 +0.010, p=0.15) | NASS +0.043 (10/1, p=0.003); judge 2 +0.035 (p=0.009) | |
| DistBridge without OT (judge 2, same masks) | not run | | -0.035/0.094/0.145 (NASS +0.32, 12/2, p=0.0006) |
| DistBridge with OT | **no number exists** | | 0.10/0.17/0.21 (CONTEXT, unpaired, averaging convention ±0.04) |

**orig**

| Dataset | Comparison | Result |
|---|---|---|
| gastr | NASS (`all`) vs DistBridge (CONTEXT) | 0.601/0.406/0.541 vs 0.585/0.406/0.541. Unpaired. Against level+obs, i>=2: +0.024 (25/2) |
| pancr | NASS vs DistBridge, same masks (judge 2), `all` | 0.506/0.308/0.454 vs 0.492/0.304/0.439: **tie**, +0.014 (8/6, p=0.63); i>=2 +0.021 (p=0.38). NASS's own claim of "+0.051" was measured against its box4+obs baseline, which sits 0.030 below DistBridge (p=0.017) |

**linear**

| Dataset | NASS vs trend, i>=2 |
|---|---|
| pancr | 0.281 vs 0.407: **loses**, -0.126 (2/10, p=0.007). Meta-selection with trend as a candidate still loses: 0.386 |
| gastr | 0.256 vs 0.201: +0.054 (14/13, p=0.44), within the mask sd of 0.06 |

**Mechanism checks** [J]:
- On Dileo's mouse builds, NASS only ties trend (pancr 0.546 vs 0.520, gastr 0.523 vs 0.568, both n.s.). Those builds hold smoothed, non-integer values with almost no count noise (s²/m about 0.03).
- On gastrulation with bins merged in pairs, the NASS lead on l1 shrinks to +0.05 / +0.14 (n.s.).

Both are consistent with the gain coming from modelling finite-cell noise. Neither is a held-out test.

**Where OT/SB belongs**

| Variant | What it reduces to | Verdict |
|---|---|---|
| Per-gene 1-D OT (DistBridge) | Mean trend plus spread. As a NASS feature: partial correlation ≤ +0.008 / -0.02, paired differences about 0 | Nowhere on node-file data |
| Population OT/UOT/SB on archetypes (PopOT, strict) | Linear in the change of proportions. OT equals linear extrapolation; the oracle composition stays below trend | Not useful |
| Latent Gaussian SB (LatentSB, **not strict**) | A Kalman predict step with drift. Pancreas l1 0.080 vs trend 0.161; linear 0.119 vs 0.365 | Not useful |
| **Legitimate place** | Cell-level data, where coupling cells across bins can expose branch and composition shifts (the non-smooth factor seen in pancreas). Or endpoint conditioning on an external atlas: the only positive probe (+0.119 on pancreas l1) is non-strict, assumes known terminal expression, and comes from the endpoint, not the coupling | Score it only as an increment over NASS |

**Flags (not reproduced, or weak)**
- DistBridge with OT on pancreas l1 was never paired: the run was killed after about 55 min. There is no DistBridge number for gastr l1, and gastr orig vs DistBridge is unpaired.
- The NASS GRN add-on:
  - was not truncation-tested, and the per-bin provenance of its edges was not verified;
  - is not significant on l1;
  - hurts linear: -0.099 (1/11, p=0.002).
- NASS claimed a Sinkhorn error ≤ 5e-4; it is 7.5e-3 at gastr i=1. This does not matter, because the OT feature was rejected.
- Design-level selection: the features and grids of NASS were chosen after the analysts had looked at the same 43 transitions. There is no valid held-out raw-count build.
- LatentSB uses the median cell count over all bins, including future ones. Under a causal fix its only significant claim goes from +0.016 (p=0.04) to +0.0002 (p=0.55).
- The judges did not re-run the signal analysts' decompositions, except the l1 noise budget and the OT-increment span.
- Averaging conventions differ between sources (`all` / i>=1 / i>=2 / i>=3), and cross-table differences carry about ±0.01. On linear the uncertainty is ±0.05-0.10.

## 4. Next steps for the HPC run inside their harness

How the harness works: `tack/hpc/dileo_harness.py` runs our forecaster inside their untouched loop (`~/tgl-grn-orig`). Their code draws the 25 % test masks and does the scoring. Output per forecaster and seed is `seed<s>.json` plus `seed<s>.npz` (predictions and test_idx).

The forecaster interface is a `task2_forecasters.REGISTRY` entry:
- `predict(hist, ts, ctx)` returns a vector of shape `(N,)`;
- `hist` holds the raw node features (mean, median, std, frac_nonzero, sum) of snapshots ≤ t;
- a strict forecaster gets `ctx = {}`, i.e. no labels and no edges;
- its `.target` must match `--target`.

Steps:

1. **Port NASS.** It is not in the registry: `nass.py` only writes score tables and saves no predictions.
   - Add a new module `tack/hpc/nass_fc.py` with entries `nass_orig`, `nass_l1`, `nass_linear` and a fixed-gain variant `*_h1`. Hook it into the registry with one try-import line, as `dist_bridge` is.
   - Rebuild every feature and every past target from `hist` only.
   - No GRN and no OT feature.
   - Use numpy RNG only, never torch: their masks are drawn from torch's RNG.
   - Align the i=0/1 fallbacks: in their loop, `hist[0]` is gastr bin 2.
2. **Local gate**, with no torch.
   - Call the ported `predict` on the local node arrays. It must match `design_best-signal/results.json` to within about 0.005 per triple.
   - Re-run the truncation test from `judge1/strict_nass.py`: replace bins > k with garbage; the prediction at k must be identical.
3. **Pre-register** with `pos decide` and a frozen file hash.
   - Primary: l1, NASS full with its 20-config rolling-origin grid. Secondary: h=1 fixed.
   - Orig is reported for comparability. A tie with DistBridge is expected; claim a win only if the paired p is below 0.05.
   - Linear is reported only with its mask sd and the library decomposition.
   - Make no OT claim.
4. **Jobs.** QoS stud allows 1 running and 2 submitted jobs, and no arrays, so chain the jobs with `--dependency=afterany`. Use seeds `1234,42,7,0,1`.
   - a. `TARGET=l1 MODELS=nass_l1,nass_l1_h1,trend_l1,distbridge_strict_l1,distbridge_strict_nosb_l1`. This gives the first paired with/without-OT l1 number. Raise `--time` above 6 h, because DistBridge with OT took over 55 min per dataset locally.
   - b. `TARGET=orig MODELS=nass_orig,obslevel4,distbridge_strict,distbridge_strict_nosb`.
   - c. `TARGET=linear MODELS=nass_linear,trend_linear,distbridge_strict_linear,distbridge_strict_nosb_linear`.
   - d. Their models on the corrected targets (`dileo_corrected.sbatch`, `PROTOCOLS="preqA preqB" TARGETS="l1 linear"`): check the logs first. Finished (model, seed) files are never recomputed, so resubmit only if the job is gone.
   - Also port the recency-weighted per-gene filter as a comparator, because on gastr l1 NASS only ties it.
5. **Score** with `dileo_rescore.py`. Our predictions are scored on their test genes for the same seed, so the comparison is paired gene by gene. Report:
   - `all`, i>=1 and i>=2;
   - tie-aware P@200 (`p_ta`) and its ceiling `p_max`, next to their argsort P@200;
   - `pearson_nan0`;
   - with and without the bins under 100 cells (pancr 3->4..7->8; gastr 2->3 and 28->29..30->31);
   - Wilcoxon over transitions, and mean ± sd over seeds.
6. **Held-out.** Only a fresh raw-count build, binned the same way and never looked at, answers the design-level selection risk. The mouse builds do not qualify. Freeze NASS before building it.
7. **Record**:
   - `pos attempt` entries: FAILED for per-gene OT, population OT and latent SB; CANDIDATE for NASS;
   - `pos checkpoint`;
   - the shared-ledger entry.

   No subagent wrote any of these.

Key files:
- `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\signal_analysis\design_best-signal\nass.py`
- `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\signal_analysis\judge1\verdict.json`
- `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\signal_analysis\judge2\judge2_verdict.json`
- `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\hpc\dileo_harness.py`
- `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\hpc\task2_forecasters.py`
- `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\hpc\dileo_rescore.py`