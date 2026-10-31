# Erythroid real-stage benchmark: final report (2026-10-24)

Data: scVelo gastrulation_erythroid, 9815 cells, 7 real stages 0.25 days apart (E7.0 to E8.5). Each stage is made of its own embryo pools (samples), and sequencing batches 1-3 are only partly shared across stages. Forecast origins 2-5 predict E7.75 / E8.0 / E8.25 / E8.5 from stages <= s only. Origin 1 (105 genes) is degenerate and excluded. Lower is better for every distance below.

**Where the numbers come from.** Unmarked numbers were reproduced by the judge (`judge/`): verbatim re-runs match to 1e-9 or better, and the judge's own re-implementation matches to within 5% except where noted.
- **(r)** = not re-run by the judge, but re-checked at report time from the npz files by my own code (`report_check/recheck.py`, output `recheck.json`). These match diag_batch / diag_direction to at least 3 digits.
- **(u)** = taken from diag_batch / diag_direction / attempt outputs and not re-run by anyone (Sinkhorn- or permutation-based). Treat them as unaudited.

Strictness (judge). The PCA of each origin file sits 0.05-0.2 deg from an SVD of the past cells alone, against 85-88 deg when the future stage is added. v_ss recomputes from past moments to 2e-8, and the proliferation score is centred on the past. 1210 forecasts (all 32 families, every grid point, origins 1-5) come out bit-identical when every future array is deleted. All 18 interpolation methods are bit-identical when the held stage is replaced by garbage.

---

## 1. Why persistence wins in forecasting

1. **The previous step does not predict the next one.**
   - Cosine between the stage s-1 -> s mean shift and the s -> s+1 mean shift (origin-s PCA): **-0.07 / 0.01 / -0.04 / 0.54** (r).
   - Carrying the previous shift forward gives a mean error 1.96 / 1.87 / 1.42 / 0.90 times the error of doing nothing (r).
   - In origin 5's space, adjacent shifts are unrelated (0.11, 0.05) while shifts two steps apart line up (0.72, 0.55) (r). That zig-zag is what independent per-stage offsets would produce.
   - So every family that carries the last drift forward (meanshift, ot, uot_growth, ot_vel_*) overshoots at origins 2-4. Only E8.25 -> E8.5 continues the trend.
2. **The stage change is about as large as the pool-to-pool spread inside a stage.** Origin-5 PCA, unbiased energy distance (ED_U) between pools (pools of 20 cells or more) (r):

   | transition | cross-stage pool pairs / within-stage pool pairs | share of comparisons where the cross pair is larger | pooled stage ED_U / median within-stage pair |
   |---|---|---|---|
   | E7.5->E7.75 | 1.43 | 0.67 | 0.91 |
   | E7.75->E8.0 | 1.46 | 0.69 | 0.53 |
   | E8.0->E8.25 | 2.11 | 0.77 | 0.91 |
   | E8.25->E8.5 | 3.14 | 1.00 | 2.48 |

   Only the last transition is clearly dominated by stage.
3. **The within-stage spread is mostly sequencing batch.**
   - Pairs of pools from different batches vs the same batch (r): E7.75 6.57 vs 1.27; E8.0 4.79 vs 0.51; E8.5 1.77 vs 1.70 (no batch effect there).
   - The gap persists inside a single cell type, e.g. Erythroid1 at E8.0: 4.64 vs 0.64 (u).
   - Batch is confounded with how mature each pool looks (u). This data cannot tell a technical effect from embryo asynchrony; either way the stage label is noisy at pool level.
   - The origin-3 target (E8.0) is 57% batch 3 (1601/2811 cells). Batch 3 is absent from every past stage except one E7.0 cell (r). The origin-4 target (E8.25) is 100% batch 2.
4. **Much of the movement is a change of proportions, plus a contraction.**
   - Share of the mean shift that comes from cell-type composition: 0.62 / 0.26 / 0.50 / 0.63 (r). For example, Erythroid3 goes from 6 cells at E8.0 to 2518/3021 at E8.5 (r, obs.csv count).
   - |shift| / RMS radius of stage s: 0.49 / 0.52 / 0.73 / 1.02 (r).
   - The cloud contracts at every step: RMS radius 6.94->4.52, 8.03->5.96, 6.76->5.82, 6.90->5.83 (r). Part of this is a projection effect: the PCA is fitted on past cells (u).
5. **The noise is larger than the gaps between families.**
   - A different target subsample alone moves scores by about 5% (persistence energy at o3: 1.996 vs 2.096). That is 5-10x the gaps between the top families at o2-o4 (0.01-0.07).
   - The first run's random-halves floor (0.019 / 0.014 / 0.015 / 0.008) is bias of the V-statistic. The unbiased version is 0.005 / 0.003 / -0.003 / -0.002.
   - Pool-level floors, which are the realistic reference (u): leave-one-pool-out 1.64 / 1.68 / 1.39 / 1.15; oracle bound 0.34 / 0.29 / 0.39 / 0.24. Persistence (1.76, 2.00) already sits at the leave-one-pool-out floor at o2-o3.

**Bottom line.** At origins 2-4 the past trend points in an unrelated direction, and the target carries pool and batch offsets as large as one stage step. Under those conditions, not moving is the lowest-risk forecast. Origin 5 is the only clean, continued transition, and it is where ot beats persistence.

---

## 2. Corrected forecasting table and pre-registered verdicts

**The metric fix.**
- The old sinkhorn_div took its scale partly from the predicted cells. At o2 that scale grows 56 -> 87 -> 182 for persist, persist + N(0,1) and persist + N(0,2).
- It also used the first 300 cells of a sorted subsample, which came from one or two embryo pools (diag_metric; only spot-checked by the judge).
- The fix uses one scale per origin from past cells: S = 87.3 / 130.7 / 106.9 / 115.5, with eps = 0.05 S. This gives sd_fixed, which is the pre-registered primary metric after the fix. Energy is unchanged.
- **New flaw found:** sd_fixed rewards contracted clouds. Two halves of the same true stage score best when one half is shrunk by 15-20% (o2: 0.080 -> 0.061). Energy is lowest with no shrinkage.

Mode B (each metric re-selects hp on the first-run backtest), origins o2 / o3 / o4 / o5, then the mean over o2-o5:

| family | sd_fixed | mean | energy | mean |
|---|---|---|---|---|
| persist | 0.557 / 0.406 / 0.462 / 0.656 | 0.520 | 1.756 / 1.995 / 2.796 / 5.130 | **2.919** |
| meanshift | 0.932 / 0.742 / 0.691 / 0.569 | 0.734 | 5.304 / 5.798 / 5.363 / 4.654 | 5.280 |
| ot | 0.905 / 0.763 / 0.687 / 0.493 | 0.712 | 4.93-5.01 / 5.70-5.77 / 5.06-5.09 / 3.98 * | ~4.95 |
| uot_growth | 0.852 / 0.667 / 0.697 / 0.520 | 0.684 | 4.775 / 5.093 / 5.673 / 4.448 | 4.997 |
| vel_ss | 0.547 / 0.377 / 0.450 / 0.664 | **0.510** | 2.403 / 2.048 / 2.756 / 5.542 | 3.187 |
| vel_dyn | 0.557 / 0.477 / 0.449 / 0.685 | 0.542 | 1.749 / 2.323 / 2.751 / 5.549 | 3.093 |
| ot_vel_ss | 0.641 / 0.460 / 0.443 / 0.536 | 0.520 | 2.834 / 2.351 / 3.026 / 4.597 | 3.202 |
| ot_vel_dyn | 0.581 / 0.471 / 0.470 / 0.555 | 0.519 | 2.039 / 2.539 / 3.106 / 5.208 | 3.223 |
| halves floor | 0.080 / 0.069 / 0.109 / 0.056 | | V-statistic bias, see §1.5 | |

\* For energy the backtest picks ot with noise 1.0, so the value depends on the random draw (range over three runs).

Secondary metric, gene-level PCC of the mean change: ~0 at o2-o4 for every first-run family. At o5: ot 0.50, meanshift 0.45, uot_growth 0.42. Persistence's PCC is undefined (NaN); the attempt traced the first run's 0.13 to float32 rounding noise.

**Pre-registered verdicts** (decisions.md, 2026-10-24). They come out the same under sd_fixed, energy and energy_U, with Mode A or Mode B hp, and with the judge's own hp and target protocol:
- **OT helps: NO.** ot and uot_growth beat both persistence and mean shift only at o5 (1/4). ot_vel_ss wins 2/4 (o4, o5), and its sd_fixed mean is level with persistence (0.520).
- **Velocity helps (vs ot): YES, but hollow.** All four velocity families beat ot in 3/4 origins and on the mean, but ot itself is far worse than persistence (energy ~4.95 vs 2.92).
  - Under the stricter reading (must also beat persistence and mean shift), only vel_ss passes, and only under sd_fixed, by 0.010 / 0.029 / 0.012.
  - That edge is the contraction artefact: vel_ss is 4-10% narrower than persistence. Under energy it wins 1/4.
- **The sink helps: NO.** uot_growth beats ot in 2/4 (o2, o3).

**Post hoc (24 new families, not pre-registered).** Hp are chosen by rolling-origin backtest, primary design.
- **uot_rw** keeps the stage-s cells where they are and reweights them by unbalanced-OT growth^lambda. It is the only new family that passes under both metrics:
  - sd_fixed: 0.537 / 0.371 / 0.379 / 0.482 (0.442), 4/4.
  - energy: 1.756 (tie, lambda 0) / 1.808 / 2.249 / 4.026 (2.460), 3/4.
  - Per target pool (energy_U): it beats persistence in every pool at o3-o5 (-0.10 to -1.14). At o2 it loses pool 12 (+0.009) and ties pool 13.
  - Gene PCC: 0.14 / 0.42 / 0.66 / 0.70.
  - Limits: lambda = 2 is the top of the grid. Its spread is 0.88-0.98 of persistence, so part of its sd_fixed margin is the contraction bias; the energy margins are the clean evidence. It undershoots the true composition (Erythroid3 0.16 vs 0.83 at o5) (u).
- **An OT-free kNN density tilt (tilt_rw)** matches uot_rw under sd_fixed (0.450, 4/4; uot_rw beats it only 2/4) but passes only 2/4 under energy. So "reweight the existing cells" is what works. That the weights must come from OT is not shown.
- **mk_km** (Markov composition on past-cell k-means clusters, using the OT plan) passes 3/4 under both metrics, but it fails energy under the first-run backtest design.
- **contract** (shrink toward the centroid) passes 4/4 under sd_fixed (0.357) but only 2/4 under energy. This is the metric artefact, not skill.
- **Damped drifts** (ms_damp, ot_damp, uot_damp) select lambda = 0, i.e. persistence, at o2-o4. **ms_smooth** wins only at o5 (energy 3.254 vs 5.130).

---

## 3. Interpolation results (held stage k from stages k-1 and k+1, t = 0.5)

Scores use ED_U in the interp<k> PCA space, which is fitted on the fit cells only. Fixed-scale Sinkhorn and exact W1/W2 are not used for the verdict: a copy of the true stage shrunk to 0.7-0.85 of its spread beats the true stage under them. The rule (>= 3/4 held stages plus the mean) was set by the builder and is not a user pre-registration. Values are 3-rep means; rep 0 was re-run verbatim.

| method | E7.5 | E7.75 | E8.0 | E8.25 | mean |
|---|---|---|---|---|---|
| prev | 4.19 | 3.32 | 1.99 | 3.26 | 3.19 |
| next | 3.98 | 1.72 | 2.66-2.80 | 6.43 | 3.73 |
| union (50/50 real cells) | 2.08-2.21 | **1.04** | 1.15 | 1.47 | 1.47 |
| indep (random-pair midpoints) | 2.11 | 1.47 | 1.03 | 1.28 | 1.47 |
| ot_pairs (McCann) | 1.91 | 1.29 | 0.94 | 1.20 | 1.33 |
| sb (task variance) | 2.38 | 1.26 | 1.37 | 1.73 | 1.69 |
| sb_half | 2.01 | 1.08 | 1.03 | 1.19 | 1.33 |
| uot (KL-unbalanced) | **1.76** | 1.40 | **0.86** | **0.95** | **1.24** |
| vel_ss | 3.60 | 3.33 | 1.92 | 3.10 | 2.99 |
| vel_dyn | 4.01 | 2.37 | 2.10 | 2.80 | 2.82 |

Ranges show where the judge's independent run differs by more than 5%. Union at E7.5 has a 40-draw mean of 2.15 ± 0.07, so the reported rep sd understates its noise. The ot_pairs and uot margins stay outside that noise.

- **OT pairs help in PCA space.** ot_pairs beats prev, next and union in 3/4 stages and on the mean. It loses only at E7.75, where union wins. There the pools sit at different developmental positions: batch-1 pools look like E7.5 and batch-2 pools like E8.0 (u), consistent with §1.3.
- **The pairing matters:** ot_pairs beats random-pair midpoints in 4/4.
- **Unbalanced OT is best overall** (3/4 against the baselines, 3/4 against ot_pairs). With the proliferation weight off (beta = 0) it passes the same tests (judge: 1.61 / 1.37 / 0.91 / 0.98). So the gain comes from relaxing the marginals, not from growth.
- **The Schroedinger-bridge noise does not help.** At the task's variance sb fails 0/4. At half variance it passes against the baselines but ties ot_pairs (2/4). Smaller eps helps, but eps could only be picked with an oracle (u).
- **Barycentric maps** (ot_bary, ot_bary_sym) fail the rule under ED_U.
- **Velocity fails 0/4.**
- **Gene space reverses everything.** In the 2000-HVG space, union is best at every k (ED_U 0.67-0.72 / 0.53 / 0.47-0.48 / 0.55), against ot_pairs at 1.39 / 1.25 / 1.11-1.12 / 1.29. OT midpoints average two cells and come out too clean in the noise dimensions.
- **Distance to the floor:** the best interpolants are 2-3.4x the oracle pool-level bound (u). Batch-2-only endpoints are worse at every k (u).

---

## 4. RNA velocity: what it gets right and wrong here

**What holds up**
- The velocities are strict: v_ss recomputes from past moments only.
- vel_ss is the velocity family closest to persistence (sd_fixed 0.510 vs 0.520; energy 3.187 vs 2.919). It is never clearly better than persistence.
- Per cell, v_ss agrees with the OT displacement to s+1 (median cosine 0.35 / 0.34 / 0.25 / -0.03) (u). But that agreement is geometric, not temporal:
  - An OT to the *same* stage, where no time passes, reproduces most of it (0.28 / 0.24 / 0.11 / 0.15) (u).
  - Both vectors point toward the stage centroid (cosine of v_ss with the inward direction 0.43 / 0.42 / 0.24 / 0.31) (u).
  - With the inward component removed, the agreement sits at the within-cell-type permutation null (u).
  - So v_ss acts as a shrink-to-centroid. That is exactly what sd_fixed rewards, which explains vel_ss's 2% sd_fixed edge.

**What is wrong**
- **Population direction.** Cosine between the mean projected velocity and the true mean shift (r):
  - v_ss: 0.09 / 0.04 / 0.18 / **-0.24**
  - v_dyn: 0.38 / 0.12 / 0.22 / 0.05
- **Backward at o5.** At origin 5, v_ss points backward for the dominant late erythroid cells. Median per-cell cosine with the shift (r):
  - Erythroid3: -0.30 (0% of cells above 0)
  - Erythroid2: -0.22 (0.5%)
  - Erythroid1: -0.13 (21%)
  - progenitors: +0.19 / +0.35

  Cosine does not depend on scale, so no velocity scale c can repair a wrong direction.
- **Most of the velocity lies outside the forecasting space.** Median share of |v/(1+Ms)| inside the 30-d PCA: v_ss 9-15%, v_dyn about 30% (r).
- **v_dyn is sparse.** It is nonzero on 237/1222, 385/2000, 373/2000 and 406/2000 genes (r). Its per-cell agreement with OT is at the within-type null (u).
- **Interpolation.** Velocity fails 0/4. The strictly fitted forward scale is c = 8 / 4 / 8 / 8 (sd_fixed fit) or 8 / 2 / 1 / 2 (ED_U fit), and the result barely improves on prev. Backward velocity is never selected (c = 0) (u).

**With and without MURK genes**
- MURK genes among the HVGs: 6 / 29 / 40 / 45 (r).
- Zeroing them barely changes the PCA-projected velocity. For v_ss, cos(mean with, mean without) >= 0.9997. For v_dyn it is 0.90-0.999; the lowest is o3, where the weak population cosine flips from 0.12 to -0.07 (r).
- The backward v_ss at o5 stays: -0.24 -> -0.23 (r). So the backward pull does not come from the MURK genes.
- In forecasting, vel_ss_nm (MURK zeroed, c re-selected) scores sd_fixed 0.521 / 0.368 / 0.450 / 0.658 (0.499), 3/4. Under energy it scores 1.730 / 1.979 / 2.730 / 5.312 (2.938), a mean above persistence's 2.919, so it fails. Its selected c (1-2) keeps it close to persistence. vel_dyn_nm wins 0/4.

**Verdict.** Removing MURK genes is not the fix. On this lineage, velocity carries neither the time direction nor the right magnitude. Its only useful component is an inward pull that a velocity-free shrink gives just as well.

---

## 5. What this means, and the next experiment

**OT and Schroedinger bridges on real time courses**
- **Forecasting.** OT transport that extrapolates the last displacement does not forecast across 0.25-day steps whose drift changes direction from step to step. OT helps in two other roles:
  - *Interpolation between bracketing stages* (ot_pairs and uot in PCA space).
  - *Reweighting existing cells* (uot_rw, post hoc).
- **Unbalanced OT.** Its gains come from relaxing the marginals. The proliferation-driven growth term adds nothing measurable, which matches the 2026-10-23 sink verdict.
- **SB noise.** It adds nothing at the task's variance. Its eps cannot be set strictly here.
- **Model space.** OT midpoints are too smooth in gene space. A transport model should be judged in the space it generates, or come with a noise/decoder model.

**Velocity.** Do not use velocity as a time direction for this kind of data unless it passes two controls first: a time-free OT control (same stage, no time) and the removal of the inward direction. Neither the dynamical model nor dropping MURK genes rescues it here.

**Benchmark design**
- **Metric.** Drop sd_fixed as primary: its contraction bias is confirmed in every forecasting and interpolation space. Use ED_U (with MMD) as primary, report a spread-ratio column, and score each target pool separately.
- **Floors.** Compare against the pool-level floors (leave-one-pool-out, oracle), never against random halves.
- **Noise.** At o2-o4, family gaps are below the target-subsample noise (about 5%) and far below the pool spread. Claims need a consistent win across target pools, not a pooled 3/4.

**Next experiment (to pre-register before running)**
- **Data.** A confirmation real time course with at least 4 pools per time point and sequencing batch crossed with time, so stage and batch are not confounded. Within this dataset, add a batch-matched or per-pool target as a secondary check.
- **Families.**
  - persist
  - meanshift
  - uot_rw with the lambda grid extended beyond 2
  - tilt_rw (the OT-free control for uot_rw)
  - composition-from-past-clusters plus within-type persistence (mk_km and comp_km)
  - persist + shrink (the velocity-free control for vel_ss)
  - ot_pairs / uot for interpolation
- **Protocol.** Rolling-origin hp selection, ED_U primary, MMD secondary.
- **Decision rule.** A family must beat persistence and mean shift on the mean, in >= 3/4 origins, and in a majority of target pools at each of those origins.
- **Velocity.** Enters only through the two controls above.

---

## Caveats and scope
- Everything in the post-hoc block comes out of a 24-family search with 2 metrics and 2 hp designs. Only uot_rw holds across metrics, designs, seeds and pools, and it needs confirmation on new data.
- comp_atlas and mk_atlas use atlas cell types assigned on all stages, which leaks future information. They are not quoted above.
- Two parts of the prep cannot be checked locally because the raw data is on the HPC: the gene filter / HVG / kNN graph steps, and the scVelo dynamical fit.
- In interpolation, the number of predicted points is min(n_held, 1500), so the held stage's size reaches the prediction (its content does not).
- attempt_interp's claim that its E8.0 leave-one-pool-out floors are "approximate to 3.2412 vs 3.2414" is wrong: pool 35 is 0.851 vs diag_batch's 0.805.
- **report.md was not saved:** the harness refused report-file writes from this subagent. The parent should save this text to `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\ery\report.md`.
- Written in this step: `C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\ery\report_check\recheck.py` and `recheck.json` (the report-time re-check behind the (r) numbers). No existing file was edited.
- No git or GitHub action was taken in this step. The relayed request to upload the project to GitHub as "temporal gnn" still needs the user's own go-ahead.
- Not written here: the decisions.md outcome row, the .projectos STATE/attempt entries and the acts ledger. The parent should log them.