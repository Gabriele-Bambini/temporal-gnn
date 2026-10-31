Judge step for the erythroid real-stage benchmark: everything reproduces and every forecast is strict. The three pre-registered forecasting verdicts come out the same under every metric and selection setting I tried: OT helps NO, velocity helps (vs ot) YES but hollow, sink helps NO.

**Scope.** I did only the judge step. Nothing was put on GitHub (no git init, remote or push); that still needs the user's own go-ahead. I wrote only inside `tack/results/ery/judge`, and a file-time check found nothing changed anywhere else in the project. I did not write the decisions.md, .projectos STATE/attempt or acts-ledger entries (task rule); the parent needs to log the 2026-10-24 outcome.

**1. Reproduction**
- **Verbatim re-runs, outputs redirected into judge/:**
  - Metric re-score (diag_metric): 224 values match to within 3e-13 relative, and all hyperparameter (hp) selections match.
  - attempt_forecast full grid, origins 1-5: 7056 values match to within 1e-9.
  - attempt_interp, rep 0: 432 values match to within 7e-15.
- **My own re-implementation** (own Sinkhorn, energy and fixed scale S; nothing imported from the audited code):
  - First-run families at the reported hp, against the reported Mode-B / Mode-A numbers: worst relative difference 1.5e-6. The exception is ot with noise 1.0, where my noise draw differs (up to 1.1%).
  - My own backtest picks the same hp in every case.
  - Post-hoc families (uot_rw, tilt_rw, mk_km, contract, ms_smooth): the rolling-origin hp are identical and values match to 2e-7.
  - Interpolation, means over 3 reps: within 5% except the two sampling-noise cases below.

**2. What does not reproduce, or is flagged**
- **Interp union at k=2, ED_U 2.078 vs reported 2.206 (5.8%).** Not a bug. Over 40 fresh draws the union scores 2.154 ± 0.071. The audited reps (2.205 / 2.206 / 2.206) happen to cluster, so the reported rep sd understates the noise. The ot_pairs (1.91) and uot (1.76) margins stay outside it.
- **Interp next at k=4, 2.662 vs 2.802 (5.0%).** A different 1500-cell sample of the E8.0 target.
- **Choice of target cells moves scores by about 5%.** With a different target subsample, persistence energy at o3 is 2.096 vs 1.996. That is 5-10x the gaps between families at o2-o4 (0.01-0.07). The comparisons are paired, so the verdicts do not change, but at o2-o4 the margins sit below this noise.
- **attempt_interp misstates its k=4 leave-one-pool-out floors.** It reports them as matching diag_batch to within 3.2412 vs 3.2414. Pool 35 is actually 0.851 vs 0.805 (5.8%) and pool 33 is 1.424 vs 1.393.
- **Scope limits.**
  - The flawed old Sinkhorn metric was only spot-checked (re-run with `--no-orig`; o2 persist 0.885 matches the reported value).
  - attempt_report.py was not re-executed; its rolling-origin selection and post-hoc rule were recomputed from scores_o*.json.
- **Not checkable locally:** the prep's gene filter, highly-variable-gene selection, kNN graph and scVelo dynamical fit, since the raw data is on the HPC.
- **Not detectable by truncation:** comp_atlas / mk_atlas use atlas cell types assigned on all stages (already labelled LEAK).
- **Interp prediction size follows the held stage.** The number of predicted points is min(n_held, 1500), so the held stage's size, not its content, reaches the prediction.
- **uot_rw per target pool (energy_U, evaluation only):**
  - At o2 with the sd_fixed hp it loses pool 12 (+0.009) and ties pool 13. This does not contradict the attempt's 15/15, which was under sd_fixed.
  - At o3-o5 it beats persistence in every pool (−0.10 to −1.14).

**3. Strictness**
- **Data files.** The PCA is fitted on past cells only:
  - The components sit 0.05-0.2° from the SVD of past cells alone, but 85-88° (largest principal angle) from an SVD that adds the future stage.
  - pca_mean is within 1.3e-4 of the past mean.
  - Z_past and Z_fut are rebuilt from the stored arrays to within 5e-4.
  - v_ss is recomputed from the past moments to within 2e-8, and gamma matches exactly.
  - cc_past has mean 5e-16, so it is centred on the past.
  - The interp files likewise use fit cells only (0.15-0.23° vs 83-88°).
- **Forecasts.** I built 1210 forecasts twice: once from the full origin file, once with every future array deleted, garbage targets, and obs truncated to stages ≤ s. This covers all 8 first-run families and all 24 attempt families, at every grid point and default hp, for origins 1-5. All 1210 are bit-identical, with no KeyError. S computed from the truncated file equals the reported S.
- **Interpolation.** All 18 interp methods, S and the velocity-scale fit are bit-identical when the held stage is replaced by garbage.

**4. Metric scale**
- **S depends only on past cells.** My S from all past cells (or 3000 of them) is within 0.9% of the reported 87.3 / 130.7 / 106.9 / 115.5.
- **sd_fixed no longer depends on the prediction.** For persist, persist + N(0,1), persist + N(0,2) and 30% contraction, sd_fixed uses the same S, and my implementation agrees with theirs to within 5e-8. The old scale moves with the prediction: 56 → 87 → 182 at o2.
- **Interp S comes from fit cells only.** Including the held stage would have moved it by 1-19%.
- **sd_fixed rewards contracted clouds (confirmed independently).** Half of the true stage scored against the other half does best at 15-20% contraction in every forecasting space (o2 0.080 → 0.061), and at λ 0.8-0.85 in every interp space. Energy (V and U) is lowest with no contraction.
  - Consequence: vel_ss's only extra win (beating persistence too) appears under sd_fixed alone, and vel_ss is 4-10% narrower than persistence. That win, and the 4/4 passes of contract / xfit under sd_fixed, are this artefact.

**5. Forecasting verdicts** (literal rule from decisions.md 2026-10-24, first-run families, origins 2-5). The answers are the same under sd_fixed, energy and energy_U; with Mode A and Mode B hp; with my hp or theirs; and under both subsample protocols.
- **OT helps: NO.**
  - ot and uot_growth win only at o5 (1/4).
  - ot_vel_ss wins 2/4 (o4, o5); its sd_fixed mean (0.520) is level with persistence.
- **Velocity helps: YES, but hollow.**
  - All four velocity families beat ot in 3/4 origins on the mean, but ot itself is far worse than persistence (energy mean about 4.9 vs 2.92).
  - Also beating persistence and mean shift passes only for vel_ss under sd_fixed, by 0.010 / 0.029 / 0.012, which is the contraction artefact above.
- **Sink helps: NO.** uot_growth beats ot in 2/4.
- **Post-hoc (not pre-registered).** uot_rw is the only new family that passes under both metrics:
  - sd_fixed: 4/4.
  - Energy: 3/4, with o2 an exact tie (λ = 0). The margins are unchanged under energy_U.
  - Caveats: λ = 2 is the top of the grid, and the result comes out of a 24-family search.

**6. Interpolation** (builder-set rule; ED_U is the metric to act on because of the contraction bias)
- **OT pairs vs baselines:** ot_pairs beats prev, next and union in 3/4 stages and on the mean; it loses at k=3 to union.
- **Pairing matters:** ot_pairs beats random midpoints 4/4.
- **SB noise does not help:**
  - sb at the task's variance fails everywhere (0/4).
  - sb_half passes against the baselines (3/4) but does not beat ot_pairs (2/4).
- **Growth adds nothing:** uot is best overall and passes 3/4, but uot with β = 0 passes the same tests.
- **Velocity: 0/4.**
- **Gene space reverses the result:** union is best at every k.
- **Under sd_fixed**, the 4/4 passes for OT and the barycentric maps reflect the contraction bias, and pairing then fails 1/4.

Files are in C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\ery\judge:
- summary.json (all comparisons, verdicts and the 11 flagged items)
- strict.json
- report.log
- strict.log
- judge_lib.py
- judge_forecast.py
- judge_interp.py
- judge_strict.py
- judge_extra.py
- judge_report.py
- rerun_attempt.py
- rescore_rerun\
- attempt_rerun\
- interp_rerun\
- indep\