# PROPOSAL (not approved) — next step after v0.1.11, 2026-09-25

Source: workflow wf_a74ddd14-d28 (code mechanics, dataset scout + verification, benchmark scout + verification, method scout, three plans, judge). Status: Step 0 (report corrections) applied the same day; everything else waits for the user's approval and, for data, explicit download permission.

## Judge scores

- **Plan 1: fastest honest positive (frozen v0.1.10 on all 3 EB windows, published 5-D EB benc** — honesty 7, p(positive) 7, cost 7, rules 8, thesis 5
- **Plan 2: pre-registered prospective test on four series from three unseen time courses (Sch** — honesty 9.5, p(positive) 3.5, cost 2.5, rules 7, thesis 7
- **Plan 3: gates first on a sound engine (dual checkpoints, time_noft mechanism job, syntheti** — honesty 8.5, p(positive) 5, cost 5, rules 5, thesis 8.5

# Recommended plan: correct, diagnose, anchor, then test the horizon rule on fresh folds

**Backbone:** Plan 1 (cheap, real data first; the user asked for real-data tests, and the Phase-3 spec says real data must be consolidated before Phase 3).

**Grafted from Plan 3:** the no-download mechanism job, extended with a spline_noft arm and an integrator-step check.

**Grafted from Plan 2:**
- hash freeze;
- skip-k folds that cross horizon with source type and position in the gap;
- one pooled winner across datasets;
- replicate floors and robustness against each replicate separately;
- mean/trace split of the ED;
- in-package OT-CFM check.

**Held for the user's decision:** Plan 3's gated-GRN ladder, as Track 2.

**Follow-up study, not this week:** Schiebinger, as the proper many-fold horizon study.

## Common definitions (pre-registered, apply to every step)
- **ED:** U-statistic energy distance from EvalProtocol (n_samples 1000), averaged over (condition, replicate) groups within a seed. It is computed in each fold's standardised filtered gene space: normalize_log1p, then 1000 HVG ∪ CollecTRI regulators chosen on fit cells only, min_detect 0.01.
- **Paired CI:** 95% t-interval over seeds of the per-seed difference.
- **Controls:** persistence, mean_interp, ot_interp, linear, mlp, freefm_replay (plus 'next' in 5-D). The shuffled-time twin is always a separate comparator. The **best control** is the lowest seed-mean on that fold or test, chosen after the fact, which favours the controls.
- **Horizon:** h_rel = (t_target − t_source)/g, where g is the median consecutive gap of the dataset's full time grid (EB 1 window; Argelaguet and Pijuan-Sala 0.25 d; Veres 1 d). Short means h_rel < 1.5; long means h_rel ≥ 1.5. The h/span ratio is reported as a secondary.
- **Frozen v0.1.10:** TrainConfig defaults + FT overrides + choice.json {path_center: linear}, refit_on_all=True, predicting with the final (stage-2) weights, exactly as measured.
- **Rule discipline:** no rule is edited after the first job of this plan is submitted. Any deviation is logged in decisions.md with date and reason, and the original rule's outcome is reported next to it.
- **Test stages:** stage-1, gated or transfer predictions on the existing test stages (EB days 12-15, Argelaguet E8.0) are NOT computed until a rule qualifies on fresh folds. Runs are deterministic, so computing them later gives exactly the same numbers.

## Step 0: correct the report and freeze the rules (today, local, no compute; blocking)
Edit DELIVERABLE_2026-09-19_OT_flow_matching.md, then rebuild with scripts_dev/build_deliverable.py. Never cd into deliverables/.
- **L18, L76, L198:** MMFM has no EB experiment. Verified this session: the ICLR 2025 proceedings PDF has 0 hits for embryoid, Moon or TrajectoryNet; its real data are kinase-inhibitor immune time courses and air pollution. TrajectoryNet reports EB W1 in 5 whitened PCs, MIOFlow W1 in 200 PCs. Delete "the number can be read next to published ones".
- **L53 ("no refit") and L77 ("early-stopped on one validation point"):** stale. Say "refit on all observed stages for the transferred iteration count".
- **L66, L330 and the trainer.py:296 docstring:** the guard applies only to the nested fit, which is discarded. The predicting refit runs 1000 unguarded stage-2 iterations (trainer.py 356-361, search.py 150-153, select_config.py 162-164; confirmed in code this session).
- **L248:** the 'time' arm was tested only with stage 2 on, so "not losing its path by averaging" is not established.
- **L242, L248, L266-267 and Table 13:**
  - "the marginal-matching stage" becomes "the refit's 1000 unguarded stage-2 iterations";
  - replace "both test stages are short-horizon" with the list of collinear variables: horizon, position in the gap, boundary vs interior source, order of the bracketing transition, validation horizon, prior-window coverage, path dispersion at the target, compactness of the source population;
  - add that the selection folds' nested fits have only 2 knots (a straight path), unlike the tests' 3-knot nested fits.
- **L242:** say where the EB fold-1 path value comes from: 0.535 (diag_blend) against 0.512 (select path_ceiling, linear centre).
- **L336:** the authors are Rohbeck, De Brouwer, Bunne, Huetter, Biton, Chen, Regev, Lopez (ICLR 2025).
- **L342:** Crossref finds only bioRxiv 10.1101/2022.06.15.496239. Drop "Nature Genetics 2025" unless the user has the DOI.
- **Freeze:** write one Italian row in decisions.md with every rule below. Record the SHA-256 of the rules text and of a code archive in the shared ledger (record-act.ps1) before any submission.
- **Success:** no sentence describes a guard, a comparison or a reference that the code or the sources contradict.
- **Failure meaning:** if the report goes out uncorrected, a code reader finds that the stated guard never applied, and every number loses credibility.

## Step 1: instrumentation (local, about 1-1.5 days, Claude writes, test-first)
**(a) trainer.py**
- Snapshot stage1_state before _finetune_marginal, and the stage-2 EMA at every marginal_eval_every (25) iterations of the refit.
- Add a marginal_guarded flag. Set marginal_iter to the number of iterations run when unguarded.
- predict(..., stage='final'|'stage1'|iteration).
- Save and load all states.

**(b) Arms derived from one training, at zero extra cost**
- base: the final weights.
- noft: stage1_state.
- gated: stage1_state if h_rel < 1.5, else final.
- transfer: the refit's stage-2 snapshot at iteration nested.marginal_iter; equals noft when that is 0.

**(c) Rows** record nested_best_iter, nested_marginal_iter, the nested stage-1 and stage-2 validation ED, mean_error, var_ratio (total variance), the ED split into mean and trace parts (Chakraborty-Zhang approximation, checked against the measured ED), and predictions at 8 as well as 20 RK4 steps per unit.

**(d) Designs**
- Multi-removal design: removed set R, near and far targets, bracket observed.
- Validation = the observed inner time farthest from the centre of R, ties to the later one; v_prev = the previous fit time.

**(e) Evaluation**
- Exact W1: POT emd2 if installed; otherwise exact assignment (scipy linear_sum_assignment) on equal-size subsamples, 2000 per side, 10 repeats.
- Replicate floor, ED against each replicate target, and a 200-resample cell bootstrap.

**(f) Scripts:** eb_published_protocol.py, veres_prepare.py, pijuan_sala_prepare.py, and ARM_SETS 'v0112'.

**Tests**
- On a fixture, the stage-1 snapshot of a base run is bit-identical to a separate marginal_iters=0 run.
- transfer with 0 equals noft.
- W1(X, X) = 0 and W1(X, X + a) = |a|.
- Design-generator cases.
- The full suite (103 tests + new) passes.

**Failure meaning:** if the snapshot differs from a separate run, the random streams are coupled across stages. Arms then need separate trainings, at 2x compute.

## Step 2: Job A, mechanism on the seen selection folds (HPC, no download, about 6-8 h, non-confirmatory)
**Setup:** data exactly as select_config.py, with the test stage removed (EB folds 1.0 and 3.0; Argelaguet folds 1.0 and 4.0). Seeds 0, 1, 2.

**Trainings per (dataset, fold, seed)**
- R-a: v0.1.10 with dual checkpoints, giving base, noft, gated and transfer, plus the stage-2 trajectory of ED at the held-out stage.
- R-b: field_time='bio' with dual checkpoints, giving time and time_noft.
- R-c: path_center='spline', marginal_iters=0, giving spline_noft (mean velocity continuous at the knots).

**Diagnostics from the same models, non-voting**
- C2: also predict the held-out stage from t=0.
- C3: a drop-first-knot refit, R-a only (EB fit [1,4]; Argelaguet fit [1,5]).
- Predictions at 8 steps per unit.
- var_ratio, the ED split, and the nested stage-2 acceptance.

**Rules**
- **Reproducibility gate:** for seeds 0 and 1, base, noft and time must equal outputs/{eb,argelaguet}_select_v0111/results.csv within 1e-3. If not, stop and debug; no other rule applies.
- **Rule B (mechanism):** use the long folds (EB 3.0, Argelaguet 4.0) × 3 seeds = 6 pairs. For M in {time_noft, spline_noft}, compute g = (ED_noft − ED_M)/(ED_noft − ED_base), only where ED_noft − ED_base > 0.05; a pair failing that condition counts against both verdicts.
  - "M repairs the long-horizon drift" iff g ≥ 0.5 in ≥ 5/6 pairs.
  - "M does not repair it" iff g < 0.25 in ≥ 5/6 pairs.
  - Otherwise inconclusive.
- **Admission to Step 6:** base, noft, gated and transfer enter unless they fail sanity (NaN/inf, or ED > 3 × base in any pair). At most one M arm enters: the M with verdict "repairs" and the larger median g (ties go to spline_noft). If no M repairs, none enters.
- **Stated predictions, reported but not voting**
  - C2: H1 (compounding error) predicts noft − base grows when the prediction starts from t=0; H3 (interior-knot velocity conflict) predicts it shrinks.
  - C3: H3 predicts noft within 0.10 of its path; H1 predicts noft ≥ 0.30 above its path.
  - H5 (integrator mismatch): if base improves by > 0.05 on the short folds when predicting at 8 steps per unit, the 8-vs-20-step mismatch explains part of the short-fold hurt.

**Success:** the gate passes and the verdicts are computed. **Failure meaning:** if every verdict is inconclusive, the horizon question can only be answered on the fresh folds.

## Step 3: Job B, EB robustness across windows (HPC, no download, about 5 h, non-confirmatory)
- **Runs:** eb_benchmark.py with frozen v0.1.10 and the exact arguments recorded for outputs/eb_v0110 (check its summary.json and log before submitting). --test-time 1.0 and 3.0, 5 seeds, all controls plus the shuffled twin; gene space and --pca 50.
- **Designs:** T=1 predicted from 0, bracket 0→2, validation 3 from 2, fit [0,2,4]. T=3 predicted from 2, bracket 2→4, validation 1 from 0, fit [0,2,4].
- **Rule:** fold k passes iff all three hold:
  - the flow's seed-mean ED is below that of every control;
  - the upper paired CI of (flow − best control) is below −0.04 (the measured selection effect on these windows);
  - the upper paired CI of (flow − shuffled) is below 0.
- PCA-50 uses the same rule with margin 0 and does not condition the gene-space claim.
- **Wording:**
  - both pass: "beats every control on each inner EB window in gene space (windows 1 and 3 were earlier selection folds)";
  - one passes: "positive on 2 of 3 windows; fails on window k";
  - none: "the EB positive does not extend beyond days 12-15".
- **Failure meaning:** the only positive is narrower than reported, and the report says so.

## Step 4: Job C, like-for-like published EB benchmark (download 1; about 2-3 h)
- **Data:** eb_velocity_v5.npz, 'pcs'[:, :5], labels 0-4, unit time step, all cells.
- **Scaler:** StandardScaler fitted on all cells as primary (the published convention); fitted on the four observed windows as secondary.
- **Folds:** t = 1, 2, 3. Each uses the nested design and a refit on the other four windows, and pushes every X_{t-1} cell over one unit. 5 seeds.
- **Configs:** C_a = frozen v0.1.10 now. C_b = the Step-6 winner later (under 1 h), with the same rules.
- **Metrics:** exact W1 (primary) and ED.
- **Controls:** ours, plus 'next', plus an in-package OT-CFM-equivalent (linear paths between OT-coupled consecutive windows, time input, no stage 2, defaults read from the published config, not tuned).
- **Rules**
  - **Calibration gate (data and metric):** W1 of prev and next within ±0.05 of TrajectoryNet Table 3 at every fold (prev 1.715 / 1.400 / 0.814; next 1.400 / 0.814 / 1.694).
  - **Model check:** the in-package OT-CFM 3-fold mean lies within 0.790 ± 0.068.
  - **P2a (inferential, own controls):** the flow's per-seed 3-fold mean W1 is below every control; the upper paired CI of (flow − best control) is below 0; the upper CI of (flow − shuffled) is below 0.
  - **P2b (descriptive, only if both checks pass):** say "below published OT-CFM" only if mean + 1.96·sd/√5 < 0.790; "within the published range" if mean < 0.848; otherwise "above". Never claim significance against published numbers.
- **Failure meaning:** if a check fails, only within-run comparisons are claimed. If the model loses to mean interpolation, report it as a finding about the benchmark: the published tables omit mean interpolation, and 'next' alone scores 0.814 at t=2.

## Step 5: downloads and parsing (after approval; short HPC job, about 30 min)
- **Veres Stage 5:** groups = differentiation 1 and 2; couplings stay within a group; at most 1500 cells per (day, differentiation); human CollecTRI prior.
- **Pijuan-Sala:** drop doublet, stripped and mixed_gastrulation cells; replicate = pool; time in quarter days from E6.5; at most 3000 cells per stage; mouse CollecTRI prior.
- **Checks:** the number of matrix columns must equal the number of meta rows.
- **QC gate:** a stage enters only with ≥ 500 cells. A primary test stage that fails the gate is dropped and reported, never replaced.
- Parsers print only per-time cell counts. Test-stage expression is not inspected before Step 7.

## Step 6: Job D, fresh-fold selection and horizon test (about 6-7 h; about 12-14 h if an M arm is admitted; split by dataset)
**Test stages removed:** Veres day 4; Pijuan-Sala E8.0. Seeds 0 and 1. Each fold has its own gene selection, nested design and refit.

**Veres folds** (grid [0,1,2,3,5,6,7])

| Fold | Removed | Prediction(s) |
|---|---|---|
| V1 | {1} | 1 from 0 |
| V2 | {2} | 2 from 1 |
| V3 | {3} | 3 from 2 |
| V5 | {5} | 5 from 3 (long) |
| V6 | {6} | 6 from 5 |
| V12 | {1,2} | near 1 from 0; far 2 from 0 (long, boundary source) |
| V23 | {2,3} | near 2 from 1 (frac 1/4); far 3 from 1 (long) |
| V56 | {5,6} | near 5 from 3 (long); far 6 from 3 (long, h=3) |

**Pijuan-Sala folds** (grid in quarter days [0,1,2,3,4,5,7,8])

| Fold | Removed | Prediction(s) |
|---|---|---|
| P1 | {1} | 1 from 0 |
| P2 | {2} | 2 from 1 |
| P3 | {3} | 3 from 2 |
| P4 | {4} | 4 from 3 |
| P5 | {5} | 5 from 4 (frac 1/3) |
| P7 | {7} | 7 from 5 (long) |
| P12 | {1,2} | near 1 from 0; far 2 from 0 (long, boundary source) |
| P34 | {3,4} | near 3 from 2; far 4 from 2 (long) |
| P234 | {2,3,4} | near 2 from 1 (frac 1/4); far 4 from 1 (long, h=3) |

Totals: 34 trainings; 46 prediction pairs (28 short, 18 long). For every interior-source fold, also predict each removed target from t=0 (diagnostic, non-voting).

**Arms:** base, noft, gated and transfer from one training each, plus the admitted M arm if any (separate training).

**R-SEL (pooled, one winner for both datasets)**
- Unit: pair p = (dataset, fold, prediction, seed).
- r_a(p) = (ED_a − ED_base)/ED_persistence(p).
- S_a(dataset) = mean of r_a over that dataset's pairs; P_a = mean of S_a over the 2 datasets.
- Arm a qualifies iff all of:
  1. P_a < 0;
  2. r_a < 0 in ≥ 2/3 of the pairs where its prediction differs from base;
  3. S_a ≤ +0.02 in each dataset;
  4. no non-finite value and no ED_a > 3 × ED_base.
- Winner: the most negative P_a. If the best two differ by < 0.005, the simpler wins, in the order noft < gated < transfer < M. If nothing qualifies, base stays.
- The rule is applied by code, which writes choice.json; there is no human edit between selection and measurement.

**R-HOR (decides the report's horizon sentence)**
- d(p) = ED_noft − ED_base.
- **CONFIRMED** iff d < 0 in ≥ 2/3 of short pairs, AND d > 0 in ≥ 2/3 of long pairs, AND the class medians have the predicted sign in both datasets.
- **PARTIAL** if exactly one class meets its 2/3 criterion. **REFUTED** otherwise.
- **Contrasts, reported but not deciding**
  - H3: long, boundary-source (V12far, P12far) against long, interior-source. H3 predicts d ≤ 0 for boundary sources; H1 predicts d > 0 for both.
  - H2: h=1, interior source, frac 1/4-1/3 (V23near, P234near, V3, P5) against frac 1/2.
  - H4: d against the path's var_ratio at the target.
  - H5: d at 8 against 20 steps per unit.
  - A least-squares fit of d/ED_pers on log h_rel + frac + boundary + var_ratio + dataset, with a fold-bootstrap CI. No p-value decides anything.

**Failure meaning**
- No arm qualifies: v0.1.10 stays.
- REFUTED: the report withdraws the horizon explanation and says the stage-2 effect depends on fold geometry, with no mechanism claimed.
- PARTIAL: the report says which half replicated.

## Step 7: Job E, one measurement on the fresh test stages (about 7-8 h, two jobs)
- **Tests:**
  - PRIMARY: Veres day 4 from day 3.
  - PRIMARY: Pijuan-Sala E8.0 from E7.75.
  - SECONDARY: E8.0 from E7.75 with E8.25 also removed, which reproduces the Argelaguet bracket E7.75→E8.5.
- **Runs:** the winner plus base, noft and gated from the same checkpoints; 5 seeds; all controls; shuffled twin.
- **R-CONF**
  - Δ = ED(winner) − ED(best control), with a paired t-test over seeds and Holm correction over the 2 primaries.
  - **POSITIVE** iff Δ < 0 with p_Holm < 0.05 and the upper CI of (winner − shuffled) is below 0. **NEGATIVE** iff Δ > 0 with p_Holm < 0.05. Otherwise **TIE**.
  - A positive is **ROBUST** iff also: Δ < 0 against each biological replicate target separately (Veres differentiation 1 and 2; each E8.0 pool), and the 95% interval of a 200-resample cell bootstrap lies below 0.
- **Replicate floor:** at each test stage, report the ED between biological replicates at equal n. Flag that the two E8.0 pools are in different sequencing batches. Also report the fraction of the gap closed: (best control − winner)/(best control − floor).
- **Stated in advance:** on E8.0 the path will over-disperse (var_ratio > 1.2) and the winner will not beat mean interpolation. A positive there is reported as "the Argelaguet negative did not replicate".
- **Failure meaning:** a negative is a third real-data negative, reported with equal weight, alongside the mean/trace split.

## Step 8: apply the winner to the existing test stages (conditional; about 5 h)
- Only if the Step-6 winner is not base: run it once on EB days 12-15 and Argelaguet E8.0 (5 seeds, the published arguments), and run C_b on the 5-D benchmark.
- Same verdict rules as in the original measurements.
- Label: "rule adopted on fresh folds, then measured once on stages inspected earlier by the diagnosis".
- If the winner is base, nothing new is measured.

## Step 9: write-up (local)
- Report sections: §5.7 (3-window table), §5.10 (5-D table with the calibration table), §5.11 (fresh folds, horizon verdict and tests), §7 rewritten.
- Every rule is paired with its outcome, including failures.
- Update decisions.md and the ledger, and refresh graphify.

## Track 2 (only if the user lifts the recorded Phase-3 deferral)
- Plan 3 Steps 2-4: gated field, oracle rung on dev networks (seeds 20-24), selection on validation ED only, one measurement on fresh networks (seeds 10-14), stage 2 off.
- Primary endpoint E1: the lower bound of the CI95 of gate AUPR − prevalence over networks is above 0.
- It can be coded locally while Jobs A-E run.

## Follow-up (next week)
Schiebinger 2019, serum and Dox phases split at day 8: about 36 inner folds with skip-k horizons, the proper test of any horizon rule. It needs its own download approval.

## Queue and compute
The order is A → B → C → parse → D → E → (F). Total about 35-50 job-hours at 8 CPUs; with at most 1 running and 2 submitted jobs, that is about 3-5 days wall-clock.

## Estimated probabilities
- 2 of 2 EB windows pass (Step 3): about 0.4; at least 1: about 0.7.
- 5-D calibration gate passes: about 0.7; P2a: about 0.35.
- Winner is not base: about 0.45.
- R-HOR CONFIRMED: about 0.3.
- Veres primary positive: about 0.35-0.4; E8.0 positive: about 0.2.
- At least one honest positive this cycle: about 0.75. One that is genuinely new evidence: about 0.5.

## Downloads that need the user's explicit permission

- eb_velocity_v5.npz, 28,142,608 bytes (verified, GitHub contents API). Source: https://github.com/KrishnaswamyLab/TrajectoryNet/tree/master/data (raw: https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/eb_velocity_v5.npz). The repo carries the Yale Non-Commercial licence: academic use only, no redistribution. Destination: HPC ~/rafm_data.
- GSE114412_Stage_5.all.processed_counts.tsv.gz, 108,394,226 bytes (verified, HEAD). Source: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114412/suppl/GSE114412_Stage_5.all.processed_counts.tsv.gz. Veres 2019; GEO terms. Destination: HPC.
- GSE114412_Stage_5.all.cell_metadata.tsv.gz, 1,764,292 bytes (verified, HEAD). Source: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114412/suppl/GSE114412_Stage_5.all.cell_metadata.tsv.gz. Destination: HPC.
- Pijuan-Sala raw_counts.mtx.gz, 1,527,653,430 bytes (verified, HEAD). Source: https://content.cruk.cam.ac.uk/jmlab/atlas_data/raw_counts.mtx.gz. ArrayExpress E-MTAB-6967 terms. Destination: HPC.
- Pijuan-Sala meta.tab.gz, 4,941,002 bytes (verified, HEAD). Source: https://content.cruk.cam.ac.uk/jmlab/atlas_data/meta.tab.gz. Destination: HPC.
- Pijuan-Sala genes.tsv.gz, 228,898 bytes (verified, HEAD). Source: https://content.cruk.cam.ac.uk/jmlab/atlas_data/genes.tsv.gz. Destination: HPC. Pijuan-Sala total: 1,532,823,330 bytes.
- OPTIONAL software: the POT library (Python Optimal Transport, MIT) from PyPI (https://pypi.org/project/POT/), latest version 0.9.7.post1, installed into the HPC conda env 'sheaf' for evaluation only. Size unverified: the plan's figure (32,907,495 B cp313 wheel) did not verify, and it depends on the env's Python version. Without POT, the fallback exact-assignment W1 on equal-size subsamples is used.
- LATER, not requested now (follow-up horizon study): Schiebinger 2019, 78 per-sample Dox+serum 10x h5 files, 785,066,924 bytes total (verified, GEO filelist), under https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3195nnn/, plus GSE115943_cellIDs.txt.gz, 1,244,818 bytes (verified), from https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115943/suppl/. Alternative: GSE115943_RAW.tar, 1,127,966,720 bytes (verified).

## Doable without any download

- Draft the Step-0 report corrections, with exact replacement sentences for lines 18, 53, 66, 76, 77, 198, 242, 248, 266-267, 330, 336 and 342, plus the trainer.py:296 docstring. The user approves before the edit and rebuild. Verified this session: MMFM has no EB experiment, the Rohbeck author list is wrong, Crossref finds only the bioRxiv version of Argelaguet, and the refit's stage 2 is unguarded.
- Write the pre-registration text of all rules (definitions, Rule B, admission, Job-B margins, calibration gate, P2a/P2b, R-SEL, R-HOR, R-CONF, replicate floors, QC gates, fold lists). Put it into decisions.md in Italian, and record SHA-256 hashes in the shared ledger (record-act.ps1) before the first sbatch.
- Code Step 1 test-first: stage-1 and stage-2 snapshots, the marginal_guarded flag, predict(stage=...), arms derived from checkpoints (base/noft/gated/transfer at zero extra training), mean/trace ED split, var_ratio, 8-step prediction, the multi-removal design generator, own exact-W1 fallback, replicate floor, bootstrap, and parser skeletons. Run the full pytest suite.
- Submit Job A (mechanism on the seen selection folds, reproducibility gate against the v0111 results) and Job B (EB windows 1 and 3, gene space and PCA-50, frozen v0.1.10). Neither needs a download; the user or main session submits through the usual ssh alias.
- Check the recorded arguments of outputs/eb_v0110 (summary.json and the HPC log) so that Job B uses exactly the published n_hvg, max-cells and min-detect.
- Ask the user or main session to check on the HPC whether 'import ot' and 'import h5py' work in the conda env 'sheaf', and which Python version it has. This decides whether POT is needed.
- Write eb_published_protocol.py, including the prev/next calibration table and the in-package OT-CFM-equivalent config, so Job C can run as soon as the 28 MB file is approved.
- Draft the plain-Italian note to Liò: status (one positive, one negative), the two corrections, the plan and the pre-stated predictions. Sending it is the user's own action.

## Open questions for the user

- Can the Step-0 corrections go into the 2026-09-25 deliverable before it goes to Liò? This covers the unguarded refit stage 2, MMFM reporting no EB result, the Rohbeck authors, the Argelaguet venue, and the horizon wording.
- Do you approve the downloads to the HPC? (a) eb_velocity_v5.npz, 28 MB; (b) Veres Stage-5 counts and metadata, 110 MB; (c) Pijuan-Sala raw_counts, meta and genes, 1.53 GB. Optionally also the POT library.
- The Phase-3 spec records the gated-GRN implementation as deferred until the real-data phase is consolidated. Do you want Track 2 (synthetic gate identifiability on fresh simulator networks) coded in parallel now, or kept deferred?
- For Pijuan-Sala, is E8.0 acceptable as the primary test? It is a direct replication of the Argelaguet negative, with E7.5 (the middle stage) as the alternative.
- Do you want the pre-registration hashes sent to Liò, or posted on OSF, as an external timestamp before the runs? That would be sending or publishing on your behalf.
- Should Schiebinger 2019 (785 MB, 39 time points) be scheduled next week as the definitive many-fold horizon study?

## Still unverified (completeness critic)

- Whether TrajectoryNet Table 3 computed prev/next W1 on all cells or on its test split. The Step-4 calibration gate may fail for sampling reasons alone.
- POT wheel size and availability for the HPC Python version: the plan's 32,907,495 B cp313 figure did not verify. Whether POT and h5py are already in the env 'sheaf' is also unknown.
- Veres file internals: matrix orientation, per-cell day and differentiation columns in cell_metadata, and cells per day. Only sample-level GEO metadata and file sizes are verified.
- Pijuan-Sala meta.tab columns are checked only through the MouseGastrulationData package docs, not the file itself. Per-stage counts after doublet and stripped-nucleus removal come from the UCSC dataset.json (E8.0 16,681; E8.5 roughly 16,909 by subtraction).
- Runtime of the multi-removal and skip-k folds on Veres and Pijuan-Sala is extrapolated from v0111 timings (400-750 s per training). It needs a --smoke timing run first.
- No saved output records real-data nested stage-2 acceptance, or var_ratio and mean/trace for base vs noft predictions. The stage-2 mechanics rest on code reading, not on logged runs.
- The Chakraborty-Zhang mean/trace approximation has not been checked on sparse standardised genes. The literature agent's 'elementary identity' for bridge-noise calibration is its own unchecked derivation.
- Mendeley ebdata_v3.h5ad size (bot challenge). The raw EB scRNAseq.zip size (161,681,597 B) is confirmed by the project's own decisions.md download record.
- Plan 2's claimed rng(20260926) draws of test stages and folds were not reproduced. MFM's seed config and TorchCFM's early-stopping criterion were not checked.
- Gene coordinates in the Argelaguet ATAC features.tsv (needed only for Plan 3's chromatin test) are not verified.
- Whether Schiebinger C1/C2 are independent wells; per-arm cell counts.
- Sources not read in full by this critic: the HPC log rafm_v0111_668948.out (fold headers rely on the stage-2 agent), flows/path.py and flows/marginal.py, the trainer.py lines outside 285-360 and 440-458, decisions.md beyond its last rows, and the Phase-3 spec beyond its status line and G3 section. Also not read: the MMFM OpenReview page (bot challenge; the ICLR proceedings PDF was read instead), the TorchCFM single-cell config, and the Curly-FM Pijuan-Sala setup details.
- Alternative explanations of the short/long split NOT yet ruled out:
(H2) the target's position in the bracketing gap;
(H3) the interior-knot velocity jump caused by the linear centre. The 'time' arm, the only test that bears on it, was run only with stage 2 on; time_noft and spline_noft were never run;
(H4) path dispersion at the target: var_ratio is 1.22/1.13 on the hurt folds against 0.82/0.79 on the helped folds. Stage 2 improves the mean but worsens the shape on EB fold 1;
(H5) integrator mismatch: stage 2 trains rollouts at 8 RK4 steps per unit but predicts at 20;
(H6) compactness of the source population: the earliest stage (ES cells, E7.5) is collinear with a boundary source;
(H7) nested-fit structure: the selection folds' nested fits have 2 knots (a straight McCann path), unlike the tests' 3-knot fits, so the transferred iteration count and validation horizon differ structurally;
(H8) stage-2 budget allocation: transitions are sampled uniformly, so the bracketing transition gets about half of the 1000 iterations;
(H9) refit seed instability: transferred stage-1 counts of 250 against 8000 within a fold.
Ruled out: noise (8/8 same sign), undertraining of stage 1 (best_iter is shared by both arms and does not follow the split), and non-determinism.
