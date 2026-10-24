---
title: "Optimal-transport conditional flow matching for time-resolved single-cell populations"
subtitle: "Engine, evaluation protocol, synthetic validation and two real time courses (deliverable for Prof. Pietro Liò)"
author: "Gabriele Bambini"
date: "25 September 2026"
---

# 1. Summary

This deliverable contains an original, tested implementation of conditional flow matching (CFM) with optimal-transport (OT) couplings for populations of cells observed as unpaired snapshots at successive times, together with a control-heavy evaluation protocol, the results of the first validation phase on synthetic data, and tests on two real single-cell time courses, one positive and one negative.

The engine learns a velocity field over gene-expression space from snapshots only (no cell tracking). Optimal transport enters in two places: (i) exact minibatch OT couplings between consecutive snapshots within each biological unit, and (ii) OT-chained pseudo-trajectories across all observed time points, through which a cubic spline is drawn so that the field is trained on curved, multi-marginal paths rather than on straight segments between pairs of snapshots.

Main result (synthetic gene-regulatory network, 40 genes, 500 cells per snapshot, seven time points, inner time point t = 3 held out, 5 seeds): the OT-spline flow model reconstructs the held-out population with an energy distance of 0.428 against 1.102 for the best control (linear interpolation of the means) and 1.171 for the McCann/OT displacement interpolant. The paired difference against the best control is −0.674 with a 95% confidence interval of [−0.681, −0.667]; against the same model trained with shuffled time labels it is −3.71 [−3.85, −3.57]. On four two-dimensional toy transformations the model is statistically indistinguishable from the target in three cases and within 0.0004 of the permutation threshold in the fourth.

That 0.428 is the second version of the model. The first reached 0.641, and rather than tune it we measured where its error lived (§5.5): the spline path the field is trained on is itself at 0.636 at the held-out time, so the network was already reproducing its target to within 0.005 and no change to the architecture could have helped. The error is a pure location error, and the path's own mean is fixed by the observed snapshot means for any coupling rule, which is also why OT brings no gain over random pairing in the ablation below on this simulator. The two corrections that follow from the diagnosis (§3.4) — putting every observed time point into the path, and a second training stage that matches the observed marginals directly while keeping the path loss as a regulariser (on this simulator its window of one time unit around the observed times still covers the whole 2 → 4 gap, so no part of the path is left unsupervised) — were selected on two validation folds with the test time point removed from the data and then measured once on it: 0.641 → 0.428, paired difference −0.213 [−0.221, −0.206], landing on the ceiling the diagnosis had predicted (0.422).

**Real data (§5.7–§5.9).** The same protocol, unchanged, on two real time courses. On the human embryoid-body course of Moon et al. (2019) — 31 054 cells in five windows over 27 days, the series of the held-out-time-point benchmark introduced with TrajectoryNet and reused by later flow-matching methods — with the middle window (days 12–15) held out, in a space of 1 197 genes chosen without looking at it, the model reaches an energy distance of 0.267 against 0.371 for the best control (mean interpolation), paired difference −0.104 [−0.116, −0.092]; against the McCann/OT interpolant −0.713 [−0.757, −0.668]; against the same model with shuffled time labels −0.519 [−0.529, −0.509]. In a 50-component PCA space, closer to the reduced spaces the published methods work in (their benchmark uses five whitened components and the Wasserstein-1 distance, so no published number is comparable to ours), it ties with mean interpolation (+0.002 [−0.006, +0.010]) and still beats every other control. On the mouse gastrulation multiome of Argelaguet et al. (2022), stage E8.0 predicted from E7.75, the result is negative: the model loses to mean interpolation (2.03 against 1.51, +0.52 [+0.44, +0.60]), ties with the OT interpolant (+0.04 [−0.03, +0.12]), and beats persistence, the learned maps and the shuffled-time model (−0.82 [−0.94, −0.70]); two E8.0 embryo pools are 0.027 apart, so no method is close to a second biological sample.

A diagnosis without training (§5.9) separates two sources of error. The first is the trained field: at both test stages and on three of the four validation folds it is further from the held-out population than the path it is trained on — by 0.14 at the embryoid-body test window (path 0.130, model 0.267), by 0.09 at the gastrulation test stage, and by 0.18 to 0.22 on those three folds; on the fourth (embryoid body, days 18–21) the two are level (path 0.522, model 0.536). The second is the path itself, and it is specific to the gastrulation test stage: there the path over-disperses E8.0 (total variance 1.54 times the truth) and is worse than mean interpolation (1.94 against 1.51), while on both gastrulation validation folds it beats mean interpolation. Mean interpolation shifts real cells and keeps their spread, which at E8.0 — a quarter day from its source and with the same spread — is close to the right answer. Two changes made the real-data numbers what they are, both fixed before the measurement they are reported with (§3.5): a gene filter that removes genes detected in fewer than 1 % of the fit cells — a third to two fifths of the first real-data gene spaces were such genes, whose few expressing cells (a median of 10 of 6 000 fit cells on the embryoid body) sit tens of units from all the others after standardisation — and a straight-line centre for the path, chosen on validation folds from which the test stage had been deleted (the candidates themselves came from a no-network look at the first embryoid-body test window, §3.5). On the test stage the centre is worth −0.85 [−0.91, −0.79] on the gastrulation series (2.88 → 2.03, and it is what makes the model use time there at all) and nothing measurable on the embryoid body (−0.004 [−0.017, +0.008]). Two changes aimed at the first source were then tried on the same validation folds (version 0.1.11, §5.9) and neither met the selection rule: giving the field biological time as an input (with the second stage kept) changed nothing, and dropping the second training stage, the one that matches the observed marginals, brought the model close to its path on one validation fold of each data set and made it much worse on the other. The two kinds of fold differ in several ways at once, so why is not yet known. The model measured in this report is therefore version 0.1.10.

What optimal transport does and does not do here (§5.3): on the regulatory-network simulator, whose populations are unimodal, the gain over the interpolant baselines comes from the curved multi-marginal paths and the autonomous field (−0.43 against straight paths), while OT pairing brings no gain over random pairing (OT minus random chaining +0.018 [+0.007, +0.029], i.e. random chaining is marginally better; OT versus independent minibatch coupling +0.009 [−0.031, +0.050]). On a two-lineage population that moves in opposite directions, where random pairing creates crossing paths, OT pairing is decisive: the OT-chained model reaches the noise floor (energy distance −0.005, indistinguishable from the target) while random chaining stays at 0.081, paired difference −0.086 [−0.101, −0.071] (§5.3, second table). We report both because the honest statement is narrower than "OT helps": on this two-lineage toy OT chaining is necessary for the chained-spline construction, on the unimodal simulator it is not, and whether it matters on the real data sets has not been tested.

The package (`cfm_grn_v2`, 103 tests) is reproducible from the command line in about 35 minutes on a 16-core CPU for the full synthetic benchmark and 3 minutes for the smoke run.

# 2. Problem and design constraints

Single-cell time courses measure populations destructively: at each time point we see a different sample of cells, never the same cell twice. The modelling task is to learn a dynamical law that transports the population observed at one time into the population observed at the next, and to use it to predict populations at unobserved times, under unobserved perturbations, and, later, to read off which regulatory interactions are active when.

Design decisions fixed before any experiment:

- **Two clocks, never mixed.** The flow-matching path coordinate s ∈ [0, 1] is a mathematical device; biological time (hours, days, stages) is data. In the final model the field is autonomous in the state and is integrated over the biological horizon (a field that also sees biological time was tested in version 0.1.11 and not adopted, §5.9); in the next phase biological time enters through the regulatory-activity gates.
- **Biological units are never split.** Whole time points, whole replicates and, later, whole perturbation targets are held out; no random cell-level splits. Couplings are computed only within a (condition, replicate, time-pair) group.
- **Mandatory controls.** Every number is reported next to persistence, mean interpolation, the McCann/OT interpolant, linear and MLP maps, a non-parametric velocity replay and the same model trained with shuffled time labels. A gain without controls is not a gain.
- **Original code.** External methods are baselines behind adapters, never copied; the package depends on PyTorch, NumPy, SciPy, scikit-learn and pandas only.

# 3. Method

![Figure 1. Pipeline of the engine. Unpaired snapshots of one biological unit are chained across time points by exact minibatch optimal transport; a natural cubic spline through every pseudo-trajectory supplies states and velocities in biological time; an autonomous velocity field is regressed on them and integrated with RK4 over the biological horizon; predictions are scored against controls and a permutation null.](figures/fig01_pipeline.png){width=100%}

## 3.1 Conditional flow matching with OT couplings

For a transition between snapshots X₀ (time t₀) and X₁ (time t₁) of the same biological unit, CFM regresses a neural velocity field on the conditional velocity of a simple path between paired samples. With the linear path x_s = (1 − s)x₀ + s x₁ + σ_min ε the target is x₁ − x₀ (Lipman et al. 2023). The pairing of x₀ with x₁ is not observed; drawing it from the exact optimal-transport plan of the minibatch (Hungarian assignment under squared Euclidean cost, Tong et al. 2024) straightens the paths, reduces the variance of the regression target and makes the learned flow approximate the displacement interpolation between the two populations. Independent (random) pairing is kept as the ablation.

Implementation: `flows/coupling.py` (`ot_coupling`, `independent_coupling`, `couple_within_groups`), `flows/path.py` (`interpolate`, `velocity_target`, `cfm_loss`), `flows/solver.py` (Euler, midpoint, RK4).

## 3.2 Multi-marginal OT-chained spline paths (the selected model)

Consecutive-pair paths cannot express curvature between two snapshots: for a held-out time point between t₂ and t₄, a model trained on the straight 2→4 segment *is* the McCann interpolant, so it can at best tie with that baseline (this was measured, see §6). We therefore build pseudo-trajectories across all observed time points of a unit by chaining exact OT assignments t₀→t₁→…→t_m (`chain_ot`), fit a natural cubic spline in biological time through each pseudo-trajectory (`SplinePath`), and train an autonomous field v(x, condition) to match the spline velocity at states sampled along the curves (`spline_loss`). Chaining exact OT assignments across all time points and drawing a natural cubic spline through each pseudo-trajectory is the transport-spline interpolation of Chewi et al. (2021); regressing a flow-matching field on such spline paths across time points and conditions is multi-marginal flow matching (Rohbeck et al. 2025). The implementation here is ours and differs in that the field is autonomous and integrated over the biological horizon (`clock="duration"`), which is what allows the same field to predict any horizon and, in the next phase, to receive time through regulatory gates (giving it biological time directly was tested in version 0.1.11 and not adopted, §5.9). Figure 2 shows what the two chaining rules do on a population made of two lineages that move apart: OT keeps each lineage on its own path, random pairing draws paths that cross and whose intermediate states collapse towards the centre.

![Figure 2. OT-chained versus randomly chained pseudo-trajectories (two lineages moving in opposite directions, five time points, t = 2 held out). Curves: cubic splines through the chained cells; crosses: the state of each path at u = 2, i.e. what the field is trained to pass through.](figures/fig02_paths_ot_vs_random.png){width=92%}

Prediction: cells observed at t_a are integrated with RK4 for a duration t_b − t_a (20 steps per unit of biological time). Training: Adam, learning rate 10⁻³, gradient-norm clipping at 1, exponential moving average of the weights (decay 0.999 with warm-up) used for validation and for the final model, early stopping on the energy distance of a held-out inner time point, no refit in this version (the early-stopped model is final; see §6 for why; version 0.1.9 reintroduces a refit on every observed time point, §3.4). The network is deliberately small (Figure 3): capacity was shown not to be the limiting factor (§5.2).

![Figure 3. Architecture of the velocity field and how it is trained and used. Only the condition enters the context in the selected model; the flow-time input (dashed) exists for the linear-path variants with the flow clock and is switched off here.](figures/fig03_architecture.png){width=100%}

## 3.3 Alternatives kept in the search space

`clock ∈ {flow, duration}` (path coordinate with Δt in the context vs. autonomous field over the biological horizon), `pair_mode ∈ {consecutive, all ordered pairs}`, `coupling ∈ {ot, independent}`, `path ∈ {linear, spline}`, `chain ∈ {ot, random}` (spline only), hidden width and σ_min. All were compared under the same budget by nested selection on validation folds only.

## 3.4 Two corrections that follow from the diagnosis (version 0.1.9)

The diagnosis in §5.5 shows that the trained field reproduces its training path to within 0.005 energy-distance units, so the way to improve the prediction is to improve the path, or to stop supervising the part of the path that is a guess. Two changes follow, both selected on the validation folds with the test time point removed from the data (§5.6).

- **All observed knots.** The nested design spends one observed time point on early stopping, and the spline is drawn through the remaining ones only. Keeping the early-stopped iteration count and then refitting the same configuration on every observed time point puts that knot back into the path, which is worth 0.21 energy-distance units of ceiling on this fold (0.636 to 0.422, Table 5).
- **Windowed path supervision with marginal matching.** A spline is anchored where it has knots and is an extrapolation in the middle of a long gap, so during the second stage the field is supervised on the spline velocity only within `marginal_window` time units of an observed time (`SplinePath.sample(..., window=)`). The long transitions are instead constrained at their endpoints: the field transports a batch of one observed snapshot over the biological horizon and the result is compared with an independent batch of the next observed snapshot through a differentiable U-statistic energy distance (`flows/marginal.py`, `training/trainer.py::_finetune_marginal`). The first stage still fits the field to the spline over the whole time range, the interior of every gap included, and the second stage starts from those weights. With `marginal_window` = 1 the window leaves nothing out on gaps of two time units or less, which covers every gap of the synthetic runs and of the embryoid-body test refit (days 6–9 to 18–21); only on longer gaps, such as the gastrulation test refit (E7.75 to E8.5, three quarter days) and the refits of the real-data validation folds, is the middle of the gap left, during the second stage, to the field's smoothness and to the match of the observed marginals. The stage runs after the flow-matching pre-training and keeps the path loss as a regulariser. In the nested fit, where a validation time point is available, it can only replace the pre-trained weights by improving that score. The refit on every observed time point, which makes every prediction of the selected model from version 0.1.9 on (all real-data predictions included; the version-0.1.8 numbers are those of the early-stopped model, §3.2), has no validation point left: there the stage runs all of its 1000 iterations unguarded, and the nested fit's verdict on it is not carried over. Matching observed marginals with a neural ODE is the idea of TrajectoryNet (Tong et al. 2020) and MIOFlow (Huguet et al. 2022); here it is a second stage on top of flow matching, with the constructed path retained as the prior and restricted to the region where it is supported by data.

## 3.5 Two corrections for real data (version 0.1.10) and one candidate (0.1.11)

- **Gene detection filter (data).** Genes are chosen on the fit cells: the most variable ones, union the regulators of the prior. The dispersion score ranks a gene seen in a handful of cells very high, and every regulator detected in a single fit cell was kept. After standardisation each such gene puts its few expressing cells (a median of 10 of the 6 000 fit cells on the embryoid body) tens of units away from all the others (|z| up to 77 on the embryoid body and 95 on the gastrulation series; one regulator with zero variance on the observed stages reached |z| = 263 822 in the held-out one). They made up 41 % and 33 % of the gene spaces of the first runs. In aggregate they carry only their share of the squared norm (0.41 on the embryoid body), but they concentrate it in a few dozen cells: 38 of the 6 000 fit cells have a squared norm above ten times the median, and the rare genes carry 81 % of it there. Those cells are far from every other cell in the squared-Euclidean cost of the OT couplings, and their rare-gene coordinates are targets no field can predict. Since 24 September a gene is a candidate only if it is detected in at least 1 % of the fit cells (`select_genes(min_detect=0.01)`), the default of SCENIC (Aibar et al. 2017), fixed before any new run and not tuned; `--min-detect 0` reproduces the first runs. The problem was found through the error of the predicted mean, not the energy distance: two halves of the same held-out stage were 176 units apart in their means (1.4 without the 13 worst genes), while the energy distance of mean interpolation moved from 1.3234 to 1.3235, because a one-cell outlier largely cancels between the cross term and the within-sample terms.
- **Straight-line centre (method).** The mean of the spline path over its trajectories is the cubic spline through the snapshot means, for any coupling rule (§5.5), and on real data that curve overshoots between distant stages. `SplinePath(center="linear")` keeps each trajectory's deviation from the spline mean and moves the centre onto straight segments between the knot means (`TrainConfig.path_center`). At the knots the two centres coincide, so the path still passes through every observed cell. It was selected against a four-times wider network and the combination of both, on validation folds (Table 9). The candidates themselves were proposed after a no-network look at the first embryoid-body run at its held-out window (`scripts_dev/eb_diagnose.py`: the cubic spline through the window means missed the held-out mean by 3.02 against 2.69 for a straight line between the neighbours, and the path family was at 0.160 against 0.278 for the trained model, both in that run's unfiltered 1 272-gene space); only the choice between them was made without the test stage.
- **Candidate: a field that sees biological time (0.1.11).** Conditional flow matching reproduces the marginals of its path only if the field is the conditional mean velocity given the state *and the time*; an autonomous field can only average the velocities of trajectories that pass through the same state at different times. `TrainConfig.field_time="bio"` gives the field the scaled biological time as one scalar input (not the high-frequency flow-time embedding) and integrates from the source's biological time (`FlowModel.field`). Interpolation stays inside the range of training times, so the reason time was taken out of the field (§6, lesson 1) does not apply. On the validation folds of both data sets, with the second stage kept, it changed nothing and was not adopted (§5.9, Table 13).

# 4. Evaluation protocol

- **Primary task: leave-one-inner-time-point-out interpolation.** The held-out time point t_k is predicted from the cells at t_{k−1} over the horizon t_k − t_{k−1}; the bracketing transition t_{k−1}→t_{k+1} is observed. This is the task of the held-out-time-point benchmark introduced with TrajectoryNet (Tong et al. 2020) and reused by later flow-matching methods, which score it in five whitened principal components with the Wasserstein-1 distance; MIOFlow (Huguet et al. 2022) uses a variant in 200 components. The terminal-time-point forecast is reported only as a diagnostic: in a time course where all cohorts start from the same state, the last time point is the frontier of the visited state space, so forecasting it is pure extrapolation and a constant-velocity replay is near-optimal by construction (measured in §5.4).
- **Nested selection.** Hyper-parameters are chosen by the mean validation energy distance over two further held-out inner time points (one on each side of the test point), two search seeds each; the test point is never touched during selection. The number of flow-matching (first-stage) iterations is set by early stopping on one validation point; up to version 0.1.8 that early-stopped model is final, and from version 0.1.9 the same configuration is instead refit on every observed time point for that number of first-stage iterations, followed by all 1000 iterations of the second stage with no validation guard (§3.4); the final model is evaluated once.
- **Controls, fitted on the same observed cells and scored in the same standardised gene space:** persistence (x_{k−1}); mean interpolation x + frac·(mean x_{k+1} − mean x_{k−1}); McCann/OT displacement interpolation along the OT pairs of the bracketing transition; ridge-linear and MLP maps fitted on observed pairs with the same horizon; velocity replay (the OT displacement of the transition arriving at t_{k−1}, at the nearest arrival position); the flow model trained with shuffled time labels.
- **Metrics.** Energy distance as a U-statistic (expectation zero for identical distributions at any sample size; primary), unbiased MMD with a median-heuristic bandwidth fixed on the observed cells, assignment-based W₂ on up to 500 cells, error of the means, mean per-gene W₁. Each metric is computed per (condition, replicate) group, averaged within seed, then summarised over five seeds as mean ± 95% t-interval; comparisons are paired by seed.
- **Indistinguishability test (toys).** A prediction is "indistinguishable" from the target when its energy distance is at or below the 95% quantile of the energy distance between two random splits of the pooled prediction ∪ target at the same sample sizes (50 permutations).
- **Verdicts.** Sanity (toys): indistinguishable from the target and better than the shuffled-time control. Benchmark (GRN): lower energy distance than every control, paired CI95 below zero against the best control, and better than the shuffled-time control.

![Figure 4. The evaluation design on the seven-time-point simulator: the held-out inner time point, the prediction step, the observed bracketing transition, the two nested validation folds used for selection, the controls and the metrics.](figures/fig04_protocol.png){width=100%}

# 5. Results

## 5.1 Toy transformations (sanity, 1500 cells per group, three replicates: fit / validation / test)

| transformation | flow model ED | permutation threshold | verdict |
|---|---|---|---|
| translation | 0.0031 | 0.0027 | fails by 0.0004 (linear map 0.0017, mean shift 0.0012) |
| variance change | 0.0014 | 0.0036 | pass |
| bimodal split | 0.0020 | 0.0082 | pass |
| condition-dependent shift | 0.0006 | 0.0028 | pass |

Table: Table 1. Toy transformations: energy distance of the flow model against the permutation threshold (mean over 5 seeds).

Every flow model beats its shuffled-time twin by a wide margin (paired CI95 far from zero, e.g. translation [−2.61, −2.45]). The bimodal split was unsolved (ED 0.27) before the training budget and weight averaging were fixed (§6).

![Figure 5. Toy transformations on a log scale: flow model, permutation threshold, best control and the shuffled-time control (5 seeds, mean ± sd).](figures/fig05_toys.png){width=88%}

## 5.2 Synthetic gene-regulatory network, held-out inner time point (gate G1)

Simulator: 40 genes, 10 regulators, 73 signed edges with Hill kinetics and linear decay, 42 of the 73 edges switching on or off at edge-specific times (the ground truth for the next phase), 500 cells per snapshot, two replicates, time points 0…6, multiplicative noise. Held out: t = 3, predicted from t = 2 over one time unit; observed bracketing transition 2→4. Figure 6 shows the ground truth of the simulator.

![Figure 6. The synthetic regulatory system: (a) activity of the 73 edges over time (42 switch on or off at edge-specific times), (b) the signed weights of the 10 regulators on the 40 genes, (c) the resulting mean expression per snapshot.](figures/fig06_grn_simulator.png){width=100%}

| model | energy distance (mean over 5 seeds) |
|---|---|
| **flow model, OT-chained spline paths, duration clock** | **0.641** (seeds 0.640–0.643) |
| mean interpolation | 1.102 |
| McCann / OT interpolant | 1.171 |
| linear map (same horizon) | 2.747 |
| MLP map (same horizon) | 2.908 |
| flow model, shuffled time labels | 4.117 |
| persistence | 4.155 |
| velocity replay | 4.402 |
| noise floor (two halves of the target) | 0.000 |

Table: Table 2. Held-out inner time point t = 3 of the regulatory simulator: energy distance of the selected flow model and of every control (mean over 5 seeds).

Paired differences (n = 5 seeds): vs. mean interpolation −0.461 [−0.463, −0.459]; vs. shuffled time −3.476 [−3.559, −3.394]. MMD, W₂, mean error and per-gene W₁ rank the models the same way (MMD 0.023 vs 0.042 for the best control; per-gene W₁ 0.136 vs 0.203). Verdict: **pass**.

Selection table (mean validation energy distance over the two validation folds): the four spline configurations occupy the first four places (0.331–0.333), followed by linear paths with the duration clock (0.58–0.62), linear paths over all ordered pairs (1.30–1.31) and every flow-clock configuration (1.94–2.03). Width (128 vs 256) and σ_min (10⁻³ vs 10⁻²) do not change the spline result, which points to the path construction, not the network capacity, as the current limit (Figure 7).

![Figure 7. (a) Energy distance to the held-out population for the flow model and every control (bars: mean ± sd over 5 seeds; dots: seeds; dashed line: permutation threshold). (b) Nested selection by configuration family: the spline family wins on both validation folds before the test point is touched.](figures/fig07_grn_interpolation.png){width=100%}

## 5.3 Optimal-transport ablation

To isolate the contribution of optimal transport, the same field, budget and validation fold were trained with (a) OT-chained spline paths, (b) spline paths through randomly paired pseudo-trajectories, (c) straight consecutive-pair paths with minibatch OT couplings and (d) the same with independent couplings, five seeds each (`scripts_dev/ot_ablation.py`, results in `outputs/ot_ablation/`).

**Regulatory-network simulator (held-out t = 3).** OT brings no gain here: random chaining is marginally better than OT chaining and independent coupling ties with OT coupling. The decisive ingredient is the path family: curved multi-marginal spline paths beat straight pair paths by 0.43 energy-distance units with a paired interval that excludes zero. The reason is largely algebraic. Chaining is a bijection that uses every cell of every snapshot, and the cubic spline is linear in its knots, so the mean of the spline path at any time equals the spline drawn through the snapshot *means* at that time, whatever the coupling rule. Since on this simulator the energy distance is almost entirely a function of the mean error (§5.5), the coupling rule can change the score only through the shape of the cloud, and only slightly: the path through randomly chained cells has the same mean error to nine digits and an energy distance 0.010 lower (0.626 against 0.636, Table 5), and the trained models differ by +0.018 [+0.007, +0.029] in favour of random chaining. Measured: OT chaining and random chaining give mean errors 1.3337591265 and 1.3337591278, identical to nine digits. Optimal transport can only matter when the coupling changes the *shape* of the interpolated cloud, which is what the two-lineage experiment below isolates.

Table: Table 3. Optimal-transport ablation on the regulatory simulator (held-out t = 3, 5 seeds). The ablation trains every arm with hidden width 256, where the selected model has 128, so its OT-chained arm is at 0.643 against 0.641 in Table 2 and its paired intervals differ slightly from those of §5.2.

OT_ABLATION_TABLE

**Two lineages moving apart (held-out t = 2, `scripts_dev/ot_crossing.py`).** Two clusters of cells drift in opposite directions over five time points; the middle time point is held out. Random pairing joins left cells with right cells, so the paths cross and their midpoints collapse towards the centre; OT pairs each cluster with itself. This is the situation of any branching or multi-population time course, and here optimal transport is what makes the interpolation correct: both OT arms sit at the noise floor (spline −0.005, straight paths −0.005, the negative values being the U-statistic's sampling noise around zero), random chaining is about seven times the permutation threshold (0.081 against 0.012, paired OT − random −0.086 [−0.101, −0.071]) and independent coupling is measurably worse than OT coupling (0.006 vs −0.005, −0.012 [−0.014, −0.010]). Mean interpolation equals persistence here (0.169) because the population mean does not move, which is the failure mode of every mean-based baseline on heterogeneous populations. The toy is deliberately easy for straight-line baselines (the McCann interpolant is exact on it); its purpose is the within-model comparison, not a benchmark.

Table: Table 4. Optimal-transport ablation on two lineages moving apart (held-out t = 2, 5 seeds).

OT_CROSSING_TABLE

![Figure 8. The same four arms on the two data sets (bars: mean ± sd over 5 seeds; dots: seeds; box: paired differences with 95% intervals). OT brings no gain on the unimodal simulator and is decisive on the two-lineage population.](figures/fig08_ot_ablation.png){width=100%}

![Figure 9. Learned velocity field and one-step prediction from t = 1 to t = 2 on the two-lineage population, OT-chained (a) versus randomly chained (b) spline paths, single seed. The randomly chained model learns a field that drags both lineages towards the centre.](figures/fig09_crossing_prediction.png){width=92%}

## 5.4 Terminal-time-point forecast (diagnostic only)

Predicting t = 6 from t = 5 after training on 0…5: the best flow models reach 0.40–0.59 against 0.56 for the linear map and 1.12 for the velocity replay, but the configuration space is unstable (the run that transfers the early-stopping iteration count to a refit produced two divergent seeds out of five, mean 5.67). This is the expected behaviour of a pure extrapolation task and is the reason the protocol uses interpolation as the primary gate.

## 5.5 Where the residual error comes from

The model beats every control by a wide margin and is still 160 times the permutation threshold. Three cheap measurements (`scripts_dev/diag_ceiling.py`, `diag_mean_vs_shape.py`, `diag_mean_curve.py`, results in `outputs/diag_ceiling/`) locate that gap precisely, and none of them involves training a network.

**The training target is the ceiling.** Evaluating the spline path family itself at the held-out time, with no network at all, gives an energy distance of 0.636. The trained field reaches 0.641. Everything the architecture, the optimiser, the weight averaging and the ODE solver contribute together is 0.005, of the order of the permutation threshold. The field reproduces its training signal essentially exactly, so no change to the network can help; the constructed path is the entire error.

Table: Table 5. The path family measured without a network, at the held-out time t = 3 (three chaining seeds).

| measurement | energy distance | error of the mean |
|---|---|---|
| spline target through the fit knots 0, 1, 2, 4, 6 (no network) | 0.636 | 1.334 |
| the trained flow model | 0.641 | 1.336 |
| spline target through all observed knots 0, 1, 2, 4, 5, 6 | 0.422 | 1.081 |
| the same target with random chaining instead of optimal transport | 0.626 | 1.334 |
| McCann / OT interpolant of the bracketing transition | 1.177 | 1.806 |
| oracle chain that includes the true t = 3 | −0.010 | 0.000 |
| noise floor (two halves of the true sample) | −0.002 | — |

The last two rows bracket the problem: with the true answer in the chain the construction is exact, and with a knot missing it is not. The improved model of §3.4 reaches 0.428 at this fold (§5.6), that is the all-knots ceiling of 0.422 to within 0.006, which is the same margin by which the first version reproduced its own target.

**The error is a pure location error.** Shifting any prediction rigidly onto the true mean drives the energy distance to the noise floor: 0.636 becomes −0.003 for the spline paths, 1.177 becomes −0.004 for the McCann interpolant, and even persistence, at 4.155, becomes +0.003. For every method whose mean error is below 3 (the flow model, both interpolants, the linear and MLP maps) the energy distance is proportional to the squared error of the predicted mean, with one constant (0.34 to 0.36); for the large errors of persistence, the shuffled-time model and velocity replay it grows more slowly (0.23 to 0.27), but it remains a function of the mean error. The reason is visible in the data: the per-gene spread inside a snapshot is 0.21 to 0.26 in the evaluation space while the population travels about one unit per unit of time, so the snapshots are near-translates of one another and the metric is almost entirely a question of where the cloud is.

**Consequences.** Two of them matter for the method and one for the benchmark.

- Any interpolant that is *linear in the observed snapshots* has its mean fixed by the snapshot means, which is why the coupling rule cannot matter here (§5.3) and why changing the spline family cannot help either: on the two validation folds the natural cubic is already the best of the families tried (natural cubic 1.011 against Akima 1.056, PCHIP 1.091, not-a-knot 1.660, linear 1.391, smoothing splines 1.011 to 1.170). Even the best affine combination of the five observed snapshot means, with weights chosen knowing the true answer, still leaves a mean error of 0.802, that is an energy distance of at least 0.23. Going below that requires leaving the class of interpolants that are linear in the data, which is what §3.4 does.
- What does move the number a lot is the *knot set*: putting the sixth observed time back into the path lowers the ceiling from 0.636 to 0.422.
- For the benchmark itself: because the snapshot shape is nearly time-invariant, this simulator cannot discriminate methods on the shape of the predicted distribution, only on its location. That is a limitation of the simulator, not of the metric, and the next version of it needs divergent lineages, of the kind the two-lineage experiment of §5.3 uses, and cell-level heterogeneity that grows with time.

![Figure 10. Why the residual error is what it is. (a) The spline path family evaluated at the held-out time, with no network, is at 0.636 and the first version of the model at 0.641; adding the sixth observed knot moves the ceiling to 0.422, and the improved model reaches 0.428. (b) Shifting any prediction onto the true mean sends the energy distance to the noise floor, bars showing the absolute value because the U-statistic fluctuates around zero. (c) The energy distance grows with the error of the predicted mean: close to one quadratic curve for small errors, and below it for persistence.](figures/fig10_ceiling.png){width=100%}

## 5.6 The improved model (version 0.1.9)

The two corrections of §3.4 were chosen on the two validation folds of a data set from which the test time point had been **removed entirely**, so that not even a nested fit could see it. Both folds have the structure of the test fold: a held-out time in the middle of a gap of two units with the bracketing transition observed (t = 1 predicted from t = 0 with 0 → 2 observed, and t = 5 predicted from t = 4 with 4 → 6 observed).

Table: Table 6. Selection on the validation folds (`scripts_dev/combo_scan.py`, 3 seeds per fold, energy distance).

COMBO_TABLE

The test fold was then measured once, with the same five seeds, the same evaluation protocol and the same controls as §5.2.

Table: Table 7. Held-out inner time point t = 3, the published model against the improved one (5 seeds).

FINAL_TEST_TABLE

## 5.7 First real data set: a held-out window of an embryoid-body time course (positive)

Everything above is synthetic. The same protocol, unchanged, was then run on the time course of Moon et al. (2019): a human embryoid body followed for 27 days and sampled in five windows of three days, 31 054 cells after quality control, no biological replicate (data card in `data_cards/moon2019_embryoid_body.md`). This is the series of the held-out-time-point benchmark introduced with TrajectoryNet and reused by later flow-matching methods (OT-CFM among them), which is why it was chosen. Those methods report their numbers in five whitened principal components with the Wasserstein-1 distance (MIOFlow in 200 components), averaged over holding out each of the three inner windows in turn, not as an energy distance over standardised genes for the middle window alone, so a like-for-like comparison with them is still to be done; the numbers below are read against our own controls.

The middle window (days 12–15) is held out and predicted from the previous one; the transition that brackets the gap is observed; early stopping uses a second inner window chosen by the design rule, not by hand; the final model is refit on every observed window. The configuration is version 0.1.10: the version-0.1.9 configuration selected on the synthetic validation folds, plus the straight-line path centre selected on validation folds of this data set with the test window deleted (Table 9). Genes are chosen on the fit cells only — among genes detected in at least 1 % of them, the most variable ones union the transcription factors of the CollecTRI prior (§3.5) — where the fit cells are the windows that are neither held out nor used for early stopping, so neither the held-out window nor the validation window touches the gene space. Normalisation is per cell and cannot move information between windows. How the numbers moved from the first run to this one is in Table 11.

Table: Table 8. Held-out window (days 12–15) of the embryoid-body time course, gene space, version 0.1.10 with the gene filter, five seeds.

EB_TABLE

EB_PCA_SENTENCE

![Figure 11. The held-out window on real data. (a) The benchmark: every control and the same model trained with shuffled time labels, five seeds, error bars one standard deviation, with the permutation threshold as the dashed line. (b) Why this data set is worth more than the simulator: ranking the methods by the error of the predicted mean and ranking them by the energy distance give different orders. The McCann/OT interpolant puts the cloud in almost the right place and still loses, because the shape of its cloud is wrong; on the simulator, where the two rankings are the same by construction (§5.5), that distinction could not be measured.](figures/fig11_real_data.png){width=100%}

## 5.8 Second real data set: a held-out stage of mouse gastrulation (negative)

The second series is the 10x Multiome time course of mouse gastrulation of Argelaguet et al. (2022): wild-type embryos at E7.5, E7.75, E8.0, E8.5 and E8.75, 71 176 cells after the authors' quality control, two biological replicates (pools of embryos) at every stage except E7.75, RNA and chromatin from the same cells (data card in `data_cards/argelaguet2022_gastrulation.md`; the Brachyury knockout samples are kept for a later phase). Biological time is counted in quarter days from E7.5, so the uneven spacing of the stages is kept (0, 1, 2, 4, 5). E8.0 is held out and predicted from E7.75; the bracketing transition E7.75 → E8.5 is observed; E8.5 serves for early stopping; replicates are pooled by stage, because their labels do not link embryos across stages, and at most 3 000 cells per stage are used. The gene space (1 428 genes, 465 of them prior regulators) is chosen on the fit cells with the same rule as above.

The version-0.1.10 configuration was chosen here exactly as on the embryoid body: the test stage was deleted from the data, the two inner remaining stages served as folds, each with its own nested design, gene selection and refit, and a rule fixed in advance was applied by code (an arm must beat the current configuration in at least three of the four fold × seed pairs and on average; the best qualifying arm wins).

Table: Table 9. Selection of the version-0.1.10 configuration on validation folds, test stage removed from the data (energy distance, two seeds per fold; change = mean paired change against version 0.1.9, with the number of the four pairs in which the arm was better).

SELECT_V0110_TABLE

Table: Table 10. Held-out stage E8.0 of the gastrulation series, predicted from E7.75, gene space, version 0.1.10 with the gene filter, five seeds.

ARG_TABLE

The result is negative against the strongest control. Mean interpolation — the E7.75 cells shifted by a third of the E7.75 → E8.5 change of the mean — is better than the flow model by half an energy-distance unit, and the flow model is only level with the McCann/OT interpolant. It is not a failure of the whole approach: the model beats persistence, the linear and MLP maps and the velocity replay, and with the straight-line centre it uses biological time (without it, the shuffled-time model was as good as the real one; Table 11). The distance between the two E8.0 embryo pools gives the scale: two biological samples of the same stage are 0.027 apart, more than fifty times closer than the best method.

Table: Table 11. What each change did on the held-out stage (energy distance, five seeds; paired differences against mean interpolation and against the shuffled-time model). The first runs used no gene filter; the gene spaces therefore differ between the first row of each data set and the others, and only rows with the same gene space are directly comparable.

HISTORY_TABLE

![Figure 12. The gastrulation benchmark and where the real-data error lives. (a) E8.0 held out, every control, five seeds; dashed line: energy distance between the two E8.0 embryo pools. (b, d) At each data set's held-out stage: mean interpolation, the path family the field is trained on, evaluated with no network, and the trained model, all on the benchmark's own evaluation draws. (c) Total variance of each construction relative to the held-out population: mean interpolation moves real cells and keeps their spread, the OT interpolant averages two cells and contracts, the spline path through chained cells over-disperses on the gastrulation series.](figures/fig12_gastrulation_and_diagnosis.png){width=100%}

## 5.9 Where the real-data error comes from

As on the simulator (§5.5), the constructions can be scored at the held-out stage without training anything, on exactly the draws the benchmark uses, so a deterministic control reproduces its benchmark value to the last digit (mean interpolation 0.3706 and 1.5131, persistence 0.9052 and 3.1162). This is a diagnosis of measured results, run after them; nothing in it was used to choose the configurations of Tables 8–11 (the earlier look at the embryoid-body test window that suggested the candidates of Table 9 is described in §3.5).

Table: Table 12. The constructions at the held-out stage, no network (energy distance, five seeds; spread = total variance of the prediction over that of the held-out population; distance to data = median distance from a predicted cell to its nearest held-out cell, over the same for real held-out cells).

DIAG_TABLE

Three things follow.

1. **The trained field loses part of its target at both test stages and on three of the four validation folds.** At the embryoid-body test window the path family is at 0.130 and the trained model at 0.267. A gap of the same size appears on three of the four validation folds (path → model; the path from the no-network diagnosis with five seeds, the model with two, each scored on its own draw of 1000 held-out cells — the selection run's own estimate of the path, 0.512, 0.526, 3.448 and 1.091, differs by up to 0.07): embryoid body days 6–9 0.535 → 0.734; gastrulation E7.75 3.445 → 3.669 and E8.5 1.159 → 1.343. On embryoid body days 18–21 the two are level (0.522 → 0.536, within the spread between seeds). It is smaller at the gastrulation test stage in energy distance (1.94 → 2.03), but there the model's mean is further from E8.0 than mean interpolation's (error of the mean 10.02 against 8.59, +1.43 [+1.23, +1.64]) while the path's mean is level with it (8.46 against 8.44 on all cells). On the simulator the same field reproduced its path to within 0.005 (§5.5), so something the simulator did not have makes the target hard to learn here. A wider network trained with the same recipe did not help (Table 9: on the embryoid body four times the width was worse in three of the four fold × seed pairs, and early stopping ended it after about a sixth of the iterations of the narrow one). The reported model also runs the marginal-matching stage after flow matching, in the refit for a fixed 1000 iterations with no validation guard (§3.4), which can move it off the path by design. The version-0.1.11 selection below looks at both: an arm without that stage, to tell the two apart, and an arm in which the field sees biological time, the one structural reason we can name (run with that stage on, so only a partial test of it; see the end of this section).
2. **At the gastrulation test stage the path itself loses.** The path over-disperses E8.0: its cells are spread 1.54 times as widely as the real ones and sit further from them than real cells do. Between E7.75 and E8.5 the spline through each chained trajectory bulges, while the real population hardly changes its spread in that quarter day (held-out over source variance 1.00). The OT interpolant has the opposite fault: each of its cells is an average of two different cells, and the cloud contracts (0.68). Mean interpolation moves real cells rigidly and keeps the spread, and that is why it wins there. This is specific to the test stage: on both gastrulation validation folds the path beats mean interpolation (E7.75 3.445 against 3.613; E8.5 1.159 against 1.892).
3. **A correction that would tie with mean interpolation on the test stage was not adopted.** Rescaling every gene's spread at the held-out time to the straight-line interpolation of its spread at the bracketing stages (the second-moment analogue of the straight-line centre) takes the path to 1.475 on E8.0, level with mean interpolation (paired over seeds −0.038 [−0.087, +0.010]). On the validation folds of the same data set it makes the path worse (DIAG_SCALE_SENTENCE). The disagreement between validation folds and test stage runs in both directions: on the embryoid body the validation folds prefer the same correction (0.481 against 0.528) and it worsens the test window (0.130 → 0.157, +0.028 [+0.023, +0.033]); on the gastrulation series the validation folds prefer Brownian-bridge noise on the path (σ = 1.0: 2.263 against 2.302), which is far worse at E8.0 (2.502 against 1.938). None of these corrections is adopted; they are reported as a diagnosis. The two gastrulation validation folds sit at other positions in longer gaps than the test stage (E7.75 a quarter of the way into a four-unit gap, E8.5 three quarters of the way; E8.0 a third of the way into a three-unit gap), and a five-stage series offers no other inner folds.

V0111_SECTION

**What the two arms say, and what they do not.** Every prediction in Table 13, as on the test stages, is made by the model refit on all observed stages, and in that refit the second stage runs its 1000 iterations with no validation guard. The runs are deterministic and the two arms share their first stage exactly, so the difference between the arm without the second stage and version 0.1.10 is the effect of those 1000 iterations. It splits the folds in the same way on both data sets. On the fold whose held-out stage is predicted from the first observed stage over one step (embryoid body days 6–9 from days 0–3; gastrulation E7.75 from E7.5) the model without it is better with both seeds and close to its own path: 0.574 against 0.734 with the stage and 0.535 for the path; 3.489 against 3.669 and 3.445. On the other fold (days 18–21 from days 6–9; E8.5 from E7.75) it is much worse, 1.195 against 0.536 and 1.709 against 1.343, and its mean drifts (on the embryoid body an error of the mean of 5.60 and 10.30 for the two seeds, against 5.28 and 5.28 with the stage). These four numbers do not say why. The two kinds of fold differ at once in the horizon (one step against two or three), in the position of the target in its gap (a quarter or a third of the way against two thirds or three quarters), in the source (the first observed stage against an interior one, where the straight-line centre makes the mean velocity of the path change abruptly), in the spread of the path at the target (1.22 and 1.13 times the truth against 0.82 and 0.79), and in the validation task of the nested fit, which sets the iteration count. The test stages mix the two kinds: one step from an interior stage. The arm in which the field sees biological time was run only with the second stage on, so it does not show that averaging over trajectories that share a state at different times is harmless; a field that sees time and is trained on the path alone was not run. Telling these explanations apart needs folds on which they are not collinear.

# 6. What the redesign log says (eleven versions in a week)

The engine was built test-first from a written plan and redesigned on validation evidence only; every change is a row in `docs/decisions.md`. The instructive ones:

1. **Absolute time in the context cannot extrapolate.** A field conditioned on (t₀, t₁) diverged on any interval outside the training range; conditioning on Δt only, then removing the time input altogether (autonomous field), fixed it.
2. **V-statistics lie about sample size.** Biased energy distance and MMD made a non-parametric replay look "better than the noise floor"; U-statistics and a permutation null replaced them.
3. **Under-training looked like divergence.** The validation curve oscillated for 1500 iterations; a weight EMA, gradient clipping and a longer budget with patience turned a failing model into a passing one on the bimodal toy.
4. **A straight-path flow model on the bracketing transition is the McCann interpolant.** It could not beat that baseline on interpolation (0.29 vs 0.24); curved multi-marginal paths could (0.04 vs 0.24 on the quick fold).
5. **Refit at a transferred iteration count is fragile — but the knot it adds is worth more.** Two seeds refitted for 500 iterations on the forecast task produced energy distances above 12, so the early-stopped model became the default and the controls kept the data advantage. On the interpolation task the same refit is stable (12 runs, no seed diverging) and the time point it puts back into the path is worth 0.21 of ceiling (§5.5), so version 0.1.9 restores it — after measuring it on two validation folds, not on the strength of the argument (§5.6).
6. **The training target, not the network, was the ceiling.** Three measurements without a network showed that the constructed path was already at 0.636 while the trained model sat at 0.641. Everything spent on architecture and optimisation would have bought 0.005. The fix had to be, and was, a better path (the observed knot that the nested design had spent on early stopping, put back in) and a second stage that matches the observed marginals directly; on this simulator no gap is longer than two units, so that stage's restriction of the path loss to within one time unit of a knot leaves out no time, and the first stage supervises the whole path.
7. **One validation fold mis-selects between path families.** Selection over two inner folds picked the spline model that the single fold rejected.
8. **A gene seen in five cells is not a feature, it is a spike.** The first real-data gene spaces were a third to two fifths genes detected in fewer than 1 % of the fit cells. The energy distance hardly noticed; the error of the mean between two halves of the same stage (176 where 1.5 was expected) did. Every real-data result in this report is from the filtered gene space, except the first runs, kept in Table 11 so that the change is visible, and the diagnostic numbers of §3.5 that led to the filter and the centre.
9. **Validation folds can disagree with the test stage, and the test stage does not get a vote.** On the gastrulation series the second-moment correction of §5.9 would tie with mean interpolation on the test stage (1.475 against 1.513) and loses on both validation folds; on the embryoid body the same correction wins on the average of the two validation folds (0.481 against 0.528: better on days 18–21, slightly worse on days 6–9) and loses on the test window. Neither was adopted.

# 7. Limitations and next steps

- **One positive and one negative real data set.** On the embryoid body the model beats every control in gene space (−0.104 [−0.116, −0.092] against mean interpolation) and ties with mean interpolation in PCA-50 (+0.002 [−0.006, +0.010]). On the gastrulation series it loses to mean interpolation (+0.52 [+0.44, +0.60]). The positive claim we can make is limited to the first data set and to gene space. The diagnosis of §5.9 locates where the error arises — in the trained field at both test stages and on three of the four validation folds, and in the path at the gastrulation test stage. On two of the four validation folds most of the first is the refit's unguarded second stage; on the other two, removing that stage moves the model much further from its path (Table 13); neither is fixed.
- **Two sources of error, not one.** The trained field loses part of its path at both test stages (0.14 at the embryoid-body test window, 0.09 at the gastrulation one) and on three of the four validation folds (0.18 to 0.22), and is level with it on the fourth (embryoid body days 18–21); giving the field biological time (with the second stage on) does not change it, and the second stage accounts for most of it on two folds while being needed on the other two (§5.9, Table 13), so neither variant met the selection rule. At the gastrulation test stage the path itself over-disperses. No-network corrections of its spread exist, but on these data the validation folds and the test stage disagree about them in both directions, and with five stages there are only two inner folds to choose on. A series with more stages, or folds built from the replicates, is needed before a construction can be chosen for its spread.
- **No real-data prediction is close to a second biological sample.** Two E8.0 embryo pools are 0.027 apart; the best method is at 1.51. On the embryoid body the model is at 0.267 against a permutation threshold of 0.006. The benchmark separates methods; it does not yet say that any of them reproduces the biology.
- **The embryoid body has no biological replicate**, so its confidence intervals are over training seeds only. The gastrulation series has replicates at four of its five stages, but E7.75, the source of the E8.0 prediction, has one, and replicate labels do not link embryos across stages, so replicates are pooled.
- **Real-data gene spaces changed once.** The first real-data runs contained a third to two fifths of genes detected in fewer than 1 % of the fit cells. They are kept in Table 11 so the effect of the filter is visible; apart from them and the diagnostics of §3.5, every real-data number is from the filtered space.
- On the synthetic benchmark the residual energy distance is attributed rather than suspected (§5.5): the field reproduces its training path to within 0.005, the error is entirely a location error, and no interpolant that is linear in the observed snapshots can go below about 0.23 on this fold. The simulator is also too easy in one dimension: its snapshots are near-translates of one another, so it ranks methods by the location of the predicted cloud and is blind to its shape — the real data sets are not (Figure 11b, Figure 12c).
- Next: (i) folds on which horizon, position in the gap and source stage are not collinear (a series with more stages, or held-out replicates), to find out when the second stage helps, and a choice of it made on those folds; (ii) a like-for-like run on the published embryoid-body benchmark (five whitened components of the processed file of TrajectoryNet, Wasserstein-1, each of the three inner windows held out in turn and the scores averaged); (iii) a construction whose spread is anchored to the data between stages, chosen on more folds than a five-stage series offers; (iv) the signed regulatory prior (CollecTRI: 13 223 mouse edges, 733 TFs) with low-rank, time-varying edge activity inside the drift, evaluated on the simulator's ground-truth switch schedule and on the chromatin half of the gastrulation multiome, which was never used in training (design note in `docs/phase3_gated_grn_design.md`). The last step is the contribution the rest serves: mean interpolation predicts a population but says nothing about which regulatory links are active when.

# 8. Reproducibility

```bash
cd cfm_grn_v2
# unit tests: 103 tests, about 3 min
python -m pytest tests -q
# smoke run of the full benchmark, about 3 min
PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic \
    --out outputs/quick --quick --seeds 2 --search-seeds 1
# full GRN interpolation benchmark, about 35 min on 16 CPU threads
PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic \
    --out outputs/full --datasets grn --tasks interpolation --seeds 5 --search-seeds 2
# optimal-transport ablations (about 15 min and 8 min) and the figures (about 3 min)
PYTHONPATH=src python scripts_dev/ot_ablation.py
PYTHONPATH=src python scripts_dev/ot_crossing.py
PYTHONPATH=src python scripts_dev/make_figures.py
# diagnosis of the residual error, no network trained (about 2 min)
PYTHONPATH=src python scripts_dev/diag_ceiling.py
PYTHONPATH=src python scripts_dev/diag_mean_vs_shape.py
PYTHONPATH=src python scripts_dev/diag_mean_curve.py
# selection of the version 0.1.9 configuration on the validation folds, test time point
# removed from the data (about 15 min), then the single measurement on it (about 17 min)
PYTHONPATH=src python scripts_dev/combo_scan.py --seeds 3
PYTHONPATH=src python scripts_dev/final_test.py --seeds 5
PYTHONPATH=src python scripts_dev/make_figure_ceiling.py
# real data: parse the embryoid-body archive into a cached sparse matrix (about 1 min),
# then the held-out-window benchmark with every control (about 2 h on 12 CPU threads)
python scripts_dev/eb_prepare.py
# selection of the version 0.1.10 configuration on validation folds, test window removed (about 3 h on 8 CPU
# threads), then the single measurement with it, the version-0.1.9 comparison and the PCA-50 control
PYTHONPATH=src python scripts_dev/select_config.py --dataset eb --cache <moon2019_eb.npz> --prior <collectri_human.tsv> \
    --out outputs/eb_select
PYTHONPATH=src python scripts_dev/eb_benchmark.py --choice outputs/eb_select/choice.json --seeds 5 --n-hvg 1000 \
    --out outputs/eb_v0110
PYTHONPATH=src python scripts_dev/eb_benchmark.py --seeds 5 --n-hvg 1000 --out outputs/eb_filter_base
PYTHONPATH=src python scripts_dev/eb_benchmark.py --choice outputs/eb_select/choice.json --pca 50 --seeds 5 \
    --n-hvg 1000 --out outputs/eb_v0110_pca50
PYTHONPATH=src python scripts_dev/make_figure_eb.py --results outputs/eb_v0110/results.csv
# gastrulation series, on the HPC: parse the 33 GEO files into a sparse cache (about 9 min), then the same steps
# (scripts_dev/hpc_v0110.sbatch and hpc_followup.sbatch run them under Slurm)
python scripts_dev/argelaguet_prepare.py --geo <GSE205117 files> --meta <metadata.txt.gz> --out <argelaguet2022_rna.npz>
PYTHONPATH=src python scripts_dev/select_config.py --dataset argelaguet --cache <npz> --prior <collectri_mouse.tsv> \
    --out outputs/argelaguet_select
PYTHONPATH=src python scripts_dev/argelaguet_benchmark.py --choice outputs/argelaguet_select/choice.json --seeds 5 \
    --n-hvg 1000 --max-cells 3000 --out outputs/argelaguet_v0110
# diagnosis without a network (minutes), the version-0.1.11 selection, and Figure 12
PYTHONPATH=src python scripts_dev/diag_blend.py --dataset eb --cache <npz> --prior <tsv> --out outputs/diag_blend_eb
PYTHONPATH=src python scripts_dev/select_config.py --dataset eb --arms v0111 --cache <npz> --prior <tsv> \
    --out outputs/eb_select_v0111
PYTHONPATH=src python scripts_dev/make_figure_argelaguet.py
```

The raw data are not redistributed with this deliverable. The embryoid-body archive is public (Mendeley `10.17632/v6n743h5ng.1`, CC BY 4.0, `scRNAseq.zip`, 161.7 MB); the gastrulation matrices are GEO `GSE205117` (33 files, 5.07 GB) with the authors' cell metadata. `scripts_dev/eb_prepare.py` and `scripts_dev/argelaguet_prepare.py` document exactly what is done to them, and `data_cards/` records sources, filters and limitations.

Outputs: `results.csv` (one row per model × seed × group with all metrics and the permutation threshold), `search.csv` (every configuration, seed and validation fold), `report.md`, `verdict.json`, `run_meta.json` (code version, arguments, selected configuration), `grn_truth.npz` (edge weights and the true activity schedule). Versions used: Python 3.13, PyTorch 2.13 (CPU), NumPy, SciPy 1.16, scikit-learn 1.9, pandas 3.0. Code version 0.1.11 (the version-0.1.10 configuration is the one measured in §5.7–§5.8; 0.1.11 only adds the `field_time` option, off by default, tested in §5.9 and not adopted).

Selected configuration (GRN interpolation): hidden 128, depth 3, SiLU MLP field without time input, learning rate 10⁻³, batch 256, σ_min 10⁻², OT-chained spline paths, duration clock, RK4 with 20 steps per time unit, EMA 0.999, gradient clip 1.0, early stopping on the validation energy distance (best iteration 1000–1750 of 8000). Version 0.1.9 adds, on top of that, the refit on every observed time point and a second stage of 1000 iterations that matches the observed marginals (run in full in the refit, where no validation point is left to stop it; batch 256, 8 RK4 steps per time unit, learning rate 3·10⁻⁴, path prior weight 1.0 restricted to a window of ±1 time unit around the observed times).

# References

- Lipman, Chen, Ben-Hamu, Nickel, Le (2023). Flow matching for generative modeling. ICLR.
- Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf, Bengio (2024). Improving and generalizing flow-based generative models with minibatch optimal transport. TMLR.
- Chewi, Clancy, Le Gouic, Rigollet, Stepaniants, Stromme (2021). Fast and smooth interpolation on Wasserstein space. AISTATS.
- Rohbeck, De Brouwer, Bunne, Huetter, Biton, Chen, Regev, Lopez (2025). Modeling complex system dynamics with flow matching across time and conditions (multi-marginal flow matching). ICLR.
- Tong, Huang, Wolf, van Dijk, Krishnaswamy (2020). TrajectoryNet. ICML.
- Huguet, Magruder, Tong, Fasina, Kuchroo, Wolf, Krishnaswamy (2022). Manifold interpolating optimal-transport flows (MIOFlow). NeurIPS.
- Schiebinger et al. (2019). Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. Cell.
- Székely, Rizzo (2013). Energy statistics. J. Stat. Plan. Inference.
- Gretton et al. (2012). A kernel two-sample test. JMLR.
- Argelaguet, Lohoff, Li, Nakhuda, Drage, Krueger, Velten, Clark, Reik (2022). Decoding gene regulation in the mouse embryo using single-cell multi-omics. bioRxiv, doi 10.1101/2022.06.15.496239.
- Moon et al. (2019). Visualizing structure and transitions in high-dimensional biological data (PHATE). Nature Biotechnology.
- Aibar et al. (2017). SCENIC: single-cell regulatory network inference and clustering. Nature Methods.
