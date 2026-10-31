---
title: "Where optimal transport helps on real single-cell time courses"
subtitle: "Cell-to-cell couplings yes, missing snapshots no: pre-registered tests, 22–30 October 2026"
author: "Gabriele Bambini"
date: "30 October 2026"
---

# 1. Summary

This report collects nine days of tests of one question: on real single-cell time courses, what does optimal transport (OT) add over simple rivals? OT was used in three roles, and the answer depends on the role.

1. **Per-stage summary statistics** (the gene-expression task of Dileo & Sottoriva 2026, run inside their released code, masks and metrics). A pre-registered model that contains an OT / Schrödinger-bridge step beats the best row of the paper's table in all six cells (two data sets × three metrics, five seeds). However, a validation on past transitions selected the OT step in only 3 % (gastrulation) and 14 % (pancreas) of transitions, and removing it changes nothing. The margin comes from a noise filter and a finite-cell observation model. On per-stage statistics OT has no role.
2. **Reconstructing the population at an unobserved time** from unpaired snapshots. Five pre-registered tests used three time courses: a mouse gastrulation atlas, zebrafish embryos and LARRY haematopoiesis in vitro; two of the tests replaced OT with alternatives (an estimated embryo age, a dynamic mixture model). In none of them does any OT-based or alternative method beat the 50/50 mixture of the two neighbouring samples. The midpoint of OT pairs beats it in 0 of 7, 1 of 9 and 0 of 4 held-out times. Reweighting cells by unbalanced OT beats persistence and mean shift in 1 of 6 and 0 of 8 forecast origins.
3. **Who becomes whom**: the coupling itself, scored against heritable lineage barcodes.
   - On LARRY in vitro the OT coupling recovers sister cells better than k-nearest neighbours (kNN) in 3 of 4 units (pre-registered).
   - An exploratory robustness check showed that the fair comparison is at equal spread, measured as effective partners per cell.
   - That comparison was then pre-registered on an independent barcoding experiment, LARRY cytokine perturbation, with 9 units. At equal spread OT beats kNN in 9 of 9 units (geometric mean of the ratios 1.95). It also beats the same Gaussian kernel without mass constraints in 9 of 9 (1.86), so the transport constraint is what helps.
   - Each method tuned to its best setting is a different matter: best OT does not beat best kNN (5 of 9). When betting on one to three partners per cell, the nearest neighbour is the better estimator in the G-csf and M-csf conditions.

**Three post hoc positives went to pre-registration in this period: OT interpolation of erythroid stages, OT-based reweighting for forecasting them, and OT > kNN couplings in vitro. Two did not replicate; one survived, with a narrower claim than it started with.**

Two further results are about method:

- **Required rivals.** The 50/50 mixture of the neighbouring samples, random pairing and persistence must be reported next to any OT interpolation. Beating each neighbour alone, which OT pairs do in 11 of 20 units, is not evidence of anything.
- **A defect in the in vivo LARRY data.** The clone matrix on GEO (GSM4185643), and the identical copy on the authors' website, carries no cell-level fate information inside recipient mice. Same-type sister enrichment is 0.98–1.00, equal to its permutation null (0.99–1.00), in all 10 mice, against 3.2–4.3 in vitro. The pre-registered in vivo replication is therefore reported as not evaluable rather than as a failure.

# 2. Ground rules shared by every test

- **Pre-registration.** Every confirmatory test has a written pre-registration (hypotheses, units, decision rule, pre-stated readings, caveats), frozen before any quantity of interest was computed on its data. The sha256 of the pre-registration and of every script, as copied to the cluster, is recorded in the decision log (§8). After the first result nothing was changed; every later analysis is labelled exploratory or post hoc.
- **Decision rule.** "A beats B" means A is better in at least about two thirds of the units (the exact count is stated per test) and better on average (mean, or geometric mean of ratios).
- **Strictness.**
  - Units are whole biological samples: held-out time points, wells, embryo pools, mice, cytokine conditions.
  - Gene selection and PCA are fitted only on the cells a method is allowed to see.
  - No method sees the held-out sample or a later time point.
- **Rivals.**
  - For populations: each neighbouring sample on its own (persistence), the equal-size mixture of the two neighbours ("mix"), and the midpoints of randomly paired cells ("random pairs").
  - For couplings: k-nearest neighbours, and a row-normalised Gaussian kernel with the same cost and temperature as OT, i.e. OT without its mass constraints ("soft"). Where informative, also a coupling that pairs each cell uniformly with cells of the same annotated type.
- **Metrics.**
  - Populations: the U-statistic energy distance (Székely & Rizzo 2013) in a PCA space fitted on the observed cells.
  - Couplings: clonal enrichment, the mass a coupling P puts on pairs of cells that share a lineage barcode, relative to random partners: E(P) = [Σ~ij~ P~ij~ 1(c~i~ = c~j~) / Σ~ij~ P~ij~] / Σ~c~ a~c~ b~c~. Here a~c~ and b~c~ are the fractions of source and target cells in clone c, and 1 is chance.
  - Spread: exp(mean row entropy of P), the effective number of partners per source cell.
- **Code.** Sinkhorn, couplings, metrics and harnesses are original. External code is used only where the comparison requires it: the Dileo & Sottoriva training loop in §3.

# 3. Per-stage statistics: the Dileo & Sottoriva expression task

**The benchmark.** Dileo & Sottoriva (2026; arXiv 2605.28659, version 1, presented as work in progress) benchmark temporal graph neural networks on gene regulatory networks inferred per pseudotime stage. Their task 2 predicts, for held-out genes, the change of expression to the next stage. It is scored by Pearson correlation (PCC) and by precision among the 200 most increased and most decreased genes (P@200 up / down).

**How we ran it.** We worked inside their released code (anonymous repository tgl-grn-1CCD): their data loaders, gene masks, metric functions and five seeds. We used the "-human" files, which are the ones that reproduce their Table 2. The check was their Linear row:

| Data set | PCC | P@200 up | P@200 down |
|---|---|---|---|
| Pancreas, our re-run | 0.1068 | 0.0618 | 0.0561 |
| Pancreas, paper | 0.107 | 0.058 | 0.057 |
| Gastrulation, our re-run | 0.0619 | 0.0414 | 0.0069 |
| Gastrulation, paper | 0.062 | 0.041 | 0.007 |

**The pre-registered model.** DistBridge predicts each gene's next-stage distribution from past stages only. It has three parts:

- a robust level estimate;
- an OT / Schrödinger-bridge drift step, which a validation on past transitions can switch on or off;
- a finite-cell observation head that returns the expected value of the clipped target.

Its gates were (G1) beating the best row of the paper in all six cells, and (G6) showing that the OT step contributes.

Table 1. Task 2 of Dileo & Sottoriva, five seeds, their code and metrics.

- **Their best**: the best of their models in each cell.
- **Their best, re-run**: their models re-run by us under their protocol.
- **Their best, corrected**: the same models with one correction, fixed before the result: a transition is scored before any of its genes is used for training.
- **No-learning rule**: the mean of the last six (gastrulation) or four (pancreas) stages, run in their loop.

| Data set | Metric | Paper, best | Their best, re-run | Their best, corrected | DistBridge (ours) | No-learning rule |
|---|---|---|---|---|---|---|
| Gastrulation-human | PCC | 0.400 | 0.386 | 0.313 | **0.585** | 0.398 |
| | P@200 up | 0.110 | 0.084 | 0.073 | **0.406** | 0.257 |
| | P@200 down | 0.387 | 0.367 | 0.329 | **0.541** | 0.448 |
| Pancreas-human | PCC | 0.434 | 0.411 | 0.249 | **0.488** | 0.332 |
| | P@200 up | 0.178 | 0.155 | 0.124 | **0.301** | 0.238 |
| | P@200 down | 0.354 | 0.334 | 0.255 | **0.442** | 0.313 |

**Result.**

- **G1 passed.** The lower 95 % bound of DistBridge is above the paper in every cell.
- **G6 failed.** The OT step was selected in 3 % (gastrulation) and 14 % (pancreas) of transitions. Without it the numbers are unchanged: gastrulation 0.585 / 0.406 / 0.541, pancreas 0.491 / 0.303 / 0.443.

**Why the OT step does nothing here** (exploratory signal analysis):

- With the original target, 91–95 % of the variance lies in the 8–9 % of genes whose stage mean crosses zero. This is an artefact of averaging a few hundred cells per stage.
- Consecutive changes are anti-correlated (lag-1 correlation −0.47 on gastrulation, −0.40 on pancreas), so any trend extrapolation hurts.
- Graph features add nothing measurable (PCC 0.610 against 0.611).
- An unbalanced OT variant with a sink for cells that die or change state does not help either.

**Notes on the released code** (each verified by re-running their code under both protocols; summarised here because they change the numbers):

- Models are trained on 75 % of the genes of the same transition on which they are scored, whereas the paper describes scoring before updating. Correcting this lowers every one of their models, by −0.07 to −0.19 PCC, and leaves ours unchanged (Table 1).
- The target in the code is a log-ratio clipped at ±5 with ε = 10^−8^, not the x(t+1) − x(t) written in the paper. It is close to three-valued, and PCC on it mostly measures switching on and off.
- P@200 depends on how ties are ordered. With ties counted fairly, their P@200 up falls to 0.06–0.08 on the original target.

Under all corrections and with a log(1 + counts) target, our model wins 5 of 6 cells: it loses pancreas P@200 down, 0.215 against 0.235.

**Lesson.** Per-stage averages leave OT nothing to transport. OT needs data where individual cells can be matched across times. This is what moved the work to cell-level time courses (§4–§5).

# 4. Reconstructing populations at unobserved times

## 4.1 Methods compared

**Interpolation.** Within each unit, cells at the two neighbouring times t−1 and t+1 are used to predict the population at t:

- **prev / next**: either neighbour as it is;
- **mix**: an equal-size sample from both neighbours;
- **random pairs**: midpoints of randomly paired cells;
- **OT pairs**: McCann displacement interpolation (McCann 1997), i.e. midpoints of pairs drawn from the entropic OT plan (Cuturi 2013);
- **SB**: the Schrödinger-bridge midpoint, i.e. OT pairs plus Brownian-bridge noise;
- **UOT**: unbalanced OT with a growth term (Chizat et al. 2018), in the spirit of Waddington-OT (Schiebinger et al. 2019).

**Forecasting** from origin s to s+1:

- **persistence**: the population at s as it is;
- **mean shift**: the population at s shifted by the last observed change of mean;
- **OT reweighting**: the cells at s reweighted by the growth estimated with unbalanced OT;
- **plain reweighting**: the same reweighting without OT.

Hyperparameters were chosen only by backtesting on earlier transitions.

## 4.2 Where the hypotheses came from: erythroid stages (exploratory)

**Data.** The erythroid lineage of the mouse gastrulation atlas: 9,815 cells at seven real embryonic stages, E7.0–E8.5, a quarter of a day apart.

**Pre-stated forecasting criterion.** An OT family must beat persistence and mean shift in at least 3 of 4 origins. It was not met: with the corrected metric, OT won 1 of 4.

**Post hoc signals** (24 method families, not pre-registered):

- OT reweighting passed on both metrics.
- OT pairs and UOT interpolation in PCA space beat the mix in 3 of 4 held-out stages. Mean energy distance: OT pairs 1.33, UOT 1.24, mix 1.47.

These two signals are what the pre-registered tests below were built to confirm.

## 4.3 Pre-registered tests

Table 2. Interpolation and forecasting, pre-registered.

- Counts are units in which the method has a lower energy distance than the rival.
- The pre-registered OT hypotheses required beating prev, next and the mix (≥ 5/7, 6/9 and 3/4 units respectively, plus the mean). They failed in all three data sets.

| Test (date) | Data | OT pairs beat the mix | UOT / SB beat the mix | OT pairs beat random pairs | Forecast: OT reweighting beats persistence and mean shift |
|---|---|---|---|---|---|
| Atlas confirmation (25 Oct) | Mouse gastrulation atlas without the blood lineage (Pijuan-Sala et al. 2019), 9 stages E6.5–E8.5, ≤ 3,000 cells per stage | 0 / 7 | 3 / 7, 0 / 7 | 7 / 7 | 1 / 6 |
| Zebrafish replication (27 Oct) | Saunders et al. 2023, uninjected controls, 11 times 18–38 hpf every 2 h, 47–48 embryos per time | 1 / 9 | 2 / 9, 2 / 9 | 9 / 9 | 0 / 8 |
| LARRY in vitro (29 Oct) | Weinreb et al. 2020, day 4 predicted from days 2 and 6, 2 populations × 2 wells | 0 / 4 | 0 / 4, 0 / 4 | 4 / 4 | — |

Two pre-registered alternatives that drop OT altogether also failed:

- **Embryo age instead of the stage label** (26 October, atlas). Each embryo pool gets a leave-one-pool-out estimated age, and the prediction is a kernel mixture of real cells weighted by age. It beats the mix in 3 of 7 held-out stages and persistence in 1 of 6 origins.
- **Dynamic mixture model** (29 October). Cells are assigned to 100 k-means states, and a multinomial-logit model of the state shares over time is fitted. It beats the mix in 0 of 9 (zebrafish) and 3 of 7 (atlas), and persistence in 1 of 8 and 0 of 6.

![Figure 1. Interpolation error relative to the 50/50 mixture of the two neighbouring samples, for every held-out time point of the three pre-registered tests (log scale; below the dashed line = better than the mix). OT pairs (blue circles) beat random pairs (orange squares) everywhere, but are worse than the mix in 19 of 20 units.](figures/fig1_interpolation_vs_mix.png){width=100%}

## 4.4 Why the mixture wins (exploratory diagnosis)

- **Neighbouring stages overlap.** On the atlas, the distance between neighbouring stage centres is 0.36–0.55 times the within-stage radius; for the erythroid lineage, where the post hoc signal came from, it is 1.18–1.93. A held-out atlas stage is a mixture of states already present on both sides, not a midpoint along a path, and the mix reproduces exactly that. Even with oracle mixture weights, OT loses to the best mixture in 7 of 7 atlas stages.
- **A stage label is not an embryo age.** A leave-one-pool-out clock places 14 of 24 atlas pools closer to another stage, and a pool's age predicts its oracle mixture weight (Spearman 0.79). Estimating age from expression (the 26 October test) was as noisy as the asynchrony it meant to remove (σ 0.6–0.8 stages) and was confounded with batch.
- **Zebrafish: there is room, but little moves.** One clock, one batch and 48 embryos per time leave room for improvement: another embryo of the same time is 3–8 times closer to the held-out sample than the mix is. But at 2-hour steps little moves (energy distance between consecutive times 0.007–0.06, except 30 → 32 hpf). The pace is also not uniform: 30 hpf sits with 28 hpf, and 32 hpf with 34 hpf, so "halfway" is the wrong target for the mix and for OT alike. Conditioning on cell type improves the mix by only 1–10 %.
- **Shares should be interpolated linearly.** Interpolating state shares in log-odds (Aitchison geometry) is worse than linear mixing, because the geometric mean wipes out states present on one side only.
- **The coupling is informative; the midpoint construction is what fails.** OT pairs beat random pairs in every unit of every test (20 of 20). In gene space the OT midpoints are also too smooth. This points to the coupling as the useful object, which §5 tests directly against lineage barcodes.

# 5. Who becomes whom: couplings scored against lineage barcodes

## 5.1 Data and methods

**LARRY** (Weinreb et al. 2020) labels progenitor cells with heritable barcodes and samples them at several days. Cells that share a barcode are sisters. A coupling between day-a and day-b cells is good if it puts its mass on sister pairs.

Couplings compared, all in a PCA space fitted on source and target cells without barcodes:

- **OT**: balanced entropic OT; Sinkhorn, ε relative to the median cost, 2,000 iterations, marginal error recorded;
- **kNN**: uniform weight over the K nearest target cells;
- **soft**: the same Gaussian kernel as OT, row-normalised, i.e. without the mass constraints;
- **cell type**: uniform weight over target cells of the same annotated type.

## 5.2 Pre-registered test in vitro (29 October)

**Units.** Four transitions: two starting populations × (day 2 → 4, day 4 → 6), with 473–846 shared clones per unit.

**Result.**

- OT at ε = 0.05 beats kNN with K = 10 in 3 of 4 units: mean enrichment 4.39 against 3.76 (unit by unit 2.89 / 6.18 / 4.23 / 4.25 against 2.12 / 3.02 / 3.12 / 6.76). **Pass.**
- OT beats the cell-type coupling in 4 of 4 units (mean 1.87). **Pass.**

This was the first pre-registered positive for OT in this work, but it rests on 4 units, and kNN wins clearly in the fourth.

## 5.3 Exploratory: did OT win by concentrating its bets?

The reading rule was written before the computation.

**OT was not concentrating its bets.** The frozen OT was more diffuse than kNN, with 133–367 effective partners per cell against 10.

**At matched spread** (log-log interpolation along each method's parameter grid):

- OT / kNN = 2.03, 3.27, 2.61, 1.28 (4 of 4 units). Every 80 % subsample was also above 1.
- OT / soft = 1.83, 2.05, 2.08, 1.11 (4 of 4 units).

**Best of each grid, each method tuned on the truth:** OT and kNN split 2–2. In the Sca1− population kNN with K = 1–3 wins: 6.56 against 5.83, and 10.09 against 6.46. A post hoc sharper OT (ε down to 0.005) does not change this: 6.20 against 6.56, and 7.24 against 10.09.

**Fixed reading: mixed.** This defined the three questions for replication:

- **H1**: OT beats kNN at matched spread;
- **H2**: OT beats soft at matched spread;
- **H3**: best OT beats best kNN.

## 5.4 In vivo replication: not evaluable

**Design.** GSM4185643 is the transplant arm of LARRY. The 8 units are day-2 cohort sample → recipient mouse of the same cohort.

**The formal verdict fails:** H1 4/8, H2 3/8, H3 3/8, with geometric means 0.99, 0.99 and 0.75. But every method is at chance: OT enrichment is 0.86–1.08, and the OT plan is almost uniform (1,513–1,778 effective partners out of about 6,000).

The exploratory integrity checks run afterwards (Table 3) locate the problem in the data. Other checks confirmed that the rest of the files line up:

- Expression and metadata are aligned: marker genes match the annotated types.
- Library blocks are aligned: re-sorting the libraries breaks the cohort structure.
- None of the within-library re-orderings tested restores the sister signal.
- The May-2020 copies on the authors' website are identical to GEO after decompression.

The original study reports that sisters split between different mice share their dominant fate far more often than chance. The distributed files cannot reproduce any sister fate concordance even inside a single mouse.

We did not try to infer the right cell order from expression: that would inject the quantity under test into the labels. The in vivo test is therefore recorded as not evaluable: a data defect, not a failed replication.

Table 3. Clone-label integrity: same-type sister enrichment (each cell paired uniformly with the other cells of its annotated type, scored with the clonal enrichment of §2) against a type-label permutation null. The null is below 1 by construction when most clones are singletons within a group, so values are read against the null, not against 1.

| Data | Groups | Enrichment | Permutation null |
|---|---|---|---|
| LARRY in vitro (positive control) | time point × population | 3.20–4.30 | 0.64–0.97 |
| LARRY in vivo, inside each mouse | 10 mice | 0.98–1.00 | 0.99–1.00 |
| LARRY in vivo, day-2 sample (sister coherence, 10 nearest neighbours) | 8 units, 2 cohorts | 26–44 | 1 (chance) |
| LARRY cytokine perturbation | 10 time × condition groups | 1.53–4.94 | 0.29–0.89 |

## 5.5 Pre-registered replication on the cytokine experiment (30 October)

**Design.** GSM4185644 (Weinreb et al. 2020) is a separate barcoding experiment from the same laboratory:

- 65,075 cells, of which 25,749 carry exactly one barcode (5,257 clones);
- Lin−Kit+ cells barcoded and grown in the full cocktail, sampled on day 2;
- then split into Epo, G-csf, M-csf, SCF and full-cocktail cultures, sampled on days 4 and 6 (SCF on day 4 only).

**Integrity check, run before freezing.** It passes in every group (Table 3).

**Units, fixed before any result.** Consecutive transitions within a condition with at least 50 shared clones. The design predicted 9 units, and the preprocessing produced exactly those 9 with the predicted sizes.

**Code.** Methods, grids and the decision rule (at least 6 of 9 units plus geometric mean above 1) were imported unchanged from the frozen in vivo script.

Table 4. Cytokine perturbation, per unit.

- "Spread": effective partners of OT at ε = 0.05.
- "OT/kNN" and "OT/soft": ratios at OT's spread.
- The last three columns compare each method tuned to its best setting.

| Unit | Source → target cells | Shared clones | Spread | OT enrich. | OT / kNN (H1) | OT / soft (H2) | Best OT | Best kNN | Best / best (H3) |
|---|---|---|---|---|---|---|---|---|---|
| Epo 2→4 | 3,776 → 1,128 | 195 | 220 | 5.52 | 3.70 | 3.90 | 9.36 (ε 0.005) | 1.79 (K 50) | 5.24 |
| G-csf 2→4 | 3,776 → 2,121 | 387 | 575 | 2.31 | 2.05 | 1.94 | 2.58 (ε 0.02) | 4.24 (K 2) | 0.61 |
| M-csf 2→4 | 3,776 → 2,109 | 396 | 325 | 1.84 | 1.70 | 1.56 | 1.84 (ε 0.05) | 2.85 (K 3) | 0.65 |
| SCF 2→4 | 3,776 → 2,432 | 403 | 179 | 2.37 | 1.77 | 2.09 | 2.70 (ε 0.02) | 1.90 (K 1) | 1.42 |
| Full 2→4 | 3,776 → 282 | 101 | 36 | 2.57 | 2.17 | 2.47 | 2.90 (ε 0.02) | 1.37 (K 20) | 2.12 |
| Epo 4→6 | 1,128 → 1,202 | 143 | 201 | 1.90 | 1.28 | 1.12 | 3.26 (ε 0.005) | 1.94 (K 2) | 1.68 |
| G-csf 4→6 | 2,121 → 1,852 | 260 | 150 | 1.37 | 1.53 | 1.52 | 1.65 (ε 0.005) | 2.95 (K 1) | 0.56 |
| M-csf 4→6 | 2,109 → 5,131 | 671 | 759 | 1.74 | 1.54 | 1.22 | 1.74 (ε 0.05) | 2.66 (K 1) | 0.66 |
| Full 4→6 | 282 → 5,716 | 163 | 590 | 3.27 | 2.73 | 2.05 | 4.38 (ε 0.005) | 1.98 (K 100) | 2.21 |
| **Verdict** | | | | | **9 / 9, 1.95: pass** | **9 / 9, 1.86: pass** | | | **5 / 9, 1.27: fail** |

**Descriptive checks:**

- On 80 % subsamples, the H1 and H2 ratios stay above 1 in 27 of 27 and 27 of 27 cases.
- All Sinkhorn plans converged (maximum marginal error 1.9 × 10^−6^).
- OT's spread lies inside the kNN and soft grids in every unit.
- The five day 2 → 4 units share one source sample, so they are correlated. The four day 4 → 6 units have different sources and pass H1 and H2 4 of 4 on their own.

![Figure 2. Clonal enrichment against spread (effective partners per source cell, log scale) for two units of the cytokine experiment. Left, Epo day 2 → 4: OT (blue circles) dominates kNN (orange squares) and the soft kernel (aqua triangles) at every spread. Right, G-csf day 2 → 4: OT wins at the pre-registered setting (ringed point, ε = 0.05) and at every grid point above about 35 partners, but a two-neighbour kNN is the best single estimator. The dotted line marks chance (enrichment 1).](figures/fig2_enrichment_vs_spread.png){width=100%}

![Figure 3. OT's advantage at matched spread, unit by unit: ratio of OT's clonal enrichment to kNN's (orange squares) and to the soft kernel's (aqua triangles) at OT's own spread (log scale; above the dashed line = OT better). Left, LARRY in vitro (exploratory robustness check, 4 units). Middle, LARRY cytokine perturbation (pre-registered, 9 units). Right, LARRY in vivo (8 units, shown in grey: the clone labels fail the integrity check, so every method is at chance and the test is not evaluable).](figures/fig3_matched_spread_ratios.png){width=100%}

**Post hoc reading of the curves** (not pre-registered):

- At every ε ≥ 0.02, i.e. spreads from about 10 to about 2,000 partners, OT is at least as good as kNN at matched spread in 9 of 9 units.
- At the sharpest settings (ε 0.005–0.01, under about 20 partners), kNN is better in three units: G-csf 2 → 4, M-csf 2 → 4 and M-csf 4 → 6.
- So OT's advantage lies in the diffuse regime, where it keeps putting mass on sisters while local similarity dilutes. It does not lie in picking the single nearest partner.

# 6. What can and cannot be claimed

**Supported:**

- On LARRY haematopoiesis in culture, at equal spread in the moderate-to-large range (tens to hundreds of partners per cell), the OT coupling puts more mass on sister cells than k-nearest neighbours and than the same kernel without mass constraints. It was seen in one experiment (exploratory, 4/4) and confirmed by pre-registration on a second, independent experiment from the same laboratory (9/9 and 9/9).
- On every data set tested, including the barcoded one where its coupling is good, OT does not reconstruct the population at an unobserved time better than the 50/50 mixture of the neighbouring samples. Neither do the two alternatives tested in its place (estimated embryo age, dynamic mixture).
- On per-stage summary statistics, the Dileo & Sottoriva task-2 margin is achievable without OT. It comes from handling the sampling noise of the stage averages.

**Not supported:**

- "OT is the best lineage-coupling estimator": best-tuned OT loses to best-tuned kNN in 4 of 9 cytokine units and 2 of 4 in vitro units.
- Any in vivo statement, since the only in vivo barcode data tested is not evaluable.
- Any statement outside haematopoiesis in culture.

**Why a good coupling gives a poor snapshot.**

- A displacement midpoint smooths the population along each pair; a mixture keeps its composition.
- When neighbouring times overlap and change is mostly a shift in the proportions of states already present, composition is what matters, and the mixture wins.
- The coupling is the useful object. Its interpolant is not.

# 7. Limitations and next steps

**Limitations:**

- There is one pre-registered confirmation of the coupling result. It comes from the same laboratory, platform and culture system as the exploratory finding, and five of its nine units share a source sample.
- Matched-spread values come from log-log interpolation along parameter grids. All OT spreads were inside the grids in the cytokine test, but not in the in vitro Sca1− units (kNN capped at K = 200 there, which makes the ratio conservative for OT).
- Enrichment is carried by the small fraction of cells with a sister in the other sample.
- All couplings were computed in PCA spaces; gene-space and learned representations were not tested.
- The couplings were scored, not used. Whether better couplings improve a downstream dynamical model, such as flow matching trained on OT pairs, is untested.

**Next steps:**

1. An independent system outside haematopoiesis with barcodes over time, e.g. CellTag direct reprogramming (Biddy et al. 2018). This requires a new download and a new pre-registration.
2. A test of whether OT couplings improve fate prediction or a trained flow over kNN couplings at equal spread.
3. Reporting the GSM4185643 cell–clone alignment issue to the data authors.

# 8. Reproducibility

Every confirmatory test below was frozen before its result. The sha256 refers to the copy on the cluster (LF line endings); the decision log `docs/superpowers/decisions.md` lists the sha256 of every script with the job identifiers. Pre-registrations, results (JSON) and the original scripts are in the `prereg/`, `results/` and `code/` folders of this deliverable.

| Test | Pre-registration (sha256, first 16) | Cluster job(s) | Results |
|---|---|---|---|
| DistBridge vs Dileo & Sottoriva task 2 | 2026-10-22-dileo-task2-distbridge-prereg.md (6140681cdd014e9a) | 712958, 712959 | dileo_task2/distbridge/ |
| Atlas confirmation | 2026-10-25-atlas-confirmation-prereg.md (7cb27ba3d237775e) | 718022 | atlas/confirm.json |
| Embryo age | 2026-10-26-embryo-age-prereg.md (84476e4b2d9808ef) | 719924 | atlas/age/age_confirm.json |
| Zebrafish replication | 2026-10-27-zebrafish-replication-prereg.md (8ee9bd6258392b3e) | 722134 | zf/confirm_zf.json |
| Dynamic mixture | 2026-10-29-dynmix-prereg.md (f84464e245b80789) | 726416, 726417 | zf/dynmix_zf.json, atlas/dynmix_atlas.json |
| LARRY in vitro | 2026-10-29-larry-prereg.md (8bf53eaf1e2bb00e) | 726981 | larry/larry_confirm.json |
| LARRY robustness (exploratory) | reading rule in the decision log | 728038, 728043 | larry/larry_robust.json, larry/larry_sharp.json |
| LARRY in vivo (not evaluable) | 2026-10-30-larry-invivo-prereg.md (fa904bfac8226c2c) | 728790 | larry_vivo/larry_vivo_confirm.json |
| LARRY cytokine perturbation | 2026-10-30-larry-cytokine-prereg.md (6e04bb9fbf27eb0f) | 729843, 729844 | larry_cyto/larry_cyto_confirm.json |

Data were downloaded with permission to the Bocconi cluster from GEO: GSM4185642 (in vitro), GSM4185643 (in vivo), GSM4185644 (cytokine perturbation), GSE202639 (zebrafish). The atlas and erythroid data were already on the cluster. Raw data and the Dileo & Sottoriva code are not included.

# References

- Biddy BA, Kong W, Kamimoto K, et al. Single-cell mapping of lineage and identity in direct reprogramming. *Nature* 564, 219–224 (2018).
- Chizat L, Peyré G, Schmitzer B, Vialard F-X. Scaling algorithms for unbalanced optimal transport problems. *Mathematics of Computation* 87, 2563–2609 (2018).
- Cuturi M. Sinkhorn distances: lightspeed computation of optimal transport. *Advances in Neural Information Processing Systems* 26 (2013).
- Dileo M, Sottoriva A. Applications of temporal graph learning for predicting the dynamics of biological systems. arXiv:2605.28659 (2026).
- McCann RJ. A convexity principle for interacting gases. *Advances in Mathematics* 128, 153–179 (1997).
- Pijuan-Sala B, Griffiths JA, Guibentif C, et al. A single-cell molecular map of mouse gastrulation and early organogenesis. *Nature* 566, 490–495 (2019).
- Saunders LM, Srivatsan SR, Duran M, et al. Embryo-scale reverse genetics at single-cell resolution. *Nature* 623, 782–791 (2023).
- Schiebinger G, Shu J, Tabaka M, et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell* 176, 928–943 (2019).
- Székely GJ, Rizzo ML. Energy statistics: a class of statistics based on distances. *Journal of Statistical Planning and Inference* 143, 1249–1272 (2013).
- Weinreb C, Rodriguez-Fraticelli A, Camargo FD, Klein AM. Lineage tracing on transcriptional landscapes links state to fate during differentiation. *Science* 367, eaaw3381 (2020).
