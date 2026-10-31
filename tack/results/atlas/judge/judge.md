# Judge: atlas confirmation of the two erythroid OT signals (2026-10-25)

Scope. I am the judge of the four checkers (verdict, audit, floor, mechanism). I re-ran the frozen pipeline myself,
recomputed the verdict with my own code, and re-derived every number the final report would lean on. All scripts and
outputs are in `tack/results/atlas/judge/`. Nothing frozen was touched (frozen code, confirm.json, prereg,
decisions.md). Everything after section 1.4 is **EXPLORATORY**. None of it changes the pre-registered verdict. Only a
bug in the frozen pipeline could do that, and that would need a new pre-registered run.

## In breve (italiano semplice)

- **Il verdetto pre-registrato è giusto e non c'è nessun errore nel codice.** Ho rifatto girare il codice congelato
  sui dati locali e ottengo gli stessi 1492 numeri (differenza massima 4.5e-13). Ho ricalcolato i cinque verdetti con
  un mio script e escono uguali: H1 FALLISCE, H1b FALLISCE, H1c PASSA, H2 FALLISCE, H2b FALLISCE.
- **Non è una sconfitta di poco.** Il mix 50/50 dei due stadi vicini batte l'OT a coppie anche cambiando le cellule
  di riferimento (0/7 in 11 prove su 12, 1/7 nell'ultima) e togliendo un gruppo di embrioni alla volta. L'unica
  eccezione è E8.25 (k7), dove i due metodi sono pari entro il rumore.
- **Perché l'OT perde sull'atlante.** Lì lo stadio di mezzo assomiglia a un miscuglio di cellule "di prima" e "di
  dopo", non a cellule a metà strada. Gli stadi vicini si sovrappongono molto: la distanza tra i loro centri è
  0.4-0.55 volte il raggio della nuvola, contro 1.2-1.9 negli eritroidi. Mettere punti a metà strada stringe la
  nuvola e peggiora il risultato. Anche con il tempo t migliore scelto guardando la risposta, l'OT non batte il
  miscuglio migliore: 0/7.
- **Il segnale eritroide era vero, ma solo per quei dati.** Sugli eritroidi resiste al rumore (95-100 % delle
  ripetizioni). Però si capovolge nello spazio dei geni, vale in solo 9 gruppi di embrioni su 15 e non vale dentro i
  singoli tipi cellulari (1/5). Anche punti a metà strada presi a caso battono il mix in 3 stadi su 4, e danno già
  il 33-70 % del guadagno: in buona parte è la geometria di una sola linea che matura in fretta, e l'accoppiamento OT
  aggiunge il resto. Come idea generale non regge.
- **Previsione (uot_rw).** Anche scegliendo i parametri a posteriori, il guadagno è circa 1 % in tre origini e conta
  solo a E8.25 (35 %). Il passo precedente non dice quale parametro andrà bene al passo dopo: correlazione 0.21
  sull'atlante contro 0.90 sugli eritroidi.
- **Cosa fare.** Fermare questa linea come risultato positivo e scriverla come risultato negativo e di metodo. Come
  base di confronto bisogna usare il mix 50/50, non solo le coppie a caso. Un altro test ha senso solo con una nuova
  pre-registrazione, su un terzo dataset e con l'ipotesi ristretta.

## 0. What I ran (all local; ~10 min total)

| script | what it tests | output |
|---|---|---|
| `python tack/hpc/atlas_confirm.py --data <scratchpad/atlas> --out judge/confirm_rerun.json` (unmodified, 112 s) | reproduction of the HPC run | `confirm_rerun.json`, `confirm_rerun.log` |
| `j_diff.py` | leaf diff of the re-run against confirm.json | `j_diff.json` |
| `j_verdict.py` | sha256, my own rule (both readings), secondary metrics, exploratory facts | `j_verdict.json` |
| `j_products.py` | stage indexing, cell identity across files, pools, PCA strictness, cc genes, Sinkhorn marginals, kernel underflow, prereg sha and mtimes | `j_products.json` |
| `j_noise.py` | ot_pairs/uot vs union under target resampling, complement target, leave-one-held-pool-out and 20 fit reps (atlas + erythroid) | `j_noise.json/.log` |
| `j_forecast.py` | hindsight grid for uot_rw/tilt_rw, rolling-choice ranks, backtest-final correlation (atlas + erythroid) | `j_forecast.json/.log` |
| `j_mixture.py` | oracle-best mixture vs McCann at t = 0.5 and at the oracle-best t, spreads, ED mixture identity (both datasets) | `j_mixture.json/.log` |
| `j_restore.py` | McCann with data-shaped residual noise (calibrated on held halves); neighbour-separation statistics | `j_restore.json/.log` |
| `j_within.py` | ot_pairs vs union inside single cell types (both datasets) | `j_within.json/.log` |
| `j_lopo.py` | leave-one-pool-out replicate vs union / ot_pairs per held pool (both datasets) | `j_lopo.json` |
| `j_comp.py` | cell-type composition: is the held stage a mixture of its neighbours? | `j_comp.json` |
| `j_eps_cc.py` | ot_pairs at eps_rel 0.01-0.2; uot with the 96-gene score or with no growth prior | `j_eps_cc.json/.log` |
| `j_ery_indep.py` | share of the erythroid ot_pairs gain over union already given by random midpoints | `j_ery_indep.json` |

## 1. The pre-registered verdict is correctly computed, and no bug can flip it

1. **Reproduction.** I ran the unmodified frozen `atlas_confirm.py` on the local atlas products.
   - All 1492 leaves of confirm.json match (1455 numeric). The largest difference is 4.5e-13 (`forecast/7/uot_rw/ess`).
   - No key is missing, no non-numeric value differs, and the verdict block is identical (`j_diff.json`).
   - Conclusion: the local products equal the HPC products in every array the run uses, and the run is deterministic.
2. **Frozen files.**
   - The LF sha256 of atlas_prep.py, atlas_confirm.py, interp_bench_frozen.py and ery_forecast.py equals prereg lines
     37-40.
   - The prereg file's own sha256 is 7cb27ba3...b5ff3b. That is the value logged in decisions.md before the result
     (`j_verdict.json`, `j_products.json`).
3. **Verdicts recomputed with my own rule.** The rule is need = ceil(2n/3) = 5 of 7 and 4 of 6, a tie counts as a
   loss, and both readings were checked: per baseline, and joint as in the frozen `beats()`. Both readings agree for
   all five hypotheses, and all five match confirm.json (`j_verdict.json`).

   | hypothesis | result | per baseline (W/T/L) | mean of A vs mean of B |
   |---|---|---|---|
   | H1 ot_pairs | **FAIL**, joint 0/7 | prev 6/0/1, next 6/0/1, union 0/0/7 | 0.6151 vs 1.2072 / 0.9369 / **0.5073** |
   | H1b uot | **FAIL**, joint 2/7 (k6, k7) | union 3/0/4 | 0.6149 vs 0.5073 |
   | H1c pairing | **PASS**, 7/7 | indep 7/0/0 | 0.6151 vs 1.0063 |
   | H2 uot_rw | **FAIL**, joint 1/6 (s6) | persist 1/3/2, meanshift 6/0/0 | 0.8154 vs persist 0.7708 |
   | H2b vs tilt_rw | **FAIL**, 2/6 | 2/2/2 | 0.8154 vs 0.7297 |

   - The interp unit value is exactly the mean of the 3 reps (max diff 0.0).
   - In H2, uot_rw equals persist bit for bit at s = 2, 3, 5, because λ = 0 was chosen there.
   - Counting those ties as wins is not the rule. It would give 4/6, and H2 would still fail on the mean
     (0.8154 vs 0.7708).
4. **Products and solver** (`j_products.json`).
   - Stage indexing:
     - `stage_fit` is exactly {k-1, k+1}, with the prev block first.
     - The held cells equal origin(k-1).fut, cell by cell (cell type and sample, in order), and the next block equals
       origin(k).fut.
     - Pools per stage are 3,1,6,3,6,4,4,3,4, as in the prereg.
   - PCA strictness: the PCA sits 0.014-0.075° from an SVD of the file's own fit cells, against 38-87° once the held
     stage is added.
   - Proliferation genes: 6, 3, 4, 6, 8, 6, 6 S/G2M genes are among the HVGs (= confirm.json `cc_genes`).
   - Sinkhorn: in the rep-0 balanced plans the largest marginal error is 2.0e-6, with no log-domain fallback.
   - kernel_regress: the largest row-minimum exponent is 11.0, far from the 745 underflow limit.
5. **The one weakness the audit flagged.** Only 3-8 HVG S/G2M genes feed the uot growth prior. It cannot flip H1b
   (`j_eps_cc.py`):
   - With the stored 96-gene score, uot beats union 2/7 (joint 1/7, mean 0.587).
   - With no prior at all (β = 0), it beats union 1/7 (joint 0/7, mean 0.575).
   - In both cases union is 0.507.
   - **No demonstrated bug.**
6. **EXPLORATORY: how robust is the H1 failure?** (`j_noise.py`). The frozen reps vary only the fit-side draws.
   - **Other targets.** I re-scored the frozen predictions against 10 other target subsamples and against the
     complement target (held cells the frozen target never used). The H1 count of ot_pairs vs union stays **0/7 at
     11 of 12 targets** (1/7 at one).
   - **20 fit reps × 11 targets** (fraction of draws in which ot_pairs < union):

     | | k1 | k2 | k3 | k4 | k5 | k6 | k7 |
     |---|---|---|---|---|---|---|---|
     | fraction ot_pairs < union | 0 | 0.08 | 0 | 0 | 0 | 0.005 | **0.35** |
     | paired gap, ot_pairs − union | 0.209 | 0.020 | 0.088 | 0.143 | 0.157 | 0.054 | 0.008 |
     | sd of the gap | 0.067 | 0.015 | 0.038 | 0.051 | 0.053 | 0.025 | 0.024 |

   - **Leave one held pool out.** The sign never flips at k2-k6. It flips at k7 only when pool 28 is dropped (−0.034).
   - **Reading.** There are 6 robust losses and one tie within noise (k7, E8.25). k2 is a small loss (2.7 % of union)
     but a consistent one, not noise.
   - **For comparison.** Under a coin-flip null, P(≥ 5/7) = 0.23; against union we observed 0/7 (P = 1/128,
     nominal; the units are not independent).
   - **Size of the gap.** The mean relative gap is +17/+3/+13/+45/+63/+26/+9 % of union. The nominal t-interval of the
     absolute gap is [0.026, 0.189]. An erythroid-sized OT gain (13-19 % below union) is excluded at 6/7 atlas units.

**H1c is a low bar.** indep loses to union at 7/7, and union < ot_pairs < indep holds at 7/7. So H1c shows only that
the OT coupling is far better than random pairing. To my knowledge that is the usual validation of OT interpolation;
I have not checked the literature. On these data it says nothing about beating the 50/50 mixture.

## 2. The checkers' claims, re-tested

**Verdict checker. Everything reproduced.**
- The verdict numbers above.
- Gene space: ot_pairs vs union is 0/7 (mean 0.7156 vs 0.388), and the other per-baseline counts are identical.
- Per pool: ot_pairs < union in 5/27 cells, uot in 10/27. uot_rw vs persist is 6/13/5, and uot_rw vs tilt_rw 10/9/5
  (its "tilt_rw 10/9/5" is this H2b comparison).
- H1c low bar; 21/21 rep pairs with ot_pairs worse than union.
- Trace ratios, recomputed as spread² in `j_mixture`: McCann 0.70-0.92 (k3 1.03), indep 0.44-0.62, union 0.95-1.28.
- One refinement: k3 is not "marginal". It is worse in 100 % of the 220 rep × target draws. Only k7 sits within noise.

**Audit. Everything I re-checked reproduces:** full reproduction, sha, stage and cell identity, pools, PCA angles, cc
gene counts, balanced marginals, kernel underflow, and the cc swap (same conclusion, k-level values within uot draw
noise, sd 0.015-0.10).
- Its eps claim survives with my own draws. ot_pairs is 0/7 against union at eps_rel 0.01 / 0.02 / 0.1 / 0.2; the
  means are 0.567 / 0.574 / 0.612 / 0.642 against union 0.507 (frozen eps 0.05: 0.615).
- Its single eps-0.01 near-win at k7 (0.188) is within k7 draw noise; my draw gives 0.219.
- Not re-run: the instrumented Sinkhorn (iteration counts, TV to a 20000-iteration reference), the S-invariance test,
  the same-target wrapper and the beats() unit tests. By code inspection S never enters `ed_uv`. My full re-run
  covers the same-target question end to end.

**Floor checker.**
- **Reproduced (model-free):**
  - Union beats the leave-one-pool-out same-stage replicate in **7/26** atlas pools against **5/15** erythroid pools.
    The median union/LOPO ratio is **1.51 vs 1.71** (`j_lopo.py`, exact match).
  - k3: the best mixture puts α = 0.85 on prev and scores 0.366 (`j_mixture.py`).
  - The ot_pairs − union gaps.
- **Not reproduced:** the τ² noise floors, union/noise ratios, off-chord residual values and forecast "room". These are
  model estimates, and the checker itself shows within-stage pool differences are batch-dominated (1-43×).
- **Rejected:** "the k2 gap (0.009) is within rep SD". Paired, ot_pairs is worse in 92 % of 220 draws and in all
  6 LOPO targets. The marginal SD is the wrong yardstick for a paired gap.
- **Rejected as an explanation of the failure:** "union sits at the noise floor, so there was little room".
  - The model-free floor does not separate the datasets (1.51 vs 1.71).
  - "No room" predicts ties, not 6 robust losses.
- **Kept as context:** the off-chord idea, in its model-free form (§3a).

**Mechanism checker.**
- **Reproduced exactly:**
  - The ED mixture identity (error ≤ 1.5e-14) and the composition TVs.
  - Best t per k: 1.0 / 0.6 / 0.1 / 0.7 / 0.3 / 0.5 / 0.6.
  - Spreads: McCann 0.911 vs union 1.038 (theirs 0.907 / 1.040).
  - Within-type results: 10/33, median ratio 1.064, sb_half 15/33 with median 1.004, ot_pairs < indep 33/33.
  - Forecast oracle: best λ>0 0.505/1.624/0.480/0.760/0.596/0.335, mean 0.716 vs 0.771, rolling ranks 1,4,12,4,1,9,
    Spearman median 0.21 vs tilt_rw 0.80.
  - Minimum ESS 1210-1414.
- **Oracle t vs oracle mixture:** I get the mixture winning 7/7 (theirs 6/7). k1 is an endpoint tie decided by the
  pair draw.
- **Partly rejected:** "k2 and k7 are within pair-draw noise". This holds for k7 only (see §1.6).
- **Not re-run** (low weight, or oracle-on-oracle): the pool-level asynchrony oracle, the stage-vote "intermediates",
  the shared-type units, and the truth-using reweighting bounds.
- **Rejected inference:** "the failure is the growth estimate, not reweighting per se".
  - The 25-77 % bound builds its weights from tomorrow's cells. So it bounds the model class; it does not show that
    a forecast is possible.
  - The erythroid weights were just as flat (minimum ESS over λ>0 at o3-o5: 1461 / 1385 / 1300) and still helped.
  - What differs between the datasets is direction consistency (§3, forecasting).

## 3. Which explanations of the failure survive (EXPLORATORY)

### Interpolation

**a. On the atlas, the held stages are mixtures of their neighbours, not midpoints. ACCEPTED, model-free.**
(`j_mixture.py`)
- McCann at t = 0.5 never beats the oracle-best α-mixture (0/7).
- Even McCann at the oracle-best t never does (0/7; gaps 0.007-0.12).
- On the erythroid, McCann at t = 0.5 beats the oracle-best mixture at 3/4 stages (by 0.25 / 0.18 / 0.10).
- So the erythroid held stages contain in-between states that no mixture reproduces. The atlas held stages do not.

**b. Separation of the neighbouring stages moderates the effect. ACCEPTED as descriptive; a hypothesis, not a
finding.** (`j_restore.py`)
- |μ_next − μ_prev| / RMS radius(held):

  | dataset | k | value |
  |---|---|---|
  | atlas | 1..7 | 0.435 / 0.415 / 0.452 / 0.509 / 0.373 / 0.524 / 0.545 |
  | erythroid | 2..5 | 1.715 / 1.183 / 1.235 / 1.922 |

  The two ranges do not overlap.
- Union's spread is 0.97-1.13 of held on the atlas against 1.16-1.64 on the erythroid.
- When the neighbour clouds overlap, the 50/50 mix already has the held spread and midpoints only contract it.
- When the clouds are far apart, the mix is bimodal and any midpoint helps. indep beats union at 3/4 erythroid
  stages (2.106 / 1.026 / 1.280 vs 2.206 / 1.154 / 1.473) and at 0/7 atlas stages.
- Spearman(relative OT gain, separation) = 0.64 over 11 non-independent units from 2 datasets.

**c. Under-dispersion is a real but partial cost. ACCEPTED as partial.**
- McCann's spread is 0.91 of held (union 1.04).
- I restored the averaging deficit with data-shaped kNN residual noise. The method is calibrated: on held halves it
  removes the averaging penalty, which is 0.000-0.022 on the atlas and 0.034-0.046 on the erythroid, down to the
  raw-half noise level (|≤ 0.011|).
- Restoration improved McCann by 0.003-0.17 (k4 0.105, k5 0.085). That is more than the calibrated averaging
  penalty (0.000-0.022), because the kNN noise also re-expands a cloud that is contracted for geometric reasons:
  McCann spread is 0.83-1.02, against 0.96-0.99 for a kNN-averaged half.
- So the restoration over-corrects relative to its calibration, which makes it a generous test for McCann. McCann
  still loses at 6/7 stages.
- Restored McCann improves at 7/7 atlas stages, for example k4 0.524 → 0.419 and k5 0.392 → 0.307. It still beats
  union only at k7 (0.195 vs 0.208, which is within noise): 1/7.
- Isotropic restoration (sb_half) wins 0/7, and sharper plans (eps 0.01) win 0/7.
- So dispersion explains part of the gap, not the loss.

**d. Timing. ACCEPTED as real, but not a rescue.**
- The held stage often sits off the midpoint: the best t ranges from 0.1 to 1.0, and at k3 the held stage is
  prev-like, α* = 0.85.
- Even so, the oracle-t McCann loses to the oracle-α mixture at 7/7 stages.

**e. Composition and mass creation. ACCEPTED as "not the cause".**
- Held mass in types that appear or disappear is only 2-14 % (`j_comp.py`).
- The held composition is mixture-like in 5/7 atlas stages and in 3/4 erythroid stages, so it does not separate the
  datasets.
- Removing one-sided types makes OT relatively worse (checker, not re-run).

**f. Within-lineage. ACCEPTED.**
- Inside single atlas types, OT beats union in only 10/33 units (median ratio 1.064).
- New: inside the erythroid types it does not win either (1/5, median 1.021; `j_within.py`).
- So the erythroid stage-level win comes from between-type geometry along the maturation continuum (blood
  progenitors → Ery1 → Ery2 → Ery3), not from within-type movement.

**g. Noise floor. Context only** (see §2).

### Forecasting (`j_forecast.py`)

The failure is model class first, with selection on top.

- **In hindsight, the best λ>0 uot_rw barely beats persistence.** Its gain over persist by origin is
  −1.0 / +1.0 / −2.5 / +0.9 / **+34.6** / +1.2 %, so only s6 matters.
  - Under hindsight selection the letter of H2 would pass: 4/6, mean 0.7165 vs 0.7708. But 3 of those 4 wins are
    ≤ 1.2 %.
- **Selection is a real second failure.**
  - The rolling choices rank 1, 4, 12, 4, 1, 9 of 13 in the final grid.
  - The backtest-vs-final Spearman has a median of **0.21** on the atlas against **0.90** on the erythroid
    (−0.94 / 0.90 / 0.91 / 1.00).
  - At s4 a backtest gain of 0.016 bought a final loss of 0.575.
- **Even the s6 win is not consistent across pools.** It wins 2 of the 3 target pools (24: 0.889 vs 1.326; 25:
  0.829 vs 1.253; 28: 0.854 vs 0.797).
- **What separates the datasets is whether the growth signal carries over to the next step.**
  - On the atlas, the growth seen from s−1 to s does not carry over to s → s+1.
  - On the erythroid it does: rolling choice rank 2 / 1 / 1 at o3-o5, gains 9.5 / 19.6 / 21.6 %.

## 4. Reassessment of the erythroid post hoc positives

**Interpolation.**
- **Not a sampling-noise false positive.** On the erythroid, ot_pairs beats union in 95 % / 100 % / 100 % of the
  20 rep × target draws at k2 / k4 / k5, and in every LOPO target there (`j_noise.py`).
- **But as evidence for a general claim it was weak from the start:**
  - The rule was set by the builder, not pre-registered by the user, and it had 4 units. Under a coin null,
    P(≥ 3/4) = 0.31 per method.
  - It was picked from at least 6 OT-type interpolants under 4 metrics.
  - The builder rule's primary metric (sd_fixed) was dropped after the results were seen. The reason was sound (its
    contraction bias), but the switch was still post hoc.
  - It reverses in gene space: union beats ot_pairs at 4/4 stages, by about 2× (0.72 / 0.53 / 0.48 / 0.55 vs
    1.39 / 1.25 / 1.11 / 1.29).
  - It holds in only 9/15 pools, and inside cell types in only 1/5 units.
  - Random-pair midpoints (indep) also beat union at the same 3 stages. They already capture 33 / 60 / 70 % of the
    ot_pairs gain over union at k2 / k4 / k5 (0.099 of 0.298, 0.128 of 0.212, 0.194 of 0.275;
    `j_ery_indep.py`). The OT pairing adds the remainder, which is consistent with ot_pairs beating indep 4/4.
- **On independent cells it reverses:** 0/7 in PCA space, 0/7 in gene space, robust at 6/7 stages.
- **Honest classification.** This is a real, dataset-specific PCA geometry effect: a single lineage whose 0.25-day
  steps move cells by 1.2-1.9 cloud radii, so the 50/50 mix is bimodal and any midpoint helps. It was selected post
  hoc. As support for "OT interpolation beats the neighbour mixture on embryo time courses" it was a false lead.
- **What it is not.** It is not mainly an OT-specific advantage over the mixture: random midpoints give one third to
  two thirds of it. The part that is OT-specific in both datasets is beating random pairing (H1c), and on the atlas
  that part is not enough to beat the mixture.

**Forecasting (uot_rw).**
- **Real on the erythroid lineage.** It beats persist by 9.4 / 19.6 / 21.6 % at o3-o5 under the frozen code
  (parity_erythroid.json), with consistent backtests.
- **But it was a rescue.** It came from a 24-family post hoc search, after the pre-registered "OT helps" test had
  failed 1/4.
- **The OT-free tilt_rw also beat persist at o4-o5.**
- **The gain rides on the late erythroid expansion**, a monotone composition trend.
- **On the atlas it fails 1/6, with a worse mean.**
- **Classification:** lineage-specific trend continuation, not a forecasting method.

## 5. What next

1. **Stop this line as a source of positive results.** The confirmation did its job, and the answer is a reversal,
   not a near miss.
2. **Write it up as a negative and methods result** (the benchmark preprint control section):
   - The 50/50 neighbour mixture beats McCann OT at 7/7 atlas held stages (robust at 6/7).
   - OT vs the independent coupling (H1c), the usual validation, passes 7/7 and is therefore too weak a bar.
   - Any interpolation benchmark here must report union, indep and the separation statistic.
   - I have not checked whether the mixture baseline is standard in the OT-interpolation literature. Check that
     before claiming novelty.
3. **Only if the user wants one more test: a NEW pre-registration of the narrowed hypothesis.**
   - The hypothesis: "midpoint/OT interpolation beats the neighbour mixture only when |Δμ|/RMS radius > ~1".
   - The data: a third dataset, neither this atlas nor scVelo erythroid, with ≥ 4 pools per time point and batch
     crossed with time. It should include units both above and below the threshold.
   - Fixed in advance:
     - union, indep, the oracle-free separation statistic, ED_U in PCA and gene space, and per-pool results;
     - a decision rule on both the sign and the moderator.
   - State up front that the prior is low, because the hypothesis comes from 11 non-independent units.
   - This would need a download and the HPC, so it needs the user's go-ahead.
4. **Forecasting: drop uot_rw.** No strict estimator has shown headroom on the atlas. If anything is kept, it is the
   descriptive diagnostic: backtest-final rank consistency (0.21 vs 0.90).
5. **No further local run is needed for the verdict.**

## 6. Completeness: not run, or not checkable

- **Done here because no checker had done it:** target-side, complement and leave-one-pool-out noise for the atlas
  and the erythroid (`j_noise.py`). All four checkers' rep spreads covered only fit-side noise.
- **Not reproduced by me:**
  - from the floor checker: the τ² floor model (union/noise ratios, off-chord values, forecast room);
  - from the mechanism checker: the pool-level timing oracle, stage-vote intermediates, the shared-type units and the
    truth-using reweighting bounds;
  - from the audit: the instrumented Sinkhorn TV-to-reference, the S-invariance test, the same-target wrapper and the
    beats() unit tests. My full re-run and my marginal checks cover the same ground.
- **Not checkable locally:**
  - the raw-data steps of atlas_prep.load: meta merge, doublet and stripped filters, and per-sample subsampling;
  - HPC provenance, i.e. that job 718022 was the first atlas result.
  - Local evidence only: the prereg sha matches the logged value, the prereg mtime precedes confirm.json by 319 s,
    and parity_erythroid.json precedes the prereg.
- **A design caveat, stated in the prereg only in part.**
  - The atlas does not meet the design the erythroid report itself recommended (≥ 4 pools per stage, batch crossed
    with time): E6.75 has 1 pool, E7.25 and E8.25 have 3, and batch is stage-confounded (E8.0 is 77 % batch 3,
    against 0 % in its neighbours; floor checker's number, not recomputed here).
  - This does not rescue H1: the reversal survives LOPO. But it limits any per-pool reading.
- **Not in the prereg and rightly not run:**
  - W1/W2 are contraction-biased in this setup (erythroid report).
  - MMD cannot change a 0/7 that already holds in both PCA and gene space.
  - The prereg had no stated effect size or power. Post hoc, §1.6 gives the excluded effect sizes.
- **Forecast parity.** The prereg's parity check covered interpolation only. The forecast values in
  parity_erythroid.json (ED_U 1.732 / 1.796 / 2.238 / 4.014) are consistent with the report's V-statistic energies
  (1.756 / 1.808 / 2.249 / 4.026). That is a difference of metric, not a bug.
