# Final judgement: three designs scored, one approach synthesised

I read all 1,627 lines of `dossiers_all.md`. This run I also checked, with 3 page fetches and 11 searches, the facts that decide which dataset to use and who might scoop the work; they are marked [verified] below.

## 0. Facts verified this run that change the plan

**sci-FATE2 (Maizels, Snell, Briscoe; GSE236512, pilot GSE236517)**
- Mouse ES cells → neuromesodermal progenitors (NMPs) → neural and mesodermal fates; 44,713 cells.
- Collected on "day 3 to 8 of the differentiation", with a 4sU pulse of "500 μM … for 2 hours immediately prior to fixation".
- "All timepoints were prepared and sequenced together, with three full-time course replicates", so time and batch are **not** confounded.
- A fourth replicate samples "between day 3 and 4 at five-hour intervals".
- Cell populations are asynchronous and mixed-fate. γ was estimated from new vs total RNA.
- Unspliced counts are partly available ("18.5% of genes lacked unspliced information").
- Not checked: the exact list of days, cells per sample, and which layers the GEO files contain.

**VeloVAE (PLoS CB 2026, PMC13021174)**
- Uses a Gaussian capture-time prior t ~ N(t₀, σ₀²) and a Poisson likelihood.
- Was validated on sci-FATE2's 2 h labelling, with transcription rates at "mean PCC of 0.72".
- Has no population, influx or growth layer, and no identifiability analysis beyond noting the scaling equivalence.
- So sci-FATE2 is both the best dataset and a direct head-to-head with the closest competitor.

**Biddy/Morris CellTag reprogramming (cellrank `reprogramming_morris`)**
- 104,887 cells (subsets of 85,010 and 48,515), 8 timepoints over days 0–28.
- Raw spliced and unspliced counts, CellTag clones, about 1.5 GiB.

**Fishman et al. 2024 (Nat Commun; GSE224113/224918/224919)**
- Only 3 stages at 4.3, 4.8 and 5.3 hpf; 8,226 cells.
- 4sUTP injected at the 1-cell stage (continuous labelling, not a pulse), Drop-seq, batch-corrected.
- **Not usable as the clock dataset.** This overrules design C's feasibility critic.

**Other works confirmed to exist**
- CardamomOT (Maugé & Ventre 2026, doi 10.64898/2026.03.31.715390): mechanistic burst-kinetic gene regulatory network (GRN) + OT + generative model on time-course data. A scoop for "mechanistic + real time + generative".
- TIDYI (Rosenbauer et al., Research Square rs-8252082): absolute proliferation and death rates from snapshots.

**Not found this run:** Navigo, and Nguyen & Malysheva (arXiv 2609.35060). No published age-structured kinetic count model for single-cell time courses turned up (narrow search).

---

## 1. Scores

Scores are out of 10, except G/C/I fit (each out of 5). "→" means after the critics' fixes. P(+) is the probability of a presentable positive result within 6–12 months.

| Design | Soundness | Feasibility | Novelty | Fit G / C / I | P(+) | One-line justification |
|---|---|---|---|---|---|---|
| **A: AKP, kinetic clock** | 5 → 7.5 | 3.5 → 6.5 | 5.5 → 6 | 4 / 5 / 3.5 | 0.35 → 0.6 | The right object: counts, two clocks, a McKendrick population. But the time-unit result (P1) is mis-derived (log-quadratic influx is not identified), the γ prior pins the time unit, the external checks are only on synchronous data, and the scope is about 3× too large. |
| **B: PALM, program-space SDE** | 4 → 6 | 3 → 5.5 | 4 → 5 | 3 / 4 / 3 | 0.3 → 0.45 | The cited theorems cover neither finite snapshot counts nor growth. The latent offset is not identified. The kinetic-lag sign is wrong. A Lyapunov drift cannot represent the cell cycle that dominates its datasets. The theory is classical and heavily scooped (Guan ICML 2026, CardamomOT, PFM). |
| **C: KAIROS, evidence-first branching model** | 5.5 → 7 | 4 → 6.5 | 5 → 5.5 | 4 / 4.5 / 3.5 | 0.3 → 0.5 | Holding out whole modalities and clones is the right evidence design. But I3 is false in general, the score is dominated by the emission model, the datasets sit in the wrong kinetic regime, and per-edge hazards depend on clustering resolution. |

**Verdict.** A and C are the same model family: a Chronocell-type kinetic decoder under a real-time age-structured population prior. That shared core is the only one whose identifiability can be stated, tested on simulators and falsified with held-out labelled RNA within 9 months. B is not the core. Its certificates, model ladder, equivalence-class reporting and declared-step ideas are grafted in, and its linear OU program graph is deferred to year 2.

---

## 2. Synthesised approach: TACK (Time-Anchored Cohort Kinetics)

Check the name for collisions before use.

### 2.1 Thesis
A count-level generative model in which each cell has:
- a lineage path l on a declared tree,
- an age a in hours since it left a progenitor reservoir, advancing at unit speed in real time,
- raw UMIs emitted by closed-form transcription/degradation kinetics with NB noise.

Real collection time T enters only through:
- the entry flux ρ(E), where E = T − a (with a known protocol onset),
- per-replicate onset shifts,
- optional declared exogenous steps such as a medium switch.

**Interpretability comes from three things only:**
1. a declared restricted class whose identifiability boundary is proved for continuous T and checked on simulators of a different family;
2. data the fit never sees: 4sU new RNA in the same cells, intermediate real times from a separate replicate, and clones;
3. certificates that refuse to answer when the conditions fail.

### 2.2 Model

**Population layer: McKendrick–von Foerster, stepped exactly with a Leslie scheme (Δ = 1 h, which divides every gap).**
```
ν_l(E) = π_l · ρ(E)
ρ(E)   = 1[E ≥ T0 + δ_b] · exp(η_j(E))    piecewise-constant on J ≤ K bins, fixed from training times;
                                          no smoothness penalty, or its influence on ĉ is reported
G_l(a) = exp(∫0^a r_l(u) du)              r_l piecewise constant per segment
n_l(a,T) = ν_l(T−a) · G_l(a)
p(l,a | T) = n_l(a,T) / N(T)
p(R | T)   = R(T) / N(T)                  log R free piecewise-linear: an OPEN reservoir
                                          (closed reservoir is a sensitivity arm only)
```
- δ_b is a per-replicate onset shift (heterochrony between replicates).
- For embryo data, add staging noise T_true,b ~ N(T_label, σ_T²) per pool or embryo, and profile σ_T.

**Emission (closed form; one species by default).**
```
dμ_g/da = α_{g,s(l,a)} − γ_g μ_g,         μ_g(0) = α_{g,R}/γ_g
μ(a) = α_k/γ + (μ(τ_k) − α_k/γ) · e^{−γ(a−τ_k)}      on segment k
y_ig ~ NB(ℓ_i · μ_g(l_i,a_i) · e^{κ_g c_i}, φ_g)
```
- ℓ_i: depth from near-Poissonian genes, binned into 32 quantiles so the dense NB term is tractable.
- c_i: a fixed cell-cycle phase score computed from excluded cycle genes. It is a nuisance, never interpreted.
- Two-species variant: dμ_u/da = α − βμ_u, dμ_s/da = βμ_u − γμ_s, with the γ = β limit guarded. β is reported only as the ratio c^u_g/β_g, because it is confounded with capture.

**Held-out labelled RNA (never fitted).**
- Expected new RNA over the pulse length L = 2 h:
  λ^new_g(l,a) = ∫_{a−L}^{a} α_g(a′,l) e^{−γ_g(a−a′)} da′, with α = α_R for a′ < 0.
- Score the new counts conditional on the totals:
  new_ig | y_ig ~ Binomial(y_ig, p_ig), with p_ig = ε₀ + q·λ^new/μ.
- At steady state p_g = q(1 − e^{−γ_g L}). This is a direct test of γ in hours.
- q (conversion efficiency) and ε₀ (background conversion) are calibrated on a disjoint 10% of cells, which are excluded from scoring.

**Tier 2 (month 6+, only if the gates pass): named programs.**
```
log α_g(a,l) = b_g + Σ_{r∈Pa_S(g)} W_gr h_r(a,l) + Σ_n W^nuis_gn h_n(a,l)
```
- S is the CollecTRI support.
- Nuisance programs may only load on genes outside every named support. This removes the rotation h_r ↔ h_n.
- Each program needs at least 3 exclusive anchor targets, counted before fitting. Scale is fixed by ‖h_r‖∞ = 1. Activities are piecewise constant over age bins with a TV penalty whose λ is fixed on SimA.
- Collinear regulators are reported as blocks.

**Declared exogenous step (grafted from B's M3).** A known medium switch at T_s may add u(T)·δ_g to log α for a small, pre-declared gene panel. It is off by default and tested.

### 2.3 Time semantics
- **T** is collection time in hours from protocol start.
- **a** is age in hours, with unit speed.
- **E = T − a** is entry time.
- There is no flow time and no pseudotime.
- Intracellular dynamics depend on (l, a) only (assumption A1, cohort invariance), except for declared steps.
- Two limits sit inside the model: persistence is the limit where the time unit c → ∞ (nobody ages), and "no dynamics" is free per-snapshot mixing.

### 2.4 Identifiability

**Assumptions**
- A1: cohort invariance.
- A2: topology lies in the declared set.
- A3: the NB kinetic kernel is injective in a and the mixture is identifiable (Chronocell's argument).
- A4: population non-degeneracy, i.e. log ρ is not quadratic on the entry window W = [T₁ − A_max, T_K], **or** there is a known onset edge T₀ inside W.
- A5: kinetic trackability. The cohort-trackability index CTI (the share of informative-gene signal with γ̂Δ < 2, times P(dwell > Δ)) is above a pre-registered threshold.
- A6: batch acts as a global per-pool capture scale.
- A7: Markov in (l, a), so no hidden heritable fate bias.

**P1′, the time unit (corrected; continuous T; growth flexible in age).**
- Under the rescaling a = c·a′ with rates divided by c, observational equivalence requires, after taking the cross-derivative ∂_T∂_{a′}:
  (log ρ′)″(u) = c·(log ρ)″(u − (c−1)a′) for all (u, a′).
- So c ≠ 1 is possible **iff** (log ρ)″ ≡ k on W and no edge of supp ρ is kinetically resolved. The compensating growth is G′(a′) = G(ca′)·exp(k·c(c−1)a′²/2 + b·a′).
- Corollaries, each with a must-fail simulator:
  - k = 0 (exponential or constant influx): age-distribution shape stationary, never identified (SimD).
  - k < 0 (Gaussian pulse): snapshots move, yet c is unidentified when growth can be age-linear (SimD2). Under piecewise-constant growth, c is only weakly identified.
  - A known onset edge identifies c with K ≥ 2, provided the age front T − T₀ is resolved by the kinetics.
  - K = 2 without an edge: never identified.
  - Discrete K ≥ 3: not proved; checked by profile likelihood.
- c is claimed in hours only if A4 **and** A5 hold.

**γ_g in 1/h.** Claimed only for "rate-identified" genes: c identified, and the gene is transient within trackable ages (per-gene profile CI narrower than ×1.5). All other genes are ratio-only. α is identified only up to capture. β is never claimed in hours.

**Population.** Identified:
- the shape and onset of ρ in hours;
- resolution-invariant functionals: root→terminal first-passage time (h), branch fractions, entry-time law;
- switch ages τ, reported as intervals.

Growth is identified only as differentials. π and growth trade off, so both extremes of the equivalence set are reported unless clones break the tie. Absolute growth (ḡ) needs total counts and is never claimed.

**Programs.** Identified up to per-program scale under the anchor/support conditions (dossier lesson L16), given that the CollecTRI support S is correct. Otherwise reported as blocks.

**Not identified, by declaration:**
- curl and oscillations (the cycle is a nuisance);
- individual cell paths;
- extrapolation;
- edges outside S;
- heterochrony along the manifold;
- birth vs death separately;
- within-(l, a) gene–gene covariance (checked, not modelled).

**Certificates (grafted from B).**

| Certificate | Purpose |
|---|---|
| C-STAT | A stationary negative control must give a flat c-profile. |
| C-BATCH | A batch-only simulator must give a flat c-profile. |
| C-TRACK | CTI must exceed its threshold. |
| C-AP | Chronocell-style stability across starts, AP ≥ 0.5. |
| C-ANCHOR | Anchors counted before fitting. |
| C-COHORT | Cells in the same (l, a) bin from different T must be exchangeable (permutation test). Detects T-specific off-manifold deviations only. |

Uncertainty comes from bootstrapping replicates/pools and then cells, with a likelihood temper calibrated on SimA to reach nominal 90% coverage. Fisher rank is never used as evidence.

### 2.5 Generation at T*
1. Compute p(l, a | T*) on the 1 h grid.
2. Draw (l, a), draw ℓ from the depth model of the nearest training pool (a declared nuisance), draw NB counts. Optionally also draw new/old counts.
3. Report the **assumption share**: newborns with E between the last observed snapshot and T* (from interpolated ρ), plus ages beyond the fitted range, plus kinetically saturated cells.

Beyond T_K, output is an envelope over three ρ scenarios (constant, last-bin trend, zero) and is labelled "assumption-driven". Clone-size predictions from a branching simulation are assumption-only, because birth and death are not separately identified.

### 2.6 Inference and compute

**Staged fitting**
1. C5-class piecewise α per segment under the age prior.
2. AP gate.
3. Programs (tier 2).
4. Joint fine-tune.

**Optimisation**
- Exact grid EM over about 200 ages × L ≤ 4 lineages; Poisson warm start, then NB refinement.
- 5 random starts plus 1 informed start.
- λ_TV and A_max fixed on SimA; only topology is selected on inner folds. No unguarded stage.

**Cost**
- One fit is about 0.5–2 A100-h (G = 300–500 genes, N ≈ 45k cells).
- Simulators: about 100 A100-h. Real data: about 150 A100-h. Total about 250–350 A100-h.
- Under the HPC QoS (1 running, no arrays), fits are packed sequentially inside each job. dyngen needs 50–200 CPU-h.

**Code.** About 5–7k original lines: decoder 500, Leslie population 300, grid EM with depth binning 500, new-RNA closed form 150, SimA 1,000, regret harness and controls 800, plus loaders.

**Reused.** `metrics.py`, `controls.py`, `designs.py` (skip-k), `prior.py` (degree- and sign-preserving random support), `real.py` loaders, the pre-registration/SHA freeze, the reproducibility gate, and the leakage tests from tgl-grn-continuous.

### 2.7 Datasets, in order
All downloads need permission and go to the HPC.

| # | Dataset | Role | Caveats |
|---|---|---|---|
| D0 | **SimA** (own agent-based model) | Main recovery test | Hill GRN, protein layer added later, telegraph bursting, division and death, toggle-switch fate, Beta capture with length bias. Violation dials: heterochrony, per-cell speed CV 0.2–0.4, open or drifting reservoir, σ_T ∈ {0, 0.5Δ, Δ}, pool batch sd up to 0.5 including on the u/s ratio, age-varying γ, size/cycle covariance. γ medians of 2 h and 20 h, not 8 h. |
| D0 | **dyngen** with a growth resampler | Recovery on an out-of-family simulator | — |
| D0 | **SimD, SimD2, batch-only, stationary** | Must fail (flat c-profile) | — |
| D0 | **SimE** (in-family) | Code sanity only | Never counted as evidence. |
| D1 | scVelo `gastrulation_erythroid` (9,815 cells; E7.0–E8.5 in 0.25 d steps, 7 stages; u/s) | **Calibration gate:** reproduce Chronocell (AP ≈ 0.79, about 24/49 genes kept). Head-to-head of uniform vs Gaussian vs age-structured priors. MURK case study. | **No clock claims:** open reservoir, staging noise, per-stage batches, MURK boosts. |
| D2 | sci-fate GSE131351 (A549 + dex, 0–10 h every 2 h; 1,054/1,049/949/1,262/1,041/1,325 cells; 2 h 4sU) | **Known-clock unit test.** With ρ free, the fit must put the onset at 0 ± 1 h. Total counts fitted; new given total held out. | Synchronous, so it cannot discriminate between priors. Pooling of timepoints and availability of u/s are unverified; one-species model. |
| D3 | **sci-FATE2 GSE236512 — PRIMARY** | CLOCK and GEN analyses (§2.8). | Exact sample list and GEO layers to verify. Protocol switch times to take from the methods. |
| D4 | Biddy/Morris CellTag | **Secondary.** Composition-based estimation only (C's route a). Barcodes withheld. Reprogrammed vs dead-end branch fractions and relative growth vs clones (thinning-corrected). | Dwell times are days, so kinetic age saturates: no clock claims. |
| Stretch / year 2 | Argelaguet multiome (node-level chromVAR, partial correlation given an age × lineage spline); Schiebinger thinning (one-species); Watermelon barcodes (growth) | — | — |
| **Dropped** | LARRY, Bastidas-Ponce, Fishman, RENGE, CD4, RAF1; Veres (optional continuity only) | — | LARRY: K = 3, normalised counts, hidden fate bias. Bastidas-Ponce: FACS sorting. Fishman: see §0. |

### 2.8 Evaluation protocol

**Two pre-registered analyses on D3,** SHA-frozen after D1–D2 and before opening D3 beyond a data card.

- **CLOCK.** Fit total counts at every timepoint for replicates 1–3, plus the day-3 sample of replicate 4. Hold out the entire new/old split.
- **GEN.** Fit days 3 and 5–8 for replicates 1–3, plus replicate 4's day 3 as its onset anchor. Predict day 4 for replicates 1–3 and replicate 4 at +5, +10, +15 and +20 h. The gap is 48 h with five intermediate real times, which genuinely tests quasi-continuity.

**Primary metrics (none dominated by the centroid)**

*E1, clock.*
- Block profile of c_kin (kinetic parameters rescaled, population fixed), computed with:
  - a scale-free γ prior (a prior on log γ_g − mean log γ only),
  - λ_smooth → 0,
  - replicate bootstrap.
- Stability of ĉ under three ablations: onset removed, closed vs open reservoir, age-flexible growth.

*E2, modality hold-out.*
- Per-cell held-out log-likelihood of new given total (nats per cell and per 10³ new UMIs).
- γ̂ against labelling-derived γ, recomputed by us from the held-out layer: Spearman, the log–log slope, and median |log2 ratio|.

*E3, generative regret with a frozen shared decoder (fixes the "emission contest").*
- Stage E: fit the emission with free per-snapshot mixing Q_k, then freeze it. Every arm outputs q̂(z | T*) on the same grid.
- LL_i(q) = log Σ_z p_θem(x_i | z) q(z).
- Oracle q* is fitted on half A of the held-out cells and scored on half B, then swapped.
- Regret(q̂) = mean_i [LL_i(q*) − LL_i(q̂)]; GapClosed = 1 − Regret(TACK) / min over controls of Regret.
- Ceiling on the clock gain at a midpoint when the age distribution only translates: KL ≈ d⁴/4, with d = Δ/(2σ_a). Primary folds are therefore only those with training-estimated Δ/σ_a ≥ 2. Single-interior folds are reported for non-inferiority only.

*E4, calibration.*
- Randomised PIT uniformity, detection-rate and Fano factor per gene, all within the replicate floor.
- var_ratio per lineage.
- Composition total variation (TV) from a classifier trained only on genes outside the panel.

*Secondary.*
- Full-count LL against C0–C2, reported as the "cost of interpretability" with a non-inferiority margin set from the replicate floor.
- C2ST. ED with its mean/trace split, kept for continuity only.

**Controls.** All arms share the decoder unless stated.

| Control | What it is |
|---|---|
| Persistence | c → ∞ |
| L-mix | Mixture interpolation of the snapshot mixing measures |
| L-disp | Age-quantile (McCann) interpolation × logit-linear lineage weights |
| Uniform-age | Chronocell's prior |
| Gaussian capture-time | a ~ N(T − T₀, σ₀²), the VeloVAE prior |
| Shuffled-T | — |
| Prior-only γ | Every gene at the prior centre |
| Free-Q oracle | No dynamics |
| C0/C1/C2 count mixtures | Persistence; mixture interpolation; shared components with logit-spline weights |
| C6 | Random signed support with matched degrees (tier 2) |
| Upper-bound arm | Time-conditioned NB-VAE, never interpreted |
| External VeloVAE | Behind a calibration gate: reproduce PCC ≈ 0.72 in our harness first |
| cfm_grn_v2 v0.1.10 | ED only |

**Decision rules (applied by code)**
- **R-ID (simulators):**
  - ĉ within ×1.25 in at least 80% of non-degenerate SimA/dyngen instances at realistic violation dials;
  - a flat profile flagged in at least 80% of SimD/SimD2/batch-only/stationary instances;
  - γ Spearman ≥ 0.7 on rate-identified genes;
  - 90% interval coverage ≥ 80% after tempering;
  - a power analysis at D3's Δ/σ_a.
- **R-UNIT (D2):** onset within ±1 h, and new-given-total LL beats the prior-only and uniform arms (CI > 0).
- **R-CLOCK (D3).** All four must hold:
  - (a) the c_kin 95% CI is narrower than ×1.5;
  - (b) ĉ moves by less than ×1.25 under the ablations;
  - (c) new-given-total LL beats uniform, Gaussian-capture, prior-only and shuffled-T, with paired cluster-bootstrap CIs > 0;
  - (d) γ Spearman ≥ 0.5, log–log slope in [0.8, 1.25], median |log2 ratio| ≤ 1, and better than prior-only.
- **R-GEN (D3).**
  - The lower CI of GapClosed is > 0 on the thinning design.
  - Single-interior folds: the upper CI of the regret difference is < +0.02 nats/cell.
  - At least 80% of genes pass the PIT test within the floor.
  - Composition TV ≤ C2 + 0.02.
- **R-PROG:** named programs beat C6 on held-out LL **and** on new-given-total LL. Otherwise they are reported as unnamed factors.
- **R-LIN (D4):** branch-fate AUROC ≥ the composition-only baseline + 0.03, and growth-sign agreement ≥ 0.8.

**Kill criteria**
- **K0 (week 3).** Regret of the best latent control on the GEN design must be > 3 SE, judged on training folds or simulators at matched Δ/σ_a. Otherwise drop the generative-clock claim a priori.
- **K1 (month 3).** If R-ID fails, stop the real-data work and write the theory + simulator paper.
- **K2 (month 4).** If R-UNIT fails, the decoder or labelling model is broken; debug only.
- **K3 (month 6).** If R-CLOCK (a) or (c) fails, drop "hours" and report ratios plus the boundary result. If R-GEN fails, report non-inferiority and the cost of interpretability.
- **K4 (any time).** If AP < 0.5 on a dataset, make no claims on it.

### 2.9 Nine-month plan

| Month | Work | Gate |
|---|---|---|
| 0 (weeks 0–3) | Fill PURPOSE.md. Write P1′ and check it symbolically in sympy (cross-derivative; SimD2 equivalence). Data cards, permission list, pre-registration skeleton. Build the regret harness with mixture controls. Run G0 on D1 and simulators. | K0 |
| 1–2 | TACK core: one- and two-species decoder, Leslie population, depth-binned grid EM, new-RNA closed form. Tests: closed form vs ODE solver, mass balance, γ = β limit. SimE sanity. Chronocell reproduction on D1. | Calibration gate |
| 2–3 | SimA (simplified first), SimD/D2, dyngen; violation-dial breakdown curves; tempering calibration. | K1 |
| 4 | D2 unit test; D1 head-to-head; freeze pre-registration v1 for D3. | K2 |
| 5–6 | D3 CLOCK and GEN; external VeloVAE run behind its calibration gate. | K3 |
| 7 | Tier-2 programs on D3 (pre-registered TF list, e.g. Sox2, Tbxt, Cdx2, Olig2, Nkx2-2, Foxa2, subject to the anchor count); C-COHORT; test of the declared step. | R-PROG |
| 8 | D4 composition-only lineage test. The audit note is finished in HPC queue gaps. | R-LIN |
| 9 | Preprint 1: theory boundary + simulators + D2/D3 validation + harness. Liò proposal with the year-2 hooks in §4. | — |

### 2.10 Risks and fallbacks

| Risk | Response |
|---|---|
| c not identified on D3 (near-quadratic influx, no resolved onset, or saturated cohorts) | Report ratios plus the certificate refusal. Fallback paper: "When do snapshot time courses give rates in hours?" (P ≈ 0.8). |
| Medium switches violate A1 | Declared steps plus the cohort test. |
| Replicate heterochrony | δ_b, fitted on each replicate's training samples only. |
| Low sci-RNA-seq depth | Smaller panel, one-species model. |
| u/s layers missing from GEO | One-species is already the default. |
| Optimisation multimodality | Staged fitting plus the AP gate. |
| Scoop (VeloVAE group adding a population prior; CardamomOT; PFM) | Lead with P1′, certificates that refuse, count-level held-out labelled RNA, and generation at intermediate real times. |

### 2.11 Honest probabilities

| Outcome | Probability |
|---|---|
| P1′ plus simulator results, including the predicted SimD/SimD2 failures | 0.8 |
| R-UNIT | 0.75 |
| R-CLOCK, full | 0.4 |
| R-CLOCK (a) + (c) only | 0.5 |
| R-GEN | 0.4 |
| R-PROG | 0.3 |
| R-LIN | 0.3 |
| **Presentable positive by month 9** (theory + simulators + at least one real external validation) | **0.6–0.65** |
| Full G + C + I on D3 | 0.2–0.25 |

### 2.12 How each critic flaw is fixed

| Flaw | Fix |
|---|---|
| A-F1: P1 mis-derived; closed reservoir and γ prior do the identifying | Corrected P1′ (A4: log-quadratic is the boundary, or an onset edge). Scale-free γ prior. No smoother whose null space is the non-identified set. Open reservoir by default. Ablations. SimD2. Prior-only arm. |
| A-F2, C-F1: decoder contest / mixture-interpolation trap | Frozen-decoder regret against an oracle q*. Thinning and skip-k folds primary. d⁴/4 headroom rule. Superiority over C2 replaced by non-inferiority. |
| A-F3, B-F3, C-S9: no asynchronous external test | sci-FATE2: asynchronous, batch-clean, held-out new given total, head-to-head with VeloVAE. sci-fate demoted to a unit test. |
| B-F1, B-F2, B-F5: borrowed theorems, growth breaks Gibbs, σ not estimable | No SDE core. Own proposition plus profiles plus simulators. Growth as differentials only. |
| B-F3: latent offset not identified | No free latent coordinates. Age is kinetic. Programs act only inside log α, with anchors and support-restricted nuisance programs. |
| B-F4: lag sign wrong; β not identified | Forward closed-form kinetics in age, no generator shortcut. β ratio-only. |
| C-F2: I3 false; age-0 entry invisible; wrong regime | Composition-based estimation only after a structural rank check (SIAN or Meshkat–Sullivant). Age-profile estimation gated by CTI. Pre-registered γ·dwell demotion. Exit-age-aware entry state, or claims restricted to predecessors with dwell ≫ 1/γ. |
| C-F3, C-S8: κ depends on clustering resolution | Continuous-age segments. Only resolution-invariant functionals are claimed, checked across S₀ ± 2 resolutions (≤ 20% change). |
| Smaller items | Depth binning (NB cost); Leslie stepping (numerical diffusion); tempered likelihood plus pool bootstrap (overconfidence); staging noise σ_T; cycle phase as a nuisance; partial correlations for chromatin; counterfactuals labelled "open-loop, direct-target"; tier 3 / closed loop moved to year 2; LARRY demoted. |

---

## 3. What the user's current projects teach

### DO

| Do | Lesson from |
|---|---|
| Score the target and its ceiling before training. Here: G0 and the oracle q*. | cfm_grn_v2 F1: path 0.636 vs model 0.641. The no-network ceiling was the most informative tool. |
| Decompose the metric so it measures the claim. | Job E: ED mean part 0.153 vs trace 0.009 (Veres) and 0.298 vs 0.003 (Pijuan-Sala). |
| Use NB on raw UMIs. | cfm_grn_v2 F9: \|z\| up to 263,822; 33–41% of genes detected in < 1% of cells. |
| Give every control its trivial analogue in the same family. | F3: the linear centre reproduced mean_interp's centroid (8.46 vs 8.44). |
| Guard every reporting stage. | F7: stage 2 was rejected in 6/6 seeds yet ran in the refit. |
| Change the data regime before the architecture. | cfm_grn v3/v4: signed prior plus velocities moved Pearson from −0.07 to 0.41–0.68. tgl: 12+ architectures moved nothing. |
| Use degree-matched, adversarial negatives. | Dileo audit: 0.94 → 0.50, below chance on new edges; a source-out-degree scorer gets 0.940/0.944. |
| Reproduce published numbers before comparing (calibration gate). | Job C failed; EdgeBank reproduced 0.7911 as 0.7912. |
| Confirm on fresh data, split by whole biological units. | Veres and Pijuan-Sala gave an honest negative and a tie. |
| Use simulators of a different family with heterogeneity. | F11: near-translate snapshots; in-house simulator from the model's own family. |
| Ablate temporal modules and require that they hurt. | tgl: GConvGRU always called with H = 0; EvolveGCN evolved across epochs. |
| Fill PURPOSE.md, keep one canonical repo, run heavy jobs on the HPC. | tgl: 5 forks, empty PURPOSE, question drift, top-8 cut set by 1.8–6.5 GB of laptop RAM. |

### DON'T

| Don't | Lesson from |
|---|---|
| Build paths that are linear in snapshot data. | F2/F3: OT and random chaining give mean errors 1.3337591265 vs …278. |
| Make a black-box autonomous deterministic drift the scientific object. | Interpretability 0/5; on Veres stage 1 (0.182) ≈ persistence (0.188); a flow map cannot branch. |
| Work from aggregate per-bin GRNs. | tgl T9: oracle-future neighbours add +0.0002 R². |
| Use learned masks or gates without a recovery check. | cfm_grn v1: edge AUC 0.538. |
| Pick fixes after looking at the test window. | EB v0.1.10. |
| Churn versions on 2 inner folds. | 12 versions in a week; F8. |
| Read seed-only CIs as biology. | cfm_grn_v2 risk 8. |
| Claim extrapolation. | F10: terminal ED > 12 in 2 of 5 seeds. |
| Use pseudotime bins when stage labels exist. | tgl anti-pattern 6. |
| Run global OT across conditions. | cfm_grn v1: ED_de 14.2 vs 6.5. |
| Announce a positive before its controls. | AUTOPSY §10.2: "+0.144 over EdgeBank". |
| Spend compute on mechanism questions off the thesis line. | Job D: 34 trainings on the stage-2 horizon. |
| Defer the scientific object. | The Phase-3 gates were never built. |

---

## 4. Pros and cons of the user's existing ideas

**RA-FM.** Drop it as the core. Freeze cfm_grn_v2 v0.1.10 as a baseline.

| Idea | Decision | Reason / what replaces it |
|---|---|---|
| I4 FM generator | Drop | Centroid pinned (F2/F3); 0/5 interpretable. |
| I5 growth | Modify | Relative differentials, profiled against π; clones (D4) as the external check. |
| I6 SDE | Modify | Stochasticity from the age mixture plus NB noise. A full SDE is deferred. |
| I7 signed prior drift | Modify | Signed CollecTRI support inside log α on counts, so decay keeps units of 1/h (the standardised decay was not kinetic). |
| I8 edge gates m(t,c) | Drop | Replaced by program activities h_r(a,l) over age with anchors. Edge-level timing is not identifiable. |
| I9 L1 + TV | Keep | For h_r, with λ fixed on SimA. |
| I10 time only through gates | Modify | Real time only through ρ(E), the onset and declared steps. |
| I11 chromatin anchor | Keep for year 2 | Node level only, partial correlations. Never use an ATAC-built prior and then validate it on ATAC. |
| I12 KO timing, I13 binding support, I21 transportability, I26 VCC | Drop for now | RENGE underpowered; binding off-window; the others are off-goal. |
| I14 interventions | Modify | Count-space α_TF → 0, labelled "open-loop". Closed loop in year 2. |
| I15 within-group coupling, I16 control battery | Keep, extend | Regret ladder, prior-only arm, uniform and Gaussian time priors. |
| I17 H1–H3 | Replace | CLOCK / GEN / PROG / LIN. |
| I18 gate-recovery metrics | Modify | Program recovery on a misspecified-prior SimA. |
| I19 identifiability theory | Keep, refocus | P1′, the realistic theory deliverable. |
| I20 TempGRN-Bench / GGE | Modify | Into the count-level regret benchmark: a natural Liò hook. |
| I22 Watermelon | Year 2 | Best dataset for identifiable growth. |
| I24 Count-FM | Covered | Subsumed by the NB likelihood. Other reserve ideas dropped. |
| I25 new Perturb-seq | Collaboration later | — |

**Engine programme (I23).** Stop. Write F2/F3 (the "centroid of any linear interpolant is pinned" identity, plus the mean/trace split) as a short methods note or a benchmark section. It is a real, publishable negative insight. Do not spend more on stage-2 horizon questions.

**tgl-grn line.** Stop every fork. Archive one canonical snapshot. Keep the leakage tests, the degree-matched sampler and the streaming cache. tgl-grn-continuous (A-004) gets a one-paragraph verdict, no more.

**Audit lane (I2).** Keep as a 2–3 week workshop or short note (0.94 → 0.50; below chance on new edges; degree-product survives rewiring). It is cheap, unclaimed, and shows rigour to Liò. Cap the effort.

---

## 5. Do the two papers add anything?

### Chronocell: yes, it is the decoder and diagnostic core (re-implemented)

**Used**
- Closed-form kinetics; near-Poissonian depth; annealed grid EM.
- The AP gate; the trajectory-vs-cluster test (the basis of C-TRACK).
- Identifiability given the mixing measure; per-gene "identified vs ratio-only" tags.
- γ-vs-labelling validation.
- The per-cell `prior` hook, which is exactly where the age-structured p(a|T) plugs in.
- Uniform-age arm and the D1 reproduction gate.

**Traps**
- The uniform prior. It "completely" fails under non-uniform sampling, so it is ablation only.
- Equating process time with embryonic day (posteriors are flat until E7.5).
- Poisson-only likelihood, and AIC on the lower bound (the PBMC failure).
- Synchronous all-gene switching read as "regulation".
- Planning 2k genes with plain EM.
- Treating its erythroid result as a clock test.

### Badalotti et al.: nothing for the 9-month core (1.5/5 as a method)

**Bounded uses**
- Framing: persistence is a model limit (here c → ∞; in their model J_D → ∞).
- Year 2 only, once programs exist and the TF-mRNA self-consistency test passes. Close the loop with h_r(a) = σ(w_r·TF_r(a−δ) + s_r h_r + b_r), then run their diagnostics:
  - self-activation index s_r divided by decay as a commitment index,
  - relaxation census,
  - canalisation curves,
  - bifurcations in real time,
  - a hysteresis *prediction* for transient knockouts.
- A Bocconi conversation (Zecchina, Mézard).

**Traps**
- "Attractor = cell type".
- The Hebbian/perceptron rule, which learns co-expression (the Dileo lesson).
- Thresholds from iid Gaussian theory (0.07 / 0.35 / 0.8).
- Binary units with no time semantics.
- Counting it as evidence.

---

## 6. Most defensible headline claim, and the minimal experiment that falsifies it

**Claim.** "In asynchronous single-cell time courses, real collection times identify absolute (hour-scale) mRNA kinetics through cohort aging, exactly when the entry-time law is non-degenerate (not log-quadratic, or with a resolved onset) and cohorts are kinetically trackable. An age-structured kinetic count model fitted only to total-RNA snapshots predicts the withheld 4sU-labelled RNA of the same cells better than a uniform process-time prior (Chronocell), a Gaussian capture-time prior (VeloVAE) and a prior-only arm. Its certificate correctly refuses on degenerate simulators (SimD/SimD2) and stationary populations."

**Minimal falsifier: the D3 CLOCK analysis, about one week of compute once the core exists.** Fit replicates 1–3 (plus replicate 4's day 3) on total counts. The claim is falsified if any one of these happens:
- (a) the c_kin 95% profile CI (replicate bootstrap, scale-free γ prior, λ → 0) spans more than ×2;
- (b) held-out new-given-total LL does not beat the uniform-age **and** Gaussian-capture arms under the same frozen decoder (paired cluster-bootstrap CI ≤ 0);
- (c) the median predicted half-life is off by more than ×2 from the labelling-derived median;
- (d) on the simulators, SimD2 gives a sharp, wrong ĉ.

---

## 7. Unresolved disagreements

1. **Odds of beating the strong NB-mixture control C2.** Designers put it at 0.35–0.4, critics at 0.2–0.3. I side with the critics and reframe it as non-inferiority plus the cost of interpretability. Still empirically open.
2. **How much information on c survives without a closed reservoir.** The theory critic says the closed reservoir does the identifying work. That remains unquantified for D3; it is handled by the open-reservoir default and the ablations.
3. **Whether day-scale gaps on D3 saturate cohort trackability for most genes.** Not knowable until the CTI is computed on training data. The 5-hour replicate is the hedge.
4. **Whether B's program-level linear OU causal graph should headline a Liò pitch.** The B critics favour it; I defer it to year 2.
5. **β.** Design A claims it in 1/h; the B and C critics say it is not identifiable. Resolved against A.
6. **Clock dataset.** Fishman vs sci-FATE2, resolved by verification in favour of sci-FATE2.
7. **Day-4 status.** Whether replicate 4's samples truly bracket day 4, and whether day 4 exists in all replicates. Must be read from the GEO sample list.

## 8. References and facts still unverified
- **Navigo:** design C cites bioRxiv 2026.06.18.733286; not found by the dossiers or by my search.
- **GRNvelo:** dossiers give MSB 22(8); not found. arXiv 2608.09722 is a different title.
- **Nguyen & Malysheva, arXiv 2609.35060:** not found this run.
- **Verified by critics, not re-checked by me:** Guan et al. arXiv 2605.08550; Balik & Lähdesmäki arXiv 2605.20989; Kucinski et al. 2024; TROUPE; Carta; Chreode numbers.
- **VeloVAE eLife 108950** (design A's citation): unverified. The PLoS CB 2026 version is verified.
- **sci-fate:** pooling of timepoints in one run and availability of spliced/unspliced counts.
- **sci-FATE2:** the exact day list, cells per sample, which GEO layers exist, and the protocol switch times.
- **Argelaguet = GSE205117.**
- **From memory only:** Kwon–Ao–Thouless, Bouchet–Reygner, Preston–Coale, Perthame–Zubelli.
- **Chronocell's erythroid set** being the same as the scVelo subset.

Sources:
- [Velvet / sci-FATE2 bioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.06.547989v1.full)
- [GSE236512](https://www.omicsdi.org/dataset/geo/GSE236512)
- [GSE236517](https://www.omicsdi.org/dataset/geo/GSE236517)
- [sci-FATE2 protocol](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11472621/)
- [VeloVAE PLoS CB 2026 (PMC13021174)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13021174/)
- [VeloVAE summary](https://letsdatascience.com/news/velovae-infers-rna-velocity-with-bayesian-model-052809ff)
- [Fishman et al. 2024 (PMC11006943)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11006943/)
- [Fishman et al. 2024 (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38600066/)
- [cellrank reprogramming_morris](https://cellrank.readthedocs.io/en/stable/api/_autosummary/datasets/cellrank.datasets.reprogramming_morris.html)
- [CardamomOT docs](https://cardamomot.readthedocs.io/en/latest/)
- [TIDYI (Research Square)](https://assets-eu.researchsquare.com/files/rs-8252082/v1/8c523a853dbaa2640d07e243.pdf)
- [Guan et al., gradient-flow SDEs (AISTATS 2026)](https://proceedings.mlr.press/v300/guan26a.html)
- [PFM, arXiv 2604.25062](https://arxiv.org/pdf/2604.25062)
- [scFM, arXiv 2605.22340](https://arxiv.org/pdf/2605.22340)
- [Chronocell](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`