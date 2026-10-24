# DESIGN
# KAIROS: a time-anchored kinetic branching-process model for falsifiable single-cell time-course dynamics (Angle C: evidence and data first, minimal model)

## 1. Thesis

KAIROS (Kinetic, Age-structured, Identifiable, Real-time, Observation-anchored, Stochastic) is a small, explicit generative model. Each cell carries a latent triple z = (s, p, a):
- s is a discrete state (a node of a small graph);
- p is its predecessor state;
- a is its age in state, in real hours.

The population moves through these compartments as a multi-type, age-dependent branching process in the experiment's real clock T, with transition hazards, division and death. Each cell emits raw UMI counts (spliced/unspliced, or new/old RNA) through closed-form first-order transcription kinetics with negative-binomial (NB) noise.

**Why it can succeed where earlier work could not.**
- **Against cfm_grn_v2:**
  - The prediction at an unobserved T comes from population dynamics acting on latent ages. It is not an interpolant linear in snapshot data, so the centroid is not pinned to an affine combination of means (F2/F3).
  - It is scored by a count likelihood, a proper scoring rule. A centroid-matching predictor loses badly on a multimodal held-out population, unlike under ED in z-space.
  - Fate choice is a stochastic hazard, so branching is real (F6).
  - Spread comes from the age mixture plus NB noise, not from path geometry (F4).
- **Against the tgl-grn family:** it models cells with capture times, not per-bin aggregates.
- **Every interpretable parameter has an out-of-fit test.** The data used for the test is never seen by the fit:
  - hazards and growth are checked against lineage barcodes;
  - kinetic rates in 1/h are checked against metabolic labelling and half-life atlases;
  - forecasts are checked against held-out whole timepoints.
- **The scientific claim is identifiability inside a deliberately restricted class.** This is the class where Weinreb-2018-type non-identifiability (curl, growth vs transport, time scale) is broken by declared assumptions plus a measurable condition: non-stationarity and observable age. The model reports equivalence classes whenever that condition fails in the data.

## 2. Model specification

**Clocks**
- T is real collection time (h or days). It is the only external clock.
- a is the real time since entering the current state. It is in the same unit as T, because the population equation transports age at unit speed in T.
- Derived quantities, not separate clocks:
  - developmental age D = the sum of dwell times since leaving the root;
  - entry time E = T − D.
- There is no flow-time and no pseudotime.

**State graph.** S ≈ 6–15 states with E ≈ S+3 directed edges. They come from fit-timepoint cells only: Leiden clustering, then merging, then time ordering into a DAG (or loops if needed). Topology is assumed, not identified.

**Population dynamics** (first-moment equation; McKendrick–von Foerster, multi-type):
```
∂_T n_s(T,a,p) + ∂_a n_s(T,a,p) = [g_s − K_s(a)] n_s ,   K_s(a) = Σ_{s'} k_{s→s'}(a)
n_s(T,0,p) = Σ_{p'} ∫ k_{p→s}(a') n_p(T,a',p') da'      (influx boundary)
n_s(T_1,a,p) = M_1 · π_1(s,a,p)                           (initial; parametric age profile per state)
k_{s→s'}(a) = κ_{ss'} · h_{m_s}(a)                        (Erlang-m_s dwell, m_s ∈ {1,2,4}; time-homogeneous by default)
g_s = b_s − d_s                                           (net growth per state, 1/h)
```
- Discretising age into compartments of width Δ turns this into a linear ODE dn/dT = A(θ)n.
- Hence p(z|T) = normalize(exp(A·(T−T_1)) n_1).
- A pre-declared alternative arm uses κ_{ss'}(T) piecewise-constant on snapshot intervals with a total-variation penalty; it is reported as limited to the snapshot grid's resolution.

**Individual-level process (the sampler).** This is a Crump–Mode–Jagers branching process. A cell in (s,a):
- switches to s' with hazard k_{ss'}(a);
- divides at rate b_s (daughters inherit (s,a,p));
- dies at rate d_s.

Fate choice is therefore stochastic per cell.

**Emission.** For gene j, entry into s with predecessor p, Chronocell-style approximation that the predecessor was at steady state:
```
u_j(a) = α_{j,s}/β_j + (α_{j,p} − α_{j,s})/β_j · e^{−β_j a}
m_j(a) = α_{j,s}/γ_j + (α_{j,p} − α_{j,s}) · [ e^{−β_j a}/(γ_j−β_j) − β_j e^{−γ_j a}/(γ_j(γ_j−β_j)) ]
x^u_{ij} ~ NB(r_i ε_{u,j} u_j(a), φ_j),   x^m_{ij} ~ NB(r_i m_j(a), φ_j)
```
- Depth r_i is fixed beforehand from total UMIs of near-Poissonian genes (Chronocell procedure).
- log ε_{u,j} = η_0 + η_1 log(gene length) captures length/internal-priming bias.
- Genes are conditionally independent given z.
- Cell-cycle genes are excluded from the emission.
- The one-species variant (total counts only) uses m_j with a single relaxation γ_j.

**New RNA from metabolic labelling** (pulse length L; used only as held-out prediction):
- If a ≥ L: E[new_j] = q·α_{j,s}(1−e^{−γ_j L})/γ_j.
- Otherwise: E[new_j] = q·[α_{j,s}(1−e^{−γ_j a}) + α_{j,p}(e^{−γ_j a} − e^{−γ_j L})]/γ_j.
- q (conversion × detection) is one scalar, calibrated on a disjoint 10% calibration subset.

**Batch.** One global scale per batch only. A sensitivity arm adds shrunken gene×batch offsets and reports how much the parameters move (time × batch confounding, L13, is declared, not solved).

**Generating a population at any T\***
1. Compute p(z|T\*) = normalize(exp(A(T\*−T_1)) n_1).
2. Draw z_i from it.
3. Draw r_i from the depth distribution of the nearest training batch. This is a declared technical nuisance.
4. Draw counts from the NB emission.
5. Population size M(T\*) is returned only if total cell counts exist; otherwise output is normalized, plus relative growth.

For T\* > T_K the same exponential is used, under an explicit label: "assumption-driven: time-homogeneous hazards, no unobserved states". The model cannot invent a state that was never observed.

**Parameter count:**
- α: G×S;
- β, γ, φ: 3G;
- κ: E;
- m, g: 2S;
- π_1: about 2S;
- η: 2.

No neural network is in the core.

**Tier-3 extension (month 10+, only if gates pass).** Regulator-modulated hazards:
```
log k_{s→s'}(a, x_i) = log κ_{ss'} + log h(a) + Σ_{r∈R_{ss'}} w_{ss'r} log(1 + x_{ir}/r_i)
```
- R_{ss'} is a small named set of TFs, taken from a prior.
- This is a Cox-type answer to "which regulators drive which transition".
- It is identifiable only with lineage or perturbation data.

## 3. Identifiability (declared class, assumptions A1–A6)

**Assumptions**
- **A1.** Discrete states; piecewise-constant transcription; first-order splicing and degradation constant per gene.
- **A2.** Markov in (s, p, a).
- **A3.** Time-homogeneous κ, g and dwell shape.
- **A4.** Batch acts as a global scale only.
- **A5.** Unbiased sampling at each T_k (no sorting).
- **A6.** Non-stationarity: composition changes over the span on the dwell-time scale.

**Claimed identifiable**

| # | Parameter | Under | Basis | How verified |
|---|---|---|---|---|
| I1 | Absolute time scale: β, γ, κ, g in 1/h | A1–A6 | The t→ct, rates/c symmetry (L9; Chronocell's reason for the uniform prior) is broken, because p(z\|T_k) at known T_k moves with c. It fails exactly when the fitted composition is stationary, mirroring Guan (AISTATS 2026, L5): identifiable iff out of equilibrium. | Profile likelihood over a global scale c. A flat profile means only ratios are reported (γ/β, κ/γ), the Chronocell/scVelo equivalence class. |
| I2 | α_{j,s}·capture, β_j, γ_j | Per Chronocell's argument (identifiable NB family plus identifiable parametric mean curve given the mixing measure); here the mixing measure is the identified population prior | Joint identifiability is not proven, only local. | Numerically: Fisher-information rank and profile curvature per gene. Genes with small \|α_p−α_s\| keep α only (Chronocell's "non-variable genes"). Absolute α is only up to capture: the product α·ε is the equivalence class. |
| I3 | Relative growth g_s − ḡ and hazards κ | A1–A6 | Two independent routes. Their agreement is an internal consistency test; clones are the external test. | See (a) and (b) below. |
| I4 | Branch probabilities κ_{ss'}/Σκ and cell-level fate given z | — | Follows from I3 and A2. | Hidden heritable fate bias (Weinreb 2020) is not representable. The gap to the clonal ceiling is reported as a measured "non-Markov fraction". |

**Route (a) for I3: composition.** Even a simple chain A→B gives dR/dT = κ + (Δg+κ)R with R = N_B/N_A, which has 2 unknowns. Fitting it from at least 3 snapshots gives parametric identifiability under A3. Weinreb 2018 and L8 non-identifiability apply to arbitrary continuous-state g(x); this restricted class avoids them by assumption. The route breaks if κ(T) is free (L7).

**Route (b) for I3: age profile** (a proposition to be proven and tested on simulators). Along characteristics:
```
p_s(T,a) = ν̃_s(T−a) · exp( (g_s − K_s)a − ∫_{T−a}^{T} ḡ )
```
- The relative influx ν̃_{s'}(T) = Σ_s ∫k_{ss'}(a)p_s(T,a)da identifies κ from the density of successor states at age 0.
- The age-slope ratio then gives g_s − ḡ.
- Requirements: age must be observable through transient kinetics over [0, dwell], i.e. 1/γ_j is not much smaller than the dwell time for a set of genes (Chronocell's failure regime when γ falls outside [1,10] process units), and successor states must contain young cells.
- Route (b) also survives a piecewise κ(T). It recovers influx history at a resolution of about 1/γ, finer than the snapshot grid. This is tested on simulators, not assumed.
- Absolute ḡ needs total counts (L8) and is never claimed.
- Related work: Ventre et al. (J. Math. Biol. 2026) show that lineage trees separate proliferation from differentiation; this is the external check.

**Not identified, reported as equivalence classes**
- Topology: the set of graphs within the held-out ΔLL band.
- When to report the affine set of (κ, g) with the same composition curves, plus its two extremes (pure transport / pure growth): when age is unobservable, i.e. cluster-like states with Chronocell AP < 0.5, or when both routes disagree.
- Curl or continuous drift within a state: not modelled (L1, L2, L12).
- Switch timing inside an interval under free κ(T), beyond the age resolution (I8/I9 lesson).
- Absolute population size, ḡ and absolute transcription.
- Regulator→transition links (core: none; tier 3 only with lineage or perturbation data).
- Extrapolation: never identified (L7). It is labelled as assumption-driven.
- Non-Markov hidden variables (L2, L11).
- Not applicable by construction: latent VAE ambiguity (L17), because there is no encoder and the E-step is exact on a grid; and factor rotation (L16), because there are no low-rank factors.

## 4. Inference, complexity, reuse

**Generalized EM on an exact latent grid**
- z ∈ {(s, p, a_ℓ)}, with A ≈ 50 age bins over [0, a_max]. That gives C ≈ (E+roots)·A ≈ 750 configurations.
- **E-step:** q_i(z) ∝ p(x_i|z) · p(z|T_i; θ_pop).
  - The NB log-likelihood decomposes into sparse X (N×G) times dense log-mean matrices (G×C), plus per-z terms.
  - It is computed chunked on GPU, with annealing temperature 3→1 (as in Chronocell).
- **Kinetic M-step:** maximize Σ_i Σ_z q_i(z) log NB(x_ij | r_i λ_j(z), φ_j) over (α_j·, β_j, γ_j, φ_j). This is vectorized over genes with L-BFGS/Adam using the closed-form λ.
- **Population M-step:** minimize Σ_k CE(Q_k(z), p(z|T_k; θ_pop)), where Q_k = mean responsibilities at T_k. Gradients pass through torch.linalg.matrix_exp of A(θ), which is about C×C.
- **Restarts:** 20 random plus 1 cluster-initialized, selected on inner held-out-timepoint likelihood (never training likelihood).
- **Stability gate:** Chronocell-style AP on posterior developmental age (Pearson > 0.8 vs the best start in ≥50% of starts).
- **Uncertainty:** profile likelihood plus a parametric and cell-bootstrap (hierarchical by replicate/embryo where available).
- **Model selection** over S, topology, m_s and homogeneous vs piecewise κ: nested inner folds only, using skip-k multi-removal designs so there are at least 3 inner folds (F8).
- **Scaling beyond 3·10⁵ cells:** stochastic EM with running sufficient statistics. No amortized encoder is needed.

**Cost**
- E-step ≈ nnz(X)·C. For 10⁵ cells, about 2k nonzeros per cell and C = 750, that is about 1.5·10¹¹ multiply-adds: seconds on an A100.
- matrix_exp is O(C³), about 4·10⁸ per timepoint.
- A full fit (300 iterations × 20 restarts) takes about 1–3 GPU-hours. A dataset with 5 seeds × about 8 folds takes about 20–80 A100-hours.

**Reused from the user's code**
- `evaluation/metrics.py`: ED, mean/trace split, var_ratio (secondary metrics).
- `evaluation/controls.py`: mean_interp, persistence, OT-interp, shuffled-time (extended to likelihood versions).
- `flows/coupling.py`: `ot_coupling`/`chain_ot`, used for the McCann control and an own unbalanced WOT baseline.
- `flows/marginal.py`: ED/sliced W2 (secondary).
- `data/real.py`: sparse CSR loaders, `select_genes`. Must be extended to layers u/s/new and to raw counts. Check whether the HPC caches hold raw UMIs; if not, re-parse from source on the HPC.
- `training/designs.py`: skip-k designs.
- The pre-registration SHA-256 freeze, the reproducibility gate and dual checkpoints.
- `data/synthetic_grn.py`: base for Sim-B (needs heterogeneity, division/death and asynchronous entry; fixes F11).
- cfm_grn_v2 v0.1.10, frozen, as the black-box generative baseline.

## 5. Data (in order of use; downloads need the user's permission)

| # | Dataset | Design / size | Extra modality | Claim supported |
|---|---|---|---|---|
| D0 | Simulators (no download) | Sim-A in-family; Sim-B own continuous Hill-GRN chemical-Langevin model with bistable switches, expression-dependent division/death, Poisson asynchronous entry, explicit u/s species, NB capture; Sim-C dyngen backbones (bifurcating/converging, known kinetics) plus a lineage-weight resampler for growth; Sim-D scMultiSim kinetic mode | Full truth: entry times, descendant fates, γ, growth | I1–I4 recovery under misspecification (different family) |
| D1 | Pijuan-Sala erythroid subset (scVelo `gastrulation_erythroid`) | 9,815 cells, stage labels E7.0–E8.5, about 0.2 GB | spliced/unspliced | Head-to-head with Chronocell on its own case; uniform vs anchored prior; held-out interior stage; γ vs mouse half-life tables (e.g. Sharova 2009 mESC, Schwanhäusser 2011, mismatched cell type, so only the improvement over the uniform prior is claimed). Full atlas (116,312 cells, 9 stages E6.5–E8.5) already parsed on the HPC; E8.0 pools are batch-confounded (floor 1.105) |
| D2 | sci-fate (Cao et al., Nat Biotech 2020; GEO GSE131351, accession to verify) | A549 + dexamethasone, 0–10 h at about 2 h spacing, >6,000 cells (≈6,680, to verify), small | 4sU new-RNA layer (2 h pulse) | I1 time scale and γ in 1/h via **modality hold-out**: the new layer is never fit. Glucocorticoid-receptor (NR3C1) response is a known driver |
| D3 | Weinreb LARRY in vitro (GSE140802; CoSpar `hematopoiesis_130K`) | ≈130k cells, ≈50k clonally labelled, days 2/4/6 | Clonal barcodes (unspliced only via FASTQ re-quantification; otherwise one-species model) | I3/I4 via **clone hold-out**: fate bias of day-2 cells, clone-size growth. Only one interior design is possible (fit d2+d6, predict d4) plus a terminal one (d6) |
| D4 | Biddy CellTag reprogramming (GSE99915; `cellrank.datasets.reprogramming_morris`) | 8 timepoints d0–d28; 85,010 10x cells (96,356 total); tags at d0/d3/d13 | Clones | Confirmatory: interior holdouts (e.g. d6, d12, d15), terminal d28 (assumption-driven), reprogrammed vs dead-end fate and growth vs clones |
| D5 | Schiebinger WOT (GSE122662) | ≈315k profiles, mostly half-day spacing over 18 days (39 timepoints per the dossier, to verify); 2i/serum arms; processed data several GB | Dense time; proliferation signatures | Horizon sweep (hold out 1/3/5 consecutive timepoints); growth vs held-out proliferation score; terminal forecast |
| D6 | Veres stage 5 (cached) | days 0–7, 2 differentiations | Replicate floor | Fresh confirmatory generative held-out-stage likelihood (one-species) |
| D7 (stretch) | Saunders et al., Nature 2023 (GSE202639) | 3.2M cells, 1,812 individually resolved embryos, 19 timepoints, 23 crispants, ≥8 embryos per condition | Embryo-level replication; perturbation × time | Proper replicate floors and partial time×batch break; tier-3 regulator–hazard layer tested on held-out crispants |

## 6. Evaluation protocol

**Primary metrics (none centroid-dominated)**
- **P1. Held-out whole-timepoint predictive log-likelihood.** Measured on raw counts (NB), in nats per cell and per 10³ UMIs, on a fixed fit-only gene panel with given depth r_i.
  - Paired per-cell differences against each control, with a hierarchical bootstrap (cells within replicate/embryo × 5 seeds).
  - Gap closed = (LL_model − LL_best control)/(LL_oracle − LL_best control).
  - Oracle-at-T = the same model family cross-fitted on half the held-out cells. The replicate floor is reported where one exists.
- **P2. Composition error.** Total-variation distance between predicted and observed state proportions at held-out T, using a fit-only state classifier. Transitional states are included.
- **P3. Shape.** Mean-removed ED plus var_ratio, in a fit-only PCA-50 of log1p counts. Full ED with its mean/trace split is secondary, kept for comparability with past work.
- **P4. Modality hold-out.**
  - (a) sci-fate new-RNA log-likelihood and per-gene Spearman of predicted vs observed new fraction.
  - (b) D1: likelihood of the unspliced layer under a model fitted on spliced only.
- **P5. Lineage hold-out.**
  - (a) Day-2 fate bias against clonal sister fates: neutrophil-vs-monocyte AUROC and Pearson with clonal fate fraction.
  - (b) Spearman of predicted expected descendants vs observed clone size, using the same sampling correction for every method.
  - (c) Ancestor-state accuracy.
- **P6. External parameter checks.**
  - γ vs half-life tables (Spearman with CI), compared against Chronocell-uniform and scVelo γ rescaled to hours.
  - g_s − ḡ vs proliferation score and vs clonal expansion.
- **P7. Terminal holdout.** Same metrics, in a separate table labelled "assumption-driven".

**Controls (all produce likelihoods)**
- C1 persistence-NB: NB mixture fitted on the nearest snapshot.
- C2 density interpolation: (1−w)p_k + w p_{k+1}.
- C3 composition interpolation: shared NB-mixture clusters, weights interpolated linearly or logit-linearly in T. This is expected to be the strongest control.
- C4 mean-shift replay: the likelihood version of mean_interp.
- C5 McCann/OT displacement plus NB kernel.
- C6 shuffled-time twin.
- **C7 KAIROS-uniform:** Chronocell prior, with time rescaled post hoc by regressing age on T. This is the key ablation.
- C8 KAIROS-no-age (CTMC compartments) and KAIROS-g≡0.
- C9 cfm_grn_v2 v0.1.10, plus optionally TIGON or DeepRUOT from official code. Its likelihood via an NB kernel around decoded cells is flagged as approximate.
- Lineage baselines: own unbalanced WOT, PRESCIENT (already reports LARRY fate bias), kNN to later cells, and a supervised clonal-label ceiling.
- **Calibration gate:** published baseline numbers must be reproduced in our harness before any comparison against them (the Job C lesson).

**Held-out designs**
- Whole timepoints, interior (single and skip-k) and terminal.
- Whole modalities: new RNA, unspliced, clones.
- Whole perturbations (D7).
- Biological-unit resampling.
- Gene set, states and depth model are always chosen on fit cells only.

**Simulator recovery gates (R-I, pre-registered; Sim-B/C/D only)**

| Target | Pass threshold |
|---|---|
| Time scale | \|log ĉ\| < log 1.25 in ≥80% of non-stationary simulations, **and** a flat profile flagged in ≥80% of stationary negative-control simulations |
| γ on transient genes | Spearman ≥ 0.7 |
| Relative growth | Sign accuracy ≥ 0.8 and Spearman ≥ 0.6 |
| Cluster-like simulations | Reported (κ, g) set covers the truth in ≥80% |
| Fate | AUROC ≥ Markov-oracle − 0.05 |
| Calibration | 90% interval coverage ≥ 80% |

A parameter that fails here is demoted to "not identified in practice" before any real data is touched.

**Decision rules (frozen, applied by code)**
- **R-G (generative):** ΔLL against the best of C1–C5 has 95% CI lower bound > 0 on ≥2 of the 3 confirmatory sets (D4, D6, D3-interior), **and** P2 is not worse than C3.
- **R-C (clock):** on D2, KAIROS beats C7 on new-RNA LL and on γ Spearman by ≥0.1 (CI excludes 0), with absolute Spearman ≥ 0.5.
- **R-L (lineage):** fate AUROC ≥ max(WOT, PRESCIENT) − 0.02 (non-inferiority), **and** clone-size Spearman beats C8 (CI).
- **Kill criteria**
  - **K1 (month 4):** if KAIROS neither beats C3 on D1/D6 LL nor beats C7 on D2, the model line stops. Pivot to the benchmark/falsification paper plus a documented negative.
  - **K2:** growth fails on the simulators → growth reported only as equivalence classes.
  - **K3:** AP < 0.5 on more than half the datasets → age is unusable; fall back to C8 plus lineage validation.
  - **K4:** no configuration change after a confirmatory set is opened. Every deviation is logged together with the original rule's outcome.

## 7. DO / DON'T, mapped to the user's past projects

**DO**

| Rule | Lesson from |
|---|---|
| Fill PURPOSE.md and freeze the pre-registration before writing model code | tgl family (question drift, empty PURPOSE) |
| Compute oracle-at-T, controls and headroom before any model code | cfm_grn_v2 F1 no-network ceiling; tgl T9 oracle features |
| Use a count likelihood on raw UMIs | cfm_grn_v2 F3/F9 (centroid-dominated ED, z-spikes up to 263,822) |
| Model stochastic branching and explicit growth with no cell caps | cfm_grn_v2 F6, I5 |
| Couple only within (condition, replicate) groups | cfm_grn v1 global OT |
| Use velocity-like and lineage observations as the levers | cfm_grn v1 v3: data regime beat architecture |
| Simulators of a different family, with heterogeneity | cfm_grn_v2 F11; I18 |
| Guard every stage that produces reported numbers | cfm_grn_v2 F7 |
| Use ≥3 inner folds via skip-k | cfm_grn_v2 F8 |
| Calibration gate before citing published numbers | cfm_grn_v2 Job C |
| Ablations that hurt when removed (no-age, no-growth, shuffled time) | tgl "temporal modules not temporal" |
| Heavy jobs on the HPC, one canonical repo | tgl #13/#14 |

**DON'T**

| Rule | Lesson from |
|---|---|
| Interpolants linear in snapshot data; standardized-space ED as the primary metric | cfm_grn_v2 F2/F3 |
| A black-box MLP drift as the scientific object | cfm_grn_v2 interpretability 0/5 |
| Architecture-first iteration | tgl-grn-2/-exp/-recurrent |
| GRN-edge claims without hard negatives or ground truth | Dileo AUPRC 0.94 → 0.50; cfm_grn v1 mask AUC 0.54 |
| A uniform process-time prior, or process time = embryonic day | Chronocell dossier |
| Claiming extrapolation as identified | cfm_grn_v2 F10; L7 |
| Tuning after seeing a test window | EB v0.1.10 |
| Reading seed-only CIs as biological generalization | cfm_grn_v2 risk 8 |
| "Attractor = cell type" claims from deterministic fixed points | Badalotti dossier |

## 8. How the two papers are used

**Chronocell**
- **Used as:** the decoder template, re-implemented from the equations (original-code rule):
  - two-species closed form;
  - near-Poissonian depth estimation;
  - annealed grid EM;
  - AP stability gate;
  - trajectory-vs-cluster test, used here as the "age observable?" diagnostic.
- **Its uniform prior** is ablation C7.
- **Its erythroid dataset** is the head-to-head D1.
- **Its γ-vs-labelling check** is scaled up into a formal modality hold-out (D2).
- **Changed:**
  - real-time-anchored age-structured prior instead of the uniform one;
  - stochastic hazards instead of shared switch times;
  - NB instead of Poisson;
  - growth;
  - length bias;
  - held-out likelihood instead of AIC (AIC failed on PBMC).

**Badalotti et al. (PRR 2026)**
- Not in the core: it has no generative or time content.
- Concrete uses:
  1. Conceptual mapping: the persistence control is the κ→0 (infinite-dwell) limit, the analogue of J_D→∞.
  2. Per-state "stability index" = expected dwell/(T_K−T_1) plus the probability of absorption. It is reported, not claimed as an attractor.
  3. Tier 3 only: pulse/hysteresis tests if perturbation-pulse data are added.
  4. A local Bocconi conversation about the tier-3 regulator-hazard layer (networking, not evidence).
- Rating unchanged (about 1.5/5 as a method).

## 9. Novelty vs closest competitors

| Method | What it has | What KAIROS adds or differs |
|---|---|---|
| PFM (Maddu/Chardès/Shelley, arXiv 2604.25062) | Continuous-state probability flow, Jacobian regulation, growth via regularizer | Discrete state + age; growth identified via route (a)/(b) and tested against clones; no Jacobian reading |
| StructureFlow / Cell-MNN | Structured drift | They have no growth and no count likelihood; their latent or PCA spaces are not tested out of fit |
| DeepRUOT / RUOT / unbalanced and branched SB / ARTEMIS / TIGON | Growth | Growth is set by a penalty or kill-rate network; MLP drift with post-hoc Jacobians. KAIROS treats growth as a declared, testable parameter |
| pseudodynamics (Fischer et al., Nat Biotech 2019) | **Closest population model:** PDE with growth on pseudotime, fitted to a real time course | Pseudodynamics fits a precomputed 1-D pseudotime density (two-step, circular). KAIROS infers age jointly from count kinetics, is multi-state branching, has a count likelihood and external validation |
| RegVelo / cell2fate / scVelo | Kinetics | Latent time, no population or growth |
| Dynamo | Absolute rates from labelling | Deterministic per cell, no population; KAIROS uses labelling as held-out validation |
| Chronocell | Process time, count model | Uniform prior, no real time, no growth |
| PRESCIENT | Potential + proliferation, LARRY fate | Used as a baseline |
| JKOnet\* | Energies | Gradient-only |
| mvOU-SB | Linear-Gaussian, non-gradient | Continuous and Gaussian |
| Navigo (bioRxiv 2026.06.18.733286, confirmed) | FM + RNA kinetics, 12.4M cells, 43 timepoints, zero-shot KO | Already covers generative + continuous at scale. Differs in having no declared identifiable class and a black-box field |
| GRNvelo | Name unverified | Treated as possibly nonexistent |
| WOT / CoSpar / moslin / LineageOT | Transport maps | Used as baselines and validation |

**Genuinely new**
1. A relative-growth/hazard identifiability proposition through observable age-in-state under real-time anchoring, with its failure condition diagnosed from the data.
2. A real-time generative count model in which every interpretable parameter has a pre-registered out-of-fit test.
3. A benchmark whose likelihood controls (density and composition interpolation) and modality/lineage hold-outs cannot be won by mean interpolation. This also extends a GGE-type evaluator, relevant to Liò.

**Honest caveats**
- Every building block exists; the contribution is the conjunction plus the falsification design.
- A real-time Chronocell from the Pachter lab is the main scoop risk. None was found as of Oct 2026.

## 10. 6–12-month plan

**Milestones**

| When | Work | Gate |
|---|---|---|
| Weeks 1–3 | PURPOSE.md; pre-registration v1 (datasets, splits, metrics, C1–C9, thresholds) frozen; download-permission list; likelihood harness C1–C5 + oracle; run on D1 and the D6 caches | **G0:** oracle − best control > 3 bootstrap SE (headroom exists) |
| Months 1–2 | KAIROS v0 with tests (closed forms vs numerical ODE, PDE mass balance, in-family recovery); Sim-B; dyngen/scMultiSim on HPC | **G1:** R-I |
| Month 3 | D1 vs Chronocell-uniform + controls; γ external check | **G2:** first half of K1 |
| Month 4 | D2 modality hold-out (frozen) | **R-C**; K1 decision |
| Months 5–6 | D3 clone hold-out | **R-L**; paper 1 (method + benchmark release) |
| Months 7–9 | D4 confirmatory (R-G); D5 horizon sweep and terminal forecasts; D6 | R-G |
| Months 10–12 | Tier 3 on D7 crispants (held-out perturbations), or stop; thesis proposal for Liò | — |

**Compute**
- Model fits: about 20–80 A100-hours per dataset; about 300–600 A100-hours in total.
- Packing: seeds and folds go into single jobs (no arrays; 1 running / 2 submitted). That is about 3–6 weeks of queue wall time spread over the year.
- dyngen: about 50–200 CPU-hours.
- Optional FASTQ re-quantification for unspliced counts: about 100–300 CPU-hours per dataset. Schiebinger FASTQs are TB-scale, so the default there is the one-species model.

**Risks and fallbacks**

| Risk | Fallback |
|---|---|
| No unspliced counts | One-species model; the D2 labelled layer is the clock |
| Cluster-like data | C8 compartments + equivalence classes + lineage validation |
| C3 cannot be beaten on likelihood | Claim parity plus validated parameters as the headline |
| Hidden fate bias | Report the non-Markov fraction as a result |
| Batch × time | Only the D7 embryo-hashed design gives a clean test; sensitivity arm elsewhere |
| EM optima | Restarts + AP gate |
| Scoop | Benchmark + identifiability proposition remain |

**Honest probabilities**

| Outcome | Probability |
|---|---|
| Benchmark + simulator identifiability study done (workshop/benchmark-paper level) | ≈0.8 |
| R-C positive | ≈0.5 |
| R-L non-inferior + growth–clone correlation | ≈0.45 |
| R-G beats composition interpolation | ≈0.35 |
| Presentable, honest positive (R-I + at least one of R-C/R-L) within 9–12 months | **≈0.6** |
| All four | ≈0.1 |

## Sources
- [Saunders et al. 2023, Embryo-scale reverse genetics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10665197/)
- [Biddy et al. CellTag (Behind the Paper)](https://communities.springernature.com/amp/posts/single-cell-mapping-of-lineage-and-identity-in-direct-reprogramming)
- [cellrank.datasets.reprogramming_morris](https://cellrank.readthedocs.io/en/stable/api/_autosummary/datasets/cellrank.datasets.reprogramming_morris.html)
- [CoSpar hematopoiesis_130K](https://cospar.readthedocs.io/en/latest/cospar.datasets.hematopoiesis_130K.html)
- [CoSpar LARRY tutorial](https://cospar.readthedocs.io/en/latest/20210121_all_hematopoietic_data_v3.html)
- [sci-fate bioRxiv](https://www.biorxiv.org/content/10.1101/666081.full.pdf)
- [pseudodynamics (Fischer 2019)](https://cris.fau.de/publications/290862181)
- [scVelo gastrulation_erythroid](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.gastrulation_erythroid.html)
- [scVelo gastrulation](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.gastrulation.html)
- [Schiebinger WOT bioRxiv](https://www.biorxiv.org/content/10.1101/191056v1)
- [Chronocell PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/)
- [Navigo (sciety)](https://sciety.org/articles/activity/10.64898/2026.06.18.733286)
- [PFM arXiv 2604.25062](https://arxiv.org/pdf/2604.25062)
- [PRESCIENT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8163769/)
- [ARTEMIS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12261462/)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`

# THEORY CRITIC
# Adversarial review: KAIROS (Angle C, evidence-first). Lens: theory, identifiability, metric validity

## (1) Verdict

**5.5/10 as written. About 7.5/10 if F1, F2 and S1–S8 below are fixed.**

The model class is the right kind of object: counts, a real-time age-structured branching process, and hold-outs of modalities and clones. It escapes cfm_grn_v2's F2, F3, F4 and F6 by construction. The prediction at T* is exp(A(T*−T₁))n₁ acting on latent ages, so it is not affine in the snapshot data.

Three problems remain:
- The primary generative metric recreates the mean-interpolation trap in a new coordinate.
- The headline identifiability proposition (I3) is false in the stated generality, and its novel route has no external test on the planned data.
- The time-scale test (I1) can certify γ in 1/h when only κ is identified.

## (2) FATAL flaws

### F1. P1, R-G and K1 rank emission flexibility, not dynamics (the mean-interpolation trap, latent edition)

**The decomposition.** The predictive density is p̂(x|T*) = Σ_z p_θ(x|z) q̂(z|T*). With sharp posteriors, the per-cell log-likelihood is about log p_θ(x_i|z_i*) + log q̂(z_i*|T*).

**Size of the dynamics term.**
- For two methods that share an emission, E[ΔLL] → KL(Q*‖q̂_B) − KL(Q*‖q̂_A).
- For adjacent-snapshot composition shifts (state proportions moving by about 0.1) this is about 0.05–0.5 nats per cell.
- The hard ceiling is log C ≈ log 750 ≈ 6.6 nats per cell.

**Size of the emission term.**
- Emission mismatch between method families is about 0.005–0.05 nats per gene. Over about 2,000 genes that is 10–100 nats per cell.
- KAIROS's emission is piecewise-constant α with first-order relaxation and conditional independence. C1/C3 are free NB mixtures with K selected on likelihood.
- So the dynamics signal is 2–3 orders of magnitude below the emission signal. This is the exact analogue of the ED mean term being 10–100× the trace term. Normalising "per 10³ UMIs" changes nothing.

**Consequences.**
- **False kill.** KAIROS can lose R-G with correct dynamics, because every panel gene must follow piecewise kinetics. Chronocell kept only 24/49 and 84/182 genes. K1 would then stop the line for a metric reason.
- **False positive.** KAIROS can win P1 through its age-continuum emission (a continuum of mixture components) with wrong dynamics.
- **The general limit.** For an interior hold-out at gap Δ, any summary that is smooth in T (composition, mixing measure) is linearly interpolable with error of about (Δ²/8)·|∂²_T q|. Dynamics can win only where curvature·Δ² is much larger than the sampling SE, or where narrow age cohorts are transported. In the second case the right control is displacement interpolation, not mixture interpolation.

**Fix: make a plug-in regret metric primary.**
- **Stage E.** Fit θ_em on fit timepoints with *free* per-snapshot mixing measures Q_k (Chronocell-like, no dynamics), then freeze it.
- **Stage D.** Every predictor outputs q̂(z|T*) on the same grid:
  - KAIROS;
  - L-mix: (1−w)Q_k + wQ_{k+1};
  - L-disp: per-state age-quantile (McCann) interpolation × logit-linear state weights;
  - persistence;
  - shuffled-T KAIROS.
- **Oracle.** q* is fit (mixing weights only, θ_em frozen) on half A of the held-out cells and scored on half B, then swapped.
- **Regret.** Regret(q̂) = mean_i[LL_i(q*) − LL_i(q̂)].
- **Gap closed.** 1 − Regret(KAIROS)/Regret(best latent control).
- **Decision rule R-G′.** On skip-k folds (k ≥ 2), the 95% hierarchical-bootstrap lower bound of the regret reduction is > 0 on at least 2 of 3 confirmatory sets. On single-interior folds, the upper CI is below +0.02 nats/cell.
- **G0′, before any model code.** Require Regret(best latent control) > 3 SE. Use fit-only curvature ‖Q_{k+1} − 2Q_k + Q_{k−1}‖ to preregister which folds have headroom.
- Report full-LL as secondary, and emission quality separately as LL at q*.
- Rewrite K1 on regret.

### F2. I3 is false as stated, and route (b) cannot be validated where external truth exists

**(a) Route (a) holds only for the 2-state chain.**
- The chain gives R′ = κ + (Δg+κ)R: 2 unknowns, R(T₁) given, so 3 snapshots suffice. That much is correct.
- For a general graph: (S−1)(K−1) constraints vs E + (S−1) + dim π₁ (+ m_s) unknowns. Example: S = 8, E = 11, π₁ ≈ 2S gives 7(K−1) ≥ 34, so K ≥ 6 *fit* snapshots, before conditioning.
- The composition curve is a sum of exponentials (the eigenvalues of A). This is a Prony-type problem: about K/2 rates are resolvable from noisy points, and conditioning explodes when eigenvalues sit within a factor of about 2.
- LARRY interior (fit d2+d6) gives zero dynamic constraints. All three LARRY days give 2(S−1) < E+S−1.

**(b) The single-snapshot age profile has the Lotka/stable-population confound.**
- p_s(T,a) = ν̃(T−a)·S_s(a)·e^{g_s a − ∫ḡ}. Tilting ν̃(E)e^{cE} with g → g+c fits identically.
- Identification needs **cohort tracking**: the same entry cohort E seen at two snapshots, at ages a and a+Δ, both resolvable by the kinetics. This is the Preston–Coale (1982) "variable-r" intercensal method, here with latent age; cite it.
- It identifies ∫(g_s − K_s(a) − ḡ)da over Δ, not g_s alone.
- Condition: there are informative genes with γ_jΔ_snap ≲ 2, and mean dwell ≥ Δ_snap.

**(c) An "age-0 successor" looks exactly like an old predecessor.**
- (u,m)(a=0 | s′, p=s) = (α_s/β, α_s/γ), which is the steady state of s.
- Entry becomes visible only after about 1/β (sparse unspliced counts) or about 1/γ (one-species model).
- So κ is read from the post-entry transient. That is exactly where the predecessor-at-steady-state approximation is wrongest (see S3).

**(d) The datasets sit in the wrong regime.**
- Mouse mRNA median half-life is about 7–9 h, so γ ≈ 0.08–0.1/h.
- D3 (Δ = 48 h) and D4 (Δ = 72–168 h) give γΔ ≈ 4–17: cohorts saturate. Route (b) is dead exactly where clones exist.
- D1 (Δ = 6 h) and D5 (Δ = 12 h) satisfy the condition but have no lineage truth. D1 also violates A5 (see S4).
- R-L's fate AUROC mostly tests the geometry of the state partition and is insensitive to the κ/g trade-off.
- So "novelty #1" has no external test in this plan.

**Fixes.**
1. **Restate I3 as a conditional proposition.**
   - Route (a): require full rank of ∂(composition at T₂..T_K)/∂(κ, g−ḡ, π₁), checked structurally for the chosen graph *before fitting* (Meshkat–Sullivant linear-compartment criteria, or SIAN/differential algebra).
   - Route (b): preregister a cohort-trackability index, CTI = share of informative-gene signal with γ̂_jΔ_snap < 2 × P(dwell > Δ), computed from fit data.
2. **Make the routes operationally separate estimators.**
   - (a) = C8 compartments fit on composition only.
   - (b) = cohort-ratio estimator on posterior ages from Stage-E fits.
   - Their agreement is a consistency check, not independent evidence.
3. **Promote D4 to the primary I3/R-L set.** The cellrank `reprogramming_morris` file has raw spliced/unspliced counts, CellTag clones, 8 timepoints over d0–28, 104,887 cells (85,010 in the 10x subset). Restrict I3 there to route (a).
4. **Find a dataset where CTI and lineage coexist**, or report I3 only as equivalence classes. Candidates (I did not verify their designs):
   - zebrafish TracerSeq (Wagner et al. 2018, hpf-scale spacing);
   - scEU-seq pulse-chase organoids (Battich 2020), where labelling acts as an intracellular age clock;
   - ChronoSeq (bioRxiv 2025.11.15.688181, minute-scale real time).

## (3) SERIOUS flaws, with concrete fixes

**S1. The I1 test conflates two symmetries.**
- A single global profile over c is curved whenever composition moves through κ and g, even if the emission carries no age information. It would then certify γ in 1/h falsely.
- The stated condition is also wrong:
  - the κ,g scale needs non-stationary composition;
  - the β,γ scale needs a non-stationary (s,a) distribution *and* age-informative transients, roughly γ·dwell ∈ [0.3, 5].
  - A synchronised cohort aging inside one state identifies γ while composition stays stationary.
- **Fix:** run block profiles c_kin (β,γ scaled, population fixed) and c_pop (κ,g scaled, emission fixed). Claim hours only if the c_kin profile rises ≥ 1.92 within |log c| < log 1.25, under the calibrated likelihood from S5. The Guan AISTATS 2026 link is an analogy, not a theorem.

**S2. β is not identifiable in these regimes.**
- Splicing takes minutes to about 1 h, much shorter than dwell, so u ≈ ε_u·α(a)/β. The steady-state ratio u/s = ε_uγ/β confounds β with capture.
- **Fix:** report β as a ratio only. The length law for ε_u is an assumption, since internal priming varies by gene.

**S3. The predecessor-at-steady-state approximation biases transitional states.**
- **Fix:** set the entry condition to E[(u,m) of p at exit], convolved with p's exit-age law. This has a closed form for Erlang × exponentials. Alternatively, use the latent (s,p,a,b_p) with 3–4 bins of predecessor exit age.
- Quantify the bias on Sim-B before any real data.
- Demote γ for genes with γ·dwell(p) < 2.

**S4. A5 and time × batch act on composition, which is the identifying signal.**
- Cell-type-specific capture, dissociation and QC removal of dying cells all distort composition per batch. A4 (one expression scale per batch) does not cover this.
- D1 is selected by annotation. Influx from non-erythroid mesoderm continues through E7.0–E8.5 and is unmodelled, so it is misattributed to g_root.
- **Fixes:**
  - add root influx ρ(T) = exp(ρ₀ + ρ₁T) and declare g_root not identified;
  - prefer the full atlas over the subset;
  - use D2 as the batch-clean reference: sci-fate indexes all timepoints in one combinatorial run, so time and batch are not confounded there;
  - add simulator negative controls with gene × batch offsets and state-specific capture bias. Gate: either a flat profile is flagged, or the CI covers the truth.

**S5. Conditional independence over about 2,000 genes overcounts information.**
- Posteriors become nearly one-hot. Profile-likelihood CIs are anti-conservative, and Fisher rank is always "full", so it says nothing.
- **Fix:**
  - make hierarchical bootstrap the primary uncertainty;
  - temper the likelihood with a weight calibrated on Sim-B so that 90% intervals reach nominal coverage;
  - declare identifiability from structural analysis + simulator recovery + bootstrap, never from Fisher rank.
- Generated cells also lack covariance within z. Either note this for P3, or add a non-interpreted low-rank lognormal nuisance factor.

**S6. The age numerics blur the clock.**
- Upwind age bins inside exp(At) make age an Erlang chain: CV = 1/√k, about 0.3 after 10 bins. That blurs exactly the T ↔ a link that identifies the time scale.
- **Fix:** step exactly with a Leslie matrix (Δ = common divisor of the snapshot gaps; per-bin exit probability 1−e^{−k(a)Δ}), or use the closed form along characteristics.
- Make the hazards consistent: k_{ss′}(a) = π_{ss′}·h_{Erlang(m_s,λ_s)}(a) with Σπ = 1.
- In the route-(b) formula, use ∫₀ᵃK_s rather than K_s·a when m_s > 1.
- Use the limit form of m_j(a) when γ = β.

**S7. The E-step cost claim is wrong for NB.**
- The NB terms φ·log(φ/(φ+r_iμ)) and x·log(r_iμ/(φ+r_iμ)) break the "sparse X × dense log-mean" factorisation, which holds only for Poisson. The dense part is about 1.5·10¹¹ transcendental operations per iteration.
- **Fix:** bin r_i into about 32 quantiles and precompute G×C×32 tables. Budget 2–5× more compute.

**S8. κ depends on clustering resolution.**
- Splitting a continuum into more Leiden clusters multiplies the boundary-crossing rates.
- **Fix:** report only coarse-graining-invariant quantities: first-passage times root→terminal in hours, branch probabilities at merged branch points, terminal relative growth, and entry-time laws. Preregister invariance across S₀ ± 2 (≤ 20% change).
- Define Sim-B truth for κ and g from the basins of the bistable switches (empirical hazards on simulated trajectories) before running. Otherwise "covers the truth" is undefined.

**S9. D2 does not discriminate the novelty.**
- sci-fate: synchronous dexamethasone at 0–10 h, 2 h 4sU labelling before harvest, about 6,000 cells (the 6,680 count and GSE131351 are still unverified).
- With synchrony and roughly even sampling across timepoints, the uniform prior is nearly correct, so C7 ≈ KAIROS. A pass would validate per-cell T anchoring, not age-structured branching or growth.
- γ is identifiable from total RNA only for responsive genes. Preregister that gene subset from fit data only (about 100–300 genes).
- **Fixes:**
  - keep R-C as an absolute-unit check: |Spearman| ≥ 0.5, and the slope of log γ̂ vs log γ_label within [0.8, 1.25];
  - add a sampling-skew stress arm that subsamples timepoints with weights ∝ 1, ½, ¼, …. Prediction: C7's Spearman drops by at least 0.15, KAIROS's by at most 0.05;
  - exclude the 10% of cells used to calibrate q from evaluation.

**S10. R-L is weak and underpowered.**
- Day-2 fate AUROC ranks cells by their position in the state partition, so kNN-type methods can do it. A Markov chain over discrete z discards within-state priming, and Weinreb 2020 shows hidden fate bias. KAIROS will likely lose to WOT/PRESCIENT.
- LARRY has 1,816 clones spanning early and late timepoints. Sampling (half at day 2, 30% at day 4) means binomial thinning dominates clone sizes.
- **Fix:**
  - simulate power on the real clone table before preregistering;
  - model thinning with the known fractions;
  - compare growth against g ≡ 0 and against signature-based growth, not against C8;
  - use a non-inferiority margin of −0.05.

**S11. Tier 3 is internally inconsistent.**
- A hazard that depends on x_i (the emitted counts) breaks the linear first-moment closure, so p(z|T) is no longer exp(AT)n₁.
- The mean-field fix (hazard on E[x|z]) makes w confounded with κ(s,a).
- **Fix:** let hazards depend on latent program activity per (s,a), tested on held-out crispants. Otherwise drop tier 3.

**S12. AP < 0.5 measures optimisation stability, not whether age is observable.**
- **Fix:** run K3 on ΔRegret(C8 − KAIROS) and on I(a; x | s,T), computed as the prior-to-posterior entropy drop.

**S13. C9 is a straw man.** An NB kernel around decoded cells in about 2,000 count dimensions guarantees a loss. Score cfm_grn_v2 on secondary metrics only.

**S14. The CMJ sampler needs b_s and d_s separately, but only g_s enters the first moments.** Fix d_s (for example to 0) or declare b_s and d_s non-identified. Any clone-size variance prediction is then assumption-driven.

## (4) Claims to downgrade to "assumption" (or to a weaker form)

| Claim | Downgrade to |
|---|---|
| Time scale identifiable iff out of equilibrium | Block-wise and conditional (S1); the Guan link is an analogy |
| I3 from "≥ 3 snapshots" | Graph-specific rank condition (F2a) |
| Route (b) "survives piecewise κ(T)" and "influx history at ~1/γ resolution" | Model-based interpolation of ν̃ between snapshots; testable only on simulators or labelling |
| Two "independent routes" | Same data; a consistency check only |
| I2 local identifiability from Fisher rank | Invalid evidence (S5) |
| "Non-Markov fraction" as a measurement | Upper bound, confounded with misspecification, coarse-graining and clonal sampling noise |
| κ and β in 1/h | Resolution-dependent (S8) or ratio-only (S2); the ε_u length law is an assumption |
| γ vs half-life atlases | Only the ordering vs C7 counts (cell types differ, and half-life studies agree only moderately with each other) |
| g_s − ḡ | Conditional on A5, no external influx, no QC bias; the correlation with proliferation score is partly built in at state level |
| "Benchmark cannot be won by mean interpolation" | Cannot be won by centroid matching; can be won by emission flexibility or latent interpolation (F1) |
| A4 batch-as-global-scale | Untestable wherever time and batch are confounded (all sets except D2 and D7) |
| Generated populations | No within-z covariance; depth taken from the nearest batch |
| pseudodynamics described as "circular" | It explicitly resolves growth vs flux using population sizes, which KAIROS lacks. The real difference is latent kinetic age plus a count likelihood |

## (5) What is genuinely strong and must be kept

- **The model object.** Raw-UMI NB likelihood, a real-time age-structured multitype branching process, stochastic fate, and spread coming from the mixture plus NB noise. Together these remove cfm_grn_v2's F2, F3, F4, F6 and F9 structurally.
- **Modality and clone hold-outs.** New RNA, clones and unspliced counts are never fit. This is evidence that marginals cannot decide, and it is the real contribution.
- **The two key ablations.** C7 (uniform prior) and C8 (no age, g ≡ 0).
- **Reporting discipline.** Equivalence-class reporting, a stationary negative control, a demotion rule after simulator gates, pre-registration, kill rules K1–K4, the calibration gate and the reproducibility gate.
- **Simulators from different families:** dyngen, scMultiSim and an own Hill-Langevin model.
- **A minimal model** with no neural core, at feasible compute.
- **Honesty about scope.** Extrapolation is labelled as assumption-driven, and there are no claims that attractors equal cell types.
- **Use of Chronocell** as a re-implemented decoder template, with held-out likelihood replacing AIC.

**Revised odds.**
- As written, K1 has about a 0.5 chance of a false kill.
- With F1 fixed, the regret version of R-G on skip-k folds has about a 0.4 chance of passing.
- With F2 fixed, I3 on D4 via route (a) is about 0.4. Route (b) is about 0.15 unless a dataset meeting the CTI condition and carrying lineage is added.

Sources:
- [sci-fate bioRxiv](https://www.biorxiv.org/content/10.1101/666081.full.pdf)
- [Weinreb 2020 LARRY (PMC7608074)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7608074)
- [cellrank reprogramming_morris](https://cellrank.readthedocs.io/en/stable/api/_autosummary/datasets/cellrank.datasets.reprogramming_morris.html)
- [pseudodynamics (Fischer 2019)](https://cris.fau.de/publications/290862181)
- [ChronoSeq bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.15.688181.full.pdf)
- [Chronocell PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`

# FEASIBILITY CRITIC
# Adversarial review of KAIROS (angle C, evidence first)

## (1) Scores

| | As written | After the cuts in §6 |
|---|---|---|
| **Feasibility** | **4/10** | **6.5/10** |
| **Novelty** | **5/10** | **5.5/10**, if the identifiability certificate plus clone validation is the headline |
| P(presentable honest positive in 9–12 months) | **about 0.3**, not 0.6 | **about 0.45** |

**Why feasibility is low as written.**
- The scope is far too wide for one person: 7 datasets, 4 simulators (two of them in R), 9 control families including TIGON, DeepRUOT and PRESCIENT, a CMJ sampler, a proof, and FASTQ re-quantification.
- More importantly, the confirmatory tests are mostly chosen so that they cannot tell a good model from a bad one (§2).

**Why novelty is moderate.**
- Every block exists: age-structured population models, pseudodynamics, Chronocell, real-time compartment models, branching rate inference from lineage, and capture-time-anchored kinetic models.
- The capture-time-anchored kinetic model was published in March 2026 (VeloVAE, PLoS CB). The design does not cite it.

---

## (2) FATAL problems as written (each can be fixed, but not with small tweaks)

### FATAL-1. The K1 kill gate is wired to tests that come out null for structural reasons, not because of model quality

**R-C on D2 (sci-fate) cannot separate KAIROS from C7.**
- Dexamethasone induction is synchronous: every cell enters at T=0, so age ≈ T for everyone.
- Cells per timepoint are nearly equal: 1,054, 1,049, 949, 1,262, 1,041 and 1,325 at 0–10 h (verified).
- So C7's uniform process-time prior, rescaled afterwards by regressing on T, is almost correct by construction. The KAIROS-vs-C7 contrast on new-RNA LL and γ Spearman has little power. The required Δρ ≥ 0.1 is unlikely.
- The main non-GR heterogeneity in A549 is the cell cycle, which the emission excludes. A549 does not branch. The age-structured, branching part of KAIROS is therefore idle on D2.

**R-G against C3 at interior one-step holdouts reproduces cfm_grn_v2's F2/F3 problem in mixture space.**
- On K = 7–9 snapshots, composition curves are smooth. Logit-linear interpolation of the weights is close to optimal at an interior point, just as mean_interp was for centroids.
- KAIROS's predicted composition is a smooth function fitted to the same composition points.
- Worse, per-cell NB LL over about 2k genes is thousands of nats per cell. The prior/composition term is about log p(z\*|T), a few nats.
- So ΔLL against C3 is dominated by differences in **decoder capacity**: KAIROS has within-state age-dependent means and C3 does not. Any "win" would be an emission win, not a dynamics win. That is the same kind of mistake as ED being dominated by the mean.

**R-L on D3 (LARRY) is under-identified.**
- The R-L design fits d2 and d6 and predicts d4, so the fit uses K = 2.
- Route (a) counting with S ≈ 10:
  - Data: about (S−1)·(K−1) = 9 independent composition numbers.
  - Parameters: about E + (S−1) + (π₁ ≈ 2S) + m_s ≈ 13 + 9 + 20 + 10 ≈ 50.
- Route (b) needs a velocity-like layer:
  - LARRY is distributed as `normed_counts` (verified that the 130,887-cell matrix is normed; whether raw UMIs are on GEO is unverified, because GEO showed a captcha).
  - It is inDrops, so there is no unspliced layer without FASTQ re-quantification.
  - Neutrophil/monocyte dwell is days while mRNA 1/γ is hours, so age information saturates after about 2–3 half-lives.
- Result: growth and hazards on LARRY are driven by the prior. "Growth–clone correlation" would then test the regulariser.
- Weinreb 2020's hidden heritable fate bias also caps every Markov model, so non-inferiority to WOT/PRESCIENT is likely and says nothing.

**Net effect:** K1 at month 4 most likely fires, or someone changes the gates, which breaks K4. That is the cfm_grn_v2 pattern again: heavy protocol producing ties.

### FATAL-2. The emission contradicts the identifiability route it is meant to serve

- Route (b) needs **dwell ≲ a few × 1/γ**, so that age is observable.
- The emission assumes the **predecessor was at steady state at entry**, which needs **dwell_p ≫ 1/γ**.
- Both hold only in a narrow window. In the regime where route (b) claims identifiability, the entry condition u,m(a=0) is wrong, and age estimates and κ come out biased in a structured way.

**Fix:** carry the predecessor's exit age.
- Make the latent (s, a, a_p), or propagate the exact piecewise-constant ODE along the path, as Chronocell does across τ intervals.
- This multiplies C by about A_p (e.g. ×10–20 with coarse bins).
- Alternatively, restrict route (b) claims to states whose predecessor passes a "dwell ≫ 1/γ" check, and test that check on simulators.

### FATAL-3. "κ in 1/h" is not a biological quantity when states are Leiden clusters on a continuum

- Splitting a state into two roughly doubles κ and halves dwell. The Erlang m absorbs the rest.
- The α_p→α_s step at a cluster boundary is a clustering artefact.
- "Identifiable within the model" does not mean meaningful.

**Fix:** claim only resolution-invariant functionals, and show they are invariant across at least 3 resolutions:
- mean first-passage time from root to terminal (h);
- terminal branch fractions;
- relative growth of terminal fates;
- the distribution of entry times into terminal states.

Report per-edge κ only as a nuisance.

---

## (3) SERIOUS problems, with fixes

### S1. Wrong primary dataset; the best-fitting one is under-used

**D4 Biddy/Morris CellTag fits the design best.** Via `cellrank.datasets.reprogramming_morris` (verified):
- raw **spliced and unspliced** counts are included;
- 104,887 cells (85k and 48k subsets);
- 8 timepoints, days 0–28;
- `CellTagDN` clone columns;
- about 1.5 GiB.

Make it **the** confirmatory dataset for R-L and R-G (skip-k). Caveats:
- passaging removes absolute growth, so claim relative growth only;
- forced Hnf4a/Foxa1 expression means A3 may fail, so the piecewise-κ arm is mandatory;
- early clonal commitment (Biddy 2018) means a non-Markov fraction should be expected.

**Replace D2 as the clock test.** Fishman et al., Nat Commun 2024 (verified): zebrafish embryogenesis, metabolically labelled scRNA-seq across real time, with cell-type-specific transcription and degradation rates.
- Specification is asynchronous and the time is real.
- The γ check is **in-system**, which removes the Sharova/Schwanhäusser cell-type mismatch.

Keep sci-fate only as a **unit test with known truth**: age ≈ T, so the model must recover a near-delta age distribution plus γ. Do not use it as a contest against C7.

**D1 erythroid** should be a calibration gate only: re-implement Chronocell and reproduce AP ≈ 0.79 with about 24/49 genes kept. Do not make it confirmatory, for three reasons:
- **MURK genes** (Barile et al., Genome Biol 2021) show multi-rate transcriptional boosts in this exact dataset, which violate constant-rate kinetics.
- The lineage subset gets influx from unobserved upstream mesoderm, violating A5. A time-varying immigration term is needed, and influx vs growth is then confounded again.
- E8.0 is batch-confounded.

**D5 Schiebinger** (one species, dwell of days, a 2i/serum medium switch at day 8 so A3 is violated by design), **D6** and **D7** (3.2M sci-RNA-seq3 cells) should be cut from the 12-month scope.

### S2. Missing decisive control: the no-dynamics twin (C3+)

- C3+ uses the **same KAIROS decoder and latent grid**, but sets p(z|T\*) to the interpolation of the posterior compositions Q_k(z) at the neighbouring fit times (linear or logit).
- Primary R-G metric: the held-out **prior log-score under a frozen shared decoder**, i.e. LL decomposed into emission plus prior, compared with C3+.
- Without this, the R-G results cannot be interpreted.

### S3. Missing control C7b: VeloVAE-style Gaussian capture-time prior

- VeloVAE already anchors latent time to capture time and reports rates in real units.
- KAIROS's I1 claim has to beat **that** prior, not only the uniform one.

### S4. The NB E-step cost is underestimated

- The design treats the E-step as nnz·C. That holds only for Poisson.
- For NB, the term Σ_j (x_ij+φ_j)·log(φ_j + r_i μ_jz) is dense in (i, j, z) even when x = 0: 10⁵ × 2·10³ × 750 ≈ 1.5·10¹¹ terms per species per E-step.
- Unfused PyTorch chunking is memory-bound, about 1 s per E-step. A fit (300 iterations × 20 restarts × 2 species) then takes about 3–10 GPU-hours, roughly 3× the plan, and the NB M-step has no sufficient statistics.
- **Fix:** quantise r_i into 16–32 depth bins (the dense term becomes B×G×C), run Poisson EM, and do a final NB refinement. Pack seeds and folds per job.

### S5. Age discretisation smears exactly the signal route (b) needs

- Upwind age advection inside expm(A·ΔT) adds numerical diffusion of about Δ/2 per unit T.
- **Fix:** use exact Leslie/characteristics stepping, with the age bin equal to the T step and Δ dividing every snapshot gap. It is exact and cheaper than expm on a C×C matrix (C can reach 3k once Erlang phases are added).

### S6. Route (b) is a known ill-posed inverse problem, with a latent age on top

- Recovering hazards and growth from age or size densities is ill-posed of degree ≥ 1 (Perthame–Zubelli 2007; Doumic–Perthame–Zubelli 2009; from memory, unverified).
- Here age is only seen through a kinetic posterior, which adds a deconvolution.
- Expect the Erlang shape m_s and fine κ(T) to be practically unidentifiable. Pre-register that, and do not promise "influx history finer than the snapshot grid".

### S7. Held-out timepoints are new batches (L13)

- Gene×batch offsets at the held-out T shift emission LL by tens to hundreds of nats per cell, more than the composition signal.
- The S2 decomposition helps. Also report a "batch-robust" score: the prior log-score, plus P2 TV computed from a fit-only classifier.

### S8. Unverified dataset claims to fix

- **LARRY:** raw UMIs on GEO are not verified; the dossier itself flags that the HPC caches may not hold raw UMIs.
- **Veres caches:** raw vs normalised status unknown.
- **Biddy:** replicate structure for the hierarchical bootstrap is unverified.
- **Pijuan-Sala erythroid stages:** I count 7 (E7.0–E8.5 in 0.25-day steps); the design implies more.

### S9. Proliferation-score validation is weaker than it looks

- Cell-cycle genes are excluded from the emission, so g_s vs proliferation score is genuinely out of fit. That part is good.
- But the score measures division, not net growth (no death term), and it saturates in culture. Make clones the primary growth test and the score secondary.

---

## (4) Scoop risks

### Verified this run

| Work | What it covers | Consequence for KAIROS |
|---|---|---|
| **VeloVAE** (Gu, Blaauw, Welch et al.), PLoS CB, Mar 2026, doi 10.1371/journal.pcbi.1014060 | Capture time as an informative Gaussian prior on latent time, giving absolute units; Poisson count likelihood on raw counts; branching ODE with cell-type-specific rates; transcription rates validated against sci-fate2 labelling (mean PCC 0.72). No growth, no population generation at unobserved T. | Scoops "time-anchored process time plus count likelihood plus labelling validation". KAIROS's I1/R-C is **not** a headline. |
| **Kucinski et al.**, Cell Stem Cell 2024 (PMID 38183977) | Persistent HSC labelling plus time-series scRNA-seq give a real-time quantitative model of differentiation and growth speeds per stage. | A reviewer will say "real-time rates on a state graph exist". The difference is that KAIROS gets them without fate mapping, with an identifiability condition. |
| **Barile et al.**, Genome Biol 2021 (bioRxiv 2020.12.21.423773) | Erythroid kinetic changes (MURK genes). | Undermines D1 as a constant-rate testbed. |
| **TROUPE**, "Inferring Cell Differentiation Dynamics with Unobserved Progenitors" (bioRxiv 10.64898/2025.12.09.693214), and **Carta** (bioRxiv 2024.09.09.611835) | Maximum-likelihood multi-type branching transition and growth rates from lineage trees. | They own "rates from lineage". Use them as lineage-side baselines or ceilings. |
| **Fishman et al.**, Nat Commun 2024 | Cell-type-specific transcription and degradation kinetics in real time. | A dataset opportunity, and a partial scoop of "γ in 1/h per cell type". |
| **Global StationaryOT** (bioRxiv 2025.12.18.694987) | Aging time courses of snapshots. | Minor overlap. |
| **ChronoSeq** (Asthana, Jambor, Wang; bioRxiv 2025.11.15.688181) | Minute-scale real-time sampling. | A data platform. Dense sampling would make the anchoring trivial. Useful as a sanity set, not a scoop. |
| **Already acknowledged by the design** | pseudodynamics (2019), TIGON, DeepRUOT, Ventre (J Math Biol 2026), Chronocell. | No new risk. |

### Not verified (from memory or snippet only)

- PhyloVelo (phylogeny as a clock for velocity).
- TemporalVAE (capture-time-supervised developmental time).
- Höfer-lab compartment models fitted to label propagation (Busch 2015 and others).
- TIDYI (Research Square rs-8252082): absolute proliferation and death rates from scRNA-seq snapshots. Seen as a snippet only; check it, because it directly overlaps relative/absolute growth from snapshots.
- arXiv 2601.22282 (time-varying branching process for self-renewing cells).
- Navigo: I did not re-check the designer's "confirmed".
- Perthame/Zubelli/Doumic inverse problems.

### Residual scoop risk

- **Medium.** No paper found combines an age-structured branching population, a kinetic count decoder, real-time anchoring, an identifiability condition and clone hold-out.
- But "VeloVAE + growth" or "Chronocell + capture time" are short steps for those groups.

---

## (5) Strongest defensible publishable claim

> "From unpaired time-course snapshots with spliced and unspliced counts, an age-structured branching model with a kinetic count decoder identifies **resolution-invariant** relative growth and branch-timing functionals exactly when a **data-computable certificate** holds:
> - non-stationary composition;
> - posterior age SD below a stated fraction of dwell;
> - Fisher information rank sufficient.
>
> On held-out CellTag clones (Biddy/Morris reprogramming, 8 days), these functionals predict reprogrammed-vs-dead-end fate fractions and relative clonal expansion **without seeing barcodes**, non-inferior to WOT and a clonal ceiling. Where the certificate fails, the reported (κ, g) equivalence set covers the clonal truth."

This is backed by:
- a simulator study on out-of-family simulators, stationary negative controls included;
- one clock result where it is not trivial: γ in 1/h against in-system labelling-derived rates (Fishman 2024), beating uniform **and** Gaussian capture-time priors.

**Venue fit**
- Cell Systems or RECOMB/ISMB-level with a positive clone result.
- A NeurIPS/ICML reviewer would say "classical McKendrick/compartment model, identifiability largely standard, small ML novelty". Target an ML workshop (e.g. ICLR GEM/LMRL) plus a bioRxiv preprint.

**Pitch for Liò**
- Good fit as "principled, falsifiable generative modelling with identifiability certificates plus an evaluation benchmark", which ties to his GGE evaluator work.
- Present tier 3 (regulator-modulated hazards on a prior graph) as the PhD extension: that is the graph-ML hook. Do not make it a deliverable.

---

## (6) Cuts that make it doable in 6–12 months

1. **Data:**
   - D4 primary (lineage plus generative, skip-k);
   - Fishman 2024 as the clock dataset;
   - D1 only as the Chronocell reproduction gate;
   - sci-fate only as a known-truth unit test;
   - drop D3 (or keep it as a one-species, secondary fate-bias check), and drop D5, D6 and D7.
2. **Model:**
   - first-moment equation with Leslie stepping;
   - drop the CMJ sampler (use branching moment or generating-function formulas for expected clone size);
   - carry the predecessor's exit age, or gate route (b) by the dwell ≫ 1/γ check;
   - depth-binned NB;
   - no tier 3.
3. **Simulators:** Sim-A (in-family) plus one out-of-family simulator (own Langevin Sim-B with division, death and asynchronous entry, or dyngen plus a growth resampler). Drop scMultiSim.
4. **Controls:**
   - keep C1, C3, **C3+ (no-dynamics twin)**, C6, C7, **C7b (Gaussian capture-time prior)** and C8, plus own WOT and a clonal ceiling;
   - drop TIGON, DeepRUOT, PRESCIENT and cfm_grn_v2 from the primary tables.
5. **Metrics:** decompose LL into emission and prior; the primary dynamic score is the prior log-score under a frozen shared decoder. Report only resolution-invariant functionals, checked across 3 clustering resolutions.
6. **Gates:**
   - replace K1 with G0 headroom on **D4** plus the simulator gate R-I;
   - kill only on a simulator recovery failure, or a D4 headroom of 0 against C3+.
7. **Timeline:**

| Months | Work |
|---|---|
| 1–2 | Harness, Chronocell reproduction, simulators |
| 3–5 | KAIROS core plus R-I |
| 6–8 | D4 confirmatory (clones held out) |
| 9–10 | Clock dataset |
| 11–12 | Write-up, benchmark release |

Compute: about 150–300 A100-hours with seeds and folds packed per job. That fits the 1 running / 2 submitted QoS.

---

Sources:
- [VeloVAE, PLoS CB 2026 (PMC13021174)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13021174/)
- [Kucinski et al., time- and single-cell-resolved hematopoiesis model](https://www.biorxiv.org/content/10.1101/2022.09.07.506735.full.pdf)
- [cellrank reprogramming_morris](https://cellrank.readthedocs.io/en/stable/api/_autosummary/datasets/cellrank.datasets.reprogramming_morris.html)
- [sci-fate (Cao et al.)](https://www.biorxiv.org/content/10.1101/666081.full.pdf)
- [LARRY dataset (normed_counts)](https://pypi.org/project/LARRY-dataset)
- [pertpy weinreb_2020](https://pertpy.readthedocs.io/en/stable/usage/data/pertpy.data.weinreb_2020.html)
- [Fishman et al. 2024, Nat Commun](https://cris.huji.ac.il/en/publications/cell-type-specific-mrna-transcription-and-degradation-kinetics-in/)
- [Barile et al., erythroid kinetics](https://www.biorxiv.org/content/10.1101/2020.12.21.423773.full.pdf)
- [TROUPE / unobserved progenitors](https://www.biorxiv.org/content/10.64898/2025.12.09.693214.full.pdf)
- [Carta](https://www.biorxiv.org/content/10.1101/2024.09.09.611835.full.pdf)
- [Global StationaryOT](https://www.biorxiv.org/content/10.64898/2025.12.18.694987v1.full)
- [ChronoSeq prereview](https://prereview.org/en-us/preprints/doi-10.1101-2025.11.15.688181/write-a-prereview)
- [TIDYI (snippet only)](https://www.researchsquare.com/article/rs-8252082/v1.pdf?c=1770709712000)
- [Time-varying branching process, arXiv 2601.22282](https://arxiv.org/html/2601.22282)
- [Chronocell](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/)
- [TIGON](https://www.nature.com/articles/s42256-023-00763-w)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`