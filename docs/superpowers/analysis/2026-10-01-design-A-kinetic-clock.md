# DESIGN
# Angle A: "Kinetic Clock First". An age-structured kinetic population model (AKP)

## 1. Thesis

AKP is a count-level generative model of a cell population observed at real times T. Each cell has three latent quantities: a lineage path `l`, an **age** `a` (hours since the cell left a progenitor reservoir), and its counts.

- **Age advances at exactly the speed of real time** (da/dT = 1).
- **Who is present at time T** comes from a McKendrick–von Foerster population: real-time influx κ(T), branch probabilities π_l, and net growth r_l(a).
- **What a cell of age a on path l looks like** comes from a closed-form transcription → splicing → degradation ODE, with rates in 1/h. Its transcription is driven by a few named regulon programs h_r(a,l), on a sparse signed CollecTRI support.
- **Generation** is: sample (l, a) from the population density at T*, then sample spliced and unspliced UMI counts from a negative binomial (NB).

**Why it can succeed where earlier work could not:**
- **It escapes the centroid pin of cfm_grn_v2 (F2/F3).** The prediction at a held-out T is `Σ_l ∫ n_l(a,T) NB(x | μ(l,a)) da`. That is nonlinear in the data: cohorts age along exponential-relaxation trajectories, so the predicted mean is not an affine combination of snapshot means.
- **Spread is calibrated, not geometric (F4).** It comes from the mixture over ages plus NB noise, not from the shape of OT-chained paths.
- **Branching is a stochastic lineage choice, not an injective flow (F6).**
- **No rare-gene spikes (F9).** A gene with near-zero counts is a harmless NB term, not a z-score of 263,822.
- **Every parameter has a unit or a name.** Rates are in 1/h; κ is in cells/h per reservoir cell; h_r belongs to a regulon.
- **It models cells, not aggregates.** That is what the tgl-grn family could never do.
- **Interpretability comes from a declared restricted class, never from post-hoc reading of a drift.** The restrictions are unit-speed age, cohort invariance, a kinetic ODE and anchored supports. Extra data (unspliced counts, metabolic labelling, ATAC, barcodes) are used only for validation.

## 2. Model specification

**Observed data.**
- Cell i has a known collection time T_i ∈ {T_1..T_K}, in one real unit (h or d).
- Raw UMIs u_ig (unspliced) and s_ig (spliced), for a gene panel G of 300–1,000 genes chosen on training stages only.
- Size factor ℓ_i, from near-Poissonian genes (Chronocell procedure).
- No normalisation, log transform or z-scoring.

**Topology.**
- A rooted tree of segments: a reservoir R (stationary progenitor state), a trunk, and branch points at ages τ_b. Lineage paths l = 1..L.
- The candidate set (≤ 3 topologies, from annotation) is declared up front. One is chosen by the log-likelihood on an inner validation stage only. AIC on a lower bound is never used.

**Latent variables per cell.**
- c_i ∈ {R, 1..L}
- a_i ∈ [0, A_max] in the same real unit as T
- Derived entry time T_e = T − a

**Population layer (McKendrick–von Foerster with branching).**

```
dR/dT = (r_R − κ(T)) R,            ρ(T) = κ(T) R(T)            (influx)
∂_T n_l + ∂_a n_l = r_l(a) n_l,    n_l(0,T) = π_l(T_e=T) ρ(T)
⇒ n_l(a,T) = π_l(T−a) ρ(T−a) exp(∫_0^a r_l) · 1[T−a ≥ T_0]   (+ optional 2-parameter initial cohort at T_0)
p(c=R|T) = R/N,   p(c=l, a | T) = n_l(a,T)/N(T),   N = R + Σ_l ∫ n_l da
```

- log κ(T): a cubic B-spline with knots at the training T_k, plus a second-difference penalty.
- r_l(a): piecewise constant per segment. Optionally it gets a proliferation-signature prior, as in WOT.
- π_l: constant by default. π_l(T_e) as a logistic in T_e is a declared alternative (see §3).

**Kinetic layer, per gene g.**

```
log α_g(a,l) = b_g + Σ_{r ∈ Pa_S(g)} W_gr h_r(a,l)
    S = CollecTRI signed support; sign(W_gr) fixed where the prior sign is confident, free otherwise
h_r(a,l): piecewise constant on age bins Δ ≈ (min gap)/4
          shared on the trunk
          h_r(0) = 0 (reservoir reference)
          total-variation penalty
+ 2–3 unnamed sparse non-negative nuisance programs (absorb non-TF drivers; never interpreted)
dμ_u/da = α_g(a,l) − β_g μ_u,    dμ_s/da = β_g μ_u − γ_g μ_s,
    initial state = reservoir steady state (α_g^0/β_g, α_g^0/γ_g)
```

- With α_m constant inside an age bin of width δ, the solution is exact:
  - `μ_u ← μ_u e^{−βδ} + (α_m/β)(1−e^{−βδ})`
  - `μ_s ← μ_s e^{−γδ} + (α_m/γ)(1−e^{−γδ}) + β(μ_u − α_m/β)(e^{−βδ} − e^{−γδ})/(γ−β)`
- β and γ are in 1/h.

**Count model.**
- `u_ig ~ NB(ℓ_i ε_g μ_u, φ^u_g)` and `s_ig ~ NB(ℓ_i μ_s, φ^s_g)`.
- `log ε_g = η_0 + η_1 log(intron length)` models the Gorin–Pachter unspliced length bias.
- Genes are conditionally independent given (c, a).
- Optional per-batch gene offsets δ_{b,g}, only where a sequencing batch contains ≥ 2 stages.
- Fallback when there are no unspliced counts: a one-species model, `dμ_s/da = α − γμ_s`.

**Time semantics.**
- T is real collection time.
- a is age in the same unit, with unit speed.
- There is **no flow time and no pseudotime**.
- Real time affects the mechanism **only** through κ(T), and optionally π_l(T_e).
- Intracellular programs depend on age, not on T. This is assumption **A1, cohort invariance**: the explicit price of identifiability (theory dossier L7).

**Sampling at an arbitrary T\*.**
1. Compute R(T\*) and n_l(·,T\*) on a grid in closed form.
2. Draw c from the composition; draw a by inverse CDF.
3. Draw ℓ from the depth model, or use a fixed reference depth (stated).
4. Draw u and s from the NB model.

**Outputs.**
- Counts plus each cell's latent (l, a, T_e).
- Optionally, genealogies from a Gillespie branching process with division rate r_l(a), giving clone-size distributions.
- Absolute N(T\*) only when total cell counts are known.

**Honesty device ("assumption share").** Every generated population reports the fraction of its cells that depend on components no data constrain:
- cells born in (T_k, T\*), whose numbers come from the interpolated κ spline;
- cells older than the maximum fitted age.

Between snapshots most cells are **cohorts observed at T_k that simply age** (mechanistic). Only newborns are interpolated.

**Extrapolation past T_K is assumption-driven.** It is shown as an envelope over three assumptions:
- (E1) κ is either held constant or linearly extrapolated;
- (E2) programs stay at their last age bin, so cells relax to the terminal steady state;
- (E3) growth is unchanged.

**Counterfactuals** (model-based, labelled as such): set h_r := 0 from age a_0 or from real time T_0, or change κ.

## 3. Identifiability

**Assumptions.**
- **A1** Cohort invariance.
- **A2** The topology lies in the declared set, and each lineage has ≥ 1 transient segment.
- **A3** The NB kernel a ↦ K(·|a) is injective, and the continuous mixture is identifiable (Chronocell's condition).
- **A4** The age distribution is non-stationary: log ρ is not affine on [T_1 − A_max, T_K], or the start is synchronised (ρ = δ at a known T_0).
- **A5** Anchors: each program has ≥ 3 targets regulated by no other selected program, and the activity matrix h (R × age bins) has full row rank.

**P1. Time unit (rates in 1/h).**
- The setup: rescale age a = c·a′. The kinetics absorb this through the rates (θ → θ_c). The marginals then need `ρ′(T−a′)G′(a′) ∝_T ρ(T−c a′)G(c a′)` for all (T, a′).
- With f = log ρ, differentiating gives `f″(T−a′) = c f″(T−c a′)` on an open set. So f″ ≡ 0 unless c = 1.
- **Statement: the time unit is identified iff ρ is not exponential (constant included).**
- Converse: an exponential influx gives a stable age distribution that does not depend on T, and then any c fits.
- This is the age-structured analogue of Guan et al. (AISTATS 2026, "identifiable iff not at equilibrium"). It generalises Chronocell's "fixed p(z)" condition.
- The proof covers continuous T. For K discrete snapshots and a spline ρ I do **not** claim a proof. It is checked by a profile likelihood over c on simulators and on every dataset.
- If the 95% profile interval for c spans more than 2×, only ratios are reported (γ/β, α/γ; the Chronocell/L9 equivalence class).

**P2. Population layer, given ages.**
- `log p(c,a|T) = log π_l + log ρ(T−a) + ∫_0^a r_l − log N(T)`.
- The tilt (ρ·e^{λu}, r_l + λ, r_R + λ) leaves the composition unchanged.
- **Identified:** the shape of κ(T), π_l, differential growth (r_l − r_l′ and r_l(a) − r_R).
- **Not identified:** the global ḡ (L8), unless N(T) is known.
- **Second equivalence:** π_l(T_e) log-linear in T_e is confounded with differential lineage growth. Either π is constant (and growth is reported) or growth is fixed (and fate drift is reported). Clone sizes (LARRY) or proliferation markers break the tie.

**P3. Programs.**
- Given identified α_g(·) for transient genes, the factorisation log α = b + W h is unique up to a per-program scale (fixed by ‖h_r‖_∞ = 1, with the sign set by the majority of prior signs) under A5.
- This follows from L16: fixed supports with distinct columns, plus anchor conditions in the style of Moran et al. 2022.
- If A5 fails, the affected programs are reported as a **co-active block** {r1, r2}. Only the sums of W over shared targets are claimed.
- Diagnostics: the posterior correlation of the h's and the rank of the Fisher information.

**Kinetic rates (from Chronocell).**
- Given an identified p(a|T), α_g(·), β_g and γ_g are identified for transient genes whose relaxation time 1/γ lies roughly between Δ_min/3 and 3× the span.
- For other genes only ratios are identified.
- Each gene is tagged "rate-identified" or "ratio-only" by the width of its profile CI.

**L1 gauge freedom is removed by construction.** The only velocity allowed is unit speed along age, and the velocity in gene space is the kinetic ODE. There is no divergence-free component to choose.

**Not identified, and reported as such:**
- the global growth ḡ;
- π(T_e) versus differential growth;
- regulon activity versus TF protein or post-translational activity. h_r is the "effective regulon activity";
- edges outside S;
- switch timing inside a gap beyond what kinetic transients resolve (reported as an interval);
- A1 itself, which marginals cannot test in general. It gets a posterior-predictive "cohort test": cells in the same (l, a) bin from different T should be exchangeable, checked with a permutation null;
- stage-specific gene batch effects when batch = stage (L13). These go to the replicate floor;
- within-(l,a) gene–gene covariance (checked, not modelled);
- curl and oscillations, except declared loops. A cell-cycle loop is the stationary negative case of P1: its period comes from outside (doubling time or labelling);
- single-cell paths.

**Verification on simulators of a different family.**

| Simulator | What it is | Purpose |
|---|---|---|
| SimA (own agent-based model) | Hill-function GRN of 30–60 TFs with an explicit protein layer and delays; telegraph bursting; splicing; division with binomial partitioning and death; asynchronous entry with a non-exponential κ (pulse); **fate set by a stochastic toggle switch, not a fixed τ_b**; Beta capture with unspliced length bias; per-stage batch offsets (sd 0.1); 30% false edges planted in the prior | Main recovery test |
| SimB | dyngen (Cannoodt 2021): linear, bifurcating and cycle backbones, with staggered starts | Recovery |
| SimC | scMultiSim kinetic mode with a time-varying GRN | Recovery |
| SimD | SimA with exponential influx | **Must fail** (flat profile over c), as P1 predicts |
| SimE | Same family as the model | Code sanity only; never counted |

## 4. Inference, complexity and code reuse

**Objective.**

```
Σ_i log Σ_{c, a_m} p(c, a_m | T_i; ψ_pop) Π_g NB(u_ig, s_ig | ℓ_i, μ(c, a_m; θ_kin))
  − TV(h) − L1(W on unsigned edges) − smooth(log κ) − weak N(log γ | log(ln2/8h), 1.5²)
```

The γ prior is generic. External half-life atlases are kept strictly for validation.

**Exact grid marginalisation.**
- An age grid of about 200 points × L lineages; no encoder.
- Minibatch Adam on the exact log-sum-exp, 5–20k steps.
- Annealed responsibilities (Chronocell's β_anneal), 0.3 → 1 over the first 30% of steps.
- Gene sums are accumulated in chunks, so the B × G × grid tensor is never materialised.

**Starts and stability.**
- 20 random starts plus 1 informed start (age = T − median per training-only cluster).
- Average-precision stability: a start counts if its posterior mean age has r > 0.9 with the best start. A dataset needs AP ≥ 0.5 before any claim is made on it.
- 20 bootstrap resamples over cells give CIs for γ, κ and h. Profile likelihoods are computed for c and the growth differentials.

**Selection and guarding (lesson F7).**
- Topology, λ_TV, the number of nuisance programs and A_max are selected on inner validation stages only.
- The final refit uses the iteration count fixed in nested CV. There is no unguarded stage.

**Ceiling diagnostics (the analogue of the no-network check).**
- On simulators, an oracle-age fit.
- On real data, a "free-age ceiling": per-snapshot unrestricted q(a|T_k), fitted including the held-out stage.
- The gap between AKP and these ceilings separates population-layer error from decoder error (lessons F1/F5).

**Cost.**
- Per step, O(B · G · |grid| · 2). With B = 1,024, G = 1,000 and 600 grid points, that is about 10 ms on an A100.
- One fit takes 5–60 min; 20 starts take 1–2 h.
- The model is small: about 10–20k parameters.

**Reused code.**
- From cfm_grn_v2:
  - `evaluation/metrics.py`: ED, var_ratio and the mean/trace split, kept as secondary metrics;
  - `evaluation/controls.py`, extended to count mixtures;
  - `flows/marginal.py`: ED;
  - `flows/coupling.py`: only for the OT-interpolation control;
  - `data/real.py`: sparse loaders and `load_collectri`;
  - `data/prior.py`: the degree- and sign-preserving random prior, used as control C6;
  - `training/designs.py`: skip-k whole-timepoint folds;
  - the pre-registration / SHA freeze and the reproducibility gate;
  - the parsed HPC caches (Veres, Pijuan-Sala, Argelaguet, EB) for one-species runs and continuity.
- From the tgl family: leakage tests, persistence-relative reporting, and the degree-matched negative sampler.

**New original code, about 2k lines.**
- Kinetic decoder: about 400 lines.
- Population layer: about 200.
- Grid likelihood: about 300.
- SimA: about 600.
- Count-level controls and metrics: about 500.

## 5. Data, in order of use

| # | Dataset | Design | Supports | Caveats |
|---|---|---|---|---|
| D0 | SimA–E | Known truth | P1–P3; the recovery thresholds in §6 | — |
| D1 | **sci-fate GSE131351** | A549 + dex at 0/2/4/6/8/10 h; 1,054/1,049/949/1,262/1,041/1,325 cells; 4sU labelling 2 h before harvest. Combinatorial indexing likely pooled all timepoints, which would reduce the batch × time confound (to verify). Small download. | Fit on **total** counts only. V1: γ in 1/h against new/old-derived γ (held out). V2: the NR3C1 regulon program switches on at dex T = 0. | Synchronised clock: tests the decoder and programs, not asynchronous inference. Cell-cycle loop: exclude cycle genes or fix the period from the A549 doubling time. Intronic counts unverified, so possibly one-species only. |
| D2 | **scNT-seq GSE141851** | Mouse cortical neurons, KCl for 0/15/30/60/120 min; 20,547 cells; 4sU labelling 2 h | V1 at minute scale; V2: the IEG program (CREB1/SRF/MEF2 regulons) switches on at T = 0 | The label window starts before stimulation |
| D3 | **Pijuan-Sala erythroid** (scVelo `gastrulation_erythroid`, subset of E-MTAB-6967) | E7.0–E8.5 at 0.25-day steps (7 stages, about 9.8k cells; count to verify); spliced/unspliced | **Primary asynchronous test**: head-to-head with Chronocell on its own dataset (AP 0.79, 24/49 genes); interior holdouts E7.75 and E8.0; terminal E8.5; Gata1/Tal1/Klf1 programs; half-lives at rank level only | Posteriors flat until E7.5, so A4 may hold only weakly |
| D4 | Bastidas-Ponce GSE132188 | Pancreas E12.5–E15.5 (4 stages), Ngn3-Venus sorted | One interior holdout; replication | Spliced/unspliced ready only for E15.5 (scVelo/CellRank); the other stages need a loom or re-quantification (download permission). FACS sorting distorts composition, so P2 claims are off. |
| D5 | Argelaguet multiome GSE205117 | E7.5–E8.75, 5 stages about 6 h apart; RNA+ATAC from nuclei (high unspliced fraction) | V3: h_r (fitted on RNA only, CollecTRI support, so not circular) against chromVAR deviations per (l, a) bin, plus the lead–lag of priming. V4: TF self-consistency. | User's earlier negative at E8.0 |
| D6 | Weinreb LARRY GSE140802 | In vitro haematopoiesis, days 2/4/6, clonal barcodes (size to verify) | V5: barcodes withheld; predicted fate of day-2 cells and growth differentials against clones; tests A1/A2 | Hidden fate bias (L2) predicts partial failure. Unspliced needs re-quantification (permission) or the one-species model. |
| D7 | Veres GSE114412 (cache) | Days 0–7 | Continuity with cfm_grn_v2 folds (d4 from d3), one-species | Probably no unspliced counts |
| D8 | Schiebinger GSE122662 | 39 timepoints at 0.5-day spacing | Thinning test (month 9+): fit every 4th day, score true intermediate times | Re-quantification is large; HPC only, with permission |
| D9 (stretch) | RENGE GSE213069, or the Pijuan-Sala Tal1⁻/⁻ chimeras | TF knockout time courses | V6: knockout as h_r := 0 on held-out KOs | Underpowered (RENGE) |

## 6. Evaluation protocol

**Primary metric M1.**
- Held-out-stage predictive log-likelihood per cell (nats/cell), on spliced + unspliced counts, conditional on each cell's observed ℓ_i, over a fixed panel chosen on training stages.
- It is a proper scoring rule that is sensitive to proportions, multimodality and dispersion, not just the centroid.
- Reported as ΔLL = AKP − the best trivial control. A per-gene-normalised ΔLL is reported too, so highly expressed genes cannot dominate.

**Secondary metrics.**
- **M2** JS divergence of composition, using a classifier trained on training stages and applied to real and generated cells.
- **M3** Shape-only ED (mean removed) in training-stage PCA-50, plus the mean/trace split.
- **M4** Detection-rate and Fano calibration, and randomised-PIT uniformity.
- **M5** Raw ED in gene space, for continuity only; never decisive.
- **M6** C2ST AUC.

**Controls.** All are count models with a likelihood, fitted on training stages only.

| Control | What it is | What it tests |
|---|---|---|
| C0 | Persistence: an NB mixture (10–30 components) of the nearest stage | Trivial floor |
| C1 | Mixture interpolation, w·p_k + (1−w)·p_{k+1} | The distributional mean_interp |
| C2 | **Shared NB-mixture components with logit-spline proportions in T** | The strongest composition control |
| C3 | Chronocell with a uniform prior, neighbour posteriors interpolated | Value of real-time anchoring |
| C4 | Shuffled-time twin | That real time is used |
| C5 | Kinetics without programs (piecewise-constant α) | Value of programs |
| C6 | Degree- and sign-preserving random prior | Value of the named support (key interpretability control) |
| C7 | Free per-gene α_g(a) with matched parameters | Program parsimony |
| C8 | Time-conditioned NB-VAE (own code) | Reference only; measures the cost of interpretability |
| C9 | OT interpolation and cfm_grn_v2 v0.1.10 | ED only |

The replicate/pool floor gives a "fraction of gap closed".

**Held-out designs.**
- Each interior stage in turn; the primary stages are fixed in the pre-registration.
- Skip-2: two adjacent stages removed.
- Terminal stage (extrapolation verdict, kept separate and labelled).
- sci-fate: 4 h and 8 h held out, 10 h as terminal.
- Schiebinger thinning.

**Simulator recovery thresholds.**
- Age: Spearman ≥ 0.8 and RMSE ≤ 15% of the span.
- γ: median |log10 error| ≤ 0.2 on rate-identified genes.
- Program switch: error ≤ 0.5 of a gap.
- Anchored edges: sign accuracy ≥ 0.8.
- Edges: lower CI of AUPRC under degree-matched negatives within S greater than prevalence + 0.1.
- SimD must show a flat profile over c.

**External validation (never used in fitting).**
- **V1** γ against labelling γ: Spearman ≥ 0.5 and median |log2 ratio| ≤ 1. It must beat both C3 rescaled by the span and steady-state γ.
- **V2** Onset of known-input programs within 1 h (sci-fate) or 15 min (scNT-seq), and sharper than under C6.
- **V3** Correlation of h_r with chromVAR higher than for C6 programs in ≥ 60% of TFs with a motif.
- **V4** Lagged cross-correlation between h_r and the TF's own spliced mRNA (the TF's own gene excluded from its targets) at lag ≥ 0, higher than under C6.
- **V5** LARRY fate AUROC above the shuffled and composition-only baselines; sign agreement of growth with clone expansion.
- **V6** Knockout ΔLL above the "no effect" and "mean shift" controls.

**Pre-registered rules (SHA-frozen, applied by code).**
- **R1 (G)** The lower 95% CI of ΔLL against max(C0–C2) is > 0 on ≥ 2/3 of interior folds (bootstrap over cells × ≥ 3 seeds), and M2 is no more than JS + 0.02 worse than C2.
- **R2 (C)** AKP beats both C3 and C4 with CIs excluding 0 on ≥ 2/3 of folds.
- **R3 (I-theory)** The simulator thresholds hold and SimD fails as predicted.
- **R4 (I-external)** V1 and V2 hold on at least one labelling dataset.
- **R5** If AKP does not beat C6 (on ΔLL or on V3/V4), programs are reported as unnamed factors only.

**Kill criteria.**
- **K1 (month 3)** SimA age Spearman < 0.6 or γ error > 0.5 log10. Go to fallback F1.
- **K2 (month 5)** On erythroid, ΔLL < 0 against C2 on every interior fold **and** AKP ≤ C3. Drop all real-data G claims.
- **K3** V1 Spearman < 0.3 on both labelling sets. Drop the "rates in hours" claim.
- **K4** AP < 0.5 on a dataset. No claims on that dataset.

## 7. DO / DON'T, mapped to past projects

**DO**
- Model raw counts with an NB likelihood; never z-scores (cfm_grn_v2 F9; the RA-FM critique of decay acting on standardised values).
- Use a proper scoring rule as the primary metric; ED is secondary (Job E: the mean term is 10–100× the trace term).
- Make every control generative and include the mixture analogue of mean_interp (F3).
- Run ceiling diagnostics before believing a component (F1/F5).
- Use whole-timepoint splits, pre-registration, SHA freeze and the reproducibility gate (cfm_grn_v2 keep list).
- Guard every stage that produces reported numbers (F7).
- Keep two clocks, T and a with da/dT = 1; never use pseudotime bins (tgl anti-pattern 6).
- Use degree-matched negatives and the random signed prior (the tgl AUPRC 0.94 artefact).
- Use a misspecified, branching simulator whose snapshots are not translates (F11).
- Change the data regime first: unspliced counts, labelling, ATAC, barcodes (cfm_grn v1: signed prior and velocity-like data moved results; architecture never did).
- Fill in PURPOSE.md and keep one canonical repo (tgl 11/13). Run heavy jobs on the HPC (tgl 14).

**DON'T**
- Build paths that are linear in snapshot data (F2).
- Use a black-box autonomous MLP drift as the core (F6, interpretability 0/5).
- Push forward source cells deterministically.
- Work from aggregate per-bin GRNs (tgl).
- Use learned masks without a recovery check (cfm_grn v1, edge AUC 0.54).
- Pick fixes after looking at the test window (EB v0.1.10), or churn versions on 2 folds (F8).
- Use a uniform process-time prior, or treat latent time as noisy capture time (Chronocell trap; VeloVAE's `tprior`).
- Select models by AIC on a Poisson lower bound (Chronocell's PBMC failure).
- Claim extrapolation as identified (F10, L7).
- Compare against published numbers without a calibration gate (Job C).
- Read seed-only CIs as biological generalisation.
- Clamp a knockout to 0 in standardised space (idea I14).

## 8. How the two papers are used

**Chronocell: core component, re-implemented from equations (it is BSD-2, but the project rule is original code).**
- **Used:**
  - the closed-form two-species kinetics, depth estimation, the AP stability diagnostic and the trajectory-vs-cluster test;
  - its identifiability argument, generalised: the fixed uniform p(z) is replaced by a population-derived p(a|T) (P1);
  - baseline C3 and a head-to-head on its own erythroid dataset;
  - its γ-vs-labelling validation, upgraded to within-experiment labels (sci-fate, scNT-seq).
- **Dropped:** the uniform prior, Poisson-only noise, AIC on the lower bound, and synchronous all-gene switching (replaced by programs).

**Badalotti et al.: zero weight in the success criteria.**
- **Not used as a method:** binary units, no time semantics, discriminative, and its Hebbian rule learns co-expression.
- **Used in three bounded ways:**
  1. A framing: the persistence control is their J_D → ∞ limit, where every state is stable.
  2. An optional extension in months 10–12, only if V4 holds. Close the loop, `h_r(a) = σ(w_r · TFmRNA_r(a−δ) + a_r)`, then run their diagnostics: a self-activation commitment index `a_r / sd(input)` per lineage TF, a relaxation census, robustness, and a hysteresis **prediction** for transient knockouts. No pulse data are planned, so this stays prediction-only.
  3. A Bocconi conversation (Zecchina, Mézard) about counting stable states in sparse, sign-constrained kinetic networks.

## 9. Novelty against the closest competitors

| Method | What it has | What it lacks, relative to AKP |
|---|---|---|
| **pseudodynamics** (Fischer et al., Nat Biotech 2019) | Closest on the population layer: a PDE over real time × pseudotime, with proliferation and death, on T-cell and β-cell time courses | A precomputed DPT coordinate with a free drift and diffusion; no count kinetics, no rates in hours, no programs |
| **VeloVAE** (Gu…Welch; eLife 2026 per search result) | Count-level, multi-lineage, uses an informative per-cell capture-time prior (`tprior`) | Treats capture time as a noisy measure of latent time, which conflates age with real time and suppresses asynchrony; no influx, growth or population; no time-unit identifiability; no named programs |
| Chronocell | Process time | Not real time, no population, genes independent |
| cell2fate | Modules with switch times | Latent time; no population, no named regulons |
| RegVelo | GRN-gated rates, CRISPR validation | Latent time; per-cell VAE, no population |
| Dynamo | Absolute rates via labelling | Needs labelling; deterministic; no population |
| PFM (Maddu/Chardès/Shelley, arXiv 2604.25062) | Biophysical regulation models via probability flow matching; growth (per dossier; the details are not verified here) | **Highest scoop risk.** Expression-space flow, no unspliced kinetic decoder with age-structured influx |
| StructureFlow, Cell-MNN | Structured drift | Balanced, continuous representation, no counts |
| DeepRUOT / RUOT / unbalanced SB / TIGON | Growth | Growth chosen by a penalty (L8); MLP drift |
| JKOnet\* | Interpretable energies | Gradient class only |
| mvOU-SB | Linear OU reference | Linear Gaussian |
| GRNvelo, Navigo | — | Not found under those names (arXiv 2608.09722 is a possible match for GRNvelo); unverified |

**What is genuinely new:**
1. A joint likelihood of count kinetics and an age-structured influx/growth population across real snapshots.
2. P1–P3, including a **predicted failure regime** (exponential influx) confirmed on a different-family simulator.
3. Rates in 1/h from unlabelled snapshot time courses, validated against metabolic labelling in the same cells.
4. Named regulon activities over age, validated against held-out chromatin and TF self-consistency.
5. A count-level, proper-scoring benchmark with generative trivial controls.

Each component exists somewhere. The defensible core is items 2 and 3.

## 10. Plan, compute, risks and odds

| Months | Work | Gate |
|---|---|---|
| 0–0.5 | PURPOSE.md; pre-registration skeleton; data cards; download requests (sci-fate, scVelo erythroid, scNT-seq processed matrices; all small) | — |
| 0.5–1.5 | Decoder, population layer and grid likelihood; unit tests (closed form vs numerical ODE); SimE sanity | — |
| 1.5–3 | SimA–D battery; profile likelihood over c | **G1 / K1** |
| 3–4 | sci-fate and scNT-seq: V1, V2 | **G2 / K3** |
| 4–5.5 | Erythroid: freeze the pre-registration, then score folds; R1, R2; Chronocell head-to-head | **G3 / K2** |
| 5.5–7 | Argelaguet V3/V4; pancreas replication | — |
| 7–9 | LARRY V5 and growth; first preprint (theory + simulators + labelling + erythroid) | — |
| 9–12 | Schiebinger thinning; knockout stretch; optional closed-loop diagnostics; write-up | — |

**Compute.**
- Simulator battery: about 4k short fits, roughly 150 A100-hours.
- Real data: about 6 datasets × 8 folds × 6 model arms × 5 seeds. The mixture controls run on CPU. Roughly 400–700 A100-hours.
- Under the QoS (1 running job, no arrays), fits are packed 20–50 per job in about 24-hour jobs, which is about 30–40 GPU-days over the year.
- Optional re-quantification (LARRY, Schiebinger, pancreas): 200–500 core-hours and 1–3 TB of FASTQ. Needs permission.

**Risks and fallbacks.**

| Risk | Response |
|---|---|
| A1 fails (embryo waves, staging noise) | Nested test with one cohort-modulation parameter per program, h_r(a) + ε_r(T_e − T̄); otherwise retreat to known-clock datasets |
| Unspliced bias | One-species model plus the labelling datasets |
| Erythroid near-synchronous | Report ratios only, as P1 dictates |
| Collinear programs | Report blocks |
| C2 too strong | Honest report; the claim shifts to I and C |
| Batch × time | Pooled sci-fate; replicate floors elsewhere |
| Scoop (PFM, VeloVAE extensions) | Lead with P1 plus the labelling validation |

**Fallback F1.** "Kinetic regulon programs with real-time clocks in synchronised perturbation time courses, plus the identifiability boundary (P1) for when snapshot time courses can or cannot give rates in hours." It is positive and honest without the asynchronous win.

**Honest probabilities.**

| Outcome | Probability |
|---|---|
| P1–P3 plus the simulator results (including the predicted SimD failure) | ≈ 0.8 |
| V1 passes on ≥ 1 labelling dataset | ≈ 0.5 |
| V2 passes | ≈ 0.7 |
| R2 (beats Chronocell-uniform and shuffled time) | ≈ 0.55 |
| R1 (beats C2 on erythroid interior folds) | ≈ 0.35–0.4 |
| V3 (chromatin) | ≈ 0.3 |
| **At least one presentable positive by month 9** (theory + simulators + one external validation) | **≈ 0.65–0.7** |
| Full G + C + I on one asynchronous dataset | ≈ 0.25 |

Sources:
- [VeloVAE, bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.08.499381.full.pdf)
- [VeloVAE, eLife](https://elifesciences.org/articles/108950)
- [VeloVAE `tprior` parameter (regvelo API docs)](https://regvelo.readthedocs.io/en/latest/api/genapi/regvelo.VELOVAE.html)
- [pseudodynamics (Fischer et al. 2019)](https://cris.fau.de/publications/290862181)
- [sci-fate (Cao et al. 2020)](https://krishna.gs.washington.edu/documents/cao_naturebio_2020.pdf)
- [scNT-seq (Qiu et al. 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8103797)
- [scVelo gastrulation_erythroid](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.gastrulation_erythroid.html)
- [GSE132188 (Bastidas-Ponce)](https://omicsdi.org/dataset/geo/GSE132188)
- [CellRank pancreas dataset](https://cellrank.readthedocs.io/en/latest/api/_autosummary/datasets/cellrank.datasets.pancreas.html)
- [MouseGastrulationData / GSE205117](https://bioconductor.statistik.tu-dortmund.de/packages/3.21/data/experiment/vignettes/MouseGastrulationData/inst/doc/MouseGastrulationData.html)
- [PFM, arXiv 2604.25062](https://arxiv.org/pdf/2604.25062)
- [Miles 2025 review, arXiv 2505.11460](https://arxiv.org/pdf/2505.11460)
- [Reddy, arXiv 2409.09548](https://arxiv.org/pdf/2409.09548)
- [arXiv 2608.09722](https://arxiv.org/pdf/2608.09722)
- [Chronocell](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`

# THEORY CRITIC
# Adversarial review of the angle A_kinetic_clock design ("Kinetic Clock First", the AKP model)

## (1) Verdict

**Scientific soundness: 5/10 as written. It would be about 7.5/10 if F1–F3 are fixed.**

This is the best-structured design in the dossier. It is a restricted, count-level model with two clocks. It uses proper scoring and generative controls, and its "assumption share" device is honest.

Three things fail as written:
- Its central identifiability result (P1, the time unit) is derived incorrectly. As implemented, the time unit would be set mainly by the γ prior, the spline penalty and a closed-reservoir assumption, not by the data.
- Its primary metric does not escape the mean-interpolation trap. It swaps it for a mixture-interpolation trap plus a contest between decoders.
- Its headline claim ("rates in 1/h from asynchronous, unlabelled snapshots") has no external test anywhere in the plan.

All three can be fixed without changing the model.

---

## (2) FATAL flaws (as written)

### F1. P1 is mis-derived. The time unit c would be identified by assumptions, the prior and the penalty, not by the data.

**The algebra is wrong.** The alternative model needs its own influx ρ′, which the design's equation does not allow for. The correct condition, for every observed T, is:

`log ρ′(T−a′) + log G′(a′) = log ρ(T−ca′) + log G(ca′) + ψ(T) + log c`

Taking ∂_T∂_{a′} gives `(log ρ′)″(u) = c·(log ρ)″(u + (1−c)a′)` for all u and a′. So for c ≠ 1, `(log ρ)″ ≡ k` (a constant) on the window of entry times.

- **If growth may vary with age, non-identification holds iff log ρ is quadratic, not iff it is affine.** The compensating growth is `log G′(a′) = log G(ca′) + k·c(c−1)a′²/2 + b·a′`.
- Exponential influx (k = 0) is only the trivial stationary case: the snapshots look identical.
- A **Gaussian influx pulse (k < 0) changes every snapshot yet leaves c unidentified.** If SimA's "pulse" is Gaussian-shaped, the main simulator sits inside the non-identified family. SimD, which tests only exponential influx, cannot detect this.

**Piecewise-constant growth r_l only hides the problem.** Excluding k ≠ 0 rests on age-linear growth being outside the model class. The identification is then borrowed from how growth is parametrised, and its strength scales roughly with |k|·(segment length)².

**The closed reservoir is what actually does the work.** Setting a′ = 0 gives κ′ = cκ. Then `ψ′(T) = Δr_R − (c−1)κ(T)`. So with ρ = κR, dR/dT = (r_R − κ)R and an observed fraction R/N, only the case "κ constant and k = 0" stays non-identified. But this rests on two weak points:
- In the erythroid subset the reservoir is not closed. Blood progenitors are fed from haemato-endothelial progenitors outside the subset (to verify against the subset's labels). With external inflow into R, the quadratic non-identification returns.
- Lineage cells at age a → 0 have exactly the reservoir's kernel. So R/N is observable only to within the fastest relaxation time.

**With discrete snapshots, K = 2 is never identified.** For any ρ, the equation `h(s+Δ) − h(s) = q(s)` can always be solved, and a free shift of r_R absorbs the reservoir term. With K ≥ 3, identification comes only from a consistency condition across the difference equations. LARRY has K = 3.

**The penalty pulls toward non-identification.** A second-difference penalty on the coefficients of log κ has affine log κ as its null space. That is the non-identified set, so the estimator is shrunk toward it.

**The "weak" γ prior is a near-delta on c.** Under global rescaling, every log γ_g shifts by log c. With prior N(log γ | log(ln2/8h), 1.5²) on G genes, the induced prior on log c has sd = 1.5/√G:
- 0.087 for 300 genes;
- **0.047 for 1,000 genes, which pins the time unit to about ±5%.**

The profile over c will therefore be the prior. V1's criterion "median |log2 ratio| ≤ 1" is passed by the prior alone wherever median half-lives are 4–16 h, which is typical for mammalian cell lines (that typical range is from memory, not checked this run).

**Fixes:**
- (a) Re-derive and state P1 for an explicit model class: age-flexible growth, piecewise growth, closed reservoir and open reservoir. Make the boundary result (a log-quadratic influx is non-identified) part of the contribution.
- (b) Compute the c-profile with the γ prior removed and with λ_smooth → 0.
- (c) Make the γ prior scale-free: a prior on log γ_g − mean_g(log γ_g) only.
- (d) Report how much of the information on c survives three ablations: an open reservoir with free inflow, age-flexible growth, and no prior. Claim hours only if c moves by less than a factor 1.25.
- (e) Add a prior-only arm to V1, with every gene at the prior centre.
- (f) Centre SimA's γ distribution away from 8 h, for example a 2 h or 20 h median.
- (g) Add SimD2: a Gaussian influx with age-linear growth and an open reservoir. It must give a flat c-profile.

### F2. M1 does not escape the trap. It replaces it with a mixture-interpolation trap plus a decoder contest.

**Age-space mixture interpolation is near-optimal under broad asynchrony.** Suppose that between snapshots p(a|T) mostly translates, with age spread σ_a. At the midpoint, the truth is N(0, σ_a²) and the interpolant is ½N(−Δ/2, σ_a²) + ½N(Δ/2, σ_a²). Then `KL ≈ d⁴/4` with d = Δ/(2σ_a). By the data-processing inequality, this bounds the possible gain in x-space.

| Setting | Gap Δ | σ_a | Maximum clock gain |
|---|---|---|---|
| Erythroid, skip-1 | 6 h | 6–12 h | about 0.001–0.016 nats/cell |
| Erythroid, skip-2 | 12 h | 6–12 h | about 0.016–0.25 nats/cell |

A perfect clock is therefore nearly invisible on interior skip-1 holdouts. This is the asynchrony paradox: the population layer is needed exactly where its predictive benefit cannot be seen.

**The rules end up testing the decoder, not the clock.**
- **R1 (vs max(C0–C2)) is a decoder contest.** Differences between per-gene NB fits over 300–1,000 genes are orders of magnitude larger than 0.01 nats/cell. C2, with 10–30 free-mean components, will likely beat a log-linear decoder built from programs on the CollecTRI support, whatever the dynamics.
- **R2 (vs C3) is passed for the wrong reason.** NB beats Chronocell's Poisson likelihood regardless of real-time anchoring.
- **The K2 conjunction never fires.** AKP will always beat a Poisson C3.
- PFM (Maddu, Chardès & Shelley; arXiv 2604.25062) itself reports that models with similar interpolation accuracy can encode different dynamics.

**Fix: a decoder-matched ladder.** Every arm uses the AKP decoder:
1. AKP, full.
2. AKP with a uniform age prior per snapshot (the honest C3).
3. AKP with free q(a|T_k) per snapshot, and q mixture-interpolated at T* (the age analogue of mean interpolation).
4. Nearest-snapshot q (persistence).
5. Shuffled T.
6. An oracle: free q(a|T*) fitted on the held-out stage.

Define `G_clock = [LL(1) − LL(3)] / [LL(6) − LL(3)]`, with pool-level bootstrap CIs.

Further changes:
- Add a dating metric: estimate the held-out stage's T̂ from its cells, report |T̂ − T| in hours, and compare it across the arms.
- Pre-register a power analysis on SimA at the dataset's Δ/σ_a.
- Make skip-2 and thinning designs primary for clock claims, with skip-1 reported only.
- Report AKP vs C2 as the "cost of interpretability", not as a pass/fail for the clock.

### F3. The headline novelty has no validation path.

**V1, V2 and R4 run only on synchronised datasets.** In sci-fate and scNT-seq, ρ = δ at a known T_0, so the clock is given. Rates in 1/h from known-clock total/intronic time courses are essentially single-cell INSPEcT− (Furlan et al., Genome Res 2020). The asynchronous inference is never tested externally.

**V2 is close to a coin flip.** With T_0 supplied and snapshots at 0 h and 2 h, "onset within 1 h" passes about half the time for an uninformative model.

**Fixes:**
- **(a) Asynchronous labelled time course.** Use the sci-FATE2 ES → neural-tube data (Maizels, Snell, Briscoe; about 45k cells; GEO superseries around GSE236512, with the timepoints and pooling still to verify). Fit AKP with free κ on total (or u/s) counts only, and validate γ against the labelling-derived γ. This is the only plan element that can externally validate the novelty, and it is also a competitor (VelvetSDE).
- **(b) Free-κ positive control on sci-fate.** Fit with free κ and T_0. The model must recover influx concentrated at T = 0 and onset at 0 h within a pre-registered CI.

---

## (3) SERIOUS flaws, with concrete fixes

**S1. The simulators share the identifying assumptions.**
- SimA, dyngen and scMultiSim (whose "time-varying" GRN changes along each cell's own trajectory, not with real time) all satisfy A1.
- SimA also has homogeneous progression speed, a closed stationary reservoir, an exactly known T and small batch effects. Recovery is "different family" only for the decoder, while the clock is circular.
- **Fix:** add violation dials to SimA and report breakdown curves for c, γ and κ:
  - heterochrony: cohort speed s(T_e) = 1 + η(T_e − T̄);
  - per-cell speed CV of 0.2–0.4;
  - an open reservoir;
  - reservoir state that drifts with T;
  - staging noise σ_T ∈ {0, 0.5Δ, Δ};
  - per-stage batch sd ∈ {0.1, 0.3, 0.5}, including a batch effect on the u/s ratio;
  - γ that changes with age (erythroid maturation changes mRNA turnover).

**S2. CIs from a cell-level bootstrap are invalid for population parameters.** κ, c, growth and π are informed by about K stage-level age distributions. **Fix:** resample pools or embryos within each stage, or fit per-sample random effects. Use a stage jackknife for κ.

**S3. Embryo staging noise.** In embryos the stage label is not exact real time. Mittnenzweig et al. (Cell 2021, 153 embryos over 36 h) show embryo-level timing must be inferred. Staging noise convolves ρ and biases its curvature k, and therefore c. **Fix:** model T_true ~ N(T_label, σ_T²) per sample and profile σ_T. Prefer in-vitro clocks for clock claims.

**S4. Batch × time has only one sink: the age distribution.** Per-stage shifts go into p(a|T), which biases κ and c. The cohort test cannot tell batch effects from A1 violations. **Fix:**
- run a negative control: a non-differentiating population sampled over time (for example sci-fate's T = 0 or vehicle cells), which must give a stationary q(a);
- add per-stage global capture offsets;
- restrict the clock-bearing panel to genes with low batch sensitivity in replicate pools.

**S5. Programs are less identified than claimed.**
- If the 2–3 nuisance programs can load on named supports, they reopen the L16 rotation: h_r → h_r + λh_n, W_n → W_n − λW_r.
- The key erythroid regulators (Gata1/Tal1/Klf1, a co-binding complex) will fail A5 and collapse into blocks.
- **Fix:**
  - restrict nuisance programs to genes outside every named support (or constrain them to be orthogonal to the named programs);
  - pre-register the TF selection rule, for example: expressed in training stages with ≥ 3 anchors in the panel;
  - count anchors before fitting.

**S6. Gene–gene covariance within (l, a) is read as age spread.** Cell cycle and cell size inflate the age dispersion and bias κ. **Fix:** add a cycle-phase nuisance latent or offsets, exclude cycle genes from the clock panel, add a residual-covariance diagnostic, and add a SimA dial.

**S7. V5 (LARRY) is unwinnable by construction.**
- On the shared trunk, p(l | trunk cell) = π_l(T_e), which is exactly the composition baseline it must beat.
- With K = 3, P1 is barely identifiable.
- **Fix:** keep only the sign test of growth differentials against clone expansion, or add a heritable fate-bias latent and say so.

**S8. Counterfactuals are open-loop.** h_r is an exogenous function of age, so setting h_r := 0 changes only direct targets. Nothing propagates, and neither fates nor composition respond. The Tal1−/− phenotype (no blood; from memory) cannot be produced. **Fix:** drop V6 until the closed-loop extension exists, and label counterfactuals "direct-target, open-loop".

**S9. The kill criteria have no teeth.** K2 is a conjunction against a Poisson C3. **Fix:** kill clock claims if the CI of G_clock includes 0 on every skip-2 fold. Kill "hours" if c is off by more than 1.5× on SimA at realistic violation levels, or if ≥ 80% of the information on c disappears under the reservoir/prior ablations.

**S10. P2's second equivalence is not exact.** Log-linear π_l(T_e) depends on T − a, while growth depends on a. They are confounded only within a single snapshot; across snapshots the parameter is weakly identified through age resolution. **Fix:** state it as weak identification and report a profile likelihood.

**S11. Hyperparameter selection repeats F8.** Topology, λ_TV, the number of nuisance programs and A_max are chosen on 1–2 inner folds. **Fix:** fix λ_TV and A_max by calibration on SimA, and select only the topology.

**S12. V3 and V4 are confounded by shared progression.** Any two monotone functions of age correlate. **Fix:** use partial correlations given an age × lineage spline, plus direction-of-change tests at branch points. Use a C6 null with matched temporal smoothness.

**S13. The datasets cannot carry clock claims as ordered.**
- **Pancreas (D4):** FACS sorting multiplies p(a|T) by an unknown acceptance s(a), which absorbs c. Use it for decoder replication only.
- **Erythroid:** it combines staging noise, an open reservoir, per-stage pools and flat posteriors before E7.5. Demote it to a secondary dataset (keep the head-to-head with Chronocell).
- **Promote instead:**
  - Schiebinger thinning with the one-species model on processed counts. The clock is exact, Δ_eff is large and no re-quantification is needed (download permission still applies). Move it to months 4–6.
  - The asynchronous labelled sci-FATE2 time course.

**S14. Numerics and smaller inconsistencies.**
- The μ_s update is singular at γ = β; use the limit βû₀·δ·e^{−γδ}.
- The profile over c needs grid resolution that rescales with c.
- The "stochastic dynamical model" is deterministic given (l, a). Entry-time asynchrony and speed heterogeneity are only weakly separable (state this as assumption A1b).

**S15. Scoop and novelty.** Several competitors overlap:
- VelvetVAE/VelvetSDE: labelling plus a neural SDE in real time;
- Balik & Lähdesmäki (ICML 2026, arXiv 2605.20989): per-cell latent time disentangling asynchrony across real times;
- Mittnenzweig 2021: embryo timing plus flows;
- INSPEcT−: rates without labelling;
- pseudodynamics.

Narrow the claimed novelty to: (i) the corrected identifiability boundary of the time unit for age-structured kinetic mixtures, and (ii) asynchronous, labelling-validated rates in hours.

---

## (4) Claims to downgrade to "assumption"

| Claim | Downgrade to |
|---|---|
| "Time unit identified iff ρ not exponential" | Identified only under a closed observed reservoir or restricted growth with a non-quadratic log ρ, K ≥ 3 and no staging noise. Otherwise it is set by assumption. |
| "L1 gauge removed by construction" | Removed within the class only. Membership of the class (unit speed, A1) cannot be tested from marginals. |
| "Between snapshots most cells are cohorts that simply age" | This is A1 plus a closed population, not an observation. |
| Cohort permutation test "checks A1" | It detects only T-specific deviations off the manifold, mostly batch effects. Heterochrony along the manifold is invisible to it. |
| Rates in 1/h on asynchronous data | Assumption- and prior-driven until validated on asynchronous labelled data. On synchronised data the clock is given. |
| Growth | Differentials only. Gillespie genealogies and clone-size distributions depend on birth and death separately, so they are a simulation under assumption. |
| Counterfactual KOs | Open-loop, direct-target predictions only. |
| Named programs | "Effective regulon activity given that the CollecTRI support is correct". Co-binding TFs are blocks; edges outside S are invisible. |
| Switch times within gaps | Intervals that depend on the TV penalty. |
| Calibrated spread | A model property to be checked with M4 and replicate floors. |
| sci-fate pooling reduces batch × time | Still to verify. |
| Chronocell's erythroid set equals the scVelo subset | Still to verify. |

---

## (5) Genuinely strong; keep

- **Likelihood:** NB on raw u/s, with depth ℓ_i from near-Poissonian genes and the intron-length bias ε_g. No z-scores, so no spike genes. The closed-form kinetics are verified correct (apart from the γ = β guard). Exact grid marginalisation keeps the model small, at about 20k parameters.
- **Time semantics:** two clocks, with unit-speed age and an explicit McKendrick–von Foerster population layer whose assumptions are declared. Real time enters only through κ and π. This is the right place to put all assumptions about the clock.
- **Honesty devices:**
  - the assumption share;
  - the extrapolation envelope (E1–E3);
  - the ratio-only fallback when the c-profile is wider than 2×;
  - per-gene "rate-identified vs ratio-only" tags;
  - reporting blocks of co-active programs.
- **Predicted failure regimes as evidence.** Keep the idea and generalise it with SimD2 and the violation dials.
- **Controls:** generative controls with a likelihood, including C1 (the mixture analogue of mean interpolation), C2, C6 (degree- and sign-preserving random prior) and C7 (free α with matched parameters). Promote the free-age ceiling into the primary decomposition (G_clock).
- **External validation:** CollecTRI (literature-curated) against chromVAR is not circular; labelling in the same cells is the right kind of external evidence.
- **Protocol:** pre-registration, guarded refit, whole-stage splits, and a positive fallback (F1).
- **The Badalotti paper** is correctly limited to framing and optional diagnostics.

**Revised core rules (suggested):**
- **R-ID:** CI for c narrower than 2×, computed with no γ prior, λ_smooth → 0 and a pool bootstrap; c must move by less than 1.25× under the open-reservoir and flexible-growth ablations.
- **R-C:** lower CI of G_clock > 0 on ≥ 2/3 of skip-2 and thinning folds, with power verified on SimA at matched Δ/σ_a.
- **R-G:**
  - M4 calibration within replicate floors;
  - M2 no more than 0.02 worse than C2;
  - the LL gap to C2 reported as the cost of interpretability.
- **R-I-ext (asynchronous V1):** on sci-FATE2, γ Spearman ≥ 0.5, and median |log2 ratio| better than both the prior-only arm and the uniform-age AKP arm.

---

**Sources**
- [Maizels, Snell, Briscoe: Velvet / sci-FATE2 (bioRxiv 2023.07.06.547989)](https://www.biorxiv.org/content/10.1101/2023.07.06.547989.full.pdf)
- [Velvet dataset record, GSE236512](https://www.omicsdi.org/dataset/geo/GSE236512)
- [Velvet, Crick publication page](https://www.crick.ac.uk/research/publications/reconstructing-developmental-trajectories-using-latent-dynamical-systems-and-time-resolved-transcriptomics)
- [sci-fate (Cao et al., Nat Biotechnol 2020)](https://krishna.gs.washington.edu/documents/cao_naturebio_2020.pdf)
- [sci-fate preprint (bioRxiv 666081)](https://www.biorxiv.org/content/10.1101/666081.full.pdf)
- [INSPEcT− (Furlan et al., Genome Res 2020)](https://genome.cshlp.org/content/genome/30/10/1492.full.pdf)
- [Balik & Lähdesmäki, arXiv 2605.20989 (ICML 2026)](https://arxiv.org/abs/2605.20989)
- [PFM (Maddu, Chardès, Shelley, arXiv 2604.25062)](https://arxiv.org/abs/2604.25062)
- [Mittnenzweig et al., Cell 2021](https://pmc.ncbi.nlm.nih.gov/articles/pmid/33932341)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`

# FEASIBILITY CRITIC
# Adversarial review of A_kinetic_clock (AKP), lens: feasibility, data, novelty, payoff

## (1) Scores
- **Feasibility: 3/10 as written, about 6.5/10 after the cuts in §6.** Most of the problem is scope. The rest is that the headline real-data claim has no dataset on which it is both identifiable and externally checkable.
- **Novelty: 5/10 as written, about 6/10 if reframed around an identifiability boundary plus the bias of existing time priors.** The pieces already exist elsewhere:
  - Chronocell: the kinetic decoder.
  - VeloVAE (PLoS CB 2026): capture-time-anchored kinetic rates in absolute units.
  - pseudodynamics, scPD and Pseudodynamics+: the age/real-time population layer.
  - RegVelo: GRN-gated kinetics.
  - What is new is the conjunction. I found no McKendrick or age-structured count-kinetic single-cell model, but that was a narrow search (no Google Scholar).

## (2) FATAL problems (as written)

**FATAL-1. P1 is stated wrongly, and the design's own regularisers push it into the non-identifiable regime.**
- Write f = log ρ and g = log G. Rescaling invariance needs f̃(T−a′) + g̃(a′) = log h(T) + f(T−ca′) + g(ca′). Taking ∂_T∂_a′ and setting u = T−a′ gives f̃″(u) = c·f″(u−(c−1)a′) for all (u, a′).
- So the condition is **f″ constant**, not f″ ≡ 0.
- Counter-case: take f = κ₂u²/2 + κ₁u (a log-quadratic, Gaussian-shaped influx wave). Then f̃ = cκ₂u²/2 + …, and the residual is absorbed by growth r̃(a′) = c·r(ca′) + c(c−1)κ₂a′ + const. The time unit is then not identified.
- Corrected statement: c is identified iff log ρ is non-quadratic on [T₁−A_max, T_K], or a hard onset T₀ is observed (the age front T−T₀), given growth flexible in age. With constant per-segment growth this is only weak identification. In practice the identifying signal is about a third derivative of a latent influx, estimated from 5–7 batch-confounded snapshots.
- The second-difference penalty on log κ shrinks log κ toward affine (exponential influx), which is exactly the non-identifiable set. Any smoothness prior shrinks toward low-order polynomials, which are all non-identifiable here.
- **Prediction:** the profile over c will be flat or penalty-driven on erythroid. Under the design's own rule, the expected outcome is "ratios only".

**FATAL-2. The γ prior makes "rates in 1/h" prior-driven, and V1 cannot tell the difference.**
- The prior log γ ~ N(log(ln2/8h), 1.5²) acts across 300–1,000 genes. When the likelihood is flat in c, this prior pins the global time scale to a median half-life of about 8 h.
- V1's Spearman is invariant to c. Only "median |log2 ratio| ≤ 1" tests the unit, and the prior delivers that almost for free.
- **Fix:**
  - Make the prior gene-relative only, i.e. a prior on log γ_g − mean_g log γ.
  - Report the posterior of c with that prior and without it.
  - Add a "c from prior alone" control.

**FATAL-3. The external validation never touches the novel regime.**
- V1 and V2 run on sci-fate (dexamethasone) and scNT-seq (KCl). Both are synchronous: a = T by design, so P1 holds trivially there.
- Kinetics from known-clock labelled time courses is already done (dynamo; Storm, PLoS CB 2024, which uses sci-fate). Passing V1 there says nothing about getting hours from asynchronous snapshots.
- V2 ("program switches on at T=0") cannot fail. Any step-input model recovers it, C5 included.
- **Fix:** add an asynchronous, labelled time course. Candidates:
  - **sci-FATE2 neural differentiation** (Maizels, Snell, Briscoe, Cell Systems 2024, doi 10.1016/j.cels.2024.04.004; about 45k mESC→neural tube; GEO GSE236512/GSE236517/GSE236518). Verified to exist; the timepoint design is not verified.
  - **dynamo scNT-seq human haematopoiesis** (PRJNA796521; probably only 2 collection times, unverified).

**FATAL-4. The primary asynchronous dataset breaks A1 for known biological reasons.**
- Erythroid maturation has MURK genes with a coordinated step change ("boost") in transcription. This made velocity run backwards (Barile…Göttgens, Genome Biol 2021, verified).
- If the boost is tied to embryo stage (T) rather than to age, A1 fails on D3.
- Embryo staging noise is about the 0.25-day grid, yet T_i is treated as exact.
- Chronocell's posteriors are flat until E7.5, so only about 4 stages carry age information.

## (3) SERIOUS problems, with fixes

1. **Scope is about 3× too large.** The plan has 9 datasets, 10 controls, 6 validations, 5 simulators, a Chronocell re-implementation, an NB-VAE and 3 re-quantifications.
   - The "2k lines" estimate is low. SimA (protein layer, delays, bursting, binomial partitioning, toggle fate, capture) alone is 800–1,200 lines. Chronocell from equations is 600–1,000. Loaders and QC run about 300 per dataset. A realistic total is 6–9k lines.
   - Fix: see §6.

2. **Compute is underestimated.** One step is 1,024 × 1,000 × 600 × 2 ≈ 1.2e9 NB terms. Even keeping only the age-dependent part (x·log μ − (x+φ)·log(φ+μ)), the step is memory-bound at about 30–100 ms forward+backward on an A100, not 10 ms.
   - Multiply 20 starts × 20 bootstraps × profile over c (≥9 points × starts) × nested CV × arms. This exceeds the 400–700 A100-hours.
   - Fix:
     - grid of 200 points, panel of 300 genes;
     - 5 random starts plus 1 informed start;
     - profile over c on the primary fold only;
     - bootstrap only the final fit.

3. **Optimisation is multimodal.** Chronocell needed 100 restarts for 24–100 genes and got AP 0.79 on erythroid. AKP adds programs, nuisance programs, a population layer and topology on top. That brings label switching between named and nuisance programs, and between π(T_e) and growth.
   - Fix: fit in stages.
     - First, piecewise α per segment with the age prior. This is C5, essentially Chronocell plus the age prior.
     - Then factorise log α into programs on the CollecTRI support.
     - Then fine-tune jointly. Run the K4 (AP) gate before any program fit.

4. **The anchor condition A5 will mostly fail.** Gata1, Tal1 and Klf1 co-bind and share targets, and CollecTRI signs are often defaults. On a 300–1,000-gene panel, "co-active blocks" will be the typical outcome.
   - Pre-specify the blocks.
   - Require named programs to carry at least a fixed share of ΔLL over C5, and to beat C6.
   - Otherwise the 2–3 nuisance programs absorb the signal and the named programs are decoration.

5. **R1 (superiority over C2) is unlikely: P ≈ 0.2–0.3, not 0.35–0.4.** A 10–30-component NB mixture with spline proportions has far more marginal capacity than a 1–2-lineage kinetic manifold.
   - Fix: pre-register non-inferiority to C2 with a margin (nats/cell), plus a parameter-count ratio. Claim superiority only over C0, C1, C3 and VeloVAE-style priors.

6. **Stage-batch effects corrupt the same signal that identifies c.** Per-stage capture or length-bias shifts move the unspliced/spliced ratios and age distributions between stages, which is exactly where c comes from.
   - Fix:
     - In SimA, sweep the stage-offset size until c is no longer identified, and report that threshold.
     - On erythroid, use the `sequencing.batch` field (present in the scVelo object). Fit δ_{b,g} only where a batch spans ≥2 stages.

7. **The kernel saturates at old ages.** Once genes reach the terminal steady state, older ages cannot be told apart. The old tail of p(a|T), and therefore growth and influx there, comes from the population prior alone.
   - Fix: count "kinetically saturated" cells in the assumption share.

8. **The reservoir assumption fails on D3.** The scVelo erythroid subset has no stationary progenitor pool; mesoderm is excluded.
   - Fix: make R optional and use an initial cohort. Initialise from a non-steady state.

9. **Dataset realities.**
   - sci-fate: about 6,680 cells (verified) at low sci-RNA-seq depth. Intronic reads exist, but splitting spliced/unspliced means re-quantifying from FASTQ (needs permission). Cell-cycle variation dominates A549.
   - LARRY: only days 2/4/6, so one interior fold. Unspliced needs re-quantification.
   - Bastidas-Ponce: 4 FACS-sorted stages; unspliced only at E15.5.
   - Argelaguet: nuclei, so the two-species model is misspecified (no nuclear export), and the user already has a negative there.
   - All four are low value per month.

10. **A better V2 that can fail: predict labelled RNA.**
    - Fit on total counts only. Then predict each cell's 4sU-labelled counts from the fitted α(a), β, γ, e.g. L_g(T) = ∫_{T−2h}^{T} α_g(s)·e^{−γ_g(T−s)} ds, with conversion efficiency estimated separately.
    - Compare with the observed new-RNA counts. This tests the absolute time scale and the kinetics on a modality the fit never saw.

11. **An optional ground-truth age check.** Zman-seq (Kirschenbaum et al., Cell 2024) carries 12/24/36/48 h interval timestamps of time since tissue entry. Monocyte→TAM, 3,108 cells, is loadable via the CellDyc docs.
    - It tests p(a|x) directly against measured age.
    - It has a single harvest time, so it cannot test P1. It is MARS-seq and low depth. Accession not verified.

## (4) Scoop risks
**Verified this run**
- **VeloVAE** (Gu, Song, Blaauw, Welch; PLoS CB, 20 Mar 2026). **HIGH** for novelty item 3.
  - Uses capture times as an informative Gaussian prior and reports time in hours/days.
  - Has a discrete Poisson model and absolute rates, with labelling comparison PCC 0.72 on one embryonic dataset; includes mouse erythroid.
  - Has no population, influx, growth or GRN layer, and no identifiability analysis.
  - Must become a baseline (C3b). It is cheap: a prior swap in your own code.
  - The design's eLife 108950 citation is unverified.
- **PFM** (Maddu, Chardès, Shelley; arXiv 2604.25062). **MEDIUM-HIGH** for the overall pitch.
  - Learns biophysical gene-regulation SDEs from time-resolved scRNA-seq.
  - Handles unbalanced proliferation and death; 3 haematopoiesis datasets.
  - Argues that "similar interpolation accuracy, different dynamics" separates biophysically consistent models. That is close to AKP's framing.
  - It has no count-kinetic decoder and no age structure.
- **Pseudodynamics+** (bioRxiv 2025.11.30.691399, PINN) and **scPD** (Bioinformatics Advances 2026, doi 10.1093/bioadv/vbag188). **MEDIUM:** they cover the population layer in real time × pseudotime with growth.
- **Velvet / sci-FATE2** (Cell Systems 2024). **MEDIUM:** real time plus labelling plus a neural SDE, but black-box. It is also the best validation dataset.
- **Chronocell authors.** No time-anchored follow-up found. But their code already accepts per-cell priors of shape (n, L, m), so a quick follow-up from the Pachter lab is a **HIGH** latent risk.

**Unverified**
- **CellDyc:** uses Zman-seq timestamps (time_weight); authors and model unknown.
- **GRNvelo (MSB 2026):** dossiers conflict and it was not found. arXiv 2608.09722 exists under another title.
- **Navigo:** not found.
- **RegVelo** (Cell 2026): latent time only, so LOW.

## (5) The strongest defensible publishable claim
"For snapshot time courses, a count-level kinetic model with an age-structured (McKendrick–von Foerster) entry prior identifies absolute kinetic time scales **iff the entry flux is non-quadratic in log, or has an observed onset** (corrected P1). On misspecified simulators the profile likelihood is flat exactly in the predicted regime and sharp outside it. Uniform process-time priors (Chronocell) and Gaussian capture-time priors (VeloVAE) are measurably biased under asynchronous entry. The fitted rates predict **held-out 4sU-labelled RNA** and labelling half-lives in a synchronous set (sci-fate) and an asynchronous set (sci-FATE2). Held-out-stage NB log-likelihood is non-inferior to strong mixture controls at far fewer parameters."

- **Venue:** Cell Systems, PLoS CB, RECOMB or ISMB, or an ICML/NeurIPS workshop.
- **What a NeurIPS/ICML main-track reviewer would say:** "the population layer is pseudodynamics, the kinetics are Chronocell, the time prior is VeloVAE, the identifiability argument is a one-line cross-derivative (and wrong as stated), and validation is on synchronous data." To fix that, they would need a finite-snapshot theorem plus scalable amortised inference.
- **Nature Methods** would require biological discovery plus a broad benchmark. That is not realistic in 12 months.

**Fallback that holds up whatever the real data show (P ≈ 0.8):** "When do snapshot time courses give kinetic rates in hours?" Contents: the identifiability boundary, plus how the Chronocell/VeloVAE priors break under asynchrony, demonstrated on simulators. This is stronger than the design's F1, which (dynamo/Storm) has already done.

**Odds overall**
- Presentable positive by month 9 (reframed): about 0.55–0.6.
- A real-data asynchronous rate-in-hours claim: about 0.25–0.3.

**Fit for a Liò application: good, if it is pitched as "identifiable, mechanistic, generative ML".** It matches his interest in explainable and physics-informed ML, and it continues the user's audit record (the Dileo artefact) into a positive method. GNN-parametrised regulatory programs belong in the PhD-extension slide, not the 12-month core.

## (6) Cuts that make it doable
- **Data.** Keep:
  - SimA (simplified: drop protein delays at first) + SimD + dyngen;
  - sci-fate (decoder plus labelled-RNA prediction);
  - **sci-FATE2 (new; the key test)**;
  - erythroid (head-to-head with Chronocell and VeloVAE on Chronocell's gene set, plus a MURK case study).
  - Stretch: Argelaguet V3 and Zman-seq age check.
  - Drop: D4, D6, D8, D9 and scMultiSim. D7 is optional (one-species continuity only).
- **Model.**
  - Core = C5-class kinetics + age prior + growth, with fixed annotated topology and constant π.
  - Programs come in month 5+, only if C5 passes the K-gates.
  - Drop Gillespie genealogies, π(T_e) and the Badalotti closed loop.
- **Controls.** Keep C0, C2, C3, C3b (VeloVAE prior), C4, and C6 once programs exist. Drop C7, C8 and C9 (ED reporting optional).
- **Theory first (weeks 0–3).**
  - Correct P1 and check it symbolically (sympy).
  - Remove the absolute γ prior.
  - Replace the second-difference κ penalty with a penalty whose influence on c is reported.
- **Process.** One frozen primary contrast per dataset; keep the pre-registration light.

Sources:
- [VeloVAE, PLoS CB 2026](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014060&rev=1)
- [PFM, arXiv 2604.25062](https://www.alphaxiv.org/abs/2604.25062)
- [Pseudodynamics+](https://www.biorxiv.org/content/10.64898/2025.11.30.691399.full.pdf)
- [scPD](https://www.citedrive.com/en/discovery/scpd-a-python-package-for-inferring-continuous-population-dynamics-from-single-cell-snapshot-data/)
- [pseudodynamics](https://cris.fau.de/publications/290862181)
- [Maizels et al., Cell Systems 2024](https://www.crick.ac.uk/research/publications/reconstructing-developmental-trajectories-using-latent-dynamical-systems-and-time-resolved-transcriptomics)
- [Velvet / sci-FATE2 bioRxiv](https://www.biorxiv.org/content/10.1101/2023.07.06.547989.full.pdf)
- [GSE236512](https://www.omicsdi.org/dataset/geo/GSE236512)
- [sci-FATE2 protocol](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11472621/)
- [Zman-seq, Cell 2024](https://portal.fis.tum.de/en/publications/time-resolved-single-cell-transcriptomics-defines-immune-trajecto/)
- [CellDyc timestamps](https://celldyc.readthedocs.io/en/latest/handling_timestamps.html)
- [scVelo gastrulation_erythroid](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.gastrulation_erythroid.html)
- [Barile et al. 2021 (MURK)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8258993)
- [sci-fate](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416490)
- [Storm](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11611270/)
- [dynamo scNT-seq haematopoiesis](https://dynamo-release.readthedocs.io/en/latest/tutorials/notebooks/301_tutorial_hsc_velocity.html)
- [Chronocell](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/)
- [arXiv 2608.09722](https://arxiv.org/pdf/2608.09722)
- [RegVelo](https://regvelo.readthedocs.io/en/latest/)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`