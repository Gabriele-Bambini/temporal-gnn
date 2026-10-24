# DESIGN
# Angle B: identifiable stochastic dynamics in an anchored program space ("PALM")

## 1. Thesis

PALM (Program-space Anchored Langevin Model) is a stochastic, time-homogeneous model of a cell population in real time. Each cell's state is a vector Z ∈ R^k (k ≈ 8–16) of log-activities of named gene programs. Each program is fixed by a pre-registered external gene set and has non-negative, masked loadings with anchor genes.

- **Dynamics.** Cells follow a non-reversible gradient Langevin equation, dZ = −(I−J)∇Ψ(Z)dt + σdW, with J skew-symmetric. The population also grows, with a growth rate tied to an anchored cell-cycle program.
- **Output.** A negative-binomial (NB) decoder turns programs into raw UMI counts. An optional kinetic layer adds unspliced/spliced counts with splicing and degradation rates in 1/h.
- **Fitting.** The model is fitted to the snapshot marginals by an unbalanced, multi-marginal Schrödinger-bridge EM whose reference process is the model itself. A second, independent estimator (backpropagation through the SDE) cross-checks it.

**Why this can succeed where earlier work could not:**
- **Against cfm_grn_v2.**
  - Predictions are pushforwards of a fitted parametric SDE, not interpolants linear in the snapshots, so the predicted centroid is not pinned to an affine combination of means (F2/F3).
  - Spread is a fitted quantity (σ and NB dispersion), not spline geometry (F4).
  - Noise in a multi-well Ψ lets one state branch into several fates (F6).
  - Raw counts replace z-scores, so rare-gene spikes disappear (F9).
  - Every parameter is named and belongs to a class with a published identifiability theorem: Guan et al. AISTATS 2026 for gradient drift plus σ out of equilibrium, and Guan et al. arXiv 2410.22729 for linear drift including curl. Per-dataset certificates check each theorem's conditions on the actual data.
- **Against the tgl-grn family.** PALM models cells at real capture times, not per-bin aggregate GRNs. It also makes no edge-level claims; claims stay at program level, where they can be identified.

Interpretation comes from the restricted class plus extra data (unspliced counts, metabolic labelling, clones, ATAC), never from reading a black-box drift. A black-box MLP drift appears only as an upper-bound arm, used to measure what the restricted class costs.

## 2. Model specification

**Observed data.** For cell i: raw UMI counts x_i ∈ N^G (G ≈ 2000 genes: highly variable genes chosen on fit stages only, plus all anchor genes), and optionally unspliced/spliced (u_i, s_i). Also its real capture time T_i ∈ {T_1..T_K}, its library size ℓ_i (observed total UMIs, or Chronocell-style depth from near-Poissonian genes), and its batch/replicate/condition labels.

**Latent state.** Z_i ∈ R^k is a vector of log program activities, a_i = e^{Z_i}. Programs are k_named + k_free, with k_free ≤ 2.
- Named program r has a mask P_r taken before seeing the data from external sources: MSigDB hallmarks (E2F/G2M cell cycle, apoptosis, p53, glucocorticoid response…), PanglaoDB/CellMarker lineage markers, and CollecTRI regulons of ≤ 6 declared lineage TFs.
- Each named program needs ≥ 3 anchor genes, i.e. genes that belong to P_r only.

**Decoder (stage 1, static).**
- Counts: x_ig ~ NB(mean ℓ_i·μ̃_g(Z_i), dispersion φ_g), with μ̃_g(Z) = b_g + Σ_r W_gr e^{Z_r}.
- Constraints: W_gr ≥ 0, W_gr = 0 for g ∉ P_r, and the mean anchor loading of each program is 1 (scale convention). In log space the scale convention is only a translation of Z_r, so it cannot change the shape of Ψ, J or σ.
- Repression lives in the dynamics (signed couplings), not in the decoder.

**Dynamics (real time t, in hours or days; no flow time anywhere).** The model class M2 is

dZ_t = −(I − J)∇Ψ(Z_t) dt + σ dW_t, with J = −Jᵀ constant and scalar σ,

Ψ(z) = ½(z−m)ᵀQ(z−m) + Σ_r [−s_r log cosh(z_r − c_r)] − Σ_{r<q} K_rq tanh(z_r)tanh(z_q).

Terms of Ψ:
- Q (positive definite, k(k+1)/2 parameters) is the restoring/linear-interaction part.
- s_r is the self-activation (commitment) index. Program r is bistable on its own when Q_rr·s_r > 1. This is the continuous Stern–Sompolinsky analogue of Badalotti's J_D.
- K is a sparse symmetric nonlinear coupling.
- In total there are about 250 dynamical parameters.

**Two exact properties of M2 (both checked in derivation):**
- **Stationary law.** For constant skew J and scalar σ, the stationary law is ∝ exp(−2Ψ/σ²), because tr(J∇²Ψ) = 0 and ∇Ψᵀ J ∇Ψ = 0. So Ψ is a true quasi-potential. J changes only transients and probability currents.
- **Attractors.** Fixed points are exactly the critical points of Ψ, since (I−J) is invertible. Minima are stable, because (I−J) has positive-definite symmetric part.

**Nested classes:**
- **M0 (linear OU).** Quadratic Ψ, which gives dZ = A(Z−m)dt + σdW with A = −(I−J)Q.
  - Lemma: the map (A,σ) ↔ (Q,J,σ) is a bijection on stable A. Concretely, Q = (σ²/2)Σ∞⁻¹ with AΣ∞ + Σ∞Aᵀ + σ²I = 0, and J = I + AQ⁻¹; one checks J + Jᵀ = 0.
  - So M0 is exactly the set of stable OU processes with isotropic noise.
- **M1 (pure gradient).** J = 0.
- **M3 (declared tilt).** Ψ(z;T) = Ψ(z) − u(T)hᵀz, where u is a known step (e.g. a media switch) or a 2-parameter logistic ramp. This adds k+2 parameters and is the only allowed time dependence (L7).
- **M4 (upper-bound arm).** M2 plus a small-norm MLP residual drift. It is never interpreted.

**Growth.** Particle weights evolve as dw = g(Z)w dt, with g(Z) = g_0 + κ·a_cc(Z) − δ·a_apo(Z). Here a_cc and a_apo are the anchored cell-cycle and apoptosis program activities.
- g is relative only (L8), unless total counts are known (which fixes ḡ(T)) or LARRY clone sizes are used (which fix κ and δ).

**Kinetic layer (when unspliced counts exist).** Let α_g = γ_g μ̃_g(Z). Use the SDE generator Lf = ∇f·b + (σ²/2)Δf. To first order in slow variation of α:
- s̄_g = μ̃_g − (Lμ̃_g)·τ_s,g
- ū_g = c^u_g(γ_g/β_g)(μ̃_g − (Lμ̃_g)·τ_u,g)
- τ_u = 1/β and τ_s = 1/β + 1/γ are lag times in real units, so γ = 1/(τ_s − τ_u).
- c^u_g absorbs length and priming bias.
- When |Lμ̃|τ is not much smaller than μ̃, (u,s) are integrated exactly along simulated paths. The kinetic ODE is linear given the path, with cost O(G) per particle-step.

**Generation at any real T\* (N cells):**
1. Draw Z⁽ⁿ⁾ from p̂_{T_1} (stage-1 deconvolved latent density), or from p̂_{T_j}, where T_j is the last observed snapshot before T\* ("data-assimilated mode").
2. Simulate with Euler–Maruyama in real time (Δt ≤ 1% of the smallest gap), carrying weights e^{∫g}. Use systematic resampling when the effective sample size falls below N/2.
3. Resample N cells in proportion to the weights. Relative population size is mean w; it becomes absolute only with counts.
4. Draw ℓ from a library-size model (log-normal). This is a declared nuisance: its parameters are interpolated linearly between batches and held at the nearest batch outside [T_1, T_K].
5. Draw x ~ NB(ℓμ̃(Z), φ), plus (u,s) from the kinetic layer.

Outputs are tagged "interpolation" for T\* ∈ [T_1, T_K]. Otherwise they are tagged "EXTRAPOLATION: rests on time-homogeneity (M2) or on the declared u(T) (M3)". The M2-vs-M3 spread is shown as an assumption-sensitivity band.

## 3. Identifiability: what is claimed, under which assumptions

Each certificate is a test computed on the real data to check that a theorem's conditions hold there; failures are reported, not hidden.

| Object | Claim | Assumptions | Basis | Per-dataset certificate |
|---|---|---|---|---|
| Latent marginals p_T(Z) and loadings W | Identified up to program permutation (fixed by the names) and per-program scale (fixed by convention) | A1: decoder linear in a, W ≥ 0, ≥ 3 anchors per program (separability). A2: NB noise independent across genes given Z, gene-specific φ. A3: library offset observed. A4: G ≥ 2k+1, so common and idiosyncratic variance separate as in factor analysis | L16: NMF uniqueness (Donoho–Stodden 2003; Fu–Huang–Sidiropoulos 2019; anchor/sparse decoding, Moran et al. 2022); classical identifiability of mixed-Poisson distributions. No nonlinear VAE, so L17 does not apply | **C-W**: anchor purity on fit cells (off-program loading < 5% of anchor loading); loadings agree across 10 starts (Tucker congruence ≥ 0.95) |
| (Ψ on the data support, σ) in M1 | Identified iff the marginals are not stationary | Scalar σ, time-homogeneous Ψ, Lipschitz gradient with linear growth, exact marginals | Guan et al. AISTATS 2026, Thm 3.2. Cor 3.3: 3 marginals suffice, but only for a countable candidate set; our class is not countable, hence the multistart check | **C-σ**: consecutive latent marginals differ beyond the bootstrap/replicate floor (permutation test on Sinkhorn divergence, p < 0.01) on ≥ 2 intervals |
| (Q, J, σ) in quadratic M2 (= M0) | Identified iff p̂_{T_1} is not auto-rotationally invariant | Linear drift, additive noise | Lemma above + Guan et al. 2410.22729, Thm 3 | **C-J**: rotating p̂_{T_1} by the fitted generalized rotation (AΣ + ΣAᵀ = 0 condition) changes it beyond bootstrap noise (sliced-W2 test) |
| J with nonlinear Ψ | **Not proven; stated as conjecture C1.** Exact counterexample: if Ψ and p_0 are both rotation-invariant in the plane where J acts, J is invisible | — | Open (theory dossier: nonlinear non-gradient is open) | J reported as a profile-likelihood interval; its sign/magnitude is interpreted only if C-J passes AND simulators recover it |
| Attractors, saddles, basin depths ΔΨ/σ² | Identified wherever Ψ is, restricted to the high-density region of the observed data | As above | Exact Gibbs property of M2 | Minima/saddles outside the 95% HPD support are flagged, not interpreted |
| Commitment and switching rates (1/time) | Derived from Ψ, J, σ through the irreversible Eyring–Kramers formula (Bouchet–Reygner 2016, from memory) | Small-noise asymptotics | — | Checked against mean first-passage times simulated from the fitted SDE; reported only if within a factor of 1.5 |
| Growth g | **Not identified** from normalized snapshots (L8). Reported as {g + c(T)} | — | L8 | κ, δ claimed only with total counts (ḡ) or clone sizes (Ventre et al. J Math Biol 2026; moslin) |
| β_g, γ_g in 1/h | Identified only for genes with \|Lμ̃_g\| above noise, given an identified drift whose time unit comes from the real T_k (L9) | Slow-variation expansion, or exact path integration | Lag-coefficient argument (§2) | Gene-level: \|Lμ̃\| signal-to-noise > 2 across cells, and the bootstrap CV of τ < 0.3 |
| M3 tilt (h, u(T)) | Only for a declared parametric u(T) | L7 | Guan AISTATS 2026, App. B (discrete parameter changes) | Is the tilt needed? Decided only by a pre-registered rule (§6) |

**Never claimed:**
- the path of an individual cell (L11);
- extrapolation (L7);
- gene-level GRN edges, or the off-support Jacobian (L15);
- non-Markov memory (L2; Weinreb 2020);
- nonlinear curl outside the J∇Ψ form;
- absolute growth without counts;
- any parameter that fails its certificate.

**How unidentified objects are reported:** as equivalence classes. Fix the non-identified coordinate on a grid (e.g. ‖J‖, κ), refit the rest, and report the range whose fit loss stays inside the 95% bootstrap band.

**Theory deliverables:**
- T1: the M0 bijection lemma (done above, needs a write-up).
- T2: the decoder identifiability lemma, assembling the cited results.
- T3: the kinetic lag lemma.
- T4: conjecture C1 for nonlinear M2. Plan: adapt Guan's "residual process" argument. Fallback: a numerical study plus the symmetric counterexample family.

## 4. Inference and training

**Stage 0 (preprocessing).**
- Raw UMIs; no standardisation and no batch correction across time (L13).
- Genes are selected on fit stages only. The library model is fitted per batch.
- Couplings are computed only within (condition, replicate) groups.

**Stage 1 (decoder and deconvolved marginals).**
- Amortized variational inference: encoder q(Z | x, T) and decoder as in §2.
- The per-stage prior p(Z | T_k) is a 20-component full-covariance Gaussian mixture. These priors are the deconvolved marginals p̂_{T_k}; posterior aggregates are not used, because they inflate σ (L14).
- 10 restarts, then certificate C-W.
- Cost: O(N·G·k) per epoch. For N ≤ 120k (≤ 3k cells per stage), G = 2000, k = 16, this is about 4·10⁹ flops per epoch, roughly 0.5–1 A100-hour per fit.

**Stage 2 (dynamics), two independent estimators.**
- **E1 (primary, an SB-EM in the style of nn-APPEX with a learned reference):**
  - Initialise θ with an M0 fit: closed-form mvOU estimate on within-group OT couplings (Zhang & Stumpf; APPEX).
  - Then repeat about 20 times:
    - (E) For each consecutive pair, compute an unbalanced entropic coupling: π_k = argmin KL(π‖R_θ,k) + τ·KL(π1 ‖ w·p̂_{T_k}) + τ·KL(πᵀ1 ‖ p̂_{T_{k+1}}).
      - R_θ,k is the current SDE's transition kernel over Δt_k (real time).
      - It is computed either by local mvOU linearisation (closed-form Gaussian), or by simulating 8 particles per source point and smoothing with a Gaussian kernel.
      - The growth weight is w = e^{gΔt}.
    - (Bridges) Sample intermediate states from closed-form linearised-OU bridges at t ~ U(T_k, T_{k+1}).
    - (M) Bridge matching (Girsanov MLE) for the parameters of Ψ and J. σ² comes from quadratic variation. Growth comes from a weighted MLE on the row masses of the coupling.
  - At convergence the reference equals the fitted model, so the coupling no longer encodes a least-action prior.
- **E2 (cross-check, direct marginal matching):**
  - Loss: L(θ) = Σ_k S_ε(Push_θ^w(p̂_{T_1}) at T_k, p̂_{T_k}) + Σ_k S_ε(Push_θ^w(p̂_{T_{k−1}}) at T_k, p̂_{T_k}), plus λ(log M̂_k − log M_k)² when cell counts are known. S_ε is the unbalanced Sinkhorn divergence.
  - Optimised with reparameterised Euler–Maruyama and backpropagation: 4096 particles, ≤ 2000 steps.
- **Estimator invariance:** parameters claimed identifiable must agree between E1 and E2 within bootstrap CIs. Disagreement is itself a reported failure.

**Stage 2b (kinetics).** Per gene, a 3-parameter NB maximum-likelihood fit of (τ_u, τ_s, c^u) on (u, s), given posterior samples of Z and Lμ̃ from the fitted drift. This is Chronocell-like, embarrassingly parallel and closed-form-friendly. The arm "M2+kin" feeds the (u,s) likelihood back into θ as velocity-like observations.

**Stage 3 (optional joint refinement).**
- Fit decoder and dynamics jointly, with the SDE pushforward (estimated by kernel density on particles in k dimensions) used as the prior p(Z | T).
- It runs only under a nested guard: it is accepted only if it improves the inner-validation score in ≥ 3/4 inner folds. This fixes cfm_grn_v2 failure F7 (an unguarded stage producing the reported predictions).

**Cost:**
- One E2 fit is about 1 A100-hour (≈ 6.5·10⁸ Sinkhorn operations per iteration, 3k iterations). One E1 fit is about 0.5–1 h.
- Stage 2b: about 2000 genes × 3 parameters, minutes on CPU.
- The dynamical model has about 250 parameters, so there is no capacity churn.

**Reuse from the user's own code:**
- `flows/coupling.py` (ot_coupling, chain_ot): E1 initialisation and the OT-interpolation control.
- `flows/marginal.py` (differentiable U-statistic ED, sliced W2): E2 losses and metrics.
- `flows/solver.py`: extend RK4 to Euler–Maruyama/Heun SDE solvers.
- `evaluation/metrics.py` and `controls.py`: add mixture-of-neighbours, density-interpolation and decoder-floor arms.
- `data/real.py`: CSR loaders; **verify the HPC caches hold raw UMIs, otherwise re-parse.**
- `data/prior.py`: degree-preserving randomisation, which becomes the random-anchor control.
- `training/designs.py`: skip-k fold designs.
- `data/synthetic_grn.py`: becomes a misspecified simulator family once heterogeneity is added.
- The pre-registration and SHA-256 freeze, the bit-exact reproducibility gate, and the leakage tests from tgl-grn-continuous.
- External methods (APPEX, mvOU-OTFM, PFM, WOT, PRESCIENT, CoSpar) are re-implemented or run only as baselines.

## 5. Datasets, in order of use

| # | Dataset (accession) | Design | Extra modality | Size / status | Claim it supports |
|---|---|---|---|---|---|
| D0 | Simulators: dyngen (gene-level stochastic simulation with pre-mRNA/mRNA, true simulation time and rates); scMultiSim (GRN, kinetic mode, s/u, Beta-Poisson noise); SERGIO (CLE plus technical noise); own misspecified families (Hill-GRN CLE in gene space with bursting counts and agent-based division/death; a repressilator-plus-branch non-gradient system); own-family PALM | Known truth | Rates, fates, commitment times | Local or HPC; installing R/py packages needs the user's OK | Identifiability recovery on families different from the model |
| D1 | sci-fate A549 dexamethasone (GSE131351) | 0/2/4/6/8/10 h; ≈6.7k cells (1054/1049/949/1262/1041/1325) | 4sU new-RNA counts (2 h pulse) | Small | Real-time rates (validated on new RNA the fit never sees). The dex step makes p_0 (untreated steady state) out of equilibrium by design, so C-σ holds a priori. Held out: interior 4 h and 6 h, terminal 10 h |
| D2 | Weinreb LARRY in vitro (GSE140802) | Days 2/4/6, ≈130k cells | Clonal barcodes, unused in the fit | Medium; download needs permission | Fate probabilities and Kramers rates vs clonal fate; growth κ from clone sizes. Only 3 marginals (the Cor 3.3 minimum), so **no** held-out stage here |
| D3 | Schiebinger reprogramming (GSE122662) | 39 timepoints over days 0–18, ≈315k cells; 2i vs serum after day 8 | Two conditions with a known media switch | ≈0.8 GB; needs permission | Dense time (the Lavenant design rule); M2 vs M3 with a known switch (Guan App. B); interior and terminal held-out stages |
| D4 | Pijuan-Sala (E-MTAB-6967; cached) | 9 stages E6.5–E8.5, ≈116k cells; erythroid subset with s/u (scVelo gastrulation_erythroid: 9,815 cells; atlas-wide spliced/unspliced via Bioconductor MouseGastrulationData) | s/u | Cached | Confirmatory held-out stages; kinetic layer; fixed-vs-tilting landscape in an embryo. Known batch issue: the E8.0 pools sit in different sequencing batches, so the replicate floor is broken |
| D5 | Veres stage 5 (GSE114412; cached) | 8 days, 2 differentiations | Replicate floor | Cached | Confirmatory, with a usable replicate floor |
| D6 | EB (Moon 2019; cached) | 5 windows | — | Cached | Engine calibration only |
| D7 | Argelaguet multiome (accession GSE205117 UNVERIFIED; GEX cached) | 5 timepoints, RNA+ATAC | chromVAR motif accessibility | ATAC download needs permission | Held-out-modality check of TF-regulon program activity (node level only) |
| D8 (neg. control) | scEU-seq RPE1-FUCCI (GSE128365) | Stationary cycling population | Labelling, FUCCI | Small | Certificates must report σ and J as **not** identifiable from snapshots |
| D9 (stretch) | RegVelo zebrafish neural-crest time course + Perturb-seq of 22 TFs (GSE256009/8) or RENGE (GSE213069) | Time + KO | KO responses | Needs permission | Program-clamp predictions vs real knockouts |

## 6. Evaluation protocol

**Ceiling diagnostic first** (cfm_grn_v2's best tool, transplanted). Before fitting any dynamics, score:
- the **decoder floor**: real held-out cells vs decode(encode(the same cells)), with a decoder fitted on fit stages only;
- the **oracle latent**: decode samples of p̂ estimated on the held-out cells.

Every arm is then reported as gap-closed = (D_control − D_model)/(D_control − D_floor).

**Metrics.** Primary metrics avoid the centroid dominance seen in cfm_grn_v2 Job E:
- **P1a, latent log predictive density.** Held-out cells' posterior samples are scored under a kernel-density estimate of the model's particles at T\*. This is a proper scoring rule.
- **P1b, composition total variation.** One classifier, trained only on fit-stage annotations, labels both real held-out cells and generated cells; compare the label proportions.
- **Secondary:**
  - count-space classifier two-sample test, C2ST-AUC (logistic regression and gradient boosting on log1p counts), with a replicate C2ST floor;
  - ED with its mean/trace split;
  - shape-only ED (both sets centred);
  - multi-bandwidth MMD;
  - var_ratio per program (pass band [0.8, 1.25]);
  - coverage of 90% predictive intervals per program (pass band [0.85, 0.95]);
  - per-gene mean, variance and zero-fraction posterior predictive checks.

**Controls.** All controls are decoded with the same decoder, plus real-cell versions where relevant:
- persistence;
- latent mean interpolation;
- McCann/OT displacement interpolation;
- **mixture of neighbouring real snapshots** weighted by time fraction (the key composition control);
- static density interpolation of p(Z|T), i.e. no dynamics;
- logit-linear extrapolation of composition (terminal stage only);
- shuffled-time twin;
- random-anchor programs (degree-matched gene sets);
- replicate floor;
- black-box arm M4 / an [SF]²M-type MLP SB;
- cfm_grn_v2 v0.1.10.

**Ablations, each of which must hurt or be reported as decoration:**
- J = 0;
- s = 0 (no self-activation);
- σ fixed to the M0 value;
- g = 0;
- kinetic layer removed.

**Held-out designs:**
- whole stages only: each interior stage removed in turn;
- terminal stage T_K removed (forecast);
- skip-k multi-removal;
- free rollout from T_1 scored at every stage, alongside one-step prediction from T_{k−1}.

Selection uses inner folds only. The configuration is frozen with SHA-256 before the confirmatory datasets (D4, D5) are opened.

**Recovery on simulators of a different family** (≥ 2 of dyngen, scMultiSim, SERGIO, own Hill-GRN; 10 instances, 3 seeds each). A parameter class is "declared identifiable" only if it passes on ≥ 2 such families:

| Parameter | Recovery threshold | Stability threshold |
|---|---|---|
| attractors vs true terminal states | ARI ≥ 0.6 | across-start CV ≤ 0.2 |
| switching rates vs true commitment-time distributions | Spearman ≥ 0.7 | across-start CV ≤ 0.2 |
| β, γ vs truth | Spearman ≥ 0.7, median \|log₂ ratio\| ≤ 1 | across-start CV ≤ 0.2 |
| relative growth per state | Spearman ≥ 0.7 | across-start CV ≤ 0.2 |
| sparsity of A_eff = −(I−J)∇²Ψ at the minima | AUPRC − prevalence > 0 under degree-matched negatives (tgl lesson) | across-start CV ≤ 0.2 |
| J | interval excludes 0 when p_0 is asymmetric; interval contains 0 when symmetric (certificate sanity) | — |

**External validation of interpretable parameters (data never seen by the fit):**
1. **Rates, sci-fate.** Predicted per-gene, per-time new-RNA fraction, ∫_{T−2h}^{T} α_g(Z_s)e^{−γ_g(T−s)}ds averaged over ancestral bridge paths, vs the observed 4sU fraction.
   - Trivial control: a constant γ with γ/β taken from the steady-state u/s ratio.
2. **Rates, other datasets.** γ vs published half-life atlases: SLAM-seq mESC (Herzog 2017), TimeLapse-seq K562, scEU-seq RPE1.
3. **Fates, LARRY.** AUROC of predicted day-2 fate bias vs clonal outcome at days 4/6.
   - Baselines re-run by us, with a calibration gate against their published numbers: WOT, PRESCIENT, CoSpar (CoSpar uses clones, so it is an upper reference).
   - Trivial control: distance to day-6 fate centroids.
4. **Growth, LARRY.** Relative growth per state vs clone-size expansion.
5. **Chromatin, D7.** Corr(TF-regulon program activity, chromVAR of the same TF in the same cell) vs random-regulon programs. Circularity check: regulons are taken from CollecTRI, not from this dataset's ATAC.

**Pre-registered decision rules:**
- **R1 (generation).** Positive on a dataset if the mean Δ (model − best control) in P1a has a lower 95% CI > 0 over seeds × held-out stages, it wins on ≥ 3/4 of stages, and P1b is non-inferior (margin 0.02 TV).
- **R2 (calibration).** ≥ 80% of programs inside the var_ratio and coverage pass bands.
- **R3 (rates).** Spearman with external half-lives > 0 (CI) and above the trivial control; median half-life within a factor of 2 of the external value.
- **R4 (fates).** AUROC ≥ trivial control + 0.03 (CI), and within the CI of the best re-run non-clonal baseline.
- **R5 (landscape).** M3 is preferred over M2 only if Δ P1a is positive at ≥ 3/4 of held-out stages with CI > 0. Otherwise the fixed landscape is retained.

**Kill criteria:**
- **K1.** Own-family recovery fails: the estimator is broken. Stop and debug.
- **K2.** A parameter class fails on 2 different-family simulators: that class is dropped from all real-data interpretation and reported as unidentified.
- **K3.** Fewer than 4 named programs pass C-W on a dataset: no interpretability claims on that dataset.
- **K4.** On D1, D3 and D4, both M2 and M3 lose to static density interpolation or the mixture-of-neighbours control: the dynamics claim is falsified. Pivot to the certificates/benchmark paper.
- **K5.** By month 6, neither the rate validation (R3) nor the fate validation (R4) beats its trivial control: pivot to the theory/benchmark paper.

## 7. DO / DON'T, mapped to the projects that taught each lesson

**DO**
- Model cells at real capture times; never per-bin aggregates (tgl-grn-1CCD/recurrent, T9).
- Score the no-dynamics ceiling (decoder floor, oracle latent) before training anything (cfm_grn_v2 F1).
- Put mean_interp, persistence, OT-interp, the shuffled twin and the replicate floor in every table, plus mixture-of-neighbours and density interpolation (cfm_grn_v2 F3; tgl lesson 3).
- Use primary metrics that are not centroid-dominated, and always print the mean/trace split (cfm_grn_v2 Job E).
- Use a count likelihood on raw UMIs (cfm_grn_v2 F9 rare-gene spikes; Chronocell).
- Couple only within (condition, replicate) groups (cfm_grn v1, global OT).
- Recover parameters on simulators of a different family before any biological reading, using degree-matched negatives (cfm_grn v1 mask AUC 0.54; Dileo audit).
- Change the data regime rather than the architecture: unspliced, labelling, clones (cfm_grn v3/v4, where a signed prior plus velocities moved results).
- Ablate every component and require that it hurts (tgl §10.3: temporal modules that were not temporal).
- Guard every training stage; pre-register with a SHA freeze; enforce the reproducibility gate (cfm_grn_v2 F7; prereg v0.2).
- Fill PURPOSE.md, keep one canonical repo, run heavy jobs on the HPC (tgl anti-patterns 11, 13, 14).

**DON'T**
- Build interpolation paths that are linear in the snapshots (cfm_grn_v2 F2/F3).
- Make a black-box MLP drift the scientific object (cfm_grn_v2: interpretable 0/5). M4 is an upper bound only.
- Read Jacobians or heatmaps of non-predictive models (tgl hub heatmaps).
- Make edge-level GRN claims (L15; tgl edge turnover).
- Use the uniform process-time prior, AIC on a lower bound, or a Poisson-only likelihood (Chronocell traps).
- Learn contextual masks without recovery evidence (cfm_grn v1).
- Select configurations on the test window (EB v0.1.10).
- Compare to published numbers without a calibration gate (Job C).
- Claim "attractor = cell type" from fixed-point counts (Badalotti dossier).
- Claim extrapolation or nonlinear curl as identified (L7; C1 is open).
- Batch-correct across timepoints (L13).
- Read seed-only CIs as biological generalisation (cfm_grn_v2 risk 8).

## 8. How the two papers are used

**Chronocell (Fang, Gorin, Pachter 2025). Used for the decoder and the kinetic idea, not for its clock.**
- *Taken:*
  - the raw-count likelihood, with depth from near-Poissonian genes;
  - the two-species kinetics, generalised to path-dependent rates, with closed-form per-gene M-steps (stage 2b);
  - the per-gene "identified vs not" reporting;
  - the multistart stability diagnostic (its "average precision" across starts, turned into our across-start CV);
  - the trajectory-vs-cluster pre-check (if latent marginals do not change beyond the floor, C-σ fails);
  - external γ validation against labelling.
- *Replaced:*
  - uniform process-time prior → real T plus the asynchrony carried in p_T(Z);
  - piecewise-constant synchronous α → α = γ·μ̃(Z(t)) driven by the SDE;
  - Poisson-only → NB;
  - genes treated independently → programs.
- *Not used:* process time as the clock; AIC on a lower bound; its code (re-implemented).

**Badalotti, Baldassi, Mézard, Scardecchia, Zecchina (PRR 2026). Used as the attractor diagnostic toolkit plus one inductive bias.**
- *Taken:*
  1. **Self-activation index s_r** in Ψ, the analogue of J_D. The persistence control is the s → ∞ limit, so the fit can be located between "no attractors" and "frozen".
  2. **Relaxation census.** Integrate the deterministic part from every cell for 5–10× the time span; compare endpoints with the minima of Ψ and with terminal annotations (ARI); controls are shuffled time and random anchors.
  3. **Canalisation curves.** Perturb late cells' Z with noise, relax, and measure the return distance relative to inter-basin distances; Hessian eigenvalues in units of σ².
  4. **Bifurcations in time**, in M3 only: saddle-node values of u(T), giving commitment times in real units.
  5. **Hysteresis prediction.** A transient program clamp persists after removal if s_r > 1; tested only where pulse/washout data exist.
  6. **Report flat high-entropy regions** (small Hessian eigenvalues) as plastic manifolds rather than forcing discrete wells.
- *Not used:* binary units, the Hebbian/perceptron rule, fixed-point counting thresholds (iid-Gaussian theory), gradient-free learning. A possible Bocconi conversation about fixed-point counting is networking, not evidence.

## 9. Novelty vs the closest competitors

| Method | What it has | What it lacks relative to PALM |
|---|---|---|
| PFM (Maddu/Chardès/Shelley, arXiv 2604.25062) | Highest scoop risk. 24 a-priori TF genes in gene space; MLP force h(x) − ℓx; CLE/multiplicative noise; growth; LARRY; held-out energy distance; in-silico KO; AUPR vs ChIP. States that models with similar interpolation accuracy can encode different dynamics | Continuous biwhitened data, no count model; TF mRNA ≠ activity; no identifiability class or certificates; no real-time kinetic rates. Its declared future work is "masked sparse layers/priors" |
| Unbalanced PFI (Zhang, Maddu, Qiu, Chardès, NeurIPS 2025) | Drift, noise and growth, disentangled | MLPs; growth identifiability unclear |
| mvOU-OTFM (Zhang & Stumpf, NeurIPS 2025) | Linear non-gradient reference | Reference is a fixed prior; in PALM the learned reference *is* the estimand, with nonlinear Ψ |
| nn-APPEX / APPEX (Guan et al.) | The estimators and theorems PALM builds on | 5 PCs, ~100–200 cells per stage, no counts, growth, kinetics or certificates; no non-reversible nonlinear class |
| StructureFlow | Structure built into the drift | Balanced mass, gene level, no counts, unproven identifiability |
| Cell-MNN | Locally linear interactions | Deterministic, non-identifiable VAE latent, no growth or noise |
| DeepRUOT / RUOT / unbalanced, branching SB (BranchSBM) / TIGON / ARTEMIS | Growth and stochasticity | MLP drifts, interpreted post hoc; ARTEMIS's VAE latent is not identifiable |
| JKOnet\* | Interpretable energy terms | Gradient only (no curl), no counts |
| RegVelo, cell2fate, Chronocell | Interpretable kinetics | Latent or process time, per-cell, no population growth |
| Dynamo | Absolute rates from labelling | Deterministic, no population |
| Count Bridges (ICLR 2026) | Count-valued bridges | Deconvolution, not dynamics identification |
| CADENCE (2605.23470) | Identifiability from snapshots | Not single-cell, not stochastic SDE classes |
| Sáez/Rand/Briscoe; Mochulska & François 2025 | Closest in spirit (landscape plus rotation) | Hand-built or non-certified, no count generation, no time-course fitting with growth |

**What is genuinely new:**
1. Per-dataset identifiability certificates (C-W, C-σ, C-J) that check the theorems' conditions on data, plus an estimator-invariance test (E1 vs E2), with a built-in negative control (RPE1).
2. The non-reversible gradient class with an exact Gibbs quasi-potential. Its linear identifiability follows via the bijection lemma; the nonlinear case is conjecture C1, with an explicit counterexample.
3. An anchored non-negative count decoder that identifies the latent marginals without VAE non-identifiability.
4. Kinetic lag times calibrated in hours by the snapshot clock, validated against metabolic labelling the fit never saw.
5. Commitment rates in real time units, validated against clonal fates.
6. A pre-registered "fixed vs tilting landscape" test.

The bare architecture is replicable. The defensible contribution is items 1, 2 and 4–6.

## 10. 6–12 month plan, compute, risks, probability

| Month | Milestone | Gate |
|---|---|---|
| 0–1 | Fill PURPOSE.md; pre-registration v1 (rules R1–R5, K1–K5, program sets); simulator suite (own families first; dyngen/scMultiSim/SERGIO after the user's OK); raw-count audit of HPC caches | — |
| 1–2.5 | Stage 1 decoder + C-W; E2, then E1; M0→M1→M2 recovery | **G1**: K1 and K2 checks |
| 2.5–3.5 | D1 sci-fate: held-out 4/6/10 h; new-RNA validation; certificates | **G2**: R1–R3 |
| 3.5–5 | D2 LARRY: fates, growth, Kramers rates, relaxation census | **G3**: R4 |
| 5–7 | D3 Schiebinger (M2 vs M3 with known switch); freeze, then D4/D5 confirmatory; kinetic layer on the D4 erythroid subset | **K4** and **K5** checks (K5 by month 6) |
| 7–9 | Theory write-up (T1–T3, C1 study); D7 chromatin; optional D9 knockouts | — |
| 9–12 | Paper(s): (a) method + validations; (b) certificates + benchmark | — |

**Compute (Bocconi QoS: 1 running / 2 submitted, no arrays).** Fits are packed sequentially inside single jobs.

| Item | Estimate |
|---|---|
| One fit | ≈ 0.5–1 A100-hour |
| Simulators | ≈ 540 small fits ≈ 110 GPU-h |
| D1 | ≈ 20 GPU-h |
| D2 | ≈ 60 GPU-h |
| D3 | ≈ 5 seeds × 8 folds × 6 arms ≈ 240 GPU-h |
| D4 + D5 | ≈ 150 GPU-h |
| **Total** | **≈ 500–700 A100-hours**, about 25–35 days of serial wall-clock, feasible within the plan |

**Risks and fallbacks:**
- **Time-homogeneity is false in embryos.** Fallback: M3. The in-vitro datasets D1–D3 are the main bets.
- **Too few timepoints.** D1 has 6, D3 has 39; LARRY is used only for lineage validation.
- **Decoder misfit dominates the count-space classifier test.** Report gap-closed against the decoder floor; latent P1a is primary.
- **Hidden heritable variables** (Weinreb 2020) cap fate AUROC. Compare against a re-run baseline, not an absolute target.
- **Growth stays prior-driven.** Report it as an equivalence class.
- **The first-order kinetic approximation fails on short timescales** (sci-fate is 2 h scale). Use exact path integration.
- **Conjecture C1 fails.** The counterexample family plus the numerical study is still a result.
- **Scoop by PFM's authors.** Lead with certificates, real-time rates and the validations, not the architecture.
- **Permissions.** Downloads and installs need the user's approval; D1, D4, D5 and D6 are small or already cached.

**Honest probabilities:**
- Recovery passes on ≥ 2 different-family simulators for Ψ minima and σ: ≈ 0.6.
- Kinetic rates beat the trivial rate control against external half-lives: ≈ 0.45.
- LARRY fate validation passes R4: ≈ 0.5.
- PALM beats all trivial controls on held-out stage generation on ≥ 1 confirmatory dataset: ≈ 0.35–0.45.
- **A presentable positive within 12 months** (certified parameters + ≥ 1 external validation passing): **≈ 0.6**.
- The full triple (generation + rates + fates): ≈ 0.15–0.2.
- The fallback certificates/benchmark/theory paper: ≈ 0.85.

**Sources checked this run:**
- [PFM, arXiv 2604.25062](https://arxiv.org/abs/2604.25062) ([details](https://www.alphaxiv.org/abs/2604.25062.md))
- [Guan et al., arXiv 2410.22729](https://arxiv.org/html/2410.22729v3)
- [Guan et al., gradient-flow SDEs, arXiv 2505.21770 (AISTATS 2026)](https://arxiv.org/html/2505.21770v3)
- [Zhang & Stumpf, mvOU Schrödinger bridges](https://arxiv.org/pdf/2505.16644)
- [Unbalanced PFI, arXiv 2505.13197](https://arxiv.org/html/2505.13197v3)
- [Multi-marginal SB with iterative reference refinement](https://arxiv.org/html/2408.06277v4)
- [ARTEMIS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12261462/)
- [Count Bridges, ICLR 2026](https://iclr.cc/virtual/2026/poster/10011529)
- [CADENCE, arXiv 2605.23470](https://arxiv.org/abs/2605.23470)
- [scFM, arXiv 2605.22340](https://arxiv.org/pdf/2605.22340)
- [sci-fate](https://www.biorxiv.org/content/10.1101/666081.full.pdf)
- [GSE128365 (scEU-seq)](https://pluto.bio/product/curated-data-sets/explore/experiment/gse128365-sequencing-of-metabolically-labeled-transcripts-in-single-cells-from-rpe1-fucci-cells-and-murine-intestinal-organoids-)
- [LARRY / CoSpar datasets](https://cospar.readthedocs.io/en/latest/cospar.datasets.hematopoiesis_130K.html)
- [MouseGastrulationData (Bioconductor)](https://bioconductor.statistik.tu-dortmund.de/packages/3.18/data/experiment/vignettes/MouseGastrulationData/inst/doc/MouseGastrulationData.html)
- [scVelo erythroid dataset](https://unitvelo.readthedocs.io/en/latest/Figure2_ErythroidMouse.html)
- Dossier: `C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md`

**Unverified items:** GSE205117 for Argelaguet; the Bouchet–Reygner 2016 and Teicher citations (from memory); exact LARRY, Schiebinger and Veres cell counts and download sizes.

# THEORY CRITIC
# Adversarial review of PALM (angle B, identifiable SDE): theory, identifiability and metric validity

## (1) Verdict: 4/10 as written; about 6.5/10 after the fixes below

The skeleton is sound: a restricted class, a count decoder, equivalence-class reporting, strong controls and kill rules. The problems are in the claims.

- Five claims fail as written. The cited theorems do not cover finite snapshots or growth. The decoder's coordinates are not identified. The kinetic-lag equations are wrong. E1 cannot estimate σ.
- The primary metric escapes the mean-interpolation trap but falls into a similar one: mixing the neighbouring snapshots is nearly optimal in exactly the regime where the landscape reading holds.

---

## (2) Fatal flaws (each sinks a headline claim as written; all can be repaired)

**FATAL-1. The cited identifiability theorems do not cover how PALM is estimated.**
- **Guan et al. AISTATS 2026, Thm 3.2** (checked in arXiv 2505.21770v3). It assumes marginals observed for **every t in [0,T]**, a gradient drift, scalar σ, time-homogeneous dynamics, Lipschitz ∇Ψ with linear growth, and **conserved mass**.
- **Cor 3.3** needs three things: a **countable** candidate set of C^∞ potentials, **randomly drawn** times t_i ~ U[T_i, T_{i+1}], and the result holds only with probability 1. A design with K fixed snapshots and about 250 continuous parameters meets none of these.
- **Guan et al. 2410.22729, Thm 3** (from the abs and v3 html pages; the fetch of the v4 html timed out).
  - It assumes the curve {p_t : t ≥ 0}, with no intercept: dX = AX dt + G dW.
  - Auto-rotational invariance is defined **about the origin, using Σ = Cov(X₀)**: there exists A ≠ 0 with AΣ + ΣAᵀ = 0 and e^{At}X ~ X.
  - There is also a **second clause**: some orthogonal P makes a component of PᵀX a mean-zero univariate Gaussian.
- **Aliasing.** With equally spaced snapshots (sci-fate is exactly every 2 h), e^{AΔ} fixes A only up to shifting complex eigenvalues by 2πn i/Δ. So the rotation speed carried by J is aliased even in M0.
- **Consequences.**
  - "Every parameter belongs to a class with a published identifiability theorem" is false for finite K.
  - The certificates test side conditions of theorems whose main hypothesis (a continuum of times) is violated.
  - C-J also tests the wrong condition. It checks one fitted rotation, when it must check the whole Σ-skew family, plus the Gaussian-component clause, with rotations about the fixed point m.
- **Fix.** Restate the claim as "identifiable in the continuous-observation limit (M1 with J = 0, and M0)". For finite K, use profile likelihood and Fisher information along each parameter direction, run own-family recovery at the real K and real time spacing, and use unequal spacing where the design allows it.

**FATAL-2. State-dependent growth breaks the Gibbs and quasi-potential properties and every theorem used. The rates-vs-growth row of the table contradicts itself.**
- With g(Z) = g₀ + κa_cc − δa_apo, the normalised marginal obeys ∂p = L\*p + (g − ḡ)p.
- Its long-run law is the principal eigenfunction of L\* + g. That is **not** ∝ exp(−2Ψ/σ²) unless g is constant.
- Every "exact property" (the quasi-potential, the basin depths ΔΨ/σ²) and both Guan theorems assume mass conservation. The fetched paper does not discuss growth at all.
- The design declares g not identified (L8), yet lists switching rates as identified. Moving mass from well A to well B fits equally well with an A→B Kramers rate or with faster growth in B. With g unidentified, the switching rates are unidentified too, and so are Ψ and σ until proven otherwise.
- **Fix.**
  - Claim Ψ, σ and the rates only for g ≡ const, declared as an assumption. It is plausible for sci-fate (10 h, under half of an A549 doubling).
  - Otherwise report (Ψ, σ, rates) jointly over the (κ, δ) grid, using the design's own equivalence-class device. Interpret only quantities whose range across that grid is narrow.
  - Report rates as identified only on LARRY, and only with a model of clone subsampling (Ventre 2026 / moslin), because raw clone-size expansion confounds growth with the sampling fraction.

**FATAL-3. The latent coordinate system, which every dynamics claim rests on, is not identified.**
- μ̃_g = b_g + Σ_r W_gr a_r admits the shift a_r → a_r − c with b_g → b_g + W_gr·c for all g in P_r. This is valid whenever c < min a_r, and with a = e^Z every a_r > 0.
- In log space that is Z_r → log(e^{Z_r} − c). This map is **nonlinear**:
  - it changes the shape of Ψ;
  - by Itô's formula it turns scalar σ into state-dependent noise;
  - it changes which wells and saddles exist.
- The design's statement that "the scale convention is only a translation, so it cannot change the shape of Ψ" covers the scale but misses the offset. It is also wrong for the coupling term: K·tanh(z_r)tanh(z_q) has no offset, so translations are not absorbed.
- **Certificate C-W is tautological.** Off-program loadings are zero by construction (W_gr = 0 for g ∉ P_r), so "off-program loading < 5%" cannot fail.
- **The k_free ≤ 2 free programs have no anchors.** Separable or NMF uniqueness needs an anchor for every column, so A1 is violated.
- The cited NMF results (Donoho–Stodden; Fu–Huang–Sidiropoulos) are about deterministic factorisation. Identifying the mixing law p_T(Z) under NB noise with unknown φ_g is **not** a published theorem; it is T2, still to be proved.
- **Fix.**
  - Fix b_g externally: set it to ℓ_i × the ambient profile from empty droplets per library, or b_g = 0 on anchors.
  - Audit anchors by fitting an unmasked or L1-relaxed NMF on fit cells and measuring leakage of anchors onto other factors.
  - Exclude free programs from the dynamics, or drop them.
  - Use tanh(z_r − c_r) in the coupling term.
  - Add a measurement-SNR certificate: posterior SD of Z_r compared with the between-stage displacement, requiring at least 10 well-expressed anchors.

**FATAL-4. The kinetic layer's equations are wrong, and β is not identifiable.**
- **(a) Wrong generator.** The lag needs E[α(Z_{t−r}) | Z_t = z], which is a statement about the **time-reversed** process, with drift b̂ = −b + σ²∇log p_t. The correct expansion is
  E[f(Z_{t−r}) | z] = f + r[(−b + σ²∇log p_t)·∇f + (σ²/2)Δf] + O(r²).
  The design instead uses f − r[b·∇f + (σ²/2)Δf].
  - Check with a stationary reversible OU process: the truth is z·e^{−θr} ≈ z(1 − θr); the design's formula gives z(1 + θr). **The sign is opposite**, so the design predicts lags at equilibrium where none exist.
  - Exact path integration along the simulated paths is correct. Only the generator shortcut is wrong.
- **(b) Z plays two roles.** The stage-1 decoder sets E[s | Z] = ℓμ̃(Z), so Z is the mRNA state. The kinetic layer sets α = γμ̃(Z), so Z is the transcription state. A Brownian Z cannot be both, because s is C¹ in time.
  - Because Z is fitted from s, it already absorbs the spliced lag. τ_s therefore has no signal.
  - The certificate |Lμ̃|τ > noise selects exactly the genes where the static decoder is misspecified.
- **(c) β is not identifiable.** Exactly, βu = γs + ṡ, so u_obs = (c^u/β)(γs + ṡ). Only γ and the product c^u/β are identified; β is confounded with priming/length bias.
- **Fix.** Claim γ_g in 1/h only, using exact path integration. Define Z as the transcriptional state, with spliced counts decoded from the path-integrated s̄. Delete the claim for β in 1/h. Verify that u/s counts exist for GSE131351 and for the atlas-wide MouseGastrulationData before planning on them; both are unverified.

**FATAL-5. E1 cannot estimate σ.**
- E1 sets "σ² from quadratic variation" of bridges **sampled from the linearised-OU bridges of the reference with σ_ref**. Their quadratic variation is exactly σ_ref², so the update is the identity map and σ stays at its M0 initial value.
- The E1-vs-E2 "estimator invariance" test on σ is therefore meaningless.
- Separately, the true θ being a fixed point of the scheme θ = Proj(SB(R_θ, p̂)) does not make it the only fixed point.
- **Fix.** Estimate σ (and the drift) by MLE of the **endpoint** transition kernel on the coupled pairs, as APPEX does, or leave σ to E2. Run multi-start fixed-point diagnostics.

---

## (3) Serious flaws, with concrete fixes

**S1. The primary metric P1a is circular and depends on the estimator.**
- Held-out "posterior samples" depend on the prior in q(Z | x, T). The held-out stage has no per-stage prior, and the encoder never saw T\*.
- If the model's own prediction is used as the prior (stage 3), the score is circular.
- The design says posterior aggregates are avoided, but the ELBO-optimal prior *is* the aggregate posterior (Hoffman & Johnson 2016). The 20-GMM "deconvolved marginal" is the GMM projection of that same inflated object.
- A KDE in 8–16 dimensions from 4096 particles is biased in its bandwidth. Choosing the bandwidth per arm changes the ranking, and log scores reward overdispersion.
- **Fix.**
  - Primary score: held-out **count** log predictive density, log ∫NB(x | ℓμ̃(Z), φ) p_arm(Z | T\*) dZ, computed by importance sampling with an encoder proposal that has no T input.
  - Fit p_arm with the same density family and hyperparameters for every arm (e.g. a 20-GMM fitted to each arm's particles).
  - Fit the stage-1 prior by marginal likelihood (IWAE or EM with the decoder fixed), and check deconvolution bias on simulators where the latent law is known.

**S2. A new trap replaces mean interpolation.**
- **Metastable regime.** When barriers are high (ΔΨ/σ² ≫ 1), p_t ≈ Σ_w π_w(t)ρ_w: fixed within-well shapes with weights following a master equation. Mixing the neighbouring snapshots is then exact up to O(Δt²·π̈_w).
  - This is exactly where the landscape and Kramers reading is valid.
  - So R1 is structurally hard to pass where the interpretation holds, and K4 can fire even when the dynamics are correct.
- **Low-barrier regime.** Transit and translation dominate, so McCann or mean interpolation is near-optimal for short gaps.
- **sci-fate (D1)** is a synchronous step stimulus. The GR response gives near-translates (cfm_grn_v2's F11 regime), and the other main axis is the cell cycle, which M2 cannot represent (S3). Expect R1 to tie or fail on the 4 h and 6 h stages.
- **Fix.**
  - Add the cfm_grn_v2-style **class ceiling**: fit PALM including T\* and score at T\*. If it cannot beat the controls, stop.
  - Make the large-gap skip-k designs (D3: gaps of 2–4 days) and the terminal forecast primary, not single interior stages.
  - Evaluate K4 only on those designs.
  - Add "novel-state mass", the share of held-out cells with low density under the mixture, as a pre-registered stratum.

**S3. M2 cannot represent oscillation.**
- Along the deterministic flow, dΨ/dt = −∇Ψᵀ(I−J)∇Ψ = −|∇Ψ|² ≤ 0. So there are no periodic orbits.
- The stationary current pJ∇Ψ vanishes at every critical point, so any curl acts only on slopes.
- The cell cycle (sci-fate, RPE1, LARRY progenitors) is badly misspecified as a result.
- g ∝ a_cc ties growth to G2/M phase, not to the cycling state.
- D8 (RPE1, stationary) tests only the test's size, i.e. how often C-σ falsely passes when nothing changes; it is not a negative control for J.
- **Fix.** Model the cycle as a separate phase variable outside Ψ, dθ = ω(Z)dt + noise, decoded through cycle genes. Base growth on a cycling/non-cycling indicator. Use a rotationally symmetric simulator as the negative control for J.

**S4. Recovery on "different-family" simulators cannot test Ψ, J or σ.**
- In dyngen, SERGIO or scMultiSim, program-level Ψ, J and σ have no ground truth. Only derived quantities do: terminal states, commitment times and γ. The A_eff "AUPRC under degree-matched negatives" test is undefined there.
- A_eff is also dense by construction (full Q and J), and degree matching on about 10 nodes is meaningless.
- **Fix: projected-truth simulators.** Simulate known program-level dynamics *outside* M2 (non-gradient nonlinear drift, a limit cycle, anisotropic multiplicative noise, slow time-variation, a hidden variable). Decode through a *different* decoder (Hill/bursty, anchor leakage, 30% wrong mask genes).
- Keep dyngen and SERGIO for derived quantities. Note that dyngen has no capture bias, so β will look recoverable in simulation when it is not on real data.

**S5. Batch confounded with time.**
- A pure batch shift passes C-σ and is absorbed as drift and σ. A permutation p < 0.01 on about 1000 cells per stage has power ≈ 1, so the certificate is uninformative.
- **Fix.**
  - Compare each change against the **replicate or same-time-library floor**, not a bootstrap.
  - Add a nuisance for technical genes (mitochondrial, ribosomal, dissociation, ambient).
  - Favour pooled-library designs. sci-fate's combinatorial indexing likely pools timepoints in one library; verify this in its methods.

**S6. Kramers and commitment rates.**
- The Bouchet–Reygner formula does apply: ℓ = −J∇Ψ is orthogonal to ∇Ψ and ∇·(ℓe^{−2Ψ/σ²}) = 0.
- But the prefactor λ\*_s(−(I−J)H_s) depends on J, which is unidentified in nonlinear M2.
- Saddles sit in low-density regions by definition, and the rate scales as e^{−2ΔΨ/σ²}. The factor-1.5 check against MFPT validates only the asymptotic approximation, not the estimate.
- **Fix.** Report simulated MFPTs with bootstrap CIs over refits, claim only orders of magnitude, and validate against LARRY.

**S7. Algebra errors.**
- **Bistability condition.** Ψ_r'' = Q_rr − s_r·sech²(z_r − c_r), so the potential is non-convex iff **s_r > Q_rr**, not Q_rr·s_r > 1. The Badalotti analogue is the gain/decay ratio s_r/Q_rr.
- **Persistence is not the s → ∞ limit.** Large s pushes the wells out to |z − c| ≈ s/Q, where cells collapse quickly. Persistence corresponds to Ψ ≈ const with σ → 0.
- The coupling term's lack of translation equivariance is covered in FATAL-3.

**S8. Load-bearing assumptions that are never tested.**
- **Scalar σ in log-program space** is required by the Gibbs property and by Guan. Test it by fitting M0 with a full H.
- **Time-homogeneity (L7)** is the central assumption, and no certificate checks it. Add C-H: fit on early intervals and on late intervals and compare the parameters with bootstrap CIs.

**S9. Count-space generation (G) will be poor, and the cluster baseline is missing.**
- HVGs outside every program mask get μ̃ = b_g, a constant. Count C2ST will be ≈ 1 for every arm, so the gap-closed ratio becomes unstable (its denominator is near 0).
- **Fix.**
  - Add a mandatory count-space **NB mixture of clusters with time-interpolated weights** (Chronocell's degenerate case). It will probably win on G, so frame PALM as "within the floor on G, while adding I".
  - Add a declared static, non-dynamic residual decoder for genes outside the masks.
  - Guard gap-closed so it is not reported when D_control − D_floor falls below the floor's standard error.

**S10. Statistical unit.** R1 uses seeds × stages, which repeats cfm_grn_v2 risk 8. With 2–3 held-out stages, "≥ 3/4 of stages" means all of them. Use a cell bootstrap within each stage plus a stage-level sign test, and set the P1b margin from the replicate floor rather than a fixed 0.02.

**S11. External-validation details.**
- The sci-fate new-RNA prediction needs a nuisance for T>C conversion efficiency.
- It is dominated by γ, so a constant-α model nearly matches it. Add an α-dynamics ablation.
- Half-life atlases from other cell types or species (mESC vs erythroid; K562 vs A549) cap correlation at the agreement between atlases, so report that ceiling.
- LARRY has only 3 marginals, so no theorem applies and there is no held-out stage.

**S12. Scope and novelty.**
- The model ladder, two estimators, kinetics, growth, 4+ simulators, 8+ datasets and a theory track are too much for one person in 6–12 months.
- **T1 / novelty #2 is known.** It is the Kwon–Ao–Thouless (PNAS 2005) decomposition. Constant-skew J∇Ψ keeping the Gibbs law is the standard *irreversible Langevin* construction (Hwang–Hwang-Ma–Sheu; Rey-Bellet–Spiliopoulos). Wang–Xu–Wang (PNAS 2008) gives the landscape-plus-flux view in cell biology.

---

## (4) Claims to downgrade to "assumption" or "conjecture"

- **(Ψ, σ) identified** → only in the continuous-time limit, for M1 with g = const. With K snapshots it is an assumption plus empirical profile likelihood.
- **Ψ (not just J) identified in nonlinear M2** → conjecture. Attractors, saddles and ΔΨ/σ² are conditional on M1, g = const, the continuum limit and the decoder fix.
- **Gibbs law and "true quasi-potential"** → only for g ≡ const and scalar σ.
- **Switching rates in 1/h** → depend on small noise, J and growth. Identified only together with growth.
- **β_g in 1/h** → not identified. **γ_g** → identified only for genes with strong dynamics, given an identified drift time scale and exact path integration.
- **Latent marginals and W identified (T2)** → a conjecture until a proof covers the offset b, NB φ and the free programs.
- **Program names = biological activity** → an assumption about the mask. Scalar σ, Markov dynamics in k programs, time-homogeneity, and exclusive anchors are assumptions too.
- **E1 = E2 agreement** → evidence of optimisation consistency, not identifiability.
- **s_r as a commitment index** → identified only as a function, not as a parameter, because s_r trades off with Q_rr and c_r on the data support.
- **"Escapes the mean-interpolation trap"** → only the metric changes; the strong baseline remains.
- **Relaxation-census ARI** → near-trivial, since wells sit at dense late-stage regions by construction. Only a forecast of unseen terminal states counts.
- **Novelty item 2, and the probabilities** → lower "presentable positive in 12 months" to about 0.35–0.45 after the fixes, and the full triple to ≤ 0.1.

---

## (5) What is strong and must be kept

- **Correct mathematics.**
  - The Gibbs stationary law for constant skew J and scalar σ (tr(J∇²Ψ) = 0 and ∇Ψᵀ J∇Ψ = 0).
  - Critical points coincide with fixed points; minima are stable, and Ψ is a Lyapunov function.
  - The M0 bijection: Q = (σ²/2)Σ∞⁻¹, J = I + AQ⁻¹, and J + Jᵀ = 0 checks out.
  - Bouchet–Reygner applies to this class.
- **Design ideas.**
  - The nested ladder M0 ⊂ M1 ⊂ M2 ⊂ M3, with M4 as an uninterpreted upper bound, which prices the cost of the restriction.
  - Reporting equivalence classes by profiling the unidentified coordinates (‖J‖, κ, δ).
  - A raw-UMI NB likelihood with a library model per batch.
  - Nonnegative anchored programs, once the offset is fixed externally.
- **Protocol.**
  - Controls including mixture-of-neighbours, density interpolation, random anchors and the shuffled twin.
  - The decoder floor and oracle latent; ablations that must hurt.
  - Pre-registered R1–R5 and K1–K5, guarded stages, the SHA freeze.
  - Extrapolation tagging and the "never claimed" list.
- **Data choices.**
  - Extra data as the source of interpretation: sci-fate 4sU withheld from the fit, LARRY clones, chromVAR at node level.
  - sci-fate's built-in out-of-equilibrium start (a dex step from the untreated steady state) and its probable pooled library.
  - Schiebinger's dense time course as the main bet for large-gap tests.
- **Suggested minimal core.**
  1. M0/M1 with g = const and the decoder fixed.
  2. The class-ceiling diagnostic first.
  3. sci-fate for γ with exact path integration.
  4. Schiebinger for large gaps and the terminal forecast.
  5. LARRY only for fate AUROC with subsampling-aware growth.
  6. Projected-truth simulators for identifiability.

Sources:
- [Guan et al., Gradient-flow SDEs, arXiv 2505.21770v3 (AISTATS 2026)](https://arxiv.org/html/2505.21770v3)
- [Guan et al., arXiv 2410.22729 (abstract, v4)](https://arxiv.org/abs/2410.22729)
- [Guan et al., arXiv 2410.22729v3 (html)](https://arxiv.org/html/2410.22729v3)
- [Irreversible perturbations of Langevin dynamics, arXiv 2108.08247](https://arxiv.org/pdf/2108.08247)
- [Optimizing irreversible perturbations of ULA, arXiv 2606.05324](https://arxiv.org/pdf/2606.05324)
- [sci-fate (Cao et al., Nat Biotech 2020) PDF](https://krishna.gs.washington.edu/documents/cao_naturebio_2020.pdf)
- [sci-fate summary (RNA-Seq Blog)](https://www.rna-seqblog.com/sci-fate-characterizes-the-dynamics-of-gene-expression-in-single-cells/)
- Dossier: C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md

# FEASIBILITY CRITIC
# Adversarial review of Angle B, "PALM" (lens: feasibility, data, novelty/scoop, payoff)

## (1) Scores

| | As written | After the cuts in §6 |
|---|---|---|
| **Feasibility** | **3.5 / 10** | **6.5 / 10** |
| **Novelty** | **4 / 10** | **5.5 / 10** |

**Why feasibility is low as written:**
- The plan has 9 datasets, 2 estimators, 4 theory deliverables, a kinetic layer, growth, an M3 tilt and about 8 re-run baselines. That is 2–3 PhD-years of work, not 6–12 months.
- Compute (500–700 A100-h) is not the bottleneck. Engineering and debugging are: cfm_grn_v2 went through 12 versions in one week, and Job E took 3–5 h per dataset.

**Why novelty is low:**
- Novelty rests on a conjunction of parts that each already exist.
- Two of the six "genuinely new" items are classical results (F2 below).
- The open theory item (C1) is now being worked on by others (§4).

**Realistic probabilities (mine, not the designer's):**

| Outcome | As written | After cuts |
|---|---|---|
| Presentable positive within 12 months | ≈ 0.30 | ≈ 0.45 |
| Full triple (generation + rates + fates) | ≈ 0.05 | — |
| Fallback paper on certificates/benchmark | ≈ 0.6–0.7 | — |

The designer's figures were 0.6, 0.15–0.2 and 0.85.

---

## (2) Fatal problems

**F1. Ψ is a strict Lyapunov function, so M2 cannot oscillate, and the cell cycle sits inside Z.**
- Along the deterministic part: dΨ/dt = −∇Ψᵀ(I−J)∇Ψ = −|∇Ψ|², because ∇ΨᵀJ∇Ψ = 0. So there are no limit cycles and no sustained rotation. J∇Ψ vanishes at every critical point. The "non-reversible" class only bends transients.
- The design uses E2F/G2M as named programs and a_cc in growth. Yet the cell cycle is the dominant non-gradient motion in D1 (A549), D2 (LARRY), D3 (MEFs) and D8 (RPE1).
- An asynchronously cycling population is close to ring-symmetric in the E2F/G2M plane. That is exactly the design's own counterexample, so J is invisible there.
- On a ring well, ∇Ψ is radial and zero along the ring, so circulation can only be driven by noise. The fit will therefore inflate σ to explain cycling. That inflated σ contaminates ΔΨ/σ², the Kramers rates (exponential in it) and the kinetic lags.
- **Fix:** take the cell cycle out of Z.
  - Option 1: model a phase nuisance θ ∈ S¹ with dθ = ω dt + σ_θ dW, where ω = 2π/T_doubling from the literature. It feeds only growth and the decoder offset.
  - Option 2: exclude the cycle programs and absorb the phase into b_g.
  - Either way, never claim anything about cycle dynamics.

**F2. Novelty item 2 and theory deliverable T1 are classical.**
- The M0 bijection is the Kwon–Ao–Thouless linear decomposition: A = −(D+Q)U with skew Q (PNAS 2005, 102:13029) [from memory].
- The Gibbs-preserving form −(I−J)∇Ψ is Ao's SDE decomposition with constant Q. It is also the standard non-reversible Langevin sampler: Hwang, Hwang-Ma & Sheu 1993/2005; Lelièvre, Nier & Pavliotis 2013 [from memory].
- Bouchet & Reygner (Ann. Henri Poincaré 2016) cover exactly the class b = −∇V + l with l·∇V = 0 and ∇·l = 0. That is M2 [from memory].
- In single-cell work, the "landscape + curl flux" framing has a 2025 review (JACS Au) and earlier preprints [verified].
- Conjecture C1 (nonlinear J) now overlaps with Nguyen & Malysheva, arXiv 2609.35060 (28 Sep 2026) [verified; summary of the arXiv listing only — they assume known diffusion]:
  - drift is constrained only through the score-weighted divergence ∇·F + F·∇log ρ;
  - a ρ-solenoidal gauge is invisible at any single time;
  - with generic Gaussian shape variation, K ≥ m time points (m = intrinsic dimension) remove every polynomial gauge direction.
- **Fix:** delete items 2 and T1 from the novelty list, cite them as tools, and present C1 only as a numerical appendix positioned against 2609.35060.

**F3. The kinetic layer is not identified as staged, and the D1 rate validation cannot run as written.**
- *Inconsistency.* Stage 1 fits μ̃(Z) to the observed counts, which are mostly spliced. Stage 2b then asserts s̄ = μ̃ − τ_s·Lμ̃. Both cannot hold.
  - If μ̃ is fitted to s, any s-lag is absorbed into W and Z, so τ_s is a gauge.
  - Z itself is inferred from the same mRNA, so a lag "between a program and its own mRNA" can be traded against a time-shift of Z.
  - Only the u–s offset is identifiable: from ū ∝ μ̃ − τ_u·Lμ̃ and s̄ = μ̃ − τ_s·Lμ̃, one gets γ = 1/(τ_s − τ_u). And only if the decoder is fitted *through* the kinetic map, so μ̃ is the latent target α/γ, never in a static stage 1.
- *Data.* sci-fate GSE131351 is documented as new vs total (4sU) counts. Spliced/unspliced layers are **unverified** (GEO was behind a CAPTCHA). If they are absent:
  - re-quantifying from SRA FASTQs needs a download plus permission;
  - otherwise the only rate signal is the new RNA, which is the held-out validation set. That is circular.
- *Bar too low.* The trivial control "constant γ" has Spearman 0 by construction, so R3 reduces to Spearman > 0. The real competitor is the steady-state u/s ratio ranking plus one global time scale. PALM's per-gene ranking comes from that same u/s ratio (c^u·γ/β). Its only new content is the **absolute unit** (1/h) taken from the snapshot clock.
- **Fix:**
  1. Confirm the D1 layers first.
  2. Fit (u, s) jointly inside the decoder.
  3. Claim only the time unit: median half-life within ×2 of labelling **in the same cells**. Control: Chronocell process time rescaled by the experiment duration.
- Validating mESC SLAM-seq half-lives against E7–E8.5 erythroid cells crosses cell types and is weak evidence.

**F4. The flagship certificates are partly vacuous, so novelty item 1 fails as written.**
- **C-W is tautological.** W_gr = 0 off the mask by construction, so "off-program loading < 5%" always passes.
  - Fix: fit a relaxed decoder (L1 on off-mask loadings) and require anchor off-mask mass below 5%. Add a residual conditional-independence test on anchor genes given Z.
- **C-σ passes on batch shifts.** Each timepoint is its own batch (L13), so "consecutive marginals differ beyond the floor" is satisfied by batch effects alone.
  - Fix: require a batch-matched floor (replicates, multiplexed stages); otherwise mark the certificate inconclusive.
- **E1-vs-E2 agreement** tests optimisation, not identifiability. Both use the same p̂ and the same model class.
- **The theorems assume exact Markov marginals of the latent process.** Here Z is an 8–16-D projection of a ~20k-D state, which is non-Markov (L2, L12). It is deconvolved by amortised VI with a 20-component full-covariance GMM per stage: 20·(16+136)+19 ≈ 3,059 parameters per stage, against roughly 1,000 sci-fate cells per stage (L14).
  - "Declared identifiable" is therefore empirical (simulator recovery), not theorem-backed. Say so explicitly.

---

## (3) Serious problems, with fixes

1. **Bistability condition is wrong.**
   - At z = c (with m = c), Ψ″ = Q_rr − s_r·sech²(0) = Q_rr − s_r. Bistability needs **s_r > Q_rr**, not Q_rr·s_r > 1.
   - s → ∞ freezes which basin a cell is in. It is **not** the persistence control: cells still relax within their basin. Persistence is Ψ flat and σ → 0.
2. **Primary metric P1a is broken.**
   - q(Z | x, T) is undefined at a held-out T*.
   - A KDE in 16-D on 4,096 particles is dominated by the bandwidth choice.
   - Latent scores cannot be compared with gene-space arms (cfm_grn_v2, [SF]²M).
   - **Fix:** use the held-out count-space marginal log-likelihood log ∫p(x|Z) p_model(Z|T*) dZ, by importance sampling with a T-free encoder. Report it in nats/cell, relative to the decoder floor and the oracle latent.
3. **Circularity in fate and composition checks.** PanglaoDB marker programs define the latent axes, and annotations are made from the same markers. So the relaxation-census ARI and the P1b composition score are partly guaranteed.
   - **Fix:** train the composition classifier on non-program genes only. Score attractors against clones, not annotations.
4. **E1's transition kernel is unreliable.**
   - Local OU linearisation over Δt fails when cells cross barriers (Schiebinger 0.5 d; sci-fate 2 h).
   - 8 particles per source plus a Gaussian KDE in 16-D is very noisy.
   - **Fix:** use closed-form APPEX for M0 and E2 for M1. Drop E1.
5. **Simulators of a different family have no named programs.** dyngen, SERGIO and scMultiSim produce no MSigDB-like anchors, so the decoder and C-W cannot be tested on them as stated, and K2 is untestable for the decoder.
   - **Fix:** plant anchored modules (known gene-to-program supports with deliberate cross-talk) in your own bursting Hill-GRN CLE and in dyngen.
6. **Growth adds no novelty.** It is WOT's signature-based growth. A cycle-program score is not a proliferation rate (G1 cells score low).
   - LARRY clone sizes fix κ and δ only if the d4/d6 sampling fractions and the re-plating are modelled (moslin and Ventre do this).
   - **Fix:** treat growth as a nuisance with a sensitivity band and make no κ/δ claims.
7. **LARRY (R4) is low-payoff.**
   - It has 3 marginals, the minimum in Cor 3.3, and that corollary covers only countable candidate sets.
   - Fate predictability is capped by hidden heritable variables. Published baselines cluster tightly: WOT 0.459, moscot 0.462, scDiffEq 0.463 (Pearson, Chreode benchmark, arXiv 2605.28111) [verified via search snippet].
   - 8–16 named programs throw away fate-predictive variance, so PALM will likely sit below baselines that use full PCA.
   - **Fix:** demote it to a stretch goal.
8. **Isotropic σ and time-homogeneity.**
   - Ao's form −(D+Q)∇Ψ keeps the Gibbs law with anisotropic constant D, but Guan's AISTATS theorem covers scalar σ only.
   - The embryo (D4) is not time-homogeneous.
   - A549 under a dex step is fine as M2 for t > 0.
9. **sci-fate depth and size.** About 1k cells per stage (verified: 1,054 / 1,049 / 949 / 1,262 / 1,041 / 1,325). sci-RNA-seq depth is low [unverified]. The latent dimension must be ≤ 6 there.
   - A possibly better dataset is the sci-FATE2 ESC → neural-tube labelling time course (Maizels, Snell & Briscoe; about 45k cells). Its accession and timepoints are **unverified**: GSE236512/7/8 appeared in search results but the mapping was not confirmed. Velvet is the natural baseline there.
10. **C-J should be quantitative, not a binary test.** Near-symmetric starting distributions give weak, not absent, identifiability. Report the profile-likelihood width of ‖J‖ or the Fisher information.
11. **Name collision.** "PaLM" is a well-known Google language model. Rename the method.

---

## (4) Scoop risks

**Verified this run:**

| Work | What it does | Risk |
|---|---|---|
| Guan, Atanackovic, Neklyudov, arXiv 2605.08550 (ICML 2026 spotlight) | Wasserstein Lagrangian mechanics beyond gradient flows (captures periodicity); embryonic development; forecast and interpolation. Same group as APPEX/nn-APPEX, so the most likely to scoop a program-space identifiable SDE. | **HIGH** |
| Nguyen & Malysheva, arXiv 2609.35060 (28 Sep 2026) | Identifiability of non-gradient dynamics from snapshots (gauge/kernel results above). Takes over the C1/theory angle. | **HIGH** for theory |
| CardamomOT (Maugé & Ventre, bioRxiv 10.64898/2026.03.31.715390, Apr 2026) | Mechanistic GRN model fitted to time-stamped snapshots, using exact time labels and literature priors on kinetic rates inside an OT framework; generative; predicts unseen perturbations. Takes the "mechanistic + real-time + generative" corner. | **HIGH** |
| nn-APPEX (Guan et al., AISTATS 2026) | Already the joint drift+diffusion SB-EM for gradient SDEs. E1 is their estimator. | — |
| Velvet / velvetSDE (Maizels, Snell, Briscoe; Cell Systems per search; bioRxiv 2023.07.06.547989) | Labelling time course plus a neural SDE generating trajectory distributions. Its drift is a black box in a VAE latent, which is the opening for PALM. | **MED-HIGH** |
| Landscape–flux from single-cell data (JACS Au 2025 review; bioRxiv 2023.08.01.551525) | Gibbs-potential + curl framing. | prior art |
| SDEvelo (Nat Commun) | Multivariate SDE kinetics with a shared latent time. | prior art |
| Chreode, arXiv 2605.28111 | LARRY benchmark numbers (used in §3.7). | — |
| PFM, arXiv 2604.25062 | Verified by the designer, not re-checked here. | **HIGH** |

**Unverified (from memory, or title seen only):**
- Kwon–Ao–Thouless 2005; Ao 2004; Hwang et al.; Lelièvre–Nier–Pavliotis 2013; Bouchet–Reygner 2016; Landim–Mariani–Seo 2019.
- **expiMap** (Lotfollahi et al., Nat Cell Biol 2023) and **Spectra** (Kunes et al., Nat Biotech) already use gene-set-masked or anchored program decoders, so item 3 is only incremental. Lotfollahi co-authored GGE with Liò, so expect a reviewer to know this.
- TopicVelo (Gao, Vaikuntanathan, Riesenfeld): programs plus transient stochastic kinetics.
- D-SPIN: spin networks over gene programs.
- "Wasserstein Residuals", arXiv 2607.04738 (title only).
- arXiv 2505.11460, "Mechanistic inference of stochastic gene expression from structured single-cell data" (title only).

**Likely reviewer reactions:**
- *NeurIPS/ICML:* "A combination of known pieces. The theory is classical (Ao, Bouchet–Reygner) or conjectural, and C1 is being addressed elsewhere. Experiments are low-dimensional."
- *Nat Methods / Cell Systems:* "Why infer rates from snapshots when labelling measures them (Dynamo, velvet)? Attractors in marker-program space are circular. Where is the new biology?"
- The only answer that survives both: **a snapshot clock gives correct absolute dynamics where no labelling exists, shown on systems where the labelling was withheld, with certificates that correctly refuse to answer when the conditions fail.**

---

## (5) The strongest defensible publishable claim

> For time courses that start out of equilibrium, a **linear-OU or gradient SDE in a small set (k ≤ 6–8) of externally anchored, non-negative programs**, fitted to raw-count snapshots at real capture times:
> 1. recovers diffusion level and landscape minima on misspecified bursting-count simulators with planted modules;
> 2. comes with per-dataset diagnostics that correctly **refuse** in three cases: stationary RPE1, near-symmetric starting populations, and batch-only differences;
> 3. generates held-out-stage populations with **calibrated spread and composition** (var_ratio in [0.8, 1.25], 90% coverage in [0.85, 0.95]), where deterministic flows and snapshot interpolants are miscalibrated, while matching them on the centroid.
>
> If a labelled time course passes, add:
>
> 4. the snapshot clock alone fixes the **absolute time unit** of program and RNA turnover, matching withheld 4sU data in the same cells (median half-life within ×2).

- **Why claim (3) is the right target:** it is the one generation result that cfm_grn_v2 showed a deterministic model cannot get (F4 there), and it does not compete on a metric dominated by the centroid.
- **Why the M0 graph claim is the right Liò hook:** the sparsity pattern of A in M0 is an **identifiable program-level causal graph** (Guan, arXiv 2410.22729) when the starting population is asymmetric. That is a graph claim that is honestly identified.
- **Positioning:** "When can you trust dynamics learned from snapshots?" The Dileo sampler-artefact audit and the cfm_grn_v2 analysis showing the mean-interpolation ceiling count as evidence of the student's rigour.
- **Do not pitch:** commitment or Kramers rates, nonlinear curl, "attractor = cell type", or the triple validation.

---

## (6) Cuts that make it doable

1. **Model.**
   - Keep M0 (OU with curl; APPEX theory is exact; Q and J via Kwon–Ao–Thouless) and M1 (nonlinear gradient; nn-APPEX/Guan theory).
   - Drop nonlinear J, the K couplings, the Kramers formula (use simulated mean first-passage times if needed) and C1 (appendix only).
   - Keep M3 only for Schiebinger's day-8 switch.
2. **Cell cycle** goes into a phase nuisance (F1).
3. **Decoder.**
   - k ≤ 6–8 named programs plus ≤ 1 free; per-stage prior a tied or diagonal GMM with ≤ 5 components; encoder without T.
   - Use the real C-W (relaxed-mask test plus residual-independence test).
4. **One estimator per class:** closed-form APPEX for M0, Sinkhorn marginal matching through the SDE (E2) for M1. Drop E1, the invariance test and Stage 3.
5. **Kinetics.** Either fit them jointly in the decoder and claim only the absolute unit, or drop the layer.
   - Before anything else, check whether sci-fate has s/u layers. If not, ask the user for permission to re-quantify, or switch to sci-FATE2 if its accession and timepoints verify.
6. **Growth** becomes a nuisance with a sensitivity band; no κ/δ claims.
7. **Datasets.**

   | Role | Datasets |
   |---|---|
   | Core | D0 (own bursting Hill-GRN with planted programs, plus dyngen with planted modules); D8 RPE1 negative control; D1 (or sci-FATE2); D3 Schiebinger (needs permission) |
   | Confirmation | D5 Veres, frozen beforehand (has a replicate floor) |
   | Secondary | D4 erythroid kinetics (s/u layers **verified** through MouseGastrulationData `get.spliced=TRUE`) |
   | Cut | LARRY (stretch only), D7, D9 |

8. **Metrics.**
   - Primary: held-out count marginal log-likelihood; composition total variation from a non-program-gene classifier; var_ratio and coverage.
   - Mandatory controls: mixture-of-neighbouring-snapshots and static density interpolation.
   - Energy distance with its mean/trace split as a secondary metric only.
9. **Timeline.**

   | Months | Work |
   |---|---|
   | 0–2 | Simulators plus decoder |
   | 2–4 | Recovery on the simulators and the RPE1 negative control (the first unit that can be written up as a certificates paper) |
   | 4–6 | D1 and D3; K5 gate at month 6 |
   | 6–8 | Frozen confirmation on Veres |
   | 8–12 | Writing |

   Pack the fits sequentially inside single HPC jobs (about 150–250 A100-h in total).
10. **Rename** the method.

Sources:
- [Beyond Gradient Flow, arXiv 2609.35060](https://arxiv.org/abs/2609.35060)
- [Learning Population Mechanics from Temporal Snapshots, arXiv 2605.08550](https://arxiv.org/abs/2605.08550)
- [Gradient-flow SDEs, arXiv 2505.21770](https://arxiv.org/pdf/2505.21770)
- [Wasserstein Residuals, arXiv 2607.04738](https://arxiv.org/pdf/2607.04738)
- [Identifying Drift, Diffusion, Causal Structure, arXiv 2410.22729](https://arxiv.org/abs/2410.22729)
- [CardamomOT, bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.31.715390v1)
- [Velvet, bioRxiv 2023.07.06.547989](https://www.biorxiv.org/content/10.1101/2023.07.06.547989.full.pdf)
- [Velvet/Crick figshare](https://crick.figshare.com/articles/journal_contribution/Reconstructing_developmental_trajectories_using_latent_dynamical_systems_and_time-resolved_transcriptomics/25846864)
- [Chreode, arXiv 2605.28111](https://arxiv.org/pdf/2605.28111)
- [Landscape–flux review, JACS Au 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12381741/)
- [Landscape–flux cell cycle, bioRxiv](https://www.biorxiv.org/content/10.1101/2023.08.01.551525v2.full)
- [EPR-Net](https://arxiv.org/pdf/2301.01946)
- [SDEvelo](https://www.nature.com/articles/s41467-024-55146-5)
- [MouseGastrulationData vignette (get.spliced)](https://bioconductor.statistik.tu-dortmund.de/packages/3.20/data/experiment/vignettes/MouseGastrulationData/inst/doc/MouseGastrulationData.html)
- [sci-fate A549 (cell counts)](https://scmer.readthedocs.io/en/latest/a549.html)
- [PFM, arXiv 2604.25062](https://arxiv.org/pdf/2604.25062)
- [Mechanistic stochastic expression, arXiv 2505.11460](https://arxiv.org/pdf/2505.11460)
- [Chronocell, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11760028/)
- [AISTATS 2026 poster](https://virtual.aistats.org/virtual/2026/poster/10812)
- [Maizels / Beddington Medal page](https://bsdb.org/page/3/)
- [Briscoe lab publications](https://briscoelab.org/publications/)

Dossier read in full: C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\dossiers_all.md