# Pre-registration v2: sci-FATE2 "do two independent clocks agree?" (adapter probe, NOT TACK)

Status: v2 DRAFT 2026-10-02 (v1 withdrawn after three adversarial reviews: detection/clock confound, decoder
confound, outcome-based QC, D4/r4 contamination, topology lottery). Frozen when its sha256 is recorded in
docs/superpowers/decisions.md, before any expression-level step. Allowed before freezing: metadata, file and
count integrity checks (tack/hpc/integrity_*.py), dynast source reading.

## 1. One primary question
Fit ONE kinetic decoder on total RNA only and freeze it. Its rates are in process units. Two independent
readings of the hour scale c (hours per process unit) are then compared:
- **clocks from collection times** (no labelled RNA): c_A (age-structured population, the TACK idea),
  c_G (Gaussian capture-time, VeloVAE-like), plus two references: c_U (protocol span) and c_E (published
  half-life);
- **the label clock** c_label: the c at which the frozen decoder best predicts each cell's held-out 4sU-labelled
  RNA.
If the TACK idea is right, c_A falls inside the label clock's uncertainty band and the references do not.

## 2. Data and integrity (stop rules)
- Decoder input: `GSE236512_processed_data_splicing.h5ad` layers `unspliced` (U), `spliced` (S).
- Labels: `GSE236512_processed_data_counting.h5ad` raw layers `new`, `total`. `*_estimated` never used.
- Detection: obs `total_TC_alpha` (dynast `estimate_alpha`: alpha = observed new-read fraction / binomial-mixture
  labelled fraction pi_c, per cell; computed without any kinetic model; read in dynast source 2026-10-02).
- STOP if: obs/var names differ across files (passed 2026-10-02); new + old != total; total(counting) !=
  total(estimate); total < U + S in more than 0.1 % of non-zero entries.
- Ambiguous reads: per-gene share a_g = 1 - sum(U+S)/sum(total) over all cells (integrity step). Genes with
  a_g > 0.3 are dropped; for the others ambiguous reads are modelled as spliced (section 5).
- Integrity results (job 679641, 2026-10-02, `tack/results/scifate2/integrity_counts.json`): ALL PASSED.
  new + old == total exactly; total(counting) == total(estimate) exactly; total >= U + S in every entry (0
  violations); ambiguous share median 0.012, 95th percentile 0.31, 5.2 % of genes (total >= 1,000) above 0.3,
  overall 0.073; 17 cells with NaN alpha/p_c; alpha median 0.30 (1-99 %: 0.074-0.58), varies by group (median
  0.18-0.43); effective K = ln(1-alpha)/ln(1-p_c) median 27.2 (10-90 %: 24.4-29.1), consistent across groups.

## 3. Cells
- Groups used: r1-r3 at D3, D5, D6, D7, D8 (15 groups; T = 72, 120, 144, 168, 192 h). **D4 (r1-r3) and all of
  r4 are excluded entirely** (neither fit nor scored), kept clean for the later GEN test. Consequence, declared:
  c_A rests on the D3 onset and D5-D8 only.
- QC (all cells of the 15 groups): genes and UMIs (from `total`) within the 10th-90th percentiles; mitochondrial
  fraction (`mt-` genes) below the 90th percentile; `total_TC_alpha` above the 10th percentile (detection, not
  outcome; the paper's label-rate filter is NOT used); cells with NaN alpha or p_c excluded; alpha winsorised to
  its 1st-99th percentiles. Scrublet not run.
- Within each group a random 50/50 split (seed 1): fit half (capped at 1,000 cells) and score half.

## 4. Frozen decoder
- Chronocell commit a827f89, model `two_species_ss`, called through its public `fit` (no internal patches).
- Topology fixed a priori to the paper's branching tree T3 = [[0,1,2],[0,1,3],[0,1,4]] (no AIC selection).
- Time grid in day-like units: tau = (0, 2.5, 5), m = 100, `fit_tau=True`, `norm_Q=False`, uniform prior.
- Genes (fit cells, U and S): Chronocell erythroid rules (near-Poissonian extra variance sp, 5 iterations; U_mean >
  0.02, S_mean > 0.1, U and S variance ratios > 1.2, |log(S_mean/U_mean)| < 4); drop Tirosh 2016 cell-cycle genes,
  `mt-` genes and genes with a_g > 0.3; keep at most 150 by spliced variance ratio. STOP if fewer than 50.
- Read depth r: mean of near-Poissonian genes per cell, divided by the fit-cell mean (sparse computation).
- Initialisation: 8 random initialisations (n_init 1, seeds 0-7, epoch 100, run in parallel); keep the best ELBO.

## 5. Label model (for any hour scale c)
- h = L / c with L = 2 h. At process time t (and lineage l) the decoder gives U(t), S(t). Old RNA surviving from
  t0 = max(t - h, 0) (root steady state before 0): old_u = U(t0) e^{-beta h};
  old_s = S(t0) e^{-gamma h} + beta U(t0) phi(h), phi(h) = (e^{-gamma h} - e^{-beta h})/(beta - gamma)
  (evaluated stably). new_u = U(t) - old_u, new_s = S(t) - old_s, clipped to [0, U] and [0, S].
- Ambiguous reads A = a_g/(1-a_g) (U + S), new fraction as spliced. pi_g(l,t) = (new_u + f_s (S + A)) / (U + S + A),
  f_s = new_s / S.
- Cell posterior q_i(l,t): frozen decoder, UNIFORM prior, from the cell's U and S counts (never its labels).
  pi_ig = sum_{l,t} q_i(l,t) pi_g(l,t).
- P(read called new) = clip(alpha_i * pi_ig, 1e-6, 1 - 1e-6).
- Score of cell i: per-read mean log-likelihood sum_g [new_ig log P + (total_ig - new_ig) log(1-P)] / sum_g total_ig.
  Group score = mean over the group's score-half cells; S(c) = equal-weight mean over the 15 groups.

## 6. The clocks
- c_label = argmax of S(c) over 41 log-spaced values in [3, 300] h per unit. Band: 2.5-97.5 % of the argmax over
  2,000 bootstrap resamples of the 15 groups, computed at alpha x 0.8, x 1.0 and x 1.25; the reported band is the
  union of the three.
- c_A: age prior (reservoir point mass at t=0 with logistic weight in T; entered cells with age a = c t, density
  proportional to exp(g a) on [0, T - T0], absorbing at t = 5; onset T0 = 48 + 36 sigmoid(z), i.e. in [48, 84] h;
  uniform over lineages) fitted by maximising the fit cells' total-count marginal likelihood under the frozen
  decoder (lineage-collapsed; Nelder-Mead from the start grid T0 in {60, 72}, c in {12, 24, 48} h, w0 in {-2, 2},
  g = 0, w1 = 0).
- c_G: t ~ Normal(mu0 + s (T - 72)/24, sd^2) truncated to [0, 5], uniform over lineages, same fitting; c_G = 24 / s
  (start grid mu0 in {0, 1}, s in {0.5, 1}, sd in {0.5, 1.5}); if s <= 0, c_G is undefined and reported as such.
- c_U = 24 h per unit (process span 0-5 mapped onto D3-D8, 120 h).
- c_E: median over decoder genes of gamma_unit / c_E = ln 2 / 7.1 h (median mRNA half-life in mouse ES cells,
  Sharova et al. 2009, DNA Research 16:45-58), i.e. c_E = median(gamma_unit) * 7.1 / ln 2.

## 7. Decision rule (applied by code)
- GATE-1: S(c) must peak strictly inside the grid; otherwise UNINFORMATIVE.
- GATE-2: the median unspliced half-life implied by c_A, ln 2 / median(beta_unit / c_A), must be below 1 h;
  otherwise UNINFORMATIVE (decoder's absolute rates implausible).
- POSITIVE: c_A inside the band, and the band excludes c_U and c_E.
- POSITIVE vs VeloVAE-like: in addition c_G is outside the band.
- UNINFORMATIVE: the band contains all four candidates (stated in advance as a real possibility, given the
  detection-clock degeneracy for slow genes).
- NEGATIVE: c_A outside the band while the band excludes at least two of the other three candidates.
- Otherwise INCONCLUSIVE, described as it is.
- Always reported: S(c) profile with c_A, c_G, c_U, c_E marked; S at each candidate; fitted beta and gamma in 1/h
  under c_label and c_A; the band at each alpha scale separately.

## 8. Declared as owed (secondary or deferred)
K-free within-cell multinomial score; Beta-binomial overdispersion; per-gene detection from 3' T content;
factorial swap (posterior source x hour scale); arm-vs-arm log-likelihood contrasts; restricted-permutation
time controls; exclusion of signalling, stress and immediate-early genes; early vs late strata; L = 1.75 h;
fresh-decoder refits. **Simulation recovery (simulate with true c = c_A and c = 2 c_A, run the whole pipeline)
is required before any public claim.**

## 9. Limitations (declared)
Chronocell's decoder (piecewise-constant transcription, given topology, no unspliced-capture parameter Ub) is
not TACK's; medium renewals every 24 h violate cohort invariance; alpha includes a small background term;
4sU at 500 uM may perturb transcription; D3 cells are near the CHIR-to-RA/SAG switch and not at steady state.
