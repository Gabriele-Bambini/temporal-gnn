# Purpose - Temporal GNN
<!-- Confirmed by the user 2026-10-02 ("vai", answering the three open decisions); D-001 ACTIVE. -->
> Owner: the user. Agents may draft, but changes need user confirmation and a DECISIONS.md entry.

## Why this project exists
The user's goal (2026-10-01, own words): "creare un sistema che permetta di modellare l'evoluzione nel tempo di una popolazione di cellule in modo generativo e quasi continuo in modo interpretable". Earlier lines (temporal GNN link prediction on aggregate GRNs; black-box flow matching cfm_grn_v2) did not reach it: see docs/ANALISI_E_APPROCCIO_2026-10-01.md.

## Goal (definition of done)
- A count-level generative model of a cell population over REAL time (TACK: age-structured population + kinetic decoder), whose parameters (entry flux, branch fractions, switch ages, degradation rates in 1/h, later named programs) are identifiable under declared assumptions and validated on evidence the fit never saw.
- A preprint: identifiability boundary (P1') + simulators that must fail + one real external validation (held-out 4sU new RNA on sci-FATE2).

## Success criteria (measurable)
- [x] Calibration gate: Chronocell reproduced on erythroid (AP 0.85 vs 0.79), 2026-10-02.
- [x] P1' checked symbolically and on a 5-timepoint simulation, 2026-10-02.
- [ ] sci-FATE2 CLOCK: held-out new-given-total log-likelihood beats uniform, Gaussian capture-time, prior-only and shuffled-T arms (paired CI > 0), rules frozen before any expression-level look.
- [ ] Predicted half-lives agree with labelling-derived ones (Spearman >= 0.5, slope 0.8-1.25).
- [ ] Generation at intermediate real times (D4, D3+5..20 h) non-inferior to the best control.

## Constraints (time, compute, data, tools, style)
- Original code only; external methods (Chronocell, VeloVAE, ...) only as baselines/engines through adapters.
- Bocconi HPC (QoS 1 running / 2 submitted, no arrays); downloads only with the user's permission.
- Pre-registration before test data; mandatory trivial controls; flow time != biological time; splits by whole timepoints.
- Reports to the user in plain Italian.

## Non-goals
- Link prediction on aggregate per-bin GRNs (tgl-grn family, closed).
- More tuning of the black-box flow engine (cfm_grn_v2 frozen at v0.1.10 as a baseline).
- Clock claims from data with one batch per stage or staging noise (e.g. gastrulation erythroid).

## Context (deadline, venue, stakeholders, links)
- PhD application, target supervisor Pietro Lio (Cambridge). Plan horizon 9 months.
- Analysis: docs/ANALISI_E_APPROCCIO_2026-10-01.md; decisions: docs/superpowers/decisions.md; data card: tack/data_cards/.
