# temporal-gnn

Research code and notes for modelling the time evolution of cell populations from single-cell data
(PhD project, Gabriele Bambini). Private working repository: code, pre-registrations, decision log and small
result files only. Raw and derived data, model checkpoints and third-party code are not included.

## What is here

| Folder | Content |
|---|---|
| `tack/` | Main line of work since October 2026. `tack/hpc/`: forecasters and runners (HPC, Slurm); `tack/tests/`; `tack/prereg/`: frozen pre-registrations; `tack/dileo_bench/`: audit and design reports for the Dileo & Sottoriva temporal-GRN benchmark; `tack/data_cards/`; `tack/results/`: small result files |
| `cfm_grn/`, `cfm_grn_v2/` | Conditional flow-matching models for gene-regulatory dynamics (earlier line of work) |
| `docs/` | Analyses and the decision log `docs/superpowers/decisions.md` (every experiment, its criteria and its outcome, in Italian) |
| `.projectos/` | Shared project state for agents and humans (purpose, map, state, decisions) |
| `scripts/` | Small utilities |

## Main threads

- **Dileo & Sottoriva benchmark (Task 2, gene-expression change).** Their original code
  (https://anonymous.4open.science/r/tgl-grn-1CCD) is used untouched as the evaluation harness and is not
  redistributed here. `tack/hpc/dileo_harness.py` runs our forecasters inside their loop;
  `tack/hpc/dileo_corrected.py` reruns their models under corrected protocols; `tack/hpc/dileo_rescore.py`
  scores everything on the same test genes. Results and the protocol audit: `docs/superpowers/decisions.md`
  (rows of 2026-10-22 and later) and `tack/dileo_bench/`.
- **Forecasting real embryonic stages (erythroid lineage, E7.0 to E8.5).** `tack/hpc/ery_prep.py`,
  `ery_prep_interp.py`, `ery_forecast.py`: optimal transport, unbalanced OT with growth, RNA velocity and
  their combinations, all fitted strictly on past stages.
- **TACK** (age-structured population with a kinetic count decoder): `tack/hpc/chrono_*`, `scifate2_*`,
  `sim_recovery.py`, `precision_study.py`.

## Data

Not included. The scripts expect the data on the Bocconi HPC: scVelo `gastrulation_erythroid` and the
Dileo & Sottoriva `GRN/` files (see `tack/data_cards/` and the docstrings of each script).
