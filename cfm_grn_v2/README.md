# cfm_grn_v2

Original conditional flow matching engine for unpaired single-cell snapshots. Two clocks, never
mixed: `flow_time` (generative path coordinate in [0, 1]) and `biological_time` (measured
hours / days / stages, enters only through the conditioning context). Since v0.1.4 the field can
also be autonomous (`TrainConfig(clock="duration")`): v(x, condition) trained on the velocity
(x1 - x0)/dt and integrated over the biological horizon, with no flow-time input at all.

Plan and rationale: `../docs/superpowers/plans/2026-09-18-rafm-phase1-cfm-engine.md` (the code
blocks tagged `# file:` in that plan are this package; regenerate with
`python ../scripts/materialize_plan.py ../docs/superpowers/plans/2026-09-18-rafm-phase1-cfm-engine.md --root ..`).

```bash
python -m pytest tests -q                    # fast suite
python -m pytest tests -q -m slow            # integration (quick toys run)
PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/quick --quick
PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/phase1_v1
```

Outputs per run: `results.csv` (one row per model × seed × group), `search.csv` (nested grid
search on the validation timepoint only), `report.md`, `verdict.json` (gate G1 per dataset),
`run_meta.json`, `grn_truth.npz` (ground-truth time-varying edge activity for Phase 3).

Controls always reported: persistence, train mean, mean shift, linear map, deterministic MLP,
FreeFM replay, flow matching trained on shuffled biological time, noise floor.

## Quickstart for readers of the 19 Sep 2026 deliverable

```bash
python -m pytest tests -q                         # 89 tests, about 20 s
PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/quick --quick --seeds 2 --search-seeds 1
PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/full --datasets grn --tasks interpolation --seeds 5 --search-seeds 2
PYTHONPATH=src python scripts_dev/ot_ablation.py  # OT vs random pairing on the GRN simulator
PYTHONPATH=src python scripts_dev/ot_crossing.py  # OT vs random pairing on two lineages moving apart
```

Map of the package: `flows/` (paths, couplings incl. exact minibatch OT and OT chaining, ODE solver),
`models/` (condition context, MLP field), `data/` (schema, splits, synthetic toys and the time-varying
GRN simulator, real-data loader, signed regulatory prior), `evaluation/` (U-statistic metrics, permutation
null, controls, report and verdicts), `training/` (config, trainer with EMA/early stopping, nested search,
forecast and interpolation benchmarks), `experiments/run_synthetic.py` (the Phase-1 benchmark CLI).
Report: `DELIVERABLE_2026-09-19_OT_flow_matching.md`.
