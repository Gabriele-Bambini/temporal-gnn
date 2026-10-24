"""Phase 1 benchmark: toy transitions + synthetic time-varying GRN, nested search, controls, report.

    PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/phase1            # full run
    PYTHONPATH=src python -m cfm_grn_v2.experiments.run_synthetic --out outputs/quick --quick     # smoke run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..data.splits import forecast_split, inner_validation_split, replicate_split
from ..data.synthetic_grn import TimeVaryingGRNSimulator
from ..data.synthetic_toys import TOY_NAMES, make_toy
from ..evaluation.metrics import EvalProtocol
from ..evaluation.report import write_report
from ..training.config import TrainConfig, expand_grid, random_subset
from ..training.search import (interpolation_design, interpolation_val_folds, nested_grid_search,
                               run_forecast_benchmark, run_interpolation_benchmark, select_best)
from ..training.trainer import replicate_task, time_task

# Toys have a single transition: delta/absolute conditioning and consecutive/all pairs coincide there.
TOY_GRID = {"hidden": [128, 256], "lr": [1e-3, 3e-4], "sigma_min": [1e-3, 1e-2], "coupling": ["ot", "independent"]}
# GRN: delta conditioning only (absolute time cannot extrapolate to an unseen interval, decision v0.1.1);
# pair_mode is the multi-marginal question (consecutive snapshots vs every ordered pair, horizon in the context).
# path="spline" chains OT across all timepoints (coupling and pair_mode irrelevant) and needs the duration clock.


def _grn_grid(hidden, lr, sigma_min, coupling, pair_mode, clock, n_linear=None, seed=0) -> list:
    """Linear-path configurations (optionally a random subset of n_linear) plus every spline configuration."""
    linear = expand_grid({"hidden": hidden, "lr": lr, "sigma_min": sigma_min, "coupling": coupling,
                          "pair_mode": pair_mode, "clock": clock, "path": ["linear"]})
    if n_linear is not None:
        linear = random_subset(linear, n_linear, seed)
    spline = expand_grid({"hidden": hidden, "lr": lr, "sigma_min": sigma_min, "coupling": ["ot"],
                          "pair_mode": ["consecutive"], "clock": ["duration"], "path": ["spline"]})
    return linear + spline


GRN_GRID = _grn_grid([128, 256], [1e-3], [1e-3, 1e-2], ["ot", "independent"], ["consecutive", "all"],
                     ["flow", "duration"], n_linear=12)                            # 12 of 32 linear + 4 spline
QUICK_TOY_GRID = {"hidden": [64, 128], "lr": [1e-3], "sigma_min": [1e-3], "coupling": ["ot"]}
QUICK_GRN_GRID = _grn_grid([128], [1e-3], [1e-3], ["ot"], ["consecutive", "all"], ["flow", "duration"])   # 4 + 1
VERDICT_MODES = {"toys": "sanity", "grn": "benchmark", "grn_interp": "benchmark"}
GRN_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def toy_setup(name: str, n_per_group: int, seed: int):
    s = make_toy(name, n_per_group, seed=seed)
    sp = replicate_split(s.replicates(), seed=seed)
    fit = s.mask(replicates=sp["fit"])
    val = replicate_task(s, sp["val"][0], 0.0, 1.0)
    test = replicate_task(s, sp["test"][0], 0.0, 1.0)
    return s, fit, [0.0, 1.0], val, test


def grn_setup(n_cells: int, seed: int, n_genes: int, n_regulators: int):
    sim = TimeVaryingGRNSimulator(n_genes=n_genes, n_regulators=n_regulators, seed=seed)
    s, truth = sim.snapshot_set(GRN_TIMES, n_cells_per_cohort=n_cells, replicates=("r1", "r2"), seed=seed)
    sp = forecast_split(GRN_TIMES)
    inner = inner_validation_split(sp["train_times"])
    fit = s.mask(times=inner["fit_times"])
    val = time_task(s, inner["fit_times"][-1], inner["val_times"][0])
    test = time_task(s, sp["train_times"][-1], sp["test_times"][0])
    return s, fit, inner["fit_times"], val, test, truth


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/phase1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--search-seeds", type=int, default=2)
    ap.add_argument("--datasets", default="toys,grn")
    ap.add_argument("--max-configs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default="forecast,interpolation", help="GRN tasks: forecast and/or interpolation")
    ap.add_argument("--interp-times", default="3.0", help="comma-separated inner timepoints held out (one fold each)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    quick = args.quick
    base = TrainConfig(n_iters=1500 if quick else 8000, eval_every=150 if quick else 250, patience=4 if quick else 8,
                       batch_size=128 if quick else 256, depth=2 if quick else 3)
    grids = {"toys": QUICK_TOY_GRID if quick else TOY_GRID, "grn": QUICK_GRN_GRID if quick else GRN_GRID}
    protocol = EvalProtocol(n_samples=300 if quick else 1000, seed=args.seed)
    seeds = list(range(args.seeds))
    search_seeds = list(range(args.search_seeds))
    mlp_kwargs = {"hidden": 64, "n_iters": 300} if quick else {"hidden": 128, "n_iters": 2000}
    meta = {"version": __version__, "args": vars(args), "base_config": base.to_dict(), "grids": grids,
            "protocol": protocol.__dict__, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "best_configs": {}}
    tables, searches, modes = [], [], {}
    datasets = args.datasets.split(",")

    def run(name, s, fit, fit_times, val, test, mode, grid):
        modes[name] = mode
        best, tab = nested_grid_search(s, fit, fit_times, val, base, grid, search_seeds, protocol,
                                       args.max_configs, args.seed)
        tab.insert(0, "dataset", name)
        searches.append(tab)
        meta["best_configs"][name] = best.to_dict()
        print(f"[{name}] best config: {best.to_dict()}", flush=True)
        tables.append(run_forecast_benchmark(s, fit, fit_times, val, test, best, seeds, protocol, name,
                                             mlp_kwargs=mlp_kwargs, verbose=True))

    if "toys" in datasets:
        for toy in TOY_NAMES:
            s, fit, fit_times, val, test = toy_setup(toy, 300 if quick else 1500, args.seed)
            run(s.name, s, fit, fit_times, val, test, VERDICT_MODES["toys"], grids["toys"])
    if "grn" in datasets:
        s, fit, fit_times, val, test, truth = grn_setup(150 if quick else 500, args.seed, 20 if quick else 40,
                                                        6 if quick else 10)
        np.savez(out / "grn_truth.npz", **{k: np.asarray(v) for k, v in truth.items() if k != "schedules"})
        (out / "grn_schedules.json").write_text(json.dumps(truth["schedules"], indent=1), encoding="utf-8")
        tasks = args.tasks.split(",")
        if "forecast" in tasks:
            run(s.name, s, fit, fit_times, val, test, VERDICT_MODES["grn"], grids["grn"])
        if "interpolation" in tasks:
            for t_hold in [float(t) for t in args.interp_times.split(",")]:
                name = f"{s.name}_interp_t{t_hold:g}"
                modes[name] = VERDICT_MODES["grn_interp"]
                fold_tables = []
                for v_time in interpolation_val_folds(GRN_TIMES, t_hold):      # selection averaged over two folds
                    d = interpolation_design(GRN_TIMES, t_hold, v_time)
                    fit_i = s.mask(times=d["fit_times"])
                    val_i = time_task(s, d["v_prev"], d["val_time"])
                    _, tab = nested_grid_search(s, fit_i, d["fit_times"], val_i, base, grids["grn"], search_seeds,
                                                protocol, args.max_configs, args.seed)
                    tab.insert(0, "dataset", name)
                    tab.insert(1, "val_time", v_time)
                    fold_tables.append(tab)
                tab = pd.concat(fold_tables, ignore_index=True)
                searches.append(tab)
                best = select_best(tab, base)
                meta["best_configs"][name] = best.to_dict()
                print(f"[{name}] best config: {best.to_dict()}", flush=True)
                tables.append(run_interpolation_benchmark(s, GRN_TIMES, t_hold, best, seeds, protocol, name,
                                                          mlp_kwargs=mlp_kwargs, verbose=True))
    df = pd.concat(tables, ignore_index=True)
    df.to_csv(out / "results.csv", index=False)
    pd.concat(searches, ignore_index=True).to_csv(out / "search.csv", index=False)
    verdicts = write_report(df, out / "report.md", out / "verdict.json", modes=modes)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for ds, v in verdicts.items():
        print(f"{ds} ({v['mode']}): {'PASS' if v['pass'] else 'FAIL'} fm={v['fm_mean']:.3f} best_control={v['best_control']} "
              f"uses_time={v['uses_time']} null_q95={v['null_q95']:.3f} matches={v['matches_target']}", flush=True)
    return verdicts


if __name__ == "__main__":
    main()
