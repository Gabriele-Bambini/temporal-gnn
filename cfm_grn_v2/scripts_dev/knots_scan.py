"""Which observed time is spent on early stopping, and should the final model be refitted on all of them?

The diagnosis (scripts_dev/diag_ceiling.py) showed that the trained field reproduces its training path to
within 0.005 energy distance, so the only way to improve the prediction is to improve the path. The path is a
spline through the FIT times, and the design currently spends one observed time on early stopping: with times
0..6 and t = 3 held out, `interpolation_design` picks the inner observed time farthest from the test time,
which ties to t = 5, and the spline is then drawn through [0, 1, 2, 4, 6].

Two leak-free levers:
  valswap  spend the OTHER admissible inner time on validation instead
  refit    keep the early-stopped iteration count, then refit on every observed time (all knots)

Measured on the two validation folds with t = 3 removed from the data, so none of this can see the test point.

Usage (from cfm_grn_v2/):  PYTHONPATH=src python scripts_dev/knots_scan.py [--seeds 3]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.experiments.run_synthetic import grn_setup
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import interpolation_design
from cfm_grn_v2.training.trainer import task_groups, time_task, train_flow

TIMES = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]           # t = 3 does not exist in this experiment
FOLDS = [1.0, 5.0]
BASE = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot")

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--out", default="outputs/knots_scan")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

s_full, *_ = grn_setup(500, 0, 40, 10)
s = s_full.subset(s_full.mask(times=TIMES))
proto = EvalProtocol(n_samples=1000, seed=0)
rows, t0 = [], time.time()


def evaluate(model, d, eval_std, x_eval, n=1000):
    task = time_task(s, d["t_prev"], d["test_time"])
    eds, mes = [], []
    for cond, rep, m0, m1 in task_groups(s, task):
        if cond not in model.registry.names:
            continue
        pred = eval_std.transform(model.predict(s.x[m0], cond, d["t_prev"], d["test_time"]))
        tgt = x_eval[m1]
        rng = np.random.default_rng(0)
        eds.append(energy_distance(pred[rng.choice(len(pred), min(n, len(pred)), replace=False)],
                                   tgt[rng.choice(len(tgt), min(n, len(tgt)), replace=False)]))
        mes.append(float(np.linalg.norm(pred.mean(0) - tgt.mean(0))))
    return float(np.mean(eds)), float(np.mean(mes))


def admissible_val_times(fold):
    d = interpolation_design(TIMES, fold)
    inner = [t for t in d["obs_times"][1:-1] if t not in (d["t_prev"], d["t_next"])]
    return d["val_time"], [t for t in inner if t != d["val_time"]]


for fold in FOLDS:
    default_val, others = admissible_val_times(fold)
    val_choices = [("default", default_val)] + [("valswap", v) for v in others[:1]]
    for label, vt in val_choices:
        d = interpolation_design(TIMES, fold, val_time=vt)
        fit_mask = s.mask(times=d["fit_times"])
        all_mask = s.mask(times=d["obs_times"])
        eval_std = Standardizer().fit(s.x[all_mask])
        x_eval = eval_std.transform(s.x)
        val_task = time_task(s, d["v_prev"], d["val_time"])
        print(f"\n=== fold t={fold:.0f} | validation on t={vt:.0f} ({label}) | knots {d['fit_times']} "
              f"| all observed {d['obs_times']} ===", flush=True)
        for seed in range(a.seeds):
            cfg = BASE.replace(seed=seed)
            model, _ = train_flow(s, fit_mask, d["fit_times"], cfg, val_task, proto)
            ed, me = evaluate(model, d, eval_std, x_eval)
            rows.append({"fold": fold, "seed": seed, "arm": label, "val_time": vt, "energy_distance": ed,
                         "mean_error": me, "best_iter": model.best_iter})
            print(f"  seed {seed}  {label:8s}            ED {ed:.4f}  mean_err {me:.3f}  "
                  f"best_iter {model.best_iter}   ({time.time() - t0:.0f}s)", flush=True)
            for mult in (1, 2):
                n_iters = max(int(model.best_iter) * mult, cfg.eval_every)
                refit, _ = train_flow(s, all_mask, d["obs_times"], cfg.replace(n_iters=n_iters), None, proto)
                ed2, me2 = evaluate(refit, d, eval_std, x_eval)
                rows.append({"fold": fold, "seed": seed, "arm": f"{label}+refit x{mult}", "val_time": vt,
                             "energy_distance": ed2, "mean_error": me2, "best_iter": n_iters})
                print(f"  seed {seed}  {label:8s}+refit x{mult}   ED {ed2:.4f}  mean_err {me2:.3f}  "
                      f"iters {n_iters}   ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(out / "results.csv", index=False)
print()
print(df.pivot_table(index="arm", columns="fold", values=["energy_distance", "mean_error"], aggfunc="mean").round(4).to_string())
print()
print(df.groupby("arm")[["energy_distance", "mean_error"]].agg(["mean", "std"]).round(4).to_string())
df.groupby("arm")[["energy_distance", "mean_error"]].mean().round(5).to_csv(out / "summary.csv")
