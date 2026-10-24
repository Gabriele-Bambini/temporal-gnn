"""Final selection on the validation folds: refit on all knots, windowed marginal fine-tune, or both.

Both levers were validated separately (scripts_dev/knots_scan.py, scripts_dev/marginal_scan.py). The final
configuration has to be chosen before the test fold is touched, so the combination is measured here, on the
same two folds, with t = 3 removed from the data.

  base        train on the fit times, early stop on the validation time
  refit       + refit on every observed time for the early-stopped number of iterations (one more spline knot)
  ft          + windowed marginal fine-tune (path supervised only near the observed times, endpoints matched)
  ft_refit    both

Usage (from cfm_grn_v2/):  PYTHONPATH=src python scripts_dev/combo_scan.py [--seeds 3]
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

TIMES = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]
FOLDS = [1.0, 5.0]
BASE = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot")
FT = dict(marginal_iters=1000, marginal_batch=256, marginal_steps=8, marginal_eval_every=25, marginal_patience=10,
          marginal_prior=1.0, marginal_lr=3e-4, marginal_window=1.0)

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--out", default="outputs/combo_scan")
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


for fold in FOLDS:
    d = interpolation_design(TIMES, fold)
    fit_mask, all_mask = s.mask(times=d["fit_times"]), s.mask(times=d["obs_times"])
    eval_std = Standardizer().fit(s.x[all_mask])
    x_eval = eval_std.transform(s.x)
    val_task = time_task(s, d["v_prev"], d["val_time"])
    print(f"\n=== fold t={fold:.0f}: knots {d['fit_times']} -> refit knots {d['obs_times']} ===", flush=True)
    for seed in range(a.seeds):
        for arm, ft in (("base", False), ("ft", True)):
            cfg = BASE.replace(seed=seed, **(FT if ft else {}))
            model, _ = train_flow(s, fit_mask, d["fit_times"], cfg, val_task, proto)
            ed, me = evaluate(model, d, eval_std, x_eval)
            rows.append({"fold": fold, "seed": seed, "arm": arm, "energy_distance": ed, "mean_error": me})
            print(f"  seed {seed}  {arm:9s} ED {ed:.4f}  mean_err {me:.3f}   ({time.time() - t0:.0f}s)", flush=True)
            n_iters = max(int(model.best_iter), cfg.eval_every)
            refit, _ = train_flow(s, all_mask, d["obs_times"], cfg.replace(n_iters=n_iters), None, proto)
            ed2, me2 = evaluate(refit, d, eval_std, x_eval)
            name = f"{arm}_refit" if ft else "refit"
            rows.append({"fold": fold, "seed": seed, "arm": name, "energy_distance": ed2, "mean_error": me2})
            print(f"  seed {seed}  {name:9s} ED {ed2:.4f}  mean_err {me2:.3f}   ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(out / "results.csv", index=False)
print()
print(df.pivot_table(index="arm", columns="fold", values=["energy_distance", "mean_error"], aggfunc="mean").round(4).to_string())
g = df.groupby("arm")[["energy_distance", "mean_error"]].agg(["mean", "std"]).round(4)
print()
print(g.to_string())
base = df[df.arm == "base"].set_index(["fold", "seed"]).energy_distance
print("\npaired change versus base (negative = better):")
for arm in [x for x in df.arm.unique() if x != "base"]:
    sub = df[df.arm == arm].set_index(["fold", "seed"]).energy_distance
    dlt = (sub - base.reindex(sub.index)).dropna()
    print(f"  {arm:9s} {dlt.mean():+.4f}  sd {dlt.std():.4f}  n {len(dlt)}")
g.to_csv(out / "summary.csv")
