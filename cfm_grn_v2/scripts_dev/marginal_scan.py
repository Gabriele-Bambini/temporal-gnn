"""Lean scan of the distributional fine-tune on the two validation folds (no controls, no permutation nulls).

The test time point t = 3 is removed from the data, so nothing here can leak into it. The pre-training is done
once per (fold, seed) and every fine-tune setting restarts from that same snapshot, which makes the comparison
paired and ten times cheaper than the full benchmark.

Folds (both have the structure of the real test fold: a held-out point in the middle of a gap of two units):
  t = 1 predicted from t = 0, bracketing 0 -> 2      t = 5 predicted from t = 4, bracketing 4 -> 6

Usage (from cfm_grn_v2/):  PYTHONPATH=src python scripts_dev/marginal_scan.py [--seeds 2] [--out outputs/marginal_scan]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.data.preprocessing import Standardizer, fit_on_train
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.experiments.run_synthetic import grn_setup
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import interpolation_design
from cfm_grn_v2.training.trainer import (_finetune_marginal, build_transitions, spline_paths, task_groups,
                                         time_task, train_flow)

TIMES = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]           # t = 3 does not exist in this experiment
FOLDS = [1.0, 5.0]
BASE = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot")
FT = dict(marginal_iters=1000, marginal_batch=256, marginal_steps=8, marginal_eval_every=25, marginal_patience=10)
SETTINGS = {
    "p1":            dict(**FT, marginal_prior=1.0, marginal_lr=3e-4, marginal_window=0.0),
    "p01":           dict(**FT, marginal_prior=0.1, marginal_lr=3e-4, marginal_window=0.0),
    "p0":            dict(**FT, marginal_prior=0.0, marginal_lr=3e-4, marginal_window=0.0),
    "win05_p1":      dict(**FT, marginal_prior=1.0, marginal_lr=3e-4, marginal_window=0.5),
    "win05_p1_lr1e3": dict(**FT, marginal_prior=1.0, marginal_lr=1e-3, marginal_window=0.5),
    "win10_p1":      dict(**FT, marginal_prior=1.0, marginal_lr=3e-4, marginal_window=1.0),
    "win05_p1_long": dict(**{**FT, "marginal_iters": 2500}, marginal_prior=1.0, marginal_lr=3e-4, marginal_window=0.5),
    "win05_sw2":     dict(**FT, marginal_prior=1.0, marginal_lr=3e-4, marginal_window=0.5, marginal_metric="sliced_w2"),
}

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--settings", default=",".join(SETTINGS))
p.add_argument("--out", default="outputs/marginal_scan")
a = p.parse_args()
names = [n for n in a.settings.split(",") if n]
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
        pa = pred[rng.choice(len(pred), min(n, len(pred)), replace=False)]
        tb = tgt[rng.choice(len(tgt), min(n, len(tgt)), replace=False)]
        eds.append(energy_distance(pa, tb))
        mes.append(float(np.linalg.norm(pred.mean(0) - tgt.mean(0))))
    return float(np.mean(eds)), float(np.mean(mes))


for fold in FOLDS:
    d = interpolation_design(TIMES, fold)
    fit_mask = s.mask(times=d["fit_times"])
    all_mask = s.mask(times=d["obs_times"])
    eval_std = Standardizer().fit(s.x[all_mask])
    x_eval = eval_std.transform(s.x)
    val_task = time_task(s, d["v_prev"], d["val_time"])
    print(f"\n=== fold t={fold:.0f}: fit {d['fit_times']}, validate on t={d['val_time']:.0f}, "
          f"predict t={fold:.0f} from t={d['t_prev']:.0f} ===", flush=True)
    for seed in range(a.seeds):
        cfg = BASE.replace(seed=seed)
        model, _ = train_flow(s, fit_mask, d["fit_times"], cfg, val_task, proto)
        state0, base_val = model.snapshot_state(), model.best_val
        ed, me = evaluate(model, d, eval_std, x_eval)
        rows.append({"fold": fold, "seed": seed, "setting": "base", "energy_distance": ed, "mean_error": me})
        print(f"  seed {seed}  {'base':16s} ED {ed:.4f}  mean_err {me:.3f}   ({time.time() - t0:.0f}s)", flush=True)
        # the training material rebuilt exactly as train_flow builds it, so the fine-tune sees the same objects
        std = fit_on_train(s.x, fit_mask)
        x_std = std.transform(s.x)
        fit = s.subset(fit_mask)
        x_fit = x_std[fit_mask]
        transitions = build_transitions(fit, d["fit_times"], x_fit, pair_mode=cfg.pair_mode)
        paths = spline_paths(fit, d["fit_times"], x_fit, np.random.default_rng(seed), chain=cfg.chain,
                             bc_type=cfg.spline_kind)
        for name in names:
            model.load_snapshot(state0)
            model.best_val = base_val
            _finetune_marginal(model, s, x_std, transitions, paths, cfg.replace(**SETTINGS[name]), val_task,
                               proto, np.random.default_rng(1000 + seed), [])
            ed, me = evaluate(model, d, eval_std, x_eval)
            rows.append({"fold": fold, "seed": seed, "setting": name, "energy_distance": ed, "mean_error": me,
                         "val": model.best_val, "stopped_at": model.marginal_iter})
            print(f"  seed {seed}  {name:16s} ED {ed:.4f}  mean_err {me:.3f}  val {model.best_val:.4f} "
                  f"stop@{model.marginal_iter}   ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(out / "results.csv", index=False)
piv = df.pivot_table(index="setting", columns="fold", values=["energy_distance", "mean_error"], aggfunc="mean")
print()
print(piv.round(4).to_string())
base = df[df.setting == "base"].set_index(["fold", "seed"]).energy_distance
delta = (df[df.setting != "base"].set_index(["fold", "seed", "setting"]).energy_distance
         - base.reindex(df[df.setting != "base"].set_index(["fold", "seed", "setting"]).index.droplevel("setting")).values)
summ = delta.groupby("setting").agg(["mean", "std", "count"]).sort_values("mean")
print("\npaired change versus base (negative = better), pooled over folds and seeds:")
print(summ.round(4).to_string())
summ.round(5).to_csv(out / "summary.csv")
