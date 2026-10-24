"""When does optimal transport matter? Two lineages that move in opposite directions.

Five time points t = 0..4, two clusters of cells: the left cluster drifts upward, the right one downward
(unit speed), two replicates. Held out: t = 2, predicted from t = 1 with the bracketing 1 -> 3 observed.
Pairing left cells with right cells (independent coupling, random chaining) creates crossing paths whose
midpoints collapse to the centre; OT pairs each cluster with itself. Same four arms as ot_ablation.py.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/ot_crossing.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.evaluation.metrics import EvalProtocol
from cfm_grn_v2.evaluation.report import paired_difference, per_seed
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import run_interpolation_benchmark

TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]


def crossing_lineages(n_per_cluster=200, seed=0, spread=0.35):
    rng = np.random.default_rng(seed)
    xs, ts, reps = [], [], []
    for rep in ("r1", "r2"):
        for t in TIMES:
            left = np.array([-3.0, t]) + spread * rng.normal(size=(n_per_cluster, 2))
            right = np.array([3.0, -t]) + spread * rng.normal(size=(n_per_cluster, 2))
            x = np.concatenate([left, right])
            xs.append(x)
            ts.append(np.full(len(x), t))
            reps.append(np.array([rep] * len(x)))
    x = np.concatenate(xs).astype(np.float32)
    n = len(x)
    return SnapshotSet(x, np.concatenate(ts), np.array(["ctrl"] * n), np.concatenate(reps), ["x", "y"], name="crossing")


out = Path("outputs/ot_crossing")
out.mkdir(parents=True, exist_ok=True)
s = crossing_lineages()
base = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=4000, eval_every=250,
                   patience=6, clock="duration", coupling="ot", pair_mode="consecutive")
ARMS = {"spline_ot": dict(path="spline", chain="ot"),
        "spline_random": dict(path="spline", chain="random"),
        "linear_ot": dict(path="linear", coupling="ot"),
        "linear_indep": dict(path="linear", coupling="independent")}
proto = EvalProtocol(n_samples=400, seed=0)
seeds = [0, 1, 2, 3, 4]
tables = []
t0 = time.time()
for arm, over in ARMS.items():
    df = run_interpolation_benchmark(s, TIMES, 2.0, base.replace(**over), seeds, proto, "crossing_t2",
                                     shuffled_control=(arm == "spline_ot"), mlp_kwargs={"hidden": 64, "n_iters": 1000})
    df["model"] = df["model"].replace({"flow_matching": arm, "flow_matching_shuffled_time": f"{arm}_shuffled_time"})
    if arm != "spline_ot":
        df = df[df["model"] == arm]
    tables.append(df)
    print(f"[{arm}] done, {time.time() - t0:.0f}s elapsed", flush=True)
res = pd.concat(tables, ignore_index=True)
res.to_csv(out / "results.csv", index=False)
summary = per_seed(res).groupby("model")["energy_distance"].agg(["mean", "std", "count"]).sort_values("mean")
print(summary.round(3).to_string())
pairs = {"spline_ot_vs_spline_random": paired_difference(res, "crossing_t2", "spline_ot", "spline_random"),
         "linear_ot_vs_linear_indep": paired_difference(res, "crossing_t2", "linear_ot", "linear_indep"),
         "spline_ot_vs_ot_interp": paired_difference(res, "crossing_t2", "spline_ot", "ot_interp"),
         "spline_ot_vs_mean_interp": paired_difference(res, "crossing_t2", "spline_ot", "mean_interp")}
for k, v in pairs.items():
    print(f"{k}: mean {v['mean']:.3f} CI95 [{v['ci_low']:.3f}, {v['ci_high']:.3f}] n={v['n']}")
(out / "summary.json").write_text(json.dumps({"summary": summary.reset_index().to_dict(orient="records"), "paired": pairs},
                                              indent=1, default=float), encoding="utf-8")
summary.round(4).to_csv(out / "summary.csv")
