"""Optimal-transport ablation on the synthetic-GRN interpolation fold (t=3 held out), full scale.

Arms (same field, same budget, same nested validation fold, 5 seeds):
  spline_ot      cubic-spline paths through OT-chained pseudo-trajectories (the selected model)
  spline_random  same, but consecutive snapshots paired at random (no optimal transport)
  linear_ot      straight consecutive-pair paths with minibatch OT coupling, duration clock
  linear_indep   straight consecutive-pair paths with independent coupling, duration clock
Controls come from run_interpolation_benchmark (persistence, mean/OT interpolants, linear, MLP, replay).

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/ot_ablation.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.evaluation.metrics import EvalProtocol
from cfm_grn_v2.evaluation.report import paired_difference, per_seed
from cfm_grn_v2.experiments.run_synthetic import GRN_TIMES, grn_setup
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import run_interpolation_benchmark

out = Path("outputs/ot_ablation")
out.mkdir(parents=True, exist_ok=True)
s, *_ = grn_setup(500, 0, 40, 10)
base = TrainConfig(hidden=256, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive")
ARMS = {"spline_ot": dict(path="spline", chain="ot"),
        "spline_random": dict(path="spline", chain="random"),
        "linear_ot": dict(path="linear", coupling="ot"),
        "linear_indep": dict(path="linear", coupling="independent")}
proto = EvalProtocol(n_samples=1000, seed=0)
seeds = [0, 1, 2, 3, 4]
tables = []
t0 = time.time()
for arm, over in ARMS.items():
    df = run_interpolation_benchmark(s, GRN_TIMES, 3.0, base.replace(**over), seeds, proto, "grn_interp_t3",
                                     shuffled_control=(arm == "spline_ot"), mlp_kwargs={"hidden": 128, "n_iters": 2000},
                                     verbose=True)
    df["model"] = df["model"].replace({"flow_matching": arm, "flow_matching_shuffled_time": f"{arm}_shuffled_time"})
    if arm != "spline_ot":                       # controls are identical across arms: keep one copy
        df = df[df["model"] == arm]
    tables.append(df)
    print(f"[{arm}] done, {time.time() - t0:.0f}s elapsed", flush=True)
res = pd.concat(tables, ignore_index=True)
res.to_csv(out / "results.csv", index=False)
ps = per_seed(res)
summary = ps.groupby("model")["energy_distance"].agg(["mean", "std", "count"]).sort_values("mean")
print(summary.round(3).to_string())
pairs = {"spline_ot_vs_spline_random": paired_difference(res, "grn_interp_t3", "spline_ot", "spline_random"),
         "spline_ot_vs_linear_ot": paired_difference(res, "grn_interp_t3", "spline_ot", "linear_ot"),
         "linear_ot_vs_linear_indep": paired_difference(res, "grn_interp_t3", "linear_ot", "linear_indep"),
         "spline_ot_vs_ot_interp": paired_difference(res, "grn_interp_t3", "spline_ot", "ot_interp"),
         "spline_ot_vs_mean_interp": paired_difference(res, "grn_interp_t3", "spline_ot", "mean_interp"),
         "spline_ot_vs_shuffled_time": paired_difference(res, "grn_interp_t3", "spline_ot", "spline_ot_shuffled_time")}
for k, v in pairs.items():
    print(f"{k}: mean {v['mean']:.3f} CI95 [{v['ci_low']:.3f}, {v['ci_high']:.3f}] n={v['n']}")
(out / "summary.json").write_text(json.dumps({"summary": summary.reset_index().to_dict(orient="records"), "paired": pairs},
                                              indent=1, default=float), encoding="utf-8")
summary.round(4).to_csv(out / "summary.csv")
