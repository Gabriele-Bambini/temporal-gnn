"""The single measurement on the held-out test fold t = 3 with the configuration selected on the validation folds.

Selection (never touching t = 3, which was removed from the data during selection):
  scripts_dev/diag_ceiling.py     the trained field reproduces its training path to within 0.005 energy
                                  distance, so the path is the ceiling; adding the sixth knot lowers it
                                  from 0.636 to 0.422
  scripts_dev/knots_scan.py       refitting on every observed time beats the fit-times-only model on both folds
  scripts_dev/marginal_scan.py    a windowed path supervision plus marginal matching beats the plain fine-tune
  scripts_dev/combo_scan.py       the combination is the best of the four arms

The previously published model (v0.1.8, outputs/phase1_v2_grn_interp) is re-scored here from its stored rows so
that the comparison is paired by seed and by (condition, replicate) group.

Usage (from cfm_grn_v2/):  PYTHONPATH=src python scripts_dev/final_test.py [--seeds 5]
"""
import argparse
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

DATASET = "grn_interp_t3"
BASE = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot")
FT = dict(marginal_iters=1000, marginal_batch=256, marginal_steps=8, marginal_eval_every=25, marginal_patience=10,
          marginal_prior=1.0, marginal_lr=3e-4, marginal_window=1.0)

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--out", default="outputs/final_test")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
seeds = list(range(a.seeds))

s, *_ = grn_setup(500, 0, 40, 10)
proto = EvalProtocol(n_samples=1000, seed=0)
t0 = time.time()
df = run_interpolation_benchmark(s, GRN_TIMES, 3.0, BASE.replace(**FT), seeds, proto, DATASET,
                                 refit_on_all=True, shuffled_control=True,
                                 mlp_kwargs={"hidden": 128, "n_iters": 2000}, verbose=True)
df["model"] = df["model"].replace({"flow_matching": "flow_v019", "flow_matching_shuffled_time": "flow_v019_shuffled_time"})
print(f"improved model done in {time.time() - t0:.0f}s", flush=True)

old = pd.read_csv("outputs/phase1_v2_grn_interp/results.csv")
old = old[old.model.isin(["flow_matching", "flow_matching_shuffled_time"])].copy()
old["model"] = old["model"].replace({"flow_matching": "flow_v018", "flow_matching_shuffled_time": "flow_v018_shuffled_time"})
old["dataset"] = DATASET
res = pd.concat([df, old[old.seed.isin(seeds)]], ignore_index=True)
res.to_csv(out / "results.csv", index=False)

summary = per_seed(res).groupby("model")["energy_distance"].agg(["mean", "std", "count"]).sort_values("mean")
print()
print(summary.round(4).to_string())
pairs = {
    "v019_vs_v018": paired_difference(res, DATASET, "flow_v019", "flow_v018"),
    "v019_vs_mean_interp": paired_difference(res, DATASET, "flow_v019", "mean_interp"),
    "v019_vs_ot_interp": paired_difference(res, DATASET, "flow_v019", "ot_interp"),
    "v019_vs_shuffled_time": paired_difference(res, DATASET, "flow_v019", "flow_v019_shuffled_time"),
}
print()
for k, v in pairs.items():
    print(f"{k:24s} mean {v['mean']:+.4f}  CI95 [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  n={v['n']}")
(out / "summary.json").write_text(json.dumps(
    {"summary": summary.reset_index().to_dict(orient="records"), "paired": pairs,
     "config": BASE.replace(**FT).to_dict(), "refit_on_all": True}, indent=1, default=float), encoding="utf-8")
summary.round(4).to_csv(out / "summary.csv")
