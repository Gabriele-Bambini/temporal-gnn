"""Does the distributional fine-tune beat the path loss? Selection on the VALIDATION folds only.

The test time point t = 3 is REMOVED FROM THE DATA here, so nothing about it can reach this choice, not even
through a nested fit. What is left, [0, 1, 2, 4, 5, 6], contains two folds with exactly the structure of the
test fold - a held-out point in the middle of a gap of two time units, with the bracketing transition observed:
  fold t = 1  predicted from t = 0, bracketing 0 -> 2
  fold t = 5  predicted from t = 4, bracketing 4 -> 6
Both fit on [0, 2, 4, 6].

Usage (from cfm_grn_v2/):
  PYTHONPATH=src python scripts_dev/marginal_validation.py [--seeds 3] [--arms base,mg_prior1,...]
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
from cfm_grn_v2.experiments.run_synthetic import grn_setup
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import run_interpolation_benchmark

BASE = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot")
FT = dict(marginal_iters=600, marginal_batch=256, marginal_steps=8, marginal_eval_every=25, marginal_patience=8)
ARMS = {
    "base": {},
    "mg_prior1": dict(**FT, marginal_prior=1.0, marginal_lr=3e-4),
    "mg_prior01": dict(**FT, marginal_prior=0.1, marginal_lr=3e-4),
    "mg_prior0": dict(**FT, marginal_prior=0.0, marginal_lr=3e-4),
    "mg_lr1e3": dict(**FT, marginal_prior=0.1, marginal_lr=1e-3),
    "mg_sw2": dict(**FT, marginal_prior=0.1, marginal_lr=3e-4, marginal_metric="sliced_w2"),
}
FOLDS = [1.0, 5.0]
TIMES = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]          # t = 3 does not exist in this experiment

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--arms", default=",".join(ARMS))
p.add_argument("--out", default="outputs/marginal_validation")
a = p.parse_args()
arms = [x for x in a.arms.split(",") if x]
seeds = list(range(a.seeds))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

s_full, *_ = grn_setup(500, 0, 40, 10)
s = s_full.subset(s_full.mask(times=TIMES))
print(f"data without t=3: {s.n_cells} cells, times {sorted(set(s.biological_time))}", flush=True)
proto = EvalProtocol(n_samples=1000, seed=0)
tables, t0 = [], time.time()
for fold in FOLDS:
    for arm in arms:
        df = run_interpolation_benchmark(s, TIMES, fold, BASE.replace(**ARMS[arm]), seeds, proto,
                                         f"val_t{fold:.0f}", shuffled_control=False,
                                         mlp_kwargs={"hidden": 128, "n_iters": 2000})
        df["model"] = df["model"].replace({"flow_matching": arm})
        if arm != arms[0]:
            df = df[df["model"] == arm]
        df["fold"] = fold
        tables.append(df)
        ed = per_seed(df[df.model == arm]).energy_distance.mean()
        print(f"[fold {fold:.0f}] {arm:11s} ED {ed:.4f}   ({time.time() - t0:.0f}s elapsed)", flush=True)
res = pd.concat(tables, ignore_index=True)
res.to_csv(out / "results.csv", index=False)

ps = per_seed(res)
summary = ps.groupby("model")["energy_distance"].agg(["mean", "std", "count"]).sort_values("mean")
print()
print(summary.round(4).to_string())
pairs = {}
for arm in arms[1:]:
    per_fold = {}
    for fold in FOLDS:
        sub = res[res.fold == fold]
        per_fold[f"t{fold:.0f}"] = paired_difference(sub, f"val_t{fold:.0f}", arm, arms[0])
    pairs[f"{arm}_vs_{arms[0]}"] = per_fold
    txt = "  ".join(f"{k}: {v['mean']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]" for k, v in per_fold.items())
    print(f"{arm:11s} minus {arms[0]:11s}  {txt}")
(out / "summary.json").write_text(json.dumps(
    {"summary": summary.reset_index().to_dict(orient="records"), "paired": pairs}, indent=1, default=float), encoding="utf-8")
summary.round(4).to_csv(out / "summary.csv")
