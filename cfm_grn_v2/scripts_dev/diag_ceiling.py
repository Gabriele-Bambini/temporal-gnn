"""Where does the residual 0.64 come from? Measure the TRAINING TARGET itself, with no network involved.

The field is regressed on states sampled along cubic splines drawn through OT-chained pseudo-trajectories.
If the marginal of that path family at the held-out time is already far from the true population, no amount
of network training can do better: the ceiling is the path construction, not the model.

Measurements (energy distance in the benchmark's evaluation space, several chaining seeds):
  ceiling_fit    spline through the FIT times [0,1,2,4,6] evaluated at u = 3  vs true t = 3   <- the ceiling
  ceiling_obs    spline through all OBSERVED times [0,1,2,4,5,6] at u = 3     vs true t = 3
  ceiling_val1   same as ceiling_obs but the validation fold u = 1            vs true t = 1
  ceiling_val5   same, u = 5                                                  vs true t = 5
  oracle_chain   spline through ALL times incl. t = 3, at u = 3               vs true t = 3   (sanity: ~0)
  mccann_2_4     straight chain between t = 2 and t = 4 at u = 3              vs true t = 3   (~ ot_interp 1.17)
  noise_floor    two halves of the true t = 3 sample

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/diag_ceiling.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.evaluation.metrics import EvalProtocol, median_bandwidth
from cfm_grn_v2.experiments.run_synthetic import GRN_TIMES, grn_setup
from cfm_grn_v2.flows.coupling import chain_ot
from cfm_grn_v2.flows.path import SplinePath
from cfm_grn_v2.training.search import interpolation_design

TEST_T = 3.0
out = Path("outputs/diag_ceiling")
out.mkdir(parents=True, exist_ok=True)

s, *_ = grn_setup(500, 0, 40, 10)
d = interpolation_design(GRN_TIMES, TEST_T)
print("design:", {k: d[k] for k in ("fit_times", "obs_times", "val_time", "v_prev", "t_prev", "t_next", "horizon")})

all_mask = s.mask(times=d["obs_times"])
eval_std = Standardizer().fit(s.x[all_mask])
x_eval = eval_std.transform(s.x)
proto = EvalProtocol(n_samples=1000, seed=0, bandwidth=median_bandwidth(x_eval[all_mask]))


def chain_marginal(times, u, chain_seed, mode="ot", fit_times_for_std=None, linear=False):
    """Spline (or straight chain) through `times` for every (condition, replicate) group, evaluated at u.
    Chaining happens in the space standardised on `fit_times_for_std` (the model's own space), and the
    resulting states are mapped back to the evaluation space, exactly as a prediction would be."""
    std = Standardizer().fit(s.x[s.mask(times=fit_times_for_std or times)])
    x_std = std.transform(s.x)
    rng = np.random.default_rng(chain_seed)
    preds, targets = [], []
    for cond in s.conditions():
        for rep in s.replicates():
            masks = [s.mask(times=[t], conditions=[cond], replicates=[rep]) for t in times]
            traj = chain_ot([x_std[m] for m in masks], rng, 1000, mode=mode)
            if linear:                                     # straight segments between the knots
                k = int(np.searchsorted(times, u, side="right") - 1)
                k = min(max(k, 0), len(times) - 2)
                w = (u - times[k]) / (times[k + 1] - times[k])
                x_u = (1 - w) * traj[k] + w * traj[k + 1]
            else:
                path = SplinePath(times, traj)
                x_u, _ = path.evaluate(np.full(traj.shape[1], u), np.arange(traj.shape[1]))
            preds.append(eval_std.transform(std.inverse_transform(x_u)))
            targets.append(x_eval[s.mask(times=[u], conditions=[cond], replicates=[rep])])
    return preds, targets


def score(name, times, u, seeds=(0, 1, 2), **kw):
    rows = []
    for cs in seeds:
        preds, targets = chain_marginal(times, u, cs, **kw)
        for p, t in zip(preds, targets):
            rows.append({"measurement": name, "u": u, "chain_seed": cs, **proto.evaluate(p, t, seed_offset=cs)})
    df = pd.DataFrame(rows)
    print(f"{name:14s} u={u:.0f}  ED {df.energy_distance.mean():.4f} +- {df.energy_distance.std():.4f}"
          f"   mean_err {df.mean_error.mean():.3f}  W2 {df.w2.mean():.3f}  null_q95 {df.null_q95.mean():.4f}", flush=True)
    return rows


FIT, OBS, ALL = d["fit_times"], d["obs_times"], list(GRN_TIMES)
rows = []
rows += score("ceiling_fit", FIT, 3.0)
rows += score("ceiling_obs", OBS, 3.0)
rows += score("ceiling_val1", OBS, 1.0)
rows += score("ceiling_val5", OBS, 5.0)
rows += score("oracle_chain", ALL, 3.0)
rows += score("mccann_2_4", [2.0, 4.0], 3.0, linear=True)
rows += score("chain_random", FIT, 3.0, mode="random")
rows += score("ceiling_fit_lin", FIT, 3.0, linear=True)

# noise floor of the measurement itself
nf = []
for cond in s.conditions():
    for rep in s.replicates():
        t = x_eval[s.mask(times=[3.0], conditions=[cond], replicates=[rep])]
        idx = np.random.default_rng(0).permutation(len(t))
        nf.append(proto.evaluate(t[idx[: len(t) // 2]], t[idx[len(t) // 2:]], seed_offset=0))
print(f"{'noise_floor':14s} u=3   ED {np.mean([r['energy_distance'] for r in nf]):.4f}")
rows += [{"measurement": "noise_floor", "u": 3.0, "chain_seed": 0, **r} for r in nf]

df = pd.DataFrame(rows)
df.to_csv(out / "results.csv", index=False)
summary = df.groupby("measurement")[["energy_distance", "mmd", "w2", "mean_error", "per_gene_w1"]].mean().sort_values("energy_distance")
summary.to_csv(out / "summary.csv")
(out / "summary.json").write_text(json.dumps(summary.reset_index().to_dict(orient="records"), indent=1, default=float), encoding="utf-8")
print()
print(summary.round(4).to_string())
