"""Is the residual error a wrong MEAN or a wrong SHAPE? (diagnostic only: uses the held-out mean as an oracle)

The spline path family already sits at energy distance 0.636 at the held-out time (scripts_dev/diag_ceiling.py).
Its mean at u = 3 is the cubic interpolation of the observed snapshot means, because chaining is a permutation
and preserves the mean. Here we separate the two error sources by shifting the predicted cloud onto the true
mean (an oracle correction that a real model may NOT use) and re-measuring.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/diag_mean_vs_shape.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.evaluation.metrics import EvalProtocol, median_bandwidth
from cfm_grn_v2.experiments.run_synthetic import GRN_TIMES, grn_setup
from cfm_grn_v2.flows.coupling import chain_ot, ot_coupling
from cfm_grn_v2.flows.path import SplinePath
from cfm_grn_v2.training.search import interpolation_design

out = Path("outputs/diag_ceiling")
out.mkdir(parents=True, exist_ok=True)
s, *_ = grn_setup(500, 0, 40, 10)
d = interpolation_design(GRN_TIMES, 3.0)
all_mask = s.mask(times=d["obs_times"])
eval_std = Standardizer().fit(s.x[all_mask])
x_eval = eval_std.transform(s.x)
proto = EvalProtocol(n_samples=1000, seed=0, bandwidth=median_bandwidth(x_eval[all_mask]))
rng = np.random.default_rng(0)
rows = []

for cond in s.conditions():
    for rep in s.replicates():
        def snap(t):
            return x_eval[s.mask(times=[t], conditions=[cond], replicates=[rep])]

        target = snap(3.0)
        # --- spline through the fit times, in the fit-standardised space, mapped back ---
        std = Standardizer().fit(s.x[s.mask(times=d["fit_times"])])
        xs = std.transform(s.x)
        traj = chain_ot([xs[s.mask(times=[t], conditions=[cond], replicates=[rep])] for t in d["fit_times"]],
                        np.random.default_rng(0), 1000)
        path = SplinePath(d["fit_times"], traj)
        spline3, _ = path.evaluate(np.full(traj.shape[1], 3.0), np.arange(traj.shape[1]))
        spline3 = eval_std.transform(std.inverse_transform(spline3))
        # --- McCann interpolant of the bracketing transition ---
        x2, x4 = snap(2.0), snap(4.0)
        i0, i1 = ot_coupling(x2, x4, rng, max_size=len(x2))
        mccann = x2[i0] + 0.5 * (x4[i1] - x2[i0])
        for name, pred in (("spline", spline3), ("mccann", mccann), ("persistence", x2)):
            rows.append({"variant": name, "correction": "none", "group": f"{cond}|{rep}",
                         **proto.evaluate(pred, target, seed_offset=0)})
            shifted = pred - pred.mean(0) + target.mean(0)                    # oracle mean correction
            rows.append({"variant": name, "correction": "oracle_mean", "group": f"{cond}|{rep}",
                         **proto.evaluate(shifted, target, seed_offset=0)})
            centred = pred - pred.mean(0)                                      # shape only, both centred
            rows.append({"variant": name, "correction": "both_centred", "group": f"{cond}|{rep}",
                         **proto.evaluate(centred, target - target.mean(0), seed_offset=0)})
        # how wrong is the interpolated mean curve, per gene?
        m = {t: snap(t).mean(0) for t in GRN_TIMES}
        true3 = m[3.0]
        cub = np.array([SplinePath(d["fit_times"], np.stack([m[t] for t in d["fit_times"]])[:, None, :])
                        .evaluate(np.array([3.0]), np.array([0]))[0][0]])[0]
        lin = 0.5 * (m[2.0] + m[4.0])
        rows.append({"variant": "mean_curve", "correction": "cubic_vs_true", "group": f"{cond}|{rep}",
                     "energy_distance": np.nan, "mean_error": float(np.linalg.norm(cub - true3) / np.sqrt(len(true3))),
                     "mmd": np.nan, "w2": np.nan, "per_gene_w1": np.nan, "null_q95": np.nan})
        rows.append({"variant": "mean_curve", "correction": "linear_vs_true", "group": f"{cond}|{rep}",
                     "energy_distance": np.nan, "mean_error": float(np.linalg.norm(lin - true3) / np.sqrt(len(true3))),
                     "mmd": np.nan, "w2": np.nan, "per_gene_w1": np.nan, "null_q95": np.nan})

df = pd.DataFrame(rows)
df.to_csv(out / "mean_vs_shape.csv", index=False)
print(df.groupby(["variant", "correction"])[["energy_distance", "mean_error", "w2"]].mean().round(4).to_string())
