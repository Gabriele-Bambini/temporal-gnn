"""The benchmark reduces to the mean trajectory: which interpolant reconstructs it best?

diag_mean_vs_shape.py showed that shifting any prediction onto the true mean drives the energy distance to
the noise floor, and that the energy distance is proportional to the squared mean error
(0.636 / 1.334^2 = 0.357 = 1.171 / 1.806^2). So the held-out-time error is, on this simulator, a question of
interpolating the mean of each gene through the observed snapshots.

This script compares interpolants and spaces, leaving out one inner time point at a time. Selection is on the
VALIDATION folds t = 1 and t = 5; the t = 3 column is printed for the record but must not drive the choice.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/diag_mean_curve.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

sys.path.insert(0, "src")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.experiments.run_synthetic import GRN_TIMES, grn_setup

out = Path("outputs/diag_ceiling")
out.mkdir(parents=True, exist_ok=True)
s, *_ = grn_setup(500, 0, 40, 10)
ED_PER_SQ_MEAN = 0.357          # measured conversion: energy distance ~ 0.357 * (L2 mean error)^2

INTERP = {
    "cubic_natural": lambda t, y: CubicSpline(t, y, axis=0, bc_type="natural"),
    "cubic_not_a_knot": lambda t, y: CubicSpline(t, y, axis=0, bc_type="not-a-knot"),
    "pchip": lambda t, y: PchipInterpolator(t, y, axis=0),
    "akima": lambda t, y: Akima1DInterpolator(t, y, axis=0),
    "linear": lambda t, y: (lambda tt: np.stack([np.interp(tt, t, y[:, j]) for j in range(y.shape[1])], -1)),
}


def curve(f, u):
    return np.asarray(f(u)).reshape(-1)


rows = []
for cond in s.conditions():
    for rep in s.replicates():
        std = Standardizer().fit(s.x[s.mask(conditions=[cond], replicates=[rep])])
        z = std.transform(s.x)
        means = {t: z[s.mask(times=[t], conditions=[cond], replicates=[rep])].mean(0) for t in GRN_TIMES}
        raw = {t: s.x[s.mask(times=[t], conditions=[cond], replicates=[rep])] for t in GRN_TIMES}
        for held in (1.0, 2.0, 3.0, 4.0, 5.0):
            for drop_extra in ([], [5.0] if held == 3.0 else []):     # the real test fold also loses t = 5
                obs = [t for t in GRN_TIMES if t != held and t not in drop_extra]
                tag = "fit_times" if drop_extra else "obs_times"
                y = np.stack([means[t] for t in obs])
                true = means[held]
                for name, fn in INTERP.items():
                    err = float(np.linalg.norm(curve(fn(np.array(obs), y), held) - true))
                    rows.append({"held": held, "knots": tag, "space": "standardised", "interp": name,
                                 "mean_error": err, "implied_ed": ED_PER_SQ_MEAN * err ** 2})
                # log space: Hill kinetics saturate, a log-scale curve may bend more naturally
                ylog = np.stack([np.log1p(np.maximum(raw[t], 0)).mean(0) for t in obs])
                tlog = np.log1p(np.maximum(raw[held], 0)).mean(0)
                for name, fn in INTERP.items():
                    pred_log = curve(fn(np.array(obs), ylog), held)
                    err_log = float(np.linalg.norm((pred_log - tlog) / std.std_ * 0 + 0))  # placeholder, see below
                    # map the log-space prediction back to expression, then to the standardised space
                    pred_raw = np.expm1(pred_log)
                    err = float(np.linalg.norm((pred_raw - raw[held].mean(0)) / std.std_))
                    rows.append({"held": held, "knots": tag, "space": "log1p", "interp": name,
                                 "mean_error": err, "implied_ed": ED_PER_SQ_MEAN * err ** 2})

df = pd.DataFrame(rows)
df.to_csv(out / "mean_curve.csv", index=False)
piv = df.groupby(["space", "interp", "knots", "held"])["mean_error"].mean().unstack("held").round(3)
print(piv.to_string())
print()
val = df[df.held.isin([1.0, 5.0])].groupby(["space", "interp"])["mean_error"].mean().sort_values()
print("SELECTION on the validation folds (t = 1 and t = 5), mean L2 error of the mean:")
print(val.round(3).to_string())
print()
test = df[(df.held == 3.0)].groupby(["space", "interp", "knots"])[["mean_error", "implied_ed"]].mean().sort_values("mean_error")
print("for the record (t = 3):")
print(test.round(3).to_string())
