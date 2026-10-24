"""Hindsight diagnostic (NOT a forecast): what gain do the data want, per expression class, on l1?

For the fixed per-gene config F_PG (k = 0, EB nu = 4, momentum, all bins) at origins t >= T0 (large-bin part of
each dataset), regress the true l1 change on [denoise term x_t - obs_t, model forecast change, last observed
change] inside each of the 9 causal expression classes (pooled over origins, in-sample). A positive denoise
coefficient means the deviation of today's bin from the filtered level reverts (noise); a negative one means it
persists (a real change the Gaussian filter has shrunk). Output: diag_gain.json.
"""
import json

import numpy as np

import lsb_common as C
from lsb_model import LatentSB

CFG = (0, "-", "-", None, 4.0, "mom")
T0 = {"gastrulation-human": 8, "pancreas-human": 8}


def main():
    out = {}
    for name in C.SNAPS:
        d = C.Data(name)
        m = LatentSB(d)
        Y = C.all_targets(d, "l1")
        rows = {c: [] for c in range(9)}
        for t in range(T0[name], d.T):
            xn, xx, parts = m.levels(t, CFG)
            cls = m.classes(t)
            den = xn - (d.X[:, t] - m.R[:, t] / 2)
            for c in range(9):
                s = cls == c
                rows[c].append((den[s], parts["res_chg"][s], Y[t - 1][s], Y[t][s]))
            m._res.clear()
        res = {}
        for c in range(9):
            D, F, L, y = (np.concatenate([r[j] for r in rows[c]]) for j in range(4))
            X = np.column_stack([np.ones_like(D), D, F, L])
            co = np.linalg.lstsq(X, y, rcond=None)[0]
            cc = lambda a: None if np.std(a) == 0 or np.std(y) == 0 else round(float(np.corrcoef(a, y)[0, 1]), 3)
            res[c] = {"n": int(len(y)), "var_y": round(float(y.var()), 5),
                      "coef_denoise": round(float(co[1]), 3), "coef_forecast_change": round(float(co[2]), 3),
                      "coef_last_change": round(float(co[3]), 3),
                      "corr_denoise": cc(D), "corr_forecast_change": cc(F), "corr_last_change": cc(L)}
            print(name, c, res[c], flush=True)
        out[name] = {"origins": f">= {T0[name]}", "classes": res}
    json.dump(out, open(f"{C.OUT}/diag_gain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
