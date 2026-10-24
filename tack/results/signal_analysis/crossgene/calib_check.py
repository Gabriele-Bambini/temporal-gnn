"""Noise-model calibration check: Var_genes(Delta_t) regressed on predicted noise mean(v_t + v_t+1), all transitions
vs transitions whose two bins both have >= 50 cells; l1 and linear. Output: calib_check.json."""
import json
import numpy as np
import common
from structure import noise_vars

out = {}
for name in common.SNAPS:
    d = common.Data(name)
    v_l1, v_lin = noise_vars(d)
    R = {}
    for t, v in (("l1", v_l1), ("linear", v_lin)):
        Y = common.all_targets(d, t)
        tot = np.array([Y[i].var() for i in range(d.T)])
        pn = np.array([v[:, i].mean() + v[:, i + 1].mean() for i in range(d.T)])
        ok = np.array([min(d.nc[i], d.nc[i + 1]) >= 50 for i in range(d.T)])
        for lab, m in (("all", np.ones(d.T, bool)), ("bins_ge50cells", ok)):
            A = np.stack([np.ones(m.sum()), pn[m]], 1)
            c = np.linalg.lstsq(A, tot[m], rcond=None)[0]
            R[f"{t}_{lab}"] = {"n": int(m.sum()), "intercept": float(c[0]), "slope": float(c[1]),
                               "corr": float(np.corrcoef(pn[m], tot[m])[0, 1]),
                               "mean_noise_share": float(np.mean(pn[m] / tot[m])),
                               "median_noise_share": float(np.median(pn[m] / tot[m]))}
    out[name] = R
    for k, x in R.items():
        print(name, k, {kk: round(vv, 4) for kk, vv in x.items()})
json.dump(out, open(f"{common.OUT}/calib_check.json", "w"), indent=1)
