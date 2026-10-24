"""Is the cheap-sink down-side gain a level rule? Per transition (i >= 1, all genes): correlation of each strict
family's forecast with -m_i (linear) and with the persistence forecast / -log1p(CP10k_i) (l1)."""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_uot as R  # noqa: E402

out = {}
for name in ("pancreas-human", "gastrulation-human"):
    X, trip, n, m = R.U.load(name)
    T = len(trip) - 1
    with open(os.path.join(R.SCRATCH, f"uot_res_{name}.pkl"), "rb") as fh:
        res = pickle.load(fh)
    J = json.load(open(os.path.join(HERE, f"results_{name}.json")))
    out[name] = {}
    for tg in ("linear", "l1"):
        Y = np.stack([R.dileo_protocol.TARGET_NP[tg](m[i], m[i + 1]).astype(np.float64) for i in range(T)])

        def pred(fam, i):
            if fam == "trend":
                return Y[i - 1]
            if fam == "persist":
                return res[(name, "persist", i, 0, 0.0)][0][0.0][tg]
            c = J["selected"][fam][tg][str(i)]
            if c[0] == "uot":
                return res[(name, "uot", i, min(c[2], i), c[1])][0][c[3]][tg]
            return res[(name, "bal", i, min(c[1], i), 0.0)][0][c[2]][tg]
        row = {}
        for fam in ("uot_lam0.0625", "uot_lam0.25", "uot_lam128", "bal", "trend") + (("persist",) if tg == "l1" else ()):
            lev = [-m[i] if tg == "linear" else -np.log1p(1e4 * m[i] / m[i].sum()) for i in range(1, T)]
            r_lev = np.nanmean([np.corrcoef(pred(fam, i), lv)[0, 1] for i, lv in zip(range(1, T), lev)])
            d = {"corr_with_minus_level": round(float(r_lev), 3)}
            if tg == "l1":
                d["corr_with_persist"] = round(float(np.nanmean([np.corrcoef(pred(fam, i), pred("persist", i))[0, 1]
                                                                 for i in range(1, T)])), 3)
            row[fam] = d
        lev_rule = np.nanmean([np.corrcoef(Y[i], -m[i] if tg == "linear" else -np.log1p(1e4 * m[i] / m[i].sum()))[0, 1]
                               for i in range(1, T)])
        row["truth_corr_with_minus_level"] = round(float(lev_rule), 3)
        out[name][tg] = row
        print(name, tg, json.dumps(row))
with open(os.path.join(HERE, "level_check.json"), "w") as fh:
    json.dump(out, fh, indent=1)
