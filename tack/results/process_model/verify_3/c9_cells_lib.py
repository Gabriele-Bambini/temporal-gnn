"""C9: per-bin cell count n_b and library per cell lib_b = sum_g s00_mean, read from the raw node CSVs (first N rows),
ALL bins (gastrulation 0..31 so the 2-point trend exists at the first scored transition 2 -> 3).

Scored transitions t -> t+1: gastrulation t = 2..30 (29), pancreas t = 0..13 (14).
persistence error |log n_{t+1} - log n_t|, trend error |log n_{t+1} - (2 log n_t - log n_{t-1})| (needs bin t-1).
dlog x_t = log x_{t+1} - log x_t over the scored transitions; autocorrelations are Pearson over consecutive pairs.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as C

BINS = {"gastrulation-human": 32, "pancreas-human": 15}
SCORED_T = {"gastrulation-human": list(range(2, 31)), "pancreas-human": list(range(0, 14))}


def read(name):
    N = len(pd.read_csv(os.path.join(C.GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    cols = [f"t{b:03d}_{s}" for b in range(BINS[name]) for s in ("s00_mean", "s04_sum")]
    df = pd.read_csv(os.path.join(C.GRN, f"{name}_node.csv"), usecols=cols, nrows=N)
    n, lib = [], []
    for b in range(BINS[name]):
        m = df[f"t{b:03d}_s00_mean"].to_numpy(np.float64)
        s = df[f"t{b:03d}_s04_sum"].to_numpy(np.float64)
        ok = m > 0
        n.append(float(np.median(s[ok] / m[ok])))
        lib.append(float(np.maximum(m, 0).sum()))
    return np.array(n), np.array(lib)


def r(a, b):
    return round(float(np.corrcoef(a, b)[0, 1]), 3), round(float(stats.spearmanr(a, b).correlation), 3), len(a)


def main():
    out = {}
    for name in C.NAMES:
        n, lib = read(name)
        ln, ll = np.log(n), np.log(lib)
        ts = SCORED_T[name]
        pers = np.array([abs(ln[t + 1] - ln[t]) for t in ts])
        has_prev = [t for t in ts if t - 1 >= 0]
        has_prev_in = [t for t in ts if t - 1 >= ts[0]]
        trend = {t: abs(ln[t + 1] - (2 * ln[t] - ln[t - 1])) for t in has_prev}
        pers_d = {t: abs(ln[t + 1] - ln[t]) for t in ts}
        o = {"n_cells_rounded": [int(round(x)) for x in n], "lib_per_cell": [round(x, 1) for x in lib]}
        for lab, sub in (("all scored with bin t-1 available", has_prev), ("t-1 inside scored range", has_prev_in)):
            o[f"median_abs_err_dlogn[{lab}]"] = {
                "n_transitions": len(sub),
                "persistence": round(float(np.median([pers_d[t] for t in sub])), 4),
                "trend_2pt": round(float(np.median([trend[t] for t in sub])), 4),
                "mean_persistence": round(float(np.mean([pers_d[t] for t in sub])), 4),
                "mean_trend_2pt": round(float(np.mean([trend[t] for t in sub])), 4),
                "trend_better_count": int(sum(trend[t] < pers_d[t] for t in sub)),
            }
        o["median_abs_err_persistence_all_scored"] = {"n": len(ts), "median": round(float(np.median(pers)), 4)}
        dn = np.array([ln[t + 1] - ln[t] for t in ts])
        dl = np.array([ll[t + 1] - ll[t] for t in ts])
        o["corr(dlogn_{t+1}, dlogn_t) [pearson, spearman, pairs]"] = r(dn[1:], dn[:-1])
        big = [k for k in range(1, len(ts)) if min(n[ts[k - 1]], n[ts[k - 1] + 1], n[ts[k] + 1]) >= 100]
        if len(big) >= 4:
            o["corr(dlogn_{t+1}, dlogn_t) pairs with all 3 bins >= 100 cells"] = r(dn[big], dn[[k - 1 for k in big]])
            o["median_abs_err_dlogn bins>=100 (t-1,t,t+1)"] = {
                "n": len(big), "persistence": round(float(np.median([pers_d[ts[k]] for k in big])), 4),
                "trend_2pt": round(float(np.median([trend[ts[k]] for k in big])), 4)}
            o["dlog_lib lag1 autocorr, bins>=100"] = r(dl[big], dl[[k - 1 for k in big]])
            o["corr(dlog lib, dlog n) same t, bins>=100"] = r(dl[big], dn[big])
        if ts[0] >= 1:  # include the change into the first scored bin
            dn_ext = np.array([ln[t + 1] - ln[t] for t in [ts[0] - 1] + ts])
            o["corr(dlogn_{t+1}, dlogn_t) incl. change into first scored bin"] = r(dn_ext[1:], dn_ext[:-1])
        o["dlog_lib"] = {"sd": round(float(dl.std(ddof=1)), 4), "mean": round(float(dl.mean()), 4),
                         "lag1_autocorr [pearson, spearman, pairs]": r(dl[1:], dl[:-1]),
                         "corr_with_dlogn_same_t": r(dl, dn),
                         "corr(dlog lib_t, dlog n_{t+1})": r(dl[:-1], dn[1:]),
                         "corr(dlog n_t, dlog lib_{t+1})": r(dn[:-1], dl[1:])}
        o["dlog_n"] = {"sd": round(float(dn.std(ddof=1)), 4), "mean": round(float(dn.mean()), 4)}
        o["corr(log lib_b, log n_b) over scored bins"] = r(ll[ts[0]:ts[-1] + 2], ln[ts[0]:ts[-1] + 2])
        out[name] = o
        print(name, json.dumps(o, indent=1))
    json.dump(out, open(os.path.join(C.HERE, "c9_cells_lib.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
