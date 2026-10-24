"""Composition dynamics in archetype space (descriptive, all bins): is the proportion trajectory smooth enough for
flow extrapolation (linear / OT-Markov) to pay off?

For K in {3, 4, 6, 8, K_cv} the weighted-KL NMF of all bins gives proportions h_b. Reported per K:
  lag-1 correlation of the proportion changes dh_t = h_t+1 - h_t with dh_t-1 (pooled over transitions and states,
  cell-weighted: transitions touching bins with < 50 cells reported separately), share of |dh| carried by
  transitions touching a < 50-cell bin, mean total-variation step 0.5 |dh|_1, and the in-sample one-step error of
  persistence vs linear extrapolation vs balanced OT-Markov (eps 0.05) on h (strict in time: dh_t-1 -> dh_t).
Output: composition.json.
"""
import json
import os

import numpy as np

import common
import nmf_kl
from forecasts import NMFMix


def analyse(d, K):
    X = d.CP.T
    om = d.nc * d.lib / 1e4
    W, H = nmf_kl.fit_subset(X, K, om)
    h = nmf_kl.to_props(W)
    LH = np.log1p(H)
    C = ((LH[:, None, :] - LH[None, :, :]) ** 2).mean(2)
    C = C / C[~np.eye(K, dtype=bool)].mean()
    dh = np.diff(h, axis=0)
    small = np.array([min(d.nc[t], d.nc[t + 1]) < 50 for t in range(d.T)])
    a, b, w = [], [], []
    for t in range(1, d.T):
        a.append(dh[t]); b.append(dh[t - 1]); w.append(not (small[t] or small[t - 1]))
    a, b, w = np.array(a), np.array(b), np.array(w)
    corr = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    err = {"persist": [], "linear": [], "ot_markov": []}
    for t in range(1, d.T):
        hp = h[: t + 1]
        tgt = h[t + 1]
        for k, f in (("persist", "persist"), ("linear", "lin1"), ("ot_markov", "ot0.05")):
            err[k].append(0.5 * np.abs(NMFMix.forecast_props(hp, C, f) - tgt).sum())
    return {"K": K, "dh_lag1_corr_all": corr(a, b), "dh_lag1_corr_bins_ge50cells": corr(a[w], b[w]) if w.sum() > 2 else None,
            "share_abs_dh_in_small_bin_transitions": float(np.abs(dh[small]).sum() / np.abs(dh).sum()),
            "mean_tv_step": float(0.5 * np.abs(dh).sum(1).mean()),
            "one_step_tv_error_mean": {k: float(np.mean(v)) for k, v in err.items()},
            "one_step_tv_error_no_small_bins": {k: float(np.mean([e for e, s in zip(v, small[1:]) if not s]))
                                                for k, v in err.items()},
            "linear_beats_persist_share": float(np.mean(np.array(err["linear"]) < np.array(err["persist"]))),
            "ot_beats_persist_share": float(np.mean(np.array(err["ot_markov"]) < np.array(err["persist"]))),
            "props": [[round(float(x), 3) for x in r] for r in h]}


def main():
    st = json.load(open(os.path.join(common.OUT, "structure.json")))
    out = {}
    for name in common.SNAPS:
        d = common.Data(name)
        kcv = st[name]["nmf_cv"]["K_within_1pct"]
        out[name] = {f"K{K}": analyse(d, K) for K in sorted({3, 4, 6, 8, kcv})}
        for k, v in out[name].items():
            print(name, k, {kk: vv for kk, vv in v.items() if kk != "props"}, flush=True)
    json.dump(out, open(os.path.join(common.OUT, "composition.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
