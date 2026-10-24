"""C3: noise-corrected l1 signal share per transition, its persistence, and two strict model pairs.

share_i = S_i / var_g(y_i), S_i = var_g(y_i) - mean_g(R_i + R_{i+1}) (all genes). R: common.R_l1, primary WITHOUT the
1/n^2 floor (that version reproduces the CONTEXT noise budget, s0_anchor.json), floored version reported too.
Pair 1 (local-level filters, R floored so zero genes get a finite gain): class Q re-estimated from the last 2
  transitions vs class Q from all past transitions (constant in time up to sampling); classes at bin b = zero
  running mean + quartiles of the log running mean (bins <= b); Q floored at 0.05 R.
Pair 2 (6-feature per-class ridge, classes as above at bin t): features lag1, lag2, rev = x_{t|t} - L_t (constant
  class-Q filter), mom = x_{t|t} - x_{t-1|t-1}, logse = 0.5 log R_t, rev x logse (z-scored product); features
  z-scored within the transition, target z-scored per transition, class intercepts, ridge 1e-3 * mean diag,
  trained on transitions j = 1..t-1. Weights: recency 0.5 ** ((t-1-j)/h), h = 1 (fixed a priori, nass default);
  analog x recency: times exp(-dist^2/2), dist = || (z_{j-1}, z_{j-2}) - (z_{t-1}, z_{t-2}) || / sd(z_0..z_{t-1}),
  z = share (or log|S|) of the transitions BEFORE the one being matched (all known at forecast time t).
Scoring: common.score (1000 s + i masks), i >= 2.
"""
import json
import os

import numpy as np
from scipy import stats

import common as C
import kf

H_REC, LAM, BW = 1.0, 1e-3, 1.0


def shares(Y, R):
    T = len(Y)
    v = np.array([Y[i].var() for i in range(T)])
    noise = np.array([(R[:, i] + R[:, i + 1]).mean() for i in range(T)])
    S = v - noise
    return S / v, S, v


def ac(x):
    x = np.asarray(x)
    return [round(float(np.corrcoef(x[1:], x[:-1])[0, 1]), 3), round(float(stats.spearmanr(x[1:], x[:-1]).correlation), 3),
            len(x) - 1]


def ac2(x, ok):
    """lag-1 correlation over consecutive pairs (i, i+1) with ok[i] and ok[i+1]."""
    x = np.asarray(x)
    k = np.where(ok[1:] & ok[:-1])[0]
    a, b = x[k + 1], x[k]
    return [round(float(np.corrcoef(a, b)[0, 1]), 3), round(float(stats.spearmanr(a, b).correlation), 3), len(k)]


def zs(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def ridge_forecasts(d, L, Y, X, R, z_hist, analog):
    T, G = Y.shape
    n_cls = 5
    design, targ = [], []
    for t in range(T):
        f = [Y[t - 1] if t >= 1 else np.zeros(G), Y[t - 2] if t >= 2 else np.zeros(G), X[:, t] - L[:, t],
             X[:, t] - X[:, t - 1] if t >= 1 else np.zeros(G), 0.5 * np.log(R[:, t])]
        f.append(zs(f[2]) * zs(f[4]))
        F = np.column_stack([zs(c) for c in f])
        cls = kf.classes_running(d, t, n_cls)
        Xd = np.hstack([F * (cls == c)[:, None] for c in range(n_cls)] + [(cls == c)[:, None].astype(float) for c in range(n_cls)])
        design.append(Xd)
        targ.append(zs(Y[t]))
    Gm = [x.T @ x for x in design]
    bv = [x.T @ y for x, y in zip(design, targ)]
    P = [None] * T
    for t in range(2, T):
        j = np.arange(1, t)
        w = 0.5 ** ((t - 1 - j) / H_REC)
        if analog:
            zpast = np.asarray(z_hist[:t])                       # z_0..z_{t-1}: needs bins <= t
            sd = zpast.std() if len(zpast) > 1 and zpast.std() > 0 else 1.0
            a = np.ones(len(j))
            for q, jj in enumerate(j):
                cur = [z_hist[t - 1]] + ([z_hist[t - 2]] if t - 2 >= 0 else [])
                ref = [z_hist[jj - 1]] + ([z_hist[jj - 2]] if jj - 2 >= 0 else [])
                k = min(len(cur), len(ref))
                dist = np.sqrt(np.sum((np.array(cur[:k]) - np.array(ref[:k])) ** 2)) / (sd * BW)
                a[q] = np.exp(-0.5 * dist ** 2)
            w = w * a
        A = sum(wi * Gm[jj] for wi, jj in zip(w, j))
        A = A + LAM * max(float(np.mean(np.diag(A))), 1e-12) * np.eye(A.shape[0])
        b = sum(wi * bv[jj] for wi, jj in zip(w, j))
        P[t] = design[t] @ np.linalg.solve(A, b)
    return P


def main():
    out = {}
    for name in C.NAMES:
        d = C.load(name)
        Y = d.Y("l1")
        L = d.L
        T = d.T
        o = {}
        b0 = d.bins[0]
        lab = [f"i={i} bins {b0 + i}->{b0 + i + 1} cells {int(round(d.n[i]))}->{int(round(d.n[i + 1]))}" for i in range(T)]
        for rk, R in (("R_nofloor", C.R_l1(d, False)), ("R_floor", C.R_l1(d, True))):
            sh, S, v = shares(Y, R)
            pos = S > 0
            logS = np.log(np.abs(S))
            o[rk] = {"share": [round(float(x), 3) for x in sh], "S": [float(f"{x:.4g}") for x in S],
                     "mean_share": round(float(sh.mean()), 3),
                     "lag1_autocorr_share [pearson, spearman, pairs]": ac(sh),
                     "lag1_autocorr_log|S| [pearson, spearman, pairs]": ac(logS),
                     "n_S<=0": int((~pos).sum()), "transitions_S<=0": [lab[i] for i in np.where(~pos)[0]],
                     "lag1_autocorr_log(S), pairs with both S>0 [pearson, spearman, pairs]": ac2(logS, pos),
                     "lag1_autocorr_share_i>=1": ac(sh[1:]),
                     "lag1_autocorr_log|S|_i>=1": ac(logS[1:])}
            if name == "gastrulation-human":
                o[rk]["share<=0.2"] = [f"{lab[i]} share {sh[i]:.3f}" for i in np.where(sh <= 0.2)[0]]
            else:
                k = int(np.argmin(sh))
                o[rk]["min_share"] = f"{lab[k]} share {sh[k]:.3f}"
                o[rk]["min_share_i>=1"] = f"{lab[1 + int(np.argmin(sh[1:]))]} share {sh[1:].min():.3f}"
            if rk == "R_nofloor":
                sh_main, S_main = sh, S
        # ---- models (R floored inside the filter)
        Rf = C.R_l1(d, True)
        mk = C.masks(d.G, T)
        q2, v2 = kf.q_class(d, L, Rf, 2)
        qc, vc = kf.q_class(d, L, Rf, None)
        X2, _, _ = kf.local_level(L, Rf, q2)
        Xc, _, _ = kf.local_level(L, Rf, qc)
        preds = {"LL classQ last-2": list(kf.ll_forecast(L, X2)), "LL classQ constant(all past)": list(kf.ll_forecast(L, Xc))}
        for k in preds:
            preds[k][0] = None
            preds[k][1] = None
        preds["ridge6 recency"] = ridge_forecasts(d, L, Y, Xc, Rf, sh_main, False)
        preds["ridge6 analog(share)xrecency"] = ridge_forecasts(d, L, Y, Xc, Rf, sh_main, True)
        preds["ridge6 analog(log|S|)xrecency"] = ridge_forecasts(d, L, Y, Xc, Rf, np.log(np.abs(S_main)), True)
        preds["trend"] = [None, None] + [Y[t - 1] for t in range(2, T)]
        rows = {k: C.score(Y, p, mk) for k, p in preds.items()}
        idx = list(range(2, T))
        o["scores_i>=2"] = {k: C.summ(r, idx) for k, r in rows.items()}
        per = {k: np.nanmean(r[:, :, 0], 0) for k, r in rows.items()}
        o["per_transition_pcc"] = {k: [None if not np.isfinite(x) else round(float(x), 3) for x in v] for k, v in per.items()}
        for a, b in (("LL classQ last-2", "LL classQ constant(all past)"), ("ridge6 analog(share)xrecency", "ridge6 recency"),
                     ("ridge6 analog(log|S|)xrecency", "ridge6 recency")):
            dl = per[a][idx] - per[b][idx]
            o[f"paired PCC {a} - {b}"] = {"mean": round(float(dl.mean()), 4), "better": int((dl > 1e-9).sum()),
                                           "worse": int((dl < -1e-9).sum()),
                                           "wilcoxon_p": round(float(stats.wilcoxon(dl).pvalue), 4)}
        o["classQ_last2_values_per_b"] = {int(k): [float(f"{x:.3g}") for x in v] for k, v in v2.items()}
        o["classQ_const_values_per_b"] = {int(k): [float(f"{x:.3g}") for x in v] for k, v in vc.items()}
        out[name] = o
        print("==", name)
        for k in ("R_nofloor", "R_floor"):
            print(" ", k, {kk: vv for kk, vv in o[k].items() if kk not in ("share", "S")})
            print("   share", o[k]["share"])
        for k, v in o["scores_i>=2"].items():
            print(f"  {k:34s} {C.fmt(v)}  (spearman {v['spearman']:+.3f})")
        for k in o:
            if k.startswith("paired"):
                print(" ", k, o[k])
    json.dump(out, open(os.path.join(C.HERE, "c3_share.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
