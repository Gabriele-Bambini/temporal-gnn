"""Judge 1: independent minimal re-implementation of the headline claims (own loader, own masks, own features).

Strict rolling origin: the prediction for transition i (bin i -> i+1) uses bins <= i only; learned weights are fit on
transitions j = 1..i-1 (labels need bins <= i). Same fallback for every method at i = 0, 1 (orig: box-4 law + finite-
cell head; l1/linear: i = 0 zero, i = 1 last change), so only i >= 2 discriminates.
Methods: trend (repeat last change); level (orig: box-4 law + finite-cell head = DistBridge-without-OT equivalent;
l1/linear: box-4 reversion); regime (lags 1-3 + box-4/8 reversion, 5 level classes, WLS, half-life 0.5, raw scale);
stack (my NASS-lite: lags, box 2/4/8 reversion, my own known-noise Kalman with empirical-Bayes signal variance
[reversion + one-bin momentum], log s.e., nonzero fraction, SNR interactions; orig adds the finite-cell head, P(all
zero), zero-today; per-transition z-scoring, class-interacted ridge, recency weights) with h fixed 0.5 and with h by
rolling origin over {0.5, 1, 2, 4, inf}.
Masks: 20 draws of a random 25 % of genes per transition, seed 4242 (NOT the designs' seed).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import dileo_protocol  # noqa: E402
import dist_bridge as DB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15)),
         # held-out non-human builds (never used by the analysts or the designs); bins with >= 7 cells
         "gastrulation": list(range(3, 35)), "pancreas": list(range(0, 15))}
HS = (0.5, 1.0, 2.0, 4.0, np.inf)
NCLS = 5


def load(name):
    cache = os.path.join(HERE, f"_x_{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"]
    n = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=[0]))
    st = ("s00_mean", "s01_median", "s02_std", "s03_frac_nonzero", "s04_sum")
    cols = [f"t{b:03d}_{s}" for b in SNAPS[name] for s in st]
    node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=cols, nrows=n)
    X = np.stack([np.stack([node[f"t{b:03d}_{s}"].to_numpy(np.float64) for s in st], 1) for b in SNAPS[name]])
    np.savez(cache, X=X)
    return X


def tfun(tg, now, nxt):
    return np.asarray(dileo_protocol.TARGET_NP[tg](np.maximum(now, 0).astype(np.float32),
                                                   np.maximum(nxt, 0).astype(np.float32)), np.float64)


def zs(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


def lclass(m):
    lv = np.where(m > 0, np.log10(np.where(m > 0, m, 1)), -9)
    return np.digitize(lv, [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])


def pearson(y, p):
    if np.std(y) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def p_at(y, p, k=200):
    up = len(np.intersect1d(np.argsort(-y)[:k], np.argsort(-p)[:k])) / k
    dn = len(np.intersect1d(np.argsort(y)[:k], np.argsort(p)[:k])) / k
    return up, dn


def J_all(y, p, K):
    r = pearson(y, p)
    r = 0.0 if np.isnan(r) else r
    u, d = p_at(y, p, K)
    return r + u + d


class Data:
    def __init__(self, name):
        X = load(name)
        self.name, self.X = name, X
        self.B, self.G = X.shape[0], X.shape[1]
        self.T = self.B - 1
        self.m = np.maximum(X[:, :, 0].T, 0)
        self.s = X[:, :, 2].T
        self.f = X[:, :, 3].T
        self.n = np.array([DB.n_cells(X[b]) for b in range(self.B)])
        self.lib = self.m.sum(0)
        self.CP = 1e4 * self.m / self.lib[None, :]


def box(Z, n, i, k):
    lo = max(0, i - k + 1)
    w = n[lo:i + 1]
    return (Z[:, lo:i + 1] * w).sum(1) / w.sum()


def kalman_eb(Z, s, n, kap):
    """My own causal local-level Kalman per gene: R = known variance of the bin mean (population sd -> /(n-1)),
    Poisson floor from the causal running mean; signal variance q_b = moment estimate of past first differences
    minus their known noise (bins <= b), shrunk toward the median of the gene's expression class."""
    G, B = Z.shape
    w = np.cumsum(n)
    zbar = np.cumsum(Z * n[None, :], 1) / w[None, :]
    R = (np.maximum((kap[None, :] * s) ** 2, kap[None, :] * zbar) + kap[None, :] ** 2 / n[None, :]) / np.maximum(n - 1, 1)[None, :]
    ex = np.diff(Z, axis=1) ** 2 - R[:, :-1] - R[:, 1:]
    cex = np.cumsum(ex, 1)
    out = np.zeros((G, B))
    x, P = Z[:, 0].copy(), R[:, 0].copy()
    out[:, 0] = x
    for b in range(1, B):
        raw = cex[:, b - 1] / b
        cls = lclass(zbar[:, b])
        pri = np.zeros(G)
        for c in range(NCLS):
            sel = cls == c
            if sel.any():
                pri[sel] = max(float(np.median(raw[sel])), 1e-12)
        q = np.maximum((b * raw + 4.0 * pri) / (b + 4.0), 0.05 * pri)
        Pm = P + q
        K = Pm / (Pm + R[:, b])
        x = x + K * (Z[:, b] - x)
        P = (1 - K) * Pm
        out[:, b] = x
    return out, R


def features(d, tg, Y):
    if tg == "l1":
        Z, kap = d.CP, 1e4 / d.lib
    else:
        Z, kap = d.m, np.ones(d.B)
    kal, R = kalman_eb(Z, d.s, d.n, kap)
    T = d.T
    if tg == "orig":
        trip = [DB.triplet(d.X[b]) for b in range(d.B)]
        law4 = DB.box_pool(trip, list(d.n), 4)
    tf = (lambda a, b: tfun(tg, a, b)) if tg != "l1" else (lambda a, b: tfun("l1", a, b))
    # for l1 the base scale is CP10k; TARGET_NP l1 renormalises, so feeding CP10k vectors is exact
    F, OBS = [], []
    for i in range(T):
        Zi = Z[:, i]
        cols = [Y[i - k] if i - k >= 0 else np.zeros(d.G) for k in (1, 2, 3)]
        r2, r4, r8 = (tf(Zi, box(Z, d.n, i, k)) for k in (2, 4, 8))
        rk = tf(Zi, kal[:, i])
        mom = tf(kal[:, i - 1], kal[:, i]) if i >= 1 else np.zeros(d.G)
        se = np.log(np.sqrt(R[:, i]) / (Zi + kap[i] / d.n[i]) + 1e-6)
        cols += [r2, r4, r8, rk, mom, se, d.f[:, i], zs(cols[0]) * zs(se), zs(r4) * zs(se)]
        if tg == "orig":
            ob = DB.expected_target(law4[i], d.m[:, i], d.n[i], np.random.default_rng([77, i]))
            p0 = (1 - np.clip(law4[i][:, 1], 0, 1)) ** max(int(round(d.n[i])), 1)
            cols += [ob, p0, (d.m[:, i] == 0).astype(float)]
            OBS.append(np.asarray(ob, np.float64))
        F.append(np.column_stack(cols))
    return F, OBS


def design(Fi, cls, standardise):
    Fz = np.column_stack([zs(c) for c in Fi.T]) if standardise else Fi
    return np.hstack([Fz * (cls == c)[:, None] for c in range(NCLS)] + [(cls == c)[:, None].astype(float) for c in range(NCLS)])


def learner(Xs, Ys, T, h, lam, i, standardise):
    j = np.arange(1, i)
    w = np.ones(len(j)) if not np.isfinite(h) else 0.5 ** ((i - 1 - j) / h)
    p = Xs[0].shape[1]
    A, b = np.zeros((p, p)), np.zeros(p)
    for wj, jj in zip(w, j):
        y = zs(Ys[jj]) if standardise else Ys[jj]
        A += wj * Xs[jj].T @ Xs[jj]
        b += wj * Xs[jj].T @ y
    A += lam * max(float(np.mean(np.diag(A))), 1e-12) * np.eye(p)
    return Xs[i] @ np.linalg.solve(A, b)


def score(Y, P, mk):
    D, T = mk.shape[0], len(Y)
    S = np.full((D, T, 3), np.nan)
    for i in range(T):
        for k in range(D):
            yy, pp = Y[i][mk[k, i]], P[i][mk[k, i]]
            S[k, i, 0] = pearson(yy, pp)
            S[k, i, 1:] = p_at(yy, pp)
    return S


def summ(S, idx):
    per = np.nanmean(S[:, idx], 1)
    return [round(float(x), 4) for x in per.mean(0)] + [round(float(per[:, 0].std()), 4)]


def paired(Sa, Sb, idx):
    a, b = np.nanmean(Sa[:, idx, 0], 0), np.nanmean(Sb[:, idx, 0], 0)
    dl = np.nan_to_num(a) - np.nan_to_num(b)
    nz = dl[np.abs(dl) > 1e-12]
    return {"dPCC": round(float(dl.mean()), 4), "better": int((dl > 0).sum()), "worse": int((dl < 0).sum()),
            "p": round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None}


def main(name):
    t0 = time.time()
    d = Data(name)
    rng = np.random.default_rng(4242)
    nt = int(round(0.25 * d.G))
    mk = np.array([[np.sort(rng.permutation(d.G)[:nt]) for _ in range(d.T)] for _ in range(20)])
    Kall = int(round(200 / nt * d.G))
    res = {"n_cells": d.n.round().tolist()}
    for tg in ("orig", "l1", "linear"):
        Y = np.stack([tfun(tg, d.m[:, i], d.m[:, i + 1]) for i in range(d.T)])
        F, OBS = features(d, tg, Y)
        cls = [lclass(d.m[:, i]) for i in range(d.T)]
        Z = d.CP if tg == "l1" else d.m
        fb0 = OBS[0] if tg == "orig" else np.zeros(d.G)
        fb1 = OBS[1] if tg == "orig" else Y[0]
        P = {"trend": [np.zeros(d.G)] + [Y[i - 1] for i in range(1, d.T)]}
        P["level"] = OBS if tg == "orig" else [fb0, fb1] + [tfun(tg, Z[:, i], box(Z, d.n, i, 4)) for i in range(2, d.T)]
        # regime: raw (unstandardised) lags + box-4/8 reversion, h = 0.5 fixed
        Xr = [design(F[i][:, [0, 1, 2, 4, 5]], cls[i], False) for i in range(d.T)]
        P["regime_h0.5"] = [fb0, fb1] + [learner(Xr, Y, d.T, 0.5, 1e-8, i, False) for i in range(2, d.T)]
        Xs = [design(F[i], cls[i], True) for i in range(d.T)]
        allh = {h: [None, None] + [learner(Xs, Y, d.T, h, 1e-3, i, True) for i in range(2, d.T)] for h in HS}
        P["stack_h0.5"] = [fb0, fb1] + allh[0.5][2:]
        Jh = {h: np.array([np.nan, np.nan] + [J_all(Y[i], allh[h][i], Kall) for i in range(2, d.T)]) for h in HS}
        ro, chosen = [fb0, fb1], []
        for i in range(2, d.T):
            if i == 2:
                h = 1.0
            else:
                lo = max(2, i - 8)
                h = max(HS, key=lambda hh: np.mean(Jh[hh][lo:i]))
            chosen.append(str(h))
            ro.append(allh[h][i])
        P["stack_RO"] = ro
        S = {k: score(Y, v, mk) for k, v in P.items()}
        rngs = {"all": list(range(d.T)), "i>=2": list(range(2, d.T)), "i>=3": list(range(3, d.T))}
        tab = {k: {r: summ(S[k], ix) for r, ix in rngs.items()} for k in S}
        pr = {f"{a} vs {b}": paired(S[a], S[b], rngs["i>=2"]) for a, b in
              (("stack_RO", "trend"), ("stack_h0.5", "trend"), ("stack_RO", "regime_h0.5"),
               ("stack_RO", "level"), ("regime_h0.5", "trend"), ("stack_h0.5", "regime_h0.5"))}
        res[tg] = {"table_pcc_up_down_pccsd": tab, "paired_i>=2": pr, "RO_h_chosen": chosen}
        print(f"== {name} {tg} ({time.time() - t0:.0f}s)", flush=True)
        for k in tab:
            print(f"  {k:12s} all {tab[k]['all'][:3]}  i>=2 {tab[k]['i>=2']}  i>=3 {tab[k]['i>=3'][:3]}", flush=True)
        for k, v in pr.items():
            print(f"  paired {k:26s} {v}", flush=True)
    fo = os.path.join(HERE, "indep_results.json")
    prev = json.load(open(fo)) if os.path.exists(fo) else {}
    prev[name] = res
    json.dump(prev, open(fo, "w"), indent=1)


if __name__ == "__main__":
    for nm in sys.argv[1:] or list(SNAPS):
        main(nm)
