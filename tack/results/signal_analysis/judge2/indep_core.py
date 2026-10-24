"""Judge 2: independent minimal re-implementation of the NASS core on the l1 target (own code, own masks, own metric
call), to check that its l1 gain over the trend is not an artefact of nass.py / dsdata.py.

Reads the node CSVs directly (first N rows, N = rows of <name>_y_outdeg.csv) and also checks the cached arrays used
by the designs. Per gene, base scale CP10k (kap_b = 1e4 / lib_b):
  known noise   R_gb = kap_b^2 max(s_gb^2, mrun_gb) / (n_b - 1)   (s = population sd; mrun = cell-weighted running
                mean of m over bins <= b, a Poisson floor)
  EB signal var q_gb = shrink(mean_{k<b} [(CP_k+1 - CP_k)^2 - R_k - R_k+1], class median, nu = 4), floor 5 % of prior
  local-level Kalman on CP with (q, R); box-k cell-weighted pools (k = 4, 8) of bins <= b
Features of transition i (bins <= i only): last three observed l1 changes, reversion moves
  log1p(pool) - log1p(CP_i) for box4, box8 and the Kalman level, and the Kalman momentum log1p(K_i) - log1p(K_i-1).
Learner: features and past targets z-scored within each transition, coefficients by 5 classes of m_i, least squares
with ridge 1e-3 (scale-free) on transitions 1..i-1, recency weights 0.5^((i-1-j)/h) with h = 1 FIXED (NASS default,
no rolling-origin search). i < 2: repeat last change.
Scoring: 20 own masks (seed 777, last 25 % of each permutation), tack/hpc/dileo_rescore.metrics.
Usage: python indep_core.py <dataset>   -> indep_<dataset>.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
HPC = os.path.join(ROOT, "tack", "hpc")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HPC)
import dileo_protocol  # noqa: E402
import dileo_rescore  # noqa: E402

SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
H, LAM, NU, NCLS = 1.0, 1e-3, 4.0, 5


def load(name):
    n = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    b = SNAPS[name]
    st = ("s00_mean", "s01_median", "s02_std", "s03_frac_nonzero", "s04_sum")
    node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=[f"t{x:03d}_{s}" for x in b for s in st], nrows=n)
    get = lambda s: node[[f"t{x:03d}_{s}" for x in b]].to_numpy(np.float64)
    m, sd, f, sm = np.maximum(get("s00_mean"), 0), get("s02_std"), get("s03_frac_nonzero"), get("s04_sum")
    nc = np.array([np.median(sm[m[:, j] > 0, j] / m[m[:, j] > 0, j]) for j in range(len(b))])
    # cache check against the designs' cached arrays
    chk = {}
    z = np.load(os.path.join(HERE, "..", "design_best-signal", f"_node_{name}.npz"))["X"]
    chk["nass_cache_mean_maxdiff"] = float(np.abs(z[:, :, 0].T - m).max())
    chk["nass_cache_sd_maxdiff"] = float(np.abs(z[:, :, 2].T - sd).max())
    z2 = np.load(os.path.join(HERE, "..", "design_latent-sb", f"_cache_{name}.npz"))
    chk["lsb_cache_mean_maxdiff"] = float(np.abs(np.maximum(z2["M"], 0) - m).max())
    return m, sd, f, nc, chk


def zs(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


def classes(mi):
    lv = np.where(mi > 0, np.log10(np.where(mi > 0, mi, 1)), -9)
    return np.digitize(lv, [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])


def main(name):
    m, sd, f, nc, chk = load(name)
    G, B = m.shape
    T = B - 1
    lib = m.sum(0)
    kap = 1e4 / lib
    CP = m * kap[None, :]
    Lg = np.log1p(CP)
    Y = np.stack([Lg[:, i + 1] - Lg[:, i] for i in range(T)])
    Yref = np.stack([dileo_protocol.TARGET_NP["l1"](m[:, i], m[:, i + 1]) for i in range(T)])
    chk["l1_target_vs_TARGET_NP_maxdiff"] = float(np.abs(Y - Yref).max())
    w = np.cumsum(nc)
    mrun = np.cumsum(m * nc[None, :], 1) / w[None, :]
    R = kap[None, :] ** 2 * np.maximum(sd ** 2, mrun) / np.maximum(nc - 1, 1)[None, :]
    cprun = mrun * kap[None, :]
    # EB signal variance, strict: at bin b uses differences between bins <= b
    ex = np.diff(CP, axis=1) ** 2 - R[:, :-1] - R[:, 1:]
    cum = np.cumsum(ex, 1)
    kal = np.zeros((G, B))
    x, P = CP[:, 0].copy(), R[:, 0].copy()
    kal[:, 0] = x
    for b in range(1, B):
        raw = cum[:, b - 1] / b
        scale = cprun[:, b] ** 2 + kap[b] * cprun[:, b] / 50 + 1e-12
        rel = raw / scale
        cls = classes(mrun[:, b])
        prior = np.zeros(G)
        for c in range(NCLS):
            s = cls == c
            if s.any():
                prior[s] = max(float(np.median(rel[s])), 1e-4)
        prior *= scale
        q = np.maximum((b * raw + NU * prior) / (b + NU), 0.05 * prior)
        Pm = P + q
        K = Pm / (Pm + R[:, b])
        x = x + K * (CP[:, b] - x)
        P = (1 - K) * Pm
        kal[:, b] = x

    def box(i, k):
        lo = max(0, i - k + 1)
        ww = nc[lo:i + 1]
        return (CP[:, lo:i + 1] * ww).sum(1) / ww.sum()

    feats, cl = [], []
    for i in range(T):
        cols = [Y[i - k] if i - k >= 0 else np.zeros(G) for k in (1, 2, 3)]
        cols += [np.log1p(box(i, 4)) - Lg[:, i], np.log1p(box(i, 8)) - Lg[:, i], np.log1p(kal[:, i]) - Lg[:, i],
                 np.log1p(kal[:, i]) - np.log1p(kal[:, i - 1]) if i >= 1 else np.zeros(G)]
        Fi = np.column_stack([zs(c) for c in cols])
        ci = classes(m[:, i])
        X = np.hstack([Fi * (ci == c)[:, None] for c in range(NCLS)] + [(ci == c)[:, None].astype(float) for c in range(NCLS)])
        feats.append(X)
        cl.append(ci)
    Gm = [X.T @ X for X in feats]
    bv = [X.T @ zs(Y[i]) for i, X in enumerate(feats)]
    preds = {"trend": [np.zeros(G)] + [Y[i - 1] for i in range(1, T)], "core_h1": [], "core_h0.5": [],
             "kal_reversion_only": [np.zeros(G)] + [np.log1p(kal[:, i]) - Lg[:, i] for i in range(1, T)]}
    for hname, h in (("core_h1", 1.0), ("core_h0.5", 0.5)):
        for i in range(T):
            if i < 2:
                preds[hname].append(Y[i - 1] if i >= 1 else np.zeros(G))
                continue
            j = np.arange(1, i)
            ww = 0.5 ** ((i - 1 - j) / h)
            A = sum(wj * Gm[jj] for wj, jj in zip(ww, j))
            A = A + LAM * max(float(np.mean(np.diag(A))), 1e-12) * np.eye(A.shape[0])
            bb = sum(wj * bv[jj] for wj, jj in zip(ww, j))
            preds[hname].append(feats[i] @ np.linalg.solve(A, bb))
    rng = np.random.default_rng(777)
    nt = int(round(0.25 * G))
    mk = np.array([[np.sort(rng.permutation(G)[G - nt:]) for _ in range(T)] for _ in range(20)])
    S = {}
    for k, P_ in preds.items():
        sc = np.zeros((20, T, 3))
        for dd in range(20):
            for i in range(T):
                r = dileo_rescore.metrics(Y[i][mk[dd, i]], P_[i][mk[dd, i]])
                sc[dd, i] = [r["pearson_nan0"], r["p_at_200_up"], r["p_at_200_down"]]
        S[k] = sc
    out = {"dataset": name, "checks": chk, "table": {}, "paired_i>=2": {}}
    for k, sc in S.items():
        for rn, lo in (("i>=2", 2), ("all", 0)):
            pd_ = sc[:, lo:].mean(1)
            out["table"].setdefault(k, {})[rn] = [round(float(v), 4) for v in pd_.mean(0)] + \
                [round(float(pd_[:, 0].std()), 4)]
    for a in ("core_h1", "core_h0.5", "kal_reversion_only"):
        da = S[a].mean(0)[2:, 0] - S["trend"].mean(0)[2:, 0]
        out["paired_i>=2"][f"{a} - trend"] = {"dPCC": round(float(da.mean()), 4), "better": int((da > 0).sum()),
                                              "worse": int((da < 0).sum()),
                                              "wilcoxon_p": round(float(stats.wilcoxon(da).pvalue), 5)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"indep_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
