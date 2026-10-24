"""Quick looks for the process-model design (diagnostics only, not forecasters; in-sample partial correlations are
labelled as such). numpy only. Reads the cached node arrays of design_best-signal (same first-N genes, snapshot bins).
"""
import os
import sys

import numpy as np

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
SRC = os.path.join(ROOT, "tack", "results", "signal_analysis", "design_best-signal")
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}


def load(name):
    X = np.load(os.path.join(SRC, f"_node_{name}.npz"))["X"]          # (B, G, 5) mean, median, sd, frac, sum
    m = np.maximum(X[:, :, 0].T, 0)
    med, s, f, tot = X[:, :, 1].T, X[:, :, 2].T, X[:, :, 3].T, X[:, :, 4].T
    n = np.array([np.median(tot[m[:, b] > 0, b] / m[m[:, b] > 0, b]) for b in range(m.shape[1])])
    return m, med, s, f, tot, n


def pcorr(y, x, Z):
    D = np.column_stack([np.ones(len(y))] + list(Z))
    ry = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    rx = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]
    den = np.sqrt((ry ** 2).sum() * (rx ** 2).sum())
    return float((ry * rx).sum() / den) if den > 0 else np.nan


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def trend_fit(x, y, nb=60):
    """Binned-median trend of y on x (within one bin, across genes), linear interpolation."""
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    q = np.unique(q)
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
    cx = np.array([np.median(x[idx == k]) if (idx == k).any() else np.nan for k in range(len(q) - 1)])
    cy = np.array([np.median(y[idx == k]) if (idx == k).any() else np.nan for k in range(len(q) - 1)])
    ok = np.isfinite(cx) & np.isfinite(cy)
    return np.interp(x, cx[ok], cy[ok])


def main(name):
    m, med, s, f, tot, n = load(name)
    G, B = m.shape
    lib = m.sum(0)
    kap = 1e4 / lib
    CP = m * kap[None, :]
    L = np.log1p(CP)
    R = (kap[None, :] ** 2) * s ** 2 / np.maximum(n - 1, 1)[None, :]           # Var of CP10k mean (known)
    RL = R / (1 + CP) ** 2                                                        # delta-method, log1p scale
    Y = np.diff(L, axis=1).T                                                      # (T, G) l1 target
    T = B - 1
    print(f"===== {name}  G={G} B={B}  n={np.round(n).astype(int).tolist()}")
    print(f"  lib (UMI/cell) = {np.round(lib).astype(int).tolist()}")
    dl = np.diff(np.log(n))
    print(f"  n: corr(dlog n_t+1, dlog n_t) = {corr(dl[1:], dl[:-1]):+.2f}; "
          f"RW |dlog n| median {np.median(np.abs(dl)):.2f}; "
          f"AR(2-pt trend) |err| median {np.median(np.abs(dl[1:] - dl[:-1])):.2f}")
    dlib = np.diff(np.log(lib))
    print(f"  lib: corr(dlog lib_t+1, dlog lib_t) = {corr(dlib[1:], dlib[:-1]):+.2f}; "
          f"corr(dlog lib, dlog n) = {corr(dlib, dl):+.2f}; sd dlog lib {dlib.std():.3f}")

    # ---- noise calibration: E[y^2] vs RL_t + RL_t+1, by level class, and model-free lag-1 check
    cls_edges = [0.0, 0.1, 1.0, 10.0, 100.0]
    print("  class (mean CP10k over the pair)  share SS(y)  noise/SS  lag1 corr obs  lag1 pred by noise")
    for c in range(len(cls_edges)):
        lo = cls_edges[c]
        hi = cls_edges[c + 1] if c + 1 < len(cls_edges) else np.inf
        ss, nz, l1o, l1p = [], [], [], []
        for i in range(T):
            lvl = 0.5 * (CP[:, i] + CP[:, i + 1])
            sel = (lvl > lo) & (lvl <= hi) if c > 0 else (lvl <= hi) & (lvl > 0)
            yc = Y[i] - Y[i].mean()
            ss.append((yc[sel] ** 2).sum() / (yc ** 2).sum())
            nz.append((RL[sel, i] + RL[sel, i + 1]).sum() / max((yc[sel] ** 2).sum(), 1e-12))
            if i >= 1:
                sel2 = sel & (0.5 * (CP[:, i - 1] + CP[:, i]) > 0)
                a, b = Y[i - 1][sel2], Y[i][sel2]
                l1o.append(corr(a, b))
                v1 = (a - a.mean()).var() * len(a)
                v2 = (b - b.mean()).var() * len(b)
                l1p.append(-RL[sel2, i].sum() / np.sqrt(v1 * v2))
        print(f"   ({lo:g},{hi:g}]   {np.mean(ss):.3f}   {np.mean(nz):.3f}   {np.nanmean(l1o):+.3f}   {np.nanmean(l1p):+.3f}")

    # ---- per-transition signal variance (noise-corrected) and its persistence
    S = np.array([((Y[i] - Y[i].mean()) ** 2).sum() - (RL[:, i] + RL[:, i + 1]).sum() for i in range(T)])
    V = np.array([((Y[i] - Y[i].mean()) ** 2).sum() for i in range(T)])
    sh = S / V
    print(f"  signal share per transition: {np.round(sh, 2).tolist()}")
    print(f"  corr(signal share_t, share_t-1) = {corr(sh[1:], sh[:-1]):+.2f}; "
          f"corr(share, log min(n_t,n_t+1)) = {corr(sh, np.log(np.minimum(n[:-1], n[1:]))):+.2f}")
    print(f"  corr(log S_t, log S_t-1) = {corr(np.log(np.maximum(S[1:],1e-9)), np.log(np.maximum(S[:-1],1e-9))):+.2f}")

    # ---- within-bin second measurement: mean vs frac-implied level, dispersion excess, median
    out = {"rev_f": [], "disp": [], "rev_med": [], "rev_f_raw": [], "rev_box4": [], "lag1": []}
    for i in range(1, T):
        mi, fi, si = m[:, i], f[:, i], s[:, i]
        ok = (fi > 0) & (fi < 1) & (mi > 0)
        lam = -np.log1p(-np.clip(fi, 0, 1 - 1e-12))
        lr = np.zeros(G)
        lr[ok] = np.log(mi[ok]) - trend_fit(np.log(lam[ok]), np.log(mi[ok]))   # excess mean given detection rate
        CPf = CP[:, i] * np.exp(-lr)
        rev_f = np.log1p(CPf) - L[:, i]                                             # move toward f-implied level
        dsp = np.zeros(G)
        dsp[ok] = np.log(si[ok] ** 2 / mi[ok]) - trend_fit(np.log(mi[ok]), np.log(si[ok] ** 2 / mi[ok]))
        okm = med[:, i] > 0
        rev_med = np.zeros(G)
        if okm.sum() > 200:
            lmr = np.log(mi[okm]) - trend_fit(np.log(med[okm, i]), np.log(mi[okm]))
            rev_med[okm] = np.log1p(CP[okm, i] * np.exp(-lmr)) - L[okm, i]
        lo = max(0, i - 3)
        w = n[lo:i + 1]
        box4 = (CP[:, lo:i + 1] * w).sum(1) / w.sum()
        rb4 = np.log1p(box4) - L[:, i]
        lag1 = Y[i - 1]
        y = Y[i]
        ctrl = [lag1, rb4, Y[i - 2] if i >= 2 else np.zeros(G)]
        out["rev_f_raw"].append(corr(y, rev_f))
        out["rev_f"].append(pcorr(y, rev_f, ctrl))
        out["disp"].append(pcorr(y, dsp * (L[:, i] > 0), ctrl))
        out["rev_med"].append(pcorr(y, rev_med, ctrl) if okm.sum() > 200 else np.nan)
        out["rev_box4"].append(corr(y, rb4))
        out["lag1"].append(corr(y, lag1))
    for k, v in out.items():
        v = np.array(v, float)
        print(f"  within-bin {k:9s}: mean {np.nanmean(v):+.3f}  (+ in {int((v > 0).sum())}/{int(np.isfinite(v).sum())})")
    print(f"  frac genes with median>0 per bin: {np.round((med > 0).mean(0), 3).tolist()}")


if __name__ == "__main__":
    for nm in sys.argv[1:] or list(SNAPS):
        main(nm)
