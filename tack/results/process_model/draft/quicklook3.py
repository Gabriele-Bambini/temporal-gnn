"""Quick look 3 (diagnostics for the claims):
 (1) integrality: K = f * n and s04_sum are integers (exact binomial observation of expressing cells);
 (2) noise mechanism: per-bin library CV^2 floor from s^2/m^2 of high genes; corrected s~^2 = s^2 - m^2 CV2_lib;
     -cov(y_{i-1}, y_i)/R ratio by class before/after the correction (quiet gastrulation transitions and all);
 (3) within-bin f-implied level (rev_f) partial correlation given the full NASS-like control set (in-sample per
     transition, diagnostic);
 (4) signal variance S_t (absolute, noise-corrected) vs log n and (dlog n)^2 (composition / effective step);
 (5) fixed local-level Kalman: Q relative floor, R floored at m = 0, noise correction c(class) from past lag-1
     covariance (strict), time-varying Q by class vs constant; PCC per transition on all genes vs NASS.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicklook import load, corr, pcorr, trend_fit  # noqa: E402

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
NASS = os.path.join(ROOT, "tack", "results", "signal_analysis", "design_best-signal", "results.json")
EDGES = [0.0, 0.1, 1.0, 10.0, 100.0, np.inf]


def cls_of(cp):
    c = np.digitize(cp, EDGES[1:-1])
    c[cp <= 0] = -1
    return c


def zs(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def main(name):
    m, med, s, f, tot, n = load(name)
    G, B = m.shape
    T = B - 1
    lib = m.sum(0)
    kap = 1e4 / lib
    CP = m * kap[None, :]
    L = np.log1p(CP)
    Y = np.diff(L, axis=1).T
    nm1 = np.maximum(n - 1, 1)
    print(f"===== {name}")
    # (1) integrality
    K = f * n[None, :]
    ex = m > 0
    print(f"  (1) max |K - round(K)| over expressed = {np.abs(K - np.round(K))[ex].max():.2e}; "
          f"max |sum - round(sum)| = {np.abs(tot - np.round(tot)).max():.2e}; "
          f"max |n - round(n)| = {np.abs(n - np.round(n)).max():.2e}; "
          f"median K among expressed = {np.median(K[ex]):.0f}; frac expressed genes with K<=3: {(K[ex] <= 3.5).mean():.3f}")
    # (2) library CV^2 floor: 5th percentile over genes with m > 1 of (s^2 - m)/m^2 (Poisson removed)
    cv2 = np.zeros(B)
    for b in range(B):
        sel = m[:, b] > 1
        cv2[b] = max(np.quantile((s[sel, b] ** 2 - m[sel, b]) / m[sel, b] ** 2, 0.05), 0) if sel.sum() > 50 else 0
    print(f"  (2) CV2_lib floor per bin: {np.round(cv2, 3).tolist()}")
    s2c = np.maximum(s ** 2 - m ** 2 * cv2[None, :], m)                     # never below Poisson
    RL0 = kap[None, :] ** 2 * s ** 2 / nm1[None, :] / (1 + CP) ** 2
    RL1 = kap[None, :] ** 2 * s2c / nm1[None, :] / (1 + CP) ** 2
    S0 = np.array([(((Y[i] - Y[i].mean()) ** 2).sum() - (RL0[:, i] + RL0[:, i + 1]).sum()) /
                   ((Y[i] - Y[i].mean()) ** 2).sum() for i in range(T)])
    quiet = [i for i in range(1, T) if S0[i] < 0.2 and S0[i - 1] < 0.2]
    print("      class   -cov/R (all)  raw -> corrected   | quiet raw -> corrected   | obs var / pred var quiet raw -> corr")
    for c in range(len(EDGES) - 1):
        r0, r1, q0, q1, v0, v1 = [], [], [], [], [], []
        for i in range(1, T):
            sel = (cls_of(CP[:, i]) == c) & (CP[:, i - 1] > 0) & (CP[:, i + 1] > 0)
            if sel.sum() < 30:
                continue
            a, b = Y[i - 1][sel], Y[i][sel]
            cv = -np.mean((a - a.mean()) * (b - b.mean()))
            r0.append(cv / RL0[sel, i].mean())
            r1.append(cv / RL1[sel, i].mean())
            if i in quiet:
                q0.append(r0[-1]); q1.append(r1[-1])
                v0.append(b.var() / (RL0[sel, i] + RL0[sel, i + 1]).mean())
                v1.append(b.var() / (RL1[sel, i] + RL1[sel, i + 1]).mean())
        md = lambda v: np.median(v) if v else np.nan
        print(f"      {EDGES[c]:g}-{EDGES[c + 1]:g}   {md(r0):6.2f} -> {md(r1):6.2f}   |   {md(q0):6.2f} -> {md(q1):6.2f} "
              f"(k={len(q0)})  |  {md(v0):5.2f} -> {md(v1):5.2f}")
    # (3) rev_f partial correlation, full control set
    se_all = np.log(np.sqrt(np.maximum(RL0, 1e-12)) + 1e-6)
    res = {"rev_f": [], "rev_f|full": [], "rev_med|full": [], "disp|full": []}
    for i in range(3, T):
        mi, fi, si = m[:, i], f[:, i], s[:, i]
        ok = (fi > 0) & (fi < 1) & (mi > 0)
        lam = -np.log1p(-np.clip(fi, 0, 1 - 1e-12))
        lr = np.zeros(G)
        lr[ok] = np.log(mi[ok]) - trend_fit(np.log(lam[ok]), np.log(mi[ok]))
        rev_f = np.log1p(CP[:, i] * np.exp(-lr)) - L[:, i]
        dsp = np.zeros(G)
        dsp[ok] = np.log(si[ok] ** 2 / mi[ok]) - trend_fit(np.log(mi[ok]), np.log(si[ok] ** 2 / mi[ok]))
        okm = med[:, i] > 0
        rev_med = np.zeros(G)
        lmr = np.log(mi[okm]) - trend_fit(np.log(med[okm, i]), np.log(mi[okm]))
        rev_med[okm] = np.log1p(CP[okm, i] * np.exp(-lmr)) - L[okm, i]
        boxes = []
        for k in (2, 4, 8):
            lo = max(0, i - k + 1)
            w = n[lo:i + 1]
            boxes.append(np.log1p((CP[:, lo:i + 1] * w).sum(1) / w.sum()) - L[:, i])
        lags = [Y[i - 1], Y[i - 2], Y[i - 3]]
        se = se_all[:, i]
        ctrl_small = [Y[i - 1], Y[i - 2], boxes[1]]
        ctrl = lags + boxes + [se, fi, zs(Y[i - 1]) * zs(se), zs(boxes[1]) * zs(se)]
        # class interactions of the main controls (5 classes as in NASS) to mimic class-wise coefficients
        cl = np.digitize(np.where(mi > 0, np.log10(np.where(mi > 0, mi, 1)), -9), [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])
        ctrl_cls = []
        for c in range(5):
            ind = (cl == c).astype(float)
            ctrl_cls += [ind] + [v * ind for v in (Y[i - 1], boxes[0], boxes[1], boxes[2])]
        y = Y[i]
        res["rev_f"].append(pcorr(y, rev_f, ctrl_small))
        res["rev_f|full"].append(pcorr(y, rev_f, ctrl + ctrl_cls))
        res["rev_med|full"].append(pcorr(y, rev_med, ctrl + ctrl_cls))
        res["disp|full"].append(pcorr(y, dsp, ctrl + ctrl_cls))
    for k, v in res.items():
        v = np.array(v)
        print(f"  (3) {k:13s} mean {np.nanmean(v):+.3f}  positive {int((v > 0).sum())}/{len(v)}  (transitions i>=3)")
    # (4) absolute signal variance per transition vs cells
    Sabs = np.array([((Y[i] - Y[i].mean()) ** 2).mean() - (RL1[:, i] + RL1[:, i + 1]).mean() for i in range(T)])
    ln = np.log(np.sqrt(n[:-1] * n[1:]))
    dln2 = np.diff(np.log(n)) ** 2
    Xd = np.column_stack([np.ones(T), ln, dln2])
    beta, *_ = np.linalg.lstsq(Xd, Sabs, rcond=None)
    fit = Xd @ beta
    r2 = 1 - ((Sabs - fit) ** 2).sum() / ((Sabs - Sabs.mean()) ** 2).sum()
    print(f"  (4) S_abs (x1e3): {np.round(Sabs * 1e3, 2).tolist()}")
    print(f"      corr(S_abs, log n_geo) {corr(Sabs, ln):+.2f}; corr(S_abs, dlog n^2) {corr(Sabs, dln2):+.2f}; "
          f"R2 both {r2:.2f}; lag-1 corr(S_abs) {corr(Sabs[1:], Sabs[:-1]):+.2f}")
    # (5) fixed local-level Kalman
    nass = np.array(json.load(open(NASS))[name]["targets"]["l1"]["per_transition_pcc_up_down"]["full"], float)[:, 0]
    zbar = np.cumsum(CP * n[None, :], 1) / np.cumsum(n)[None, :]
    s2f = np.maximum(s2c, zbar / kap[None, :])                               # floor: running mean (raw scale)
    RLf = (kap[None, :] ** 2 * s2f + kap[None, :] ** 2 / n[None, :]) / nm1[None, :] / (1 + CP) ** 2

    def ccorr(t):
        """noise correction per class from past lag-1 covariances (transitions j-1, j with j+1 <= t)."""
        out = np.ones(len(EDGES))
        for c in range(-1, len(EDGES) - 1):
            r = []
            for j in range(1, t):
                sel = (cls_of(CP[:, j]) == c) & (CP[:, j - 1] > 0) & (CP[:, j + 1] > 0)
                if sel.sum() < 30:
                    continue
                a, b = Y[j - 1][sel], Y[j][sel]
                r.append(-np.mean((a - a.mean()) * (b - b.mean())) / RLf[sel, j].mean())
            out[c + 1] = np.clip(np.median(r), 0.3, 1.2) if r else 1.0
        return out

    def run(qmode, use_c, win=2, frel=0.05):
        x = L[:, 0].copy()
        P = RLf[:, 0].copy()
        preds = [None] * T
        for t in range(1, B):
            c = cls_of(CP[:, t - 1])
            cc = ccorr(t) if use_c else np.ones(len(EDGES))
            Rg = lambda j: cc[cls_of(CP[:, j]) + 1] * RLf[:, j]
            Q = frel * RLf[:, t - 1]
            if t >= 2:
                js = range(max(0, t - 1 - win), t - 1) if qmode == "tv" else range(0, t - 1)
                Rs = {j: Rg(j) for j in set(list(js) + [j + 1 for j in js])}
                for k in range(-1, len(EDGES) - 1):
                    sel = c == k
                    if sel.sum() < 30:
                        continue
                    est = np.mean([(Y[j][sel] ** 2 - Rs[j][sel] - Rs[j + 1][sel]).mean() for j in js])
                    Q[sel] = np.maximum(est, frel * Rg(t - 1)[sel].mean())
            Pm = P + Q
            Rt = np.maximum(Rg(t), 1e-10)
            Kg = Pm / (Pm + Rt)
            x = x + Kg * (L[:, t] - x)
            P = (1 - Kg) * Pm
            if t < B - 1:
                preds[t] = x - L[:, t]
        return np.array([corr(Y[t], preds[t]) if preds[t] is not None else np.nan for t in range(T)])
    idx = list(range(2, T))
    q = [i for i in idx if i in quiet]
    for qmode in ("const", "tv"):
        for use_c in (False, True):
            pc = run(qmode, use_c)
            print(f"  (5) LL Q={qmode:5s} c={int(use_c)}: i>=2 {np.nanmean(pc[idx]):+.3f} (NASS {np.nanmean(nass[idx]):+.3f}) | "
                  f"quiet {np.nanmean(pc[q]) if q else np.nan:+.3f} (NASS {np.nanmean(nass[q]) if q else np.nan:+.3f}, k={len(q)}) "
                  f"| per-t {np.round(pc, 2).tolist()}")


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["gastrulation-human", "pancreas-human"]:
        main(nm)
