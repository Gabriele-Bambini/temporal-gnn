"""Quick look 2: (a) is R = kappa^2 s^2/(n-1) the right l1 noise (ratio estimator), checked in the quiet phase
and by the model-free lag-1 covariance; (b) does a time-varying process speed (strict, per class) in a plain local
level Kalman reach NASS in quiet transitions. Diagnostic only; scored on all genes (no masks), PCC per transition.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicklook import load, corr  # noqa: E402

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
NASS = os.path.join(ROOT, "tack", "results", "signal_analysis", "design_best-signal", "results.json")
EDGES = [0.0, 0.1, 1.0, 10.0, 100.0, np.inf]


def cls_of(cp):
    c = np.digitize(cp, EDGES[1:-1])
    c[cp <= 0] = -1
    return c


def main(name):
    m, med, s, f, tot, n = load(name)
    G, B = m.shape
    lib = m.sum(0)
    kap = 1e4 / lib
    CP = m * kap[None, :]
    L = np.log1p(CP)
    RL = (kap[None, :] ** 2) * s ** 2 / np.maximum(n - 1, 1)[None, :] / (1 + CP) ** 2
    Y = np.diff(L, axis=1).T
    T = B - 1
    nass = np.array(json.load(open(NASS))[name]["targets"]["l1"]["per_transition_pcc_up_down"]["full"], float)[:, 0]
    S = np.array([(((Y[i] - Y[i].mean()) ** 2).sum() - (RL[:, i] + RL[:, i + 1]).sum()) /
                  ((Y[i] - Y[i].mean()) ** 2).sum() for i in range(T)])
    print(f"===== {name}")
    # (a) noise calibration: lag-1 covariance noise estimate  -Cov(y_{i-1}, y_i | class of bin i)  vs  mean RL_i
    print("  class  | -cov(y_i-1,y_i)/mean RL_i  all transitions  | quiet transitions (share<0.2) | obs var/pred var quiet")
    quiet = [i for i in range(1, T) if S[i] < 0.2 and S[i - 1] < 0.2]
    for c in range(len(EDGES) - 1):
        ra, rq, vq = [], [], []
        for i in range(1, T):
            sel = (cls_of(CP[:, i]) == c) & (CP[:, i - 1] > 0) & (CP[:, i + 1] > 0)
            if sel.sum() < 30:
                continue
            a, b = Y[i - 1][sel], Y[i][sel]
            cv = -np.mean((a - a.mean()) * (b - b.mean()))
            r = cv / np.mean(RL[sel, i])
            ra.append(r)
            if i in quiet:
                rq.append(r)
                vq.append(np.var(b) / np.mean(RL[sel, i] + RL[sel, i + 1]))
        print(f"   {EDGES[c]:g}-{EDGES[c + 1]:g}  {np.median(ra) if ra else np.nan:6.2f}   "
              f"{np.median(rq) if rq else np.nan:6.2f} (k={len(rq)})   {np.median(vq) if vq else np.nan:6.2f}")
    # (b) local level Kalman in L scale, R = RL (floored), Q_{c,t} = class signal variance estimated strictly from
    # the last transition(s) (bins <= t): q_c(t) = mean over class c of (y_{t-1}^2 - RL_{t-1} - RL_t), floored.
    def run(qmode, win=2, floor=1e-6):
        x = L[:, 0].copy()
        P = RL[:, 0] + 1e-3
        xs = [x.copy()]
        preds = [None] * T
        for t in range(1, B):
            c = cls_of(CP[:, t - 1])
            Q = np.full(G, floor)
            if qmode == "tv" and t >= 2:
                for k in range(-1, len(EDGES) - 1):
                    sel = c == k
                    if sel.sum() < 30:
                        continue
                    vals = []
                    for j in range(max(0, t - 1 - win), t - 1):     # transitions j -> j+1 with j+1 <= t-1
                        v = Y[j][sel] ** 2 - RL[sel, j] - RL[sel, j + 1]
                        vals.append(v.mean())
                    Q[sel] = max(np.mean(vals), floor)
            elif qmode == "const" and t >= 2:
                for k in range(-1, len(EDGES) - 1):
                    sel = c == k
                    if sel.sum() < 30:
                        continue
                    vals = [(Y[j][sel] ** 2 - RL[sel, j] - RL[sel, j + 1]).mean() for j in range(0, t - 1)]
                    Q[sel] = max(np.mean(vals), floor)
            Pm = P + Q
            Rt = np.maximum(RL[:, t], 1e-8)
            K = Pm / (Pm + Rt)
            x = x + K * (L[:, t] - x)
            P = (1 - K) * Pm
            xs.append(x.copy())
            if t < B - 1:
                preds[t] = x - L[:, t]                       # local level forecast of y_t
        return preds, xs
    for qmode in ("const", "tv"):
        pr, xs = run(qmode)
        pc = np.array([corr(Y[t], pr[t]) if pr[t] is not None else np.nan for t in range(T)])
        mom = np.array([corr(Y[t], xs[t] - xs[t - 1]) if t >= 1 else np.nan for t in range(T)])
        idx = list(range(2, T))
        q = [i for i in idx if i in quiet]
        sig = [i for i in idx if S[i - 1] >= 0.2]
        print(f"  LL-Kalman Q={qmode:5s}: PCC i>=2 {np.nanmean(pc[idx]):+.3f} | quiet-prev {np.nanmean(pc[q]):+.3f} "
              f"(NASS {np.nanmean(nass[q]):+.3f}, k={len(q)}) | prev-signal {np.nanmean(pc[sig]):+.3f} "
              f"(NASS {np.nanmean(nass[sig]):+.3f}, k={len(sig)}) | NASS i>=2 {np.nanmean(nass[idx]):+.3f}")
        print(f"     per transition: {np.round(pc, 2).tolist()}")
        print(f"     filtered momentum PCC: {np.round(mom, 2).tolist()}")
    print(f"  NASS per transition:   {np.round(nass, 2).tolist()}")
    print(f"  signal share:          {np.round(S, 2).tolist()}")


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["gastrulation-human", "pancreas-human"]:
        main(nm)
