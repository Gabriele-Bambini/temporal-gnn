"""Quick look 4: in gastrulation quiet transitions, which simple strict predictors work (pooled and per class)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicklook import load, corr
EDGES = [0.0, 0.1, 1.0, 10.0, 100.0, np.inf]
for name in ["gastrulation-human", "pancreas-human"]:
    m, med, s, f, tot, n = load(name)
    G, B = m.shape; T = B - 1
    lib = m.sum(0); kap = 1e4 / lib; CP = m * kap[None, :]; L = np.log1p(CP)
    Y = np.diff(L, axis=1).T
    RL = kap[None, :] ** 2 * s ** 2 / np.maximum(n - 1, 1)[None, :] / (1 + CP) ** 2
    S = np.array([1 - (RL[:, i] + RL[:, i + 1]).sum() / ((Y[i] - Y[i].mean()) ** 2).sum() for i in range(T)])
    quiet = [i for i in range(3, T) if S[i] < 0.2 and S[i - 1] < 0.2] or list(range(3, T))
    print(f"===== {name}  transitions used: {quiet}")
    rows = {}
    for i in quiet:
        c = np.digitize(CP[:, i], EDGES[1:-1]); c[CP[:, i] <= 0] = -1
        P = {"-lag1": -Y[i - 1], "-lag1-lag2": -(Y[i - 1] + Y[i - 2]) , "lag1": Y[i-1]}
        for k in (2, 3, 4, 8):
            lo = max(0, i - k + 1)
            P[f"boxU{k}"] = L[:, lo:i + 1].mean(1) - L[:, i]
            w = n[lo:i + 1]
            P[f"boxW{k}"] = np.log1p((CP[:, lo:i + 1] * w).sum(1) / w.sum()) - L[:, i]
            P[f"pastU{k}"] = L[:, max(0, i - k):i].mean(1) - L[:, i]      # mean of the k bins BEFORE t minus today
        for k_, v in P.items():
            rows.setdefault(k_, {"all": []})
            rows[k_]["all"].append(corr(Y[i], v))
            for cc in range(-1, 4):
                sel = c == cc
                if sel.sum() > 100:
                    rows[k_].setdefault(cc, []).append(corr(Y[i][sel], v[sel]))
        # SS share by class
        yc = Y[i] - Y[i].mean()
        for cc in range(-1, 4):
            rows.setdefault("SSshare", {}).setdefault(cc, []).append((yc[c == cc] ** 2).sum() / (yc ** 2).sum())
    hdr = "  pred        all    " + "  ".join(f"c{cc:+d}" for cc in range(-1, 4))
    print(hdr)
    for k_, d in rows.items():
        print(f"  {k_:10s} " + (f"{np.mean(d['all']):+.3f}  " if 'all' in d else "        ") +
              "  ".join(f"{np.mean(d[cc]):+.3f}" if cc in d else "  nan " for cc in range(-1, 4)))
