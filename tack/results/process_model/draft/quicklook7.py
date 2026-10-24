"""Quick look 7 (strict mini-forecaster of quicklook5, level groups): training-transition weights
 uniform | recency 0.5**((i-1-j)/h), h=1 | analog: exp(-(S_j - S_hat_i)^2 / (2*0.15^2)) with S_j the noise-corrected
 signal share of past transition j (bins j, j+1 <= i) and S_hat_i = S_{i-1} | recency x analog.
All quantities use bins <= i. PCC per transition, all genes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicklook import load, corr
from quicklook5 import feats, groups

for name in ["gastrulation-human", "pancreas-human"]:
    m, med, s, f, tot, n = load(name)
    G, B = m.shape; T = B - 1
    lib = m.sum(0); kap = 1e4 / lib; CP = m * kap[None, :]; L = np.log1p(CP)
    Y = np.diff(L, axis=1).T
    zbar = np.cumsum(m * n[None, :], 1) / np.cumsum(n)[None, :]
    s2 = np.maximum(s ** 2, zbar) + 1.0 / n[None, :]
    RL = kap[None, :] ** 2 * s2 / np.maximum(n - 1, 1)[None, :] / (1 + CP) ** 2
    S = np.array([1 - (RL[:, i] + RL[:, i + 1]).sum() / ((Y[i] - Y[i].mean()) ** 2).sum() for i in range(T)])
    Fs = [feats(L, CP, n, Y, i) for i in range(T)]
    Gs = [groups("level", m, RL, Y, i) for i in range(T)]
    print(f"===== {name}")
    for wk in ("uniform", "recency", "analog", "rec_x_analog"):
        pcs = np.full(T, np.nan)
        for i in range(2, T):
            js = np.arange(1, i)
            if wk == "uniform":
                wj = np.ones(len(js))
            elif wk == "recency":
                wj = 0.5 ** ((i - 1 - js) / 1.0)
            else:
                wa = np.exp(-(S[js] - S[i - 1]) ** 2 / (2 * 0.15 ** 2))
                wj = wa if wk == "analog" else wa * 0.5 ** ((i - 1 - js) / 2.0)
            pred = np.zeros(G)
            for g in np.unique(Gs[i]):
                A = np.zeros((7, 7)); bvec = np.zeros(7)
                for j, ww in zip(js, wj):
                    sel = Gs[j] == g
                    if sel.sum() < 20:
                        continue
                    X = np.column_stack([Fs[j][sel], np.ones(sel.sum())])
                    w = ww / max(Y[j].var(), 1e-12)
                    A += w * X.T @ X; bvec += w * X.T @ Y[j][sel]
                if A[6, 6] == 0:
                    continue
                A += 1e-3 * np.mean(np.diag(A)) * np.eye(7)
                beta = np.linalg.solve(A, bvec)
                sel = Gs[i] == g
                pred[sel] = np.column_stack([Fs[i][sel], np.ones(sel.sum())]) @ beta
            pcs[i] = corr(Y[i], pred)
        print(f"  weights={wk:12s}: PCC i>=2 {np.nanmean(pcs[2:]):+.3f}  i>=3 {np.nanmean(pcs[3:]):+.3f}  per-t {np.round(pcs, 2).tolist()}")
