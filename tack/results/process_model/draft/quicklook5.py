"""Quick look 5 (strict mini-forecaster, diagnostic): same 6 l1 features (lag1, lag2, lag3, rev box2/4/8), ridge per
group fitted on past transitions (each transition weighted equally via 1/Var(y_j)), grouping of genes by
 (a) expression level of bin i (NASS's 5 classes on m_i), (b) quantiles of the known noise variance RL_{g,i},
 (c) quantiles of an EB signal-to-noise ratio v_g / RL_{g,i} (v_g = past noise-corrected mean squared change, strict),
 (d) (c) crossed with zero-today.  PCC per transition on all genes; mean over i >= 2 (and i >= 3)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicklook import load, corr

def feats(L, CP, n, Y, i):
    G = L.shape[0]
    F = [Y[i - k] if i - k >= 0 else np.zeros(G) for k in (1, 2, 3)]
    for k in (2, 4, 8):
        lo = max(0, i - k + 1); w = n[lo:i + 1]
        F.append(np.log1p((CP[:, lo:i + 1] * w).sum(1) / w.sum()) - L[:, i])
    return np.column_stack(F)

def groups(kind, m, RL, Y, i, ng=5):
    G = m.shape[0]
    mi = m[:, i]
    if kind == "level":
        lv = np.where(mi > 0, np.log10(np.where(mi > 0, mi, 1)), -9)
        return np.digitize(lv, [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])
    zero = mi <= 0
    if kind == "noise":
        x = np.log(RL[:, i] + 1e-12)
    else:
        if i < 2:
            v = np.ones(G)
        else:
            d = np.stack([Y[j] ** 2 - RL[:, j] - RL[:, j + 1] for j in range(0, i)])   # transitions j<i: bins <= i
            v = d.mean(0)
            prior = np.median(v[~zero]) if (~zero).any() else 1e-6
            k = i
            v = np.maximum((k * v + 4 * prior) / (k + 4), 1e-3 * abs(prior) + 1e-9)
        x = np.log(v / (RL[:, i] + 1e-9))
    g = np.zeros(G, int)
    q = np.quantile(x[~zero], np.linspace(0, 1, ng + 1)[1:-1]) if (~zero).sum() > 10 else []
    g[~zero] = 1 + np.searchsorted(q, x[~zero])
    return g          # 0 = zero today, 1..ng by quantile

for name in ["gastrulation-human", "pancreas-human"]:
    m, med, s, f, tot, n = load(name)
    G, B = m.shape; T = B - 1
    lib = m.sum(0); kap = 1e4 / lib; CP = m * kap[None, :]; L = np.log1p(CP)
    Y = np.diff(L, axis=1).T
    zbar = np.cumsum(m * n[None, :], 1) / np.cumsum(n)[None, :]
    s2 = np.maximum(s ** 2, zbar) + 1.0 / n[None, :]
    RL = kap[None, :] ** 2 * s2 / np.maximum(n - 1, 1)[None, :] / (1 + CP) ** 2
    Fs = [feats(L, CP, n, Y, i) for i in range(T)]
    print(f"===== {name}")
    for kind in ("level", "noise", "snr"):
        Gs = [groups(kind, m, RL, Y, i) for i in range(T)]
        pcs = np.full(T, np.nan)
        for i in range(2, T):
            pred = np.zeros(G)
            for g in np.unique(Gs[i]):
                A = np.zeros((7, 7)); bvec = np.zeros(7)
                for j in range(1, i):
                    sel = Gs[j] == g
                    if sel.sum() < 20:
                        continue
                    X = np.column_stack([Fs[j][sel], np.ones(sel.sum())])
                    w = 1.0 / max(Y[j].var(), 1e-12)
                    A += w * X.T @ X; bvec += w * X.T @ Y[j][sel]
                if A[6, 6] == 0:
                    continue
                A += 1e-3 * np.mean(np.diag(A)) * np.eye(7)
                beta = np.linalg.solve(A, bvec)
                sel = Gs[i] == g
                pred[sel] = np.column_stack([Fs[i][sel], np.ones(sel.sum())]) @ beta
            pcs[i] = corr(Y[i], pred)
        print(f"  group={kind:5s}: PCC i>=2 {np.nanmean(pcs[2:]):+.3f}  i>=3 {np.nanmean(pcs[3:]):+.3f}  per-t {np.round(pcs, 2).tolist()}")
