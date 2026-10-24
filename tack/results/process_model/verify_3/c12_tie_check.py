"""C12 check: is the plug-in Spearman < 1 (0.999994) only floating-point tie noise? Gastrulation, tau = 0.1,
mu0 = -6 (the setting chosen at most t), t = 10 and t = 20: order violations of log E_n[(1-p_hat)^n] along p_hat."""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

import common as C
import c12_binomial_head as H

d = C.load("gastrulation-human")
K = np.rint(d.f * d.n[None, :])
import sys
TAU, TS = (1.0, (1,)) if "--t1" in sys.argv else (0.1, (10, 20))
_, kept = H.run_filter(K, d.n, TAU, -6.0, keep=set(TS))
xg, wg = H.gh()
for t in TS:
    p = kept[t] @ H.PG
    pos = K[:, t] > 0
    ph = p[pos]
    a = d.n[t] * np.log1p(-ph)
    b = logsumexp((d.n[t] * np.exp(0.8 * xg))[None, :] * np.log1p(-ph)[:, None] + np.log(wg)[None, :], axis=1)
    o = np.argsort(ph, kind="stable")
    viol_a = np.sum(np.diff(a[o]) > 1e-9 * np.abs(a[o][1:]))
    viol_b = np.sum(np.diff(b[o]) > 1e-9 * np.abs(b[o][1:]))
    r = stats.spearmanr(a, b).correlation
    r_round = stats.spearmanr(np.round(ph, 12), np.round(ph, 12)).correlation
    r_vs_p = (stats.spearmanr(-ph, a).correlation, stats.spearmanr(-ph, b).correlation)
    near = np.sum(np.diff(np.sort(ph)) < 1e-14)
    print(f"t={t} genes={pos.sum()} raw spearman(a,b)={r:.7f} order violations a={viol_a} b={viol_b} "
          f"spearman(-p_hat,a)={r_vs_p[0]:.7f} (-p_hat,b)={r_vs_p[1]:.7f} near-equal p_hat pairs (<1e-14)={near}")
    a2, b2 = np.round(a, 6), np.round(b, 6)
    print(f"   after rounding logs to 1e-6: spearman={stats.spearmanr(a2, b2).correlation:.7f}")
