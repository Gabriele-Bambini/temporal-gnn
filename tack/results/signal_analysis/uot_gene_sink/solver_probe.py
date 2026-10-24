"""Per-bucket convergence of the augmented sink / source Sinkhorn (lambda scale = max(mu_B, mean positive-part var))."""
import sys
import time

import numpy as np

import uot_core as U
import uot_sinkhorn

name, i, L, lam0 = "gastrulation-human", 10, int(sys.argv[1]), float(sys.argv[2])
ist = int(sys.argv[3]) if len(sys.argv) > 3 else 12
X, trip, n, m = U.load(name)
A, B = trip[i - L], trip[i]
fA, muA, vA, fB, muB, vB = U._laws(A, B)
idx = np.where((fA > 0) & (fB > 0))[0]
for sel, z, pA, pB in U.grids(muA[idx], vA[idx], muB[idx], vB[idx]):
    ii = idx[sel]
    g, K = z.shape
    a, b = fA[ii, None] * pA, fB[ii, None] * pB
    tot = fA[ii] + fB[ii]
    a2 = np.concatenate([a, fB[ii, None]], 1) / tot[:, None]
    b2 = np.concatenate([b, fA[ii, None]], 1) / tot[:, None]
    lam = lam0 * np.maximum(muB[ii], 0.5 * (vA[ii] + vB[ii]))
    C = np.zeros((g, K + 1, K + 1))
    C[:, :K, :K] = (z[:, :, None] - z[:, None, :]) ** 2 / 2
    C[:, :K, K] = lam[:, None]
    C[:, K, :K] = lam[:, None]
    eps = np.maximum(0.25 * L * muB[ii], 1e-3)
    t0 = time.time()
    pi, err, it = uot_sinkhorn.solve(a2, b2, C, eps, tol=1e-7, iters_stage=ist, max_final=3000)
    rowerr = np.abs(pi.sum(2) - a2).max(1)
    print(f"K={K:3d} n={g:5d} {time.time() - t0:.1f}s it={it} err={err:.1e} n_bad(>1e-5)={(rowerr > 1e-5).sum()} "
          f"lam/eps med={np.median(lam / eps):.1f} max={np.max(lam / eps):.1f} Cmax/eps med={np.median(C.max((1, 2)) / eps):.0f}", flush=True)
