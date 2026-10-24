"""Which genes make the unbalanced forecast depart from the trend; per-bucket convergence."""
import sys

import numpy as np

import uot_core as U

name, i, L, lam0 = "gastrulation-human", 10, int(sys.argv[1]), float(sys.argv[2])
X, trip, n, m = U.load(name)
A, B = trip[i - L], trip[i]
cp = U.uot_coupling(A, B, L, lam0)
H = U.uot_step(B, cp, L, 1.0)
cpb = U.bal_coupling(A, B, L)
Hb = U.bal_step(B, cpb, L, 1.0)
du, db, dt = H[:, 0] - B[:, 0], Hb[:, 0] - B[:, 0], (B[:, 0] - A[:, 0]) / L
print("corr uot/trend", np.corrcoef(du, dt)[0, 1], "bal/trend", np.corrcoef(db, dt)[0, 1])
for bk in cp["buckets"]:
    ii = bk["ii"]
    K = bk["z"].shape[1]
    tot = cp["fA"][ii] + cp["fB"][ii]
    r = np.corrcoef(du[ii], dt[ii])[0, 1]
    print(f"K={K:3d} n={len(ii):5d} corr(uot,trend)={r:.3f} corr(bal,trend)={np.corrcoef(db[ii], dt[ii])[0, 1]:.3f} "
          f"sd uot {du[ii].std():.3g} trend {dt[ii].std():.3g} mean m {B[ii, 0].mean():.3g}")
big = np.argsort(-np.abs(du - dt))[:8]
for g in big:
    print(f"gene {g}: mA={A[g, 0]:.3f} mB={B[g, 0]:.3f} fA={A[g, 1]:.3f} fB={B[g, 1]:.3f} uot dm={du[g]:.3f} bal dm={db[g]:.3f} trend dm={dt[g]:.3f} fhat={H[g, 1]:.3f}")
    for bk in cp["buckets"]:
        w = np.where(bk["ii"] == g)[0]
        if len(w):
            k = w[0]
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            print("   z", bk["z"][k][:12], "\n   a", bk["a"][k][:12], "\n   b", (bk["c"] + bk["beta"])[k][:12],
                  "\n   s", bk["s"][k][:12], "\n   beta", bk["beta"][k][:12], "\n   d", bk["d"][k][:12], "\n   v", bk["v"][k][:12])
