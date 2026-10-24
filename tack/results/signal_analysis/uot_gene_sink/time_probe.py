"""Timing / sanity probe on one transition: couplings (balanced, unbalanced at several lam0), agreement of the
bucketed balanced ablation with dist_bridge.coupling + sb_step on a gene subset, observation-head cost."""
import sys
import time

import numpy as np

import uot_core as U

name = sys.argv[1] if len(sys.argv) > 1 else "gastrulation-human"
i = int(sys.argv[2]) if len(sys.argv) > 2 else 10
X, trip, n, m = U.load(name)
print(name, "n cells", np.round(n).astype(int).tolist(), flush=True)
for L in (1, 3):
    A, B = trip[i - L], trip[i]
    t0 = time.time()
    cpb = U.bal_coupling(A, B, L)
    Hb = U.bal_step(B, cpb, L, 1.0)
    print(f"L={L} balanced {time.time() - t0:.1f}s err={cpb['err']:.1e} it={cpb['iters']} "
          f"buckets={[len(b['ii']) for b in cpb['buckets']]}", flush=True)
    # reference: dist_bridge on a subset of 600 genes (its own grid, tol 1e-9)
    sub = np.where((A[:, 1] > 1e-6) & (B[:, 1] > 1e-6))[0][::15][:600]
    t0 = time.time()
    cpr = U.DB.coupling(A[sub], B[sub], L, scale="count")
    Hr = U.DB.sb_step(B[sub], cpr, L, 1.0)
    print(f"   dist_bridge subset {time.time() - t0:.1f}s; corr of m-change {np.corrcoef(Hr[:, 0] - B[sub, 0], Hb[sub, 0] - B[sub, 0])[0, 1]:.4f}"
          f" max|diff| {np.abs(Hr[:, 0] - Hb[sub, 0]).max():.2e} (scale {np.abs(Hr[:, 0] - B[sub, 0]).mean():.2e})", flush=True)
    tr = 2 * B[:, 0] - A[:, 0]
    for lam0 in (0.5, 2, 8, 32, 128):
        t0 = time.time()
        cp = U.uot_coupling(A, B, L, lam0)
        dt = time.time() - t0
        dg = U.uot_diag(cp)
        H = U.uot_step(B, cp, L, 1.0)
        print(f"L={L} lam0={lam0} {dt:.1f}s err={cp["err"]:.1e} over={cp.get("over",0):.1e} it={cp['iters']} sink/fA={dg['sink_over_fA']:.3f} "
              f"forced={dg['forced_death_over_fA']:.3f} src/fB={dg['source_over_fB']:.3f} forcedB={dg['forced_birth_over_fB']:.3f} "
              f"dlvl={dg['death_level_ratio_median']:.2f} blvl={dg['birth_level_ratio_median']:.2f} "
              f"d3={np.round(dg['death_rate_by_third'], 3).tolist()} b3={np.round(dg['birth_rate_by_third'], 3).tolist()}", flush=True)
        print(f"   corr(dm, trend dm) uot {np.corrcoef(H[:, 0] - B[:, 0], tr - B[:, 0])[0, 1]:.3f} bal "
              f"{np.corrcoef(Hb[:, 0] - B[:, 0], tr - B[:, 0])[0, 1]:.3f}; corr(uot, bal) {np.corrcoef(H[:, 0] - B[:, 0], Hb[:, 0] - B[:, 0])[0, 1]:.3f}", flush=True)
t0 = time.time()
p = U.heads(H, m[i], n[i], [0, i], ("l1",))
print(f"l1 head {time.time() - t0:.1f}s", flush=True)
t0 = time.time()
p = U.heads(H, m[i], n[i], [0, i], ("orig",))
print(f"orig head {time.time() - t0:.1f}s", flush=True)
