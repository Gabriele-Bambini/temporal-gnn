"""OT / Schroedinger-bridge increment as ONE candidate feature for the stacker (strict, per transition).

For transition i (bins <= i only): today's law H = box-4 cell-weighted mixture at bin i (dist_bridge.box_pool), the law
L = min(3, i) bins earlier pooled the same way, per-gene entropic OT plan between their positive parts (Brownian cost on
the Anscombe scale, eps = 0.25 L, dist_bridge.coupling), today's law pushed delta = 1 bin ahead (dist_bridge.sb_step).
Feature = E_obs[target | pushed law] - E_obs[target | unpushed law], with the finite-cell observation of n_i cells
(dist_bridge.expected_target for orig, dist_bridge.expected_value with the l1 formula for l1). The linear target is
left out: there the prediction is the law mean, and by the 1-D identity the pushed mean is the mean trend.
Monte Carlo: 1024 draws per gene, generator keyed by (dataset, transition, target, arm).
Output: _ot_<dataset>.npz with inc_orig, inc_l1 (T, G; row 0 = 0) and timings.
"""
import sys
import time

import numpy as np

import dsdata as D
from dsdata import DB

DRAWS = 1024


def l1_fn(m_now, Lh):
    Lt = max(float(np.maximum(m_now, 0).sum()), 1e-12)
    return lambda m0, s: (np.log1p(1e4 * s / Lh) - np.log1p(1e4 * np.maximum(m0, 0) / Lt)).astype(np.float32)


def main(names):
    for name in names:
        d = D.Data(name)
        trip = [DB.triplet(d.X[b]) for b in range(d.B)]
        ns = list(d.n)
        box = DB.box_pool(trip, ns, 4)
        inc = {"orig": np.zeros((d.T, d.G), np.float32), "l1": np.zeros((d.T, d.G), np.float32)}
        tim = []
        key = 1 + (name == "pancreas-human")
        for i in range(1, d.T):
            t0 = time.time()
            L = min(3, i)
            H = box[i]
            cp = DB.coupling(box[i - L], H, L)
            Hot = DB.sb_step(H, cp, L, 1.0)
            m_now = d.m[:, i]
            for arm, HH in ((0, H), (1, Hot)):
                rng = np.random.default_rng([key, i, 0, arm])
                eo = DB.expected_target(HH, m_now, d.n[i], rng, draws=DRAWS)
                rng = np.random.default_rng([key, i, 1, arm])
                el = DB.expected_value(HH, m_now, d.n[i], rng, l1_fn(m_now, max(float(HH[:, 0].sum()), 1e-12)),
                                       draws=DRAWS)
                sgn = 1 if arm else -1
                inc["orig"][i] += sgn * eo
                inc["l1"][i] += sgn * el
            tim.append(time.time() - t0)
            print(f"{name} i={i} L={L} coupled={len(cp['idx'])} err={cp['err']:.1e} "
                  f"|inc_orig|={np.abs(inc['orig'][i]).mean():.4f} |inc_l1|={np.abs(inc['l1'][i]).mean():.5f} "
                  f"{tim[-1]:.1f}s", flush=True)
        np.savez(f"{D.OUT}/_ot_{name}.npz", inc_orig=inc["orig"], inc_l1=inc["l1"], seconds=np.array(tim))


if __name__ == "__main__":
    main(sys.argv[1:] or list(D.SNAPS))
