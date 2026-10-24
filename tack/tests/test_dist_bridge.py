"""Checks for the DistBridge forecaster (tack/hpc/dist_bridge.py). Run: python tack/tests/test_dist_bridge.py

1 Sinkhorn: both marginals to 1e-8 on random problems, finite plans with zero-mass entries.
2 OT identity: for any coupling with these marginals, sum_k pB_k v_k = (E_B[z] - E_A[z]) / L.
3 Nesting on the real pancreas-human node file: Box(k), delta = 0 with obs='clt' equals ObsLevel(k) bitwise and
  with obs='inf' equals Level(k), for k in {2, 4, 6}, transitions i >= 1.
4 Observation model: expected_target agrees with a brute-force simulation of n cells within 4 standard errors.
5 No leak: predictions for transitions <= 6 do not change when every bin > 6 is replaced by garbage.
6 Synthetic known drift: with a true per-gene drift, the OT step (delta = 1) beats delta = 0 on mean J and MSE
  over transitions i >= 4; with no drift, rolling selection keeps delta = 0 in >= 80 % of those transitions.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "hpc"))
import dist_bridge as db  # noqa: E402
import sinkhorn  # noqa: E402
import task2_forecasters as tf  # noqa: E402

GRN = os.environ.get("GRN_DIR", os.path.join(ROOT, "..", "tgl-grn-1CCD", "GRN"))


def real_hist(name="pancreas-human", snaps=range(0, 15)):
    node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"))
    n = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    stats_ = ("s00_mean", "s01_median", "s02_std", "s03_frac_nonzero", "s04_sum")
    return [node[[f"t{b:03d}_{s}" for s in stats_]].to_numpy(np.float32)[:n].astype(np.float64) for b in snaps]


def test_sinkhorn():
    rng = np.random.default_rng(0)
    a = rng.random((50, 12))
    b = rng.random((50, 12))
    a[:, :3] = 0
    a /= a.sum(1, keepdims=True)
    b /= b.sum(1, keepdims=True)
    z = np.sort(rng.random((50, 12)) * 10, 1)
    C = (z[:, :, None] - z[:, None, :]) ** 2 / 2
    for eps in (0.25, 0.75, 5.0):  # the step uses eps = 0.25 L with L in {1, 2, 3}
        pi, err = sinkhorn.solve(a, b, C, eps)
        assert np.all(np.isfinite(pi))
        assert np.abs(pi.sum(2) - a).max() < 1e-8 and np.abs(pi.sum(1) - b).max() < 1e-8, eps


def test_identity():
    hist = real_hist()
    A = db.box_pool([db.triplet(x) for x in hist[:6]], [db.n_cells(x) for x in hist[:6]], 2)[2]
    B = db.box_pool([db.triplet(x) for x in hist[:6]], [db.n_cells(x) for x in hist[:6]], 2)[5]
    for s2 in (0.25 / 3, 0.25, 4 / 3):
        old = db.SIGMA2_SB
        db.SIGMA2_SB = s2
        t = time.time()
        cp = db.coupling(A[:2000], B[:2000], 3)
        print(f"  eps={3 * s2:.2f}: 2000 genes in {time.time() - t:.1f}s, row error {cp['err']:.1e}")
        db.SIGMA2_SB = old
        _, muA, vA = db.positive_part(A[:2000][cp["idx"]])
        _, muB, vB = db.positive_part(B[:2000][cp["idx"]])
        z, pA, pB = db.grid_masses(muA, vA, muB, vB)
        lhs = (cp["pB"] * cp["v"]).sum(1)
        rhs = ((pB * z).sum(1) - (pA * z).sum(1)) / 3
        assert np.abs(lhs - rhs).max() < 1e-7, (s2, np.abs(lhs - rhs).max())


def test_nesting():
    hist = real_hist()
    for k in (2, 4, 6):
        db._CACHE.clear()
        clt = db.DistBridge(fixed=("box", k, 0.0), obs="clt")
        inf = db.DistBridge(fixed=("box", k, 0.0), obs="inf")
        ol, lv = tf.ObsLevel(k), tf.Level(k)
        for i in range(1, len(hist)):
            h = hist[:i + 1]
            assert np.array_equal(clt.predict(h, None, {}), ol.predict(h, None, {})), (k, i, "clt")
            assert np.allclose(inf.predict(h, None, {}), lv.predict(h, None, {}), atol=1e-6), (k, i, "inf")


def test_observation():
    rng = np.random.default_rng(3)
    G, n = 200, 60
    f = rng.uniform(0.002, 0.6, G)
    mu = 1 + rng.gamma(1.0, 2.0, G)
    var = np.where(rng.random(G) < 0.8, (mu - 1) * rng.uniform(1.2, 4, G), (mu - 1) * 0.7)
    H = np.stack([f * mu, f, f * (var + mu ** 2)], 1)
    m_now = np.where(rng.random(G) < 0.15, 0.0, rng.uniform(0.01, 2, G))
    got = db.expected_target(H, m_now, n, np.random.default_rng(5), draws=4096)
    R = 6000
    kind = db._law_kind(mu, var)
    lam = mu - 1
    r_nb = np.where(kind == 2, lam ** 2 / np.maximum(var - lam, 1e-12), 1.0)
    p_nb = np.where(kind == 2, lam / np.maximum(var, 1e-12), 0.5)
    ys = np.empty((G, R), np.float32)
    for g in range(G):
        pos = rng.random((R, n)) < f[g]
        E = rng.negative_binomial(r_nb[g], p_nb[g], size=(R, n)) if kind[g] == 2 else rng.poisson(max(lam[g], 0), size=(R, n))
        ys[g] = db.log_delta(np.full(R, m_now[g]), (pos * (1 + E)).sum(1) / n)
    se = np.sqrt(ys.var(1) / R + ys.var(1) / 4096)
    err = np.abs(got - ys.mean(1))
    assert np.all(err < 4 * se + 1e-3), (err.max(), (err / (se + 1e-12)).max())


def test_no_leak():
    hist = real_hist()
    db._CACHE.clear()
    m = db.DistBridge()
    p1 = [m.predict(hist[:i + 1], None, {}) for i in range(7)]
    junk = [h.copy() for h in hist]
    rng = np.random.default_rng(0)
    for b in range(7, len(junk)):
        junk[b] = junk[b][rng.permutation(len(junk[b]))]
    db._CACHE.clear()
    m2 = db.DistBridge()
    p2 = [m2.predict(junk[:i + 1], None, {}) for i in range(7)]
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))


def synthetic(drift, G=2000, T=14, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(-1.5, 1.0, G)
    c = rng.normal(0.5, 0.8, G)
    s = rng.normal(0, 0.15, G) * drift
    r = rng.normal(0, 0.15, G) * drift
    hist = []
    for b in range(T):
        n = int(rng.integers(50, 601))
        f = 1 / (1 + np.exp(-(a + s * b)))
        lam = np.exp(c + r * b)
        pos = rng.random((G, n)) < f[:, None]
        E = rng.negative_binomial(2, 2 / (2 + lam)[:, None], size=(G, n))
        cells = pos * (1 + E)
        hist.append(np.stack([cells.mean(1), np.median(cells, 1), cells.std(1), (cells > 0).mean(1), cells.sum(1)], 1).astype(float))
    return hist


def test_synthetic_drift():
    q = 200.0 / 500
    for drift in (1.0, 0.0):
        hist = synthetic(drift)
        res = {}
        for d in (0.0, 1.0):
            db._CACHE.clear()
            m = db.DistBridge(fixed=("box", 4, d))
            Js, mses = [], []
            for i in range(len(hist) - 1):
                p = m.predict(hist[:i + 1], None, {})
                y = db.log_delta(hist[i][:, 0], hist[i + 1][:, 0])
                if i >= 4:
                    Js.append(db.criterion(y, p, q))
                    mses.append(float(np.mean((y - p) ** 2)))
            res[d] = (np.mean(Js), np.mean(mses))
        print(f"  drift={drift}: delta0 J={res[0.0][0]:.4f} MSE={res[0.0][1]:.4f} | delta1 J={res[1.0][0]:.4f} MSE={res[1.0][1]:.4f}")
        if drift:
            assert res[1.0][0] > res[0.0][0] and res[1.0][1] < res[0.0][1], res
        else:
            db._CACHE.clear()
            m = db.DistBridge()
            for i in range(len(hist) - 1):
                m.predict(hist[:i + 1], None, {})
            picks = [cfg[2] for (_, i), cfg in m.selected.items() if i >= 4]
            share = np.mean([p == 0 for p in picks])
            print(f"  null: delta = 0 selected in {share:.0%} of transitions i >= 4")
            assert share >= 0.8, picks


if __name__ == "__main__":
    only = sys.argv[1:]
    for fn in (test_sinkhorn, test_identity, test_nesting, test_observation, test_no_leak, test_synthetic_drift):
        if only and fn.__name__ not in only:
            continue
        t = time.time()
        fn()
        print(f"PASS {fn.__name__} ({time.time() - t:.0f}s)", flush=True)
