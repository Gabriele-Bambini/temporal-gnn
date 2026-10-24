"""Judge 2 truncation test: replace every bin > i with garbage (genes permuted, levels rescaled, cell counts x3) and
check that each design's prediction for transition i (bins i -> i+1) is bit-identical to the one on real data.
Covers bins > t AND the label of t+1 (bin i+1 is garbage). Runs on the judge2 copies of the prototype scripts.
Usage: python trunc_test.py <design> <dataset> <i>   design in {nass, lsb, popot}
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTJ = os.path.join(HERE, "trunc_results.jsonl")


def garbage_rows(rng, G):
    return rng.permutation(G), rng.uniform(0.2, 5.0, G)


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.nanmax(np.abs(a - b))) if a.size else 0.0


def test_nass(name, i):
    import dsdata as D
    import nass as N
    out = {}
    d0 = D.Data(name)
    d1 = D.Data(name)
    X = d1.X.copy()
    rng = np.random.default_rng(7)
    for b in range(i + 1, d1.B):
        perm, sc = garbage_rows(rng, d1.G)
        Xb = X[b][perm].copy()
        Xb[:, 0] *= sc
        Xb[:, 1] *= sc
        Xb[:, 2] *= sc
        Xb[:, 3] = rng.permutation(Xb[:, 3])
        Xb[:, 4] = Xb[:, 0] * 3 * D.DB.n_cells(X[b])
        X[b] = Xb
    d1.X = X
    d1.m = np.maximum(X[:, :, 0].T, 0)
    d1.s = X[:, :, 2].T
    d1.f = X[:, :, 3].T
    d1.n = np.array([D.DB.n_cells(X[b]) for b in range(d1.B)])
    d1.lib = d1.m.sum(0)
    d1.CP = 1e4 * d1.m / d1.lib[None, :]
    d1.L = np.log1p(d1.CP)
    for tg in ("l1", "orig", "linear"):
        res = []
        for d in (d0, d1):
            Y = d.Y(tg)
            F, cls = N.build_features(d, tg, Y, None, None)
            G_ = N.GROUPS
            full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"] + (G_["obs"] if tg == "orig" else [])
            st = N.Stacker(F, cls, Y, full)
            fb = N.fallback(d, tg, Y, F)
            pred, chosen = N.run_learned(st, [full], Y, d, fb)
            res.append((F, pred, chosen, Y))
        (F0, p0, c0, Y0), (F1, p1, c1, Y1) = res
        fd = {k: maxdiff(F0[k][i], F1[k][i]) for k in F0}
        out[tg] = {"pred_maxabs_diff_at_i": maxdiff(p0[i], p1[i]),
                   "pred_diff_at_i+1 (should be >0, garbage visible)": maxdiff(p0[i + 1], p1[i + 1]) if i + 1 < len(p0) else None,
                   "label_diff_at_i (garbage label)": maxdiff(Y0[i], Y1[i]),
                   "feature_maxabs_diff_at_i": max(fd.values()), "chosen_same": c0[i] == c1[i]}
        print(name, "nass", tg, out[tg], flush=True)
    return out


def test_lsb(name, i):
    import lsb_common as C
    from lsb_model import LatentSB
    import run_lsb as R
    d0 = C.Data(name)
    d1 = C.Data(name)
    rng = np.random.default_rng(7)
    for s in range(i + 1, d1.S):
        perm, sc = garbage_rows(rng, d1.N)
        d1.M[:, s] = d1.M[perm, s] * sc
        d1.SD[:, s] = d1.SD[perm, s] * sc
        d1.F[:, s] = rng.permutation(d1.F[:, s])
        d1.SUM[:, s] = d1.M[:, s] * 3 * d0.nc[s]
    ok = d1.M > 0
    d1.nc = np.array([np.round(np.median(d1.SUM[ok[:, s], s] / d1.M[ok[:, s], s])) for s in range(d1.S)])
    d1.lib = d1.M.sum(0)
    d1.CP = 1e4 * d1.M / d1.lib
    d1.X = np.log1p(d1.CP)
    d1.trip = [np.stack([d1.M[:, s], d1.F[:, s], d1.SD[:, s] ** 2 + d1.M[:, s] ** 2], 1) for s in range(d1.S)]
    m0, m1 = LatentSB(d0), LatentSB(d1)
    out = {"nmed_real": m0.nmed, "nmed_garbage": m1.nmed, "R_diff_bins<=i": maxdiff(m0.R[:, :i + 1], m1.R[:, :i + 1])}
    worst, worst_k0, worst_cfg = 0.0, 0.0, None
    for c in R.CFGS:
        x0 = m0.levels(i, c)[1]
        x1 = m1.levels(i, c)[1]
        dd = maxdiff(m0.head_l1(i, x0), m1.head_l1(i, x1))
        if c[0] == 0:
            worst_k0 = max(worst_k0, dd)
        if dd > worst:
            worst, worst_cfg = dd, c
    out["l1_head_maxdiff_all_cfgs"] = worst
    out["l1_head_maxdiff_k0_cfgs"] = worst_k0
    out["worst_cfg"] = str(worst_cfg)
    # orig head (alpha = 0 and alpha = 1 toward a latent level)
    c = R.CFGS[R.F_LAT]
    xn0, xn1 = m0.levels(i, c)[1], m1.levels(i, c)[1]
    out["orig_head_alpha1_diff"] = maxdiff(m0.head_orig(i, xn0, 4, 1.0, [1, i, 0]), m1.head_orig(i, xn1, 4, 1.0, [1, i, 0]))
    out["orig_head_alpha0_diff"] = maxdiff(m0.head_orig(i, xn0, 4, 0.0, [1, i, 0]), m1.head_orig(i, xn1, 4, 0.0, [1, i, 0]))
    # same check with a causal nmed (median of n over bins <= i) on both: isolates the nmed effect
    for m, d in ((m0, d0), (m1, d1)):
        m.nmed = float(np.median(d.nc[:i + 1]))
        m._fac.clear(); m._zf.clear(); m._res.clear()
    worst_c = 0.0
    for c in R.CFGS:
        dd = maxdiff(m0.head_l1(i, m0.levels(i, c)[1]), m1.head_l1(i, m1.levels(i, c)[1]))
        worst_c = max(worst_c, dd)
    out["l1_head_maxdiff_all_cfgs_with_causal_nmed"] = worst_c
    print(name, "lsb", out, flush=True)
    return out


def test_popot(name, i):
    import popot as PP
    import common
    d0 = common.Data(name)
    d1 = common.Data(name)
    rng = np.random.default_rng(7)
    for s in range(i + 1, d1.S):
        perm, sc = garbage_rows(rng, d1.N)
        d1.M[:, s] = d1.M[perm, s] * sc
        d1.SD[:, s] = d1.SD[perm, s] * sc
        d1.F[:, s] = rng.permutation(d1.F[:, s])
        d1.nc[s] = 3 * d1.nc[s]
    d1.lib = d1.M.sum(0)
    d1.CP = 1e4 * d1.M / d1.lib
    d1.L = np.log1p(d1.CP)
    p0, p1 = PP.PopOT(d0), PP.PopOT(d1)
    cfgs = [("box4", 6, "ot0.05", 0.0), ("mixavg3", 8, "sb3", 0.0), ("box1", 3, "uot0.05t1", 0.5),
            ("box2", 8, "lin1", 0.0), ("box4", 3, "sbW", 0.0)]
    out = {}
    for cfg in cfgs:
        r = {}
        for t in ("orig", "l1", "linear"):
            a = p0.target_pred(i, p0.law(i, cfg), t, PP.label(cfg))
            b = p1.target_pred(i, p1.law(i, cfg), t, PP.label(cfg))
            r[t] = maxdiff(a, b)
        out[PP.label(cfg)] = r
        print(name, "popot", PP.label(cfg), r, flush=True)
    return out


if __name__ == "__main__":
    design, name, i = sys.argv[1], sys.argv[2], int(sys.argv[3])
    t0 = time.time()
    res = {"nass": test_nass, "lsb": test_lsb, "popot": test_popot}[design](name, i)
    with open(OUTJ, "a") as fh:
        fh.write(json.dumps({"design": design, "dataset": name, "i": i, "result": res,
                             "seconds": round(time.time() - t0, 1)}) + "\n")
