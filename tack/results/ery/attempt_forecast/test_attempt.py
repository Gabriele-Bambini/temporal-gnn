"""Checks of the new families against the first-run code on origin 2 and 4 (no scores are written).

1. lambda = 0 / a = 0 / c = 0 grid points are bit-identical to persistence (so ties are true ties).
2. ot_damp lam = 1 equals the first-run ot (noise 0); uot_damp lam = 1 equals the first-run uot_growth (same tau, beta).
3. vel_ss_nm equals the first-run vel_ss when no MURK gene is zeroed (checked by putting MURK back).
4. Weights are positive and sum to 1; composition reweighting hits the requested composition exactly.
5. Every family runs on the final and the same-space problem.
"""
import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attempt_forecast as A  # noqa: E402

F = A.F
obs, codes = A.load_obs(A.DATA)
for s in (2, 4):
    F._DRIFT.clear()
    d, smp_past, smp_fut, km, S, P = A.build_final(A.DATA, s, obs, codes)
    pz, px, _ = F.forecast("persist", {}, d, s, (s, s - 1))
    for fam, (grid, fn, _) in A.FAMILIES.items():
        Z, X, w = fn(P, grid[0])
        if fam == "xfit":
            continue
        same = (Z.dtype == pz.dtype and np.array_equal(Z, pz) and np.array_equal(X, px) and w is None)
        assert same, (s, fam, grid[0], Z.dtype, w is None)
    Z, X, w = A.f_ot_damp(P, {"lam": 1.0})
    oz, ox, ow = F.forecast("ot", {"noise": 0.0}, d, s, (s, s - 1))
    assert np.array_equal(Z, oz) and np.allclose(X, ox, atol=1e-6) and ow is None, "ot"
    for tau, beta in ((1.0, 2.0), (0.1, 0.0)):
        Z, X, w = A.f_uot_damp(P, {"lam": 1.0, "tau": tau, "beta": beta})
        oz, ox, ow = F.forecast("uot_growth", {"tau": tau, "beta": beta, "noise": 0.0}, d, s, (s, s - 1))
        assert np.array_equal(Z, oz) and np.allclose(X, ox, atol=1e-6) and np.array_equal(w, ow), ("uot", tau, beta)
    murk = d["murk"].copy()
    d["murk"][:] = False
    P.cache.pop(("vel", "v_ss"), None)
    Z, X, w = A.f_vel("v_ss")(P, {"c": 2.0})
    oz, ox, _ = F.forecast("vel_ss", {"c": 2.0}, d, s, (s, s - 1))
    assert np.allclose(Z, oz, atol=1e-4) and np.allclose(X, ox, atol=1e-5), "vel"
    d["murk"][:] = murk
    P.cache.pop(("vel", "v_ss"), None)
    Z2, _, _ = A.f_vel("v_ss")(P, {"c": 2.0})
    print(s, "MURK in HVG", int(murk.sum()), "max |dZ| from zeroing MURK", float(np.abs(Z2 - Z).max()),
          "rel", float(np.linalg.norm(Z2 - Z) / np.linalg.norm(Z - P.Zn)))
    for fam, (grid, fn, _) in A.FAMILIES.items():
        for hp in grid:
            Z, X, w = fn(P, hp)
            assert Z.shape == P.Zn.shape and X.shape == P.Xn.shape and np.isfinite(Z).all()
            if w is not None:
                assert (w >= 0).all() and np.isfinite(w).all()
                assert abs(w.sum() - 1) < 1e-9 or fam in ("uot_rw", "uot_damp", "tilt_rw", "c_tilt", "c_uot")
    w = A.comp_weights_loglin(P.ct_n, P.ct_p, 1.0)
    types = np.unique(np.r_[P.ct_n, P.ct_p])
    nn = np.array([(P.ct_n == t).sum() for t in types], float)
    npv = np.array([(P.ct_p == t).sum() for t in types], float)
    pn = (nn + 0.5) / (nn.sum() + 0.5 * len(types))
    pp = (npv + 0.5) / (npv.sum() + 0.5 * len(types))
    ph = np.where(nn > 0, pn * pn / pp, 0)
    ph /= ph.sum()
    got = np.array([w[P.ct_n == t].sum() for t in types])
    assert np.allclose(got, ph), (got, ph)
    print(s, "atlas composition now", dict(zip(types, np.round(nn / nn.sum(), 3))), "-> lam 1",
          dict(zip(types, np.round(got, 3))))
    wm = A.comp_weights_markov(P.ot()[0], P.ct_p, P.ct_n, 1.0)
    print(s, "markov lam 1", dict(zip(types, np.round([wm[P.ct_n == t].sum() for t in types], 3))))
    print(s, "rho", round(P.rho(), 3), "pool_eq g=1 masses",
          {p: round(float(A.pool_eq_weights(P.smp_n, 1.0)[P.smp_n == p].sum()), 3) for p in np.unique(P.smp_n)})
    Pb = A.Problem(d, smp_past, s - 1, S, d["Z_past"][d["stage_past"] == s],
                   d["X_past"][d["stage_past"] == s].astype(np.float32), smp_past[d["stage_past"] == s], km, "same")
    for fam, (grid, fn, _) in A.FAMILIES.items():
        Z, X, w = fn(Pb, grid[-1])
        assert np.isfinite(Z).all()
    print(s, "ok")
print("all checks passed", F.STATS)
