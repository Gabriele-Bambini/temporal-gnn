"""Judge: convergence of the gene-sink TV-UOT solver (p2/uot_sinkhorn.solve_tv) and its effect on the forecasts.
For pancreas transitions, Le = 1, lam0 in {128, 1, 0.0625}: per-gene final potential move / eps (fraction of genes
still >= tol when MAX_FINAL = 1500 sweeps stop), marginal overshoot relative to the gene's own expressing mass, and
the change of the forecast mean / linear target when MAX_FINAL is raised to 8000.
Usage: python conv_p2.py  ->  conv_p2.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "p2"))
import uot_core as U  # noqa: E402
import uot_sinkhorn as S  # noqa: E402

REC = []
_orig = S.solve_tv


def solve_tv_rec(a, b, C, eps, rho, tol=1e-7, iters_stage=12, max_final=3000, check_every=10):
    """Same iterations as uot_sinkhorn.solve_tv, recording the per-gene final move and relative overshoot."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    G = a.shape[0]
    eps = np.broadcast_to(np.asarray(eps, float), (G,)).copy()
    rho = np.broadcast_to(np.asarray(rho, float), (G,)).copy()
    ta, tb = a.sum(1), b.sum(1)
    m0 = np.minimum(ta, tb)
    with np.errstate(divide="ignore"):
        lat, lbt = np.log(a / ta[:, None]), np.log(b / tb[:, None])
    ca, cb = np.log(ta / m0), np.log(tb / m0)
    a_pos, b_pos = a > 0, b > 0
    cmax = np.max(np.where(a_pos[:, :, None] & b_pos[:, None, :], C, 0.0), axis=(1, 2))
    cmax = np.maximum(cmax, rho)
    f, g = np.zeros_like(a), np.zeros_like(b)
    for e, last in S._schedule(eps, cmax):
        if last:
            break
        for _ in range(iters_stage):
            f, g = S._sweep_tv(f, g, lat, lbt, ca, cb, a_pos, b_pos, C, e, rho)
    act = np.arange(G)
    n_it, move = 0, np.zeros(G)
    for it in range(max_final):
        sl = (f[act], g[act], lat[act], lbt[act], ca[act], cb[act], a_pos[act], b_pos[act], C[act], eps[act], rho[act])
        fa, ga = S._sweep_tv(*sl)
        n_it = it + 1
        if n_it % check_every == 0:
            with np.errstate(invalid="ignore"):
                df = np.where(a_pos[act], np.abs(fa - f[act]), 0.0).max(1)
                dg = np.where(b_pos[act], np.abs(ga - g[act]), 0.0).max(1)
            mv = np.maximum(df, dg) / eps[act]
            move[act] = mv
            f[act], g[act] = fa, ga
            act = act[mv >= tol]
            if len(act) == 0:
                break
        else:
            f[act], g[act] = fa, ga
    pi = m0[:, None, None] * np.exp(lat[:, :, None] + lbt[:, None, :] + (f[:, :, None] + g[:, None, :] - C) / eps[:, None, None])
    pi = np.where(a_pos[:, :, None] & b_pos[:, None, :], pi, 0.0)
    over_rel = np.maximum((pi.sum(2) - a).max(1) / ta, (pi.sum(1) - b).max(1) / tb)
    REC.append({"move": move.copy(), "unconv": move >= tol, "over_rel": over_rel, "ta": ta})
    over = max(float(np.max(pi.sum(2) - a)), float(np.max(pi.sum(1) - b)))
    return pi, float(move.max()), over, n_it


S.solve_tv = solve_tv_rec
U.uot_sinkhorn.solve_tv = solve_tv_rec


def run(A, B, Le, lam0, max_final):
    REC.clear()
    U.MAX_FINAL = max_final
    cp = U.uot_coupling(A, B, Le, lam0)
    H = U.uot_step(B, cp, Le, 1.0)
    unconv = np.concatenate([r["unconv"] for r in REC])
    over = np.concatenate([r["over_rel"] for r in REC])
    mv = np.concatenate([r["move"] for r in REC])
    return H, {"n_genes_ot": int(len(unconv)), "frac_unconverged": round(float(unconv.mean()), 5),
               "n_unconverged": int(unconv.sum()), "max_move_over_eps": round(float(mv.max()), 4),
               "max_overshoot_rel_to_own_mass": round(float(over.max()), 4),
               "p99_overshoot_rel": round(float(np.quantile(over, 0.99)), 6)}


def main():
    out = {}
    for name in ("pancreas-human",):
        X, trip, n, m = U.load(name)
        for i in (4, 9, 12):
            for lam0 in (128.0, 1.0, 0.0625):
                H1, d1 = run(trip[i - 1], trip[i], 1, lam0, 1500)
                H2, d2 = run(trip[i - 1], trip[i], 1, lam0, 8000)
                lin1, lin2 = H1[:, 0] - m[i], H2[:, 0] - m[i]
                key = f"{name}_i{i}_lam{lam0:g}"
                out[key] = {"max_final_1500": d1, "max_final_8000": d2,
                            "forecast_mean_max_abs_change": float(np.abs(H1[:, 0] - H2[:, 0]).max()),
                            "linear_target_corr_1500_vs_8000": float(np.corrcoef(lin1, lin2)[0, 1]),
                            "fhat_max_abs_change": float(np.abs(H1[:, 1] - H2[:, 1]).max())}
                print(key, json.dumps(out[key]), flush=True)
    with open(os.path.join(HERE, "conv_p2.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
