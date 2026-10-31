"""Erythroid real-stage forecasting with a drift field learned from ALL past transitions (strict rolling origin).

ery_forecast.py learnt the OT drift from the last transition only. Here, at origin s (forecast stage s+1 from
stages <= s, origin-s PCA space from ery_prep.py), every past transition j -> j+1 (j < s) gets its own entropic OT
coupling (own log-domain Sinkhorn, from ery_forecast.py); the barycentric displacements of all past source cells
form a training set (state, displacement, transition index). The drift at a stage-s cell is a recency-weighted
Gaussian kernel regression over that set (weight rho^(s-1-j)), shrunk by lambda (lambda = 0 is persistence), plus
optional Brownian noise (the Schroedinger-bridge reference). rho, lambda and noise are chosen by backtest on the
transition s-1 -> s using only transitions < s-1. Velocity (scVelo dynamical / own steady-state, MURK genes zeroed)
gets the same backtest of scale and sign (scale 0 = persistence). Metrics use scales that do not depend on the
prediction: energy distance, Sinkhorn divergence with a fixed per-origin scale (median squared distance among past
cells), RBF MMD with that scale; plus the gene-level PCC / P@200 of the mean change.
Usage: python ery_forecast2.py <dir with origin*.npz> <out.json>
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ery_forecast import sinkhorn, sqdist, sub  # noqa: E402

NSUB = 1200
RNG = np.random.default_rng(0)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fixed_scale(Zpast):
    i = sub(len(Zpast), 600, seed=9)
    D = sqdist(Zpast[i], Zpast[i])
    return float(np.median(D[D > 0]))


def sinkhorn_div_fixed(X, Y, scale, eps_rel=0.05):
    wx, wy = np.full(len(X), 1 / len(X)), np.full(len(Y), 1 / len(Y))

    def ot(A, B, wa, wb):
        C = sqdist(A, B) / scale
        return float((sinkhorn(wa, wb, C, eps_rel) * C).sum())
    return ot(X, Y, wx, wy) - 0.5 * ot(X, X, wx, wx) - 0.5 * ot(Y, Y, wy, wy)


def energy(X, Y):
    d = lambda A, B: np.sqrt(sqdist(A, B))
    return float(2 * d(X, Y).mean() - d(X, X).mean() - d(Y, Y).mean())


def mmd(X, Y, scale):
    k = lambda A, B: np.exp(-sqdist(A, B) / scale)
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


def transition_pairs(Z, X, st, j):
    """OT coupling stage j -> j+1: source states, PCA and gene displacements."""
    Z0, Z1, X0, X1 = Z[st == j], Z[st == j + 1], X[st == j], X[st == j + 1]
    C = sqdist(Z0, Z1)
    P = sinkhorn(np.full(len(Z0), 1 / len(Z0)), np.full(len(Z1), 1 / len(Z1)), C / np.median(C), 0.05)
    r = P.sum(1)[:, None]
    return Z0, (P @ Z1) / r - Z0, (P @ X1) / r - X0


def drift_at(tgt, pairs, upto, rho, k=15):
    """Recency-weighted kernel regression of displacements from transitions 0..upto-1, evaluated at tgt."""
    src = np.concatenate([pairs[j][0] for j in range(upto)])
    dz = np.concatenate([pairs[j][1] for j in range(upto)])
    dx = np.concatenate([pairs[j][2] for j in range(upto)])
    wj = np.concatenate([np.full(len(pairs[j][0]), rho ** (upto - 1 - j)) for j in range(upto)])
    ref = src[sub(len(src), 800, seed=5)]
    h = np.median(np.sort(np.sqrt(sqdist(ref, src)), 1)[:, min(k, len(src) - 1)]) + 1e-9
    W = np.exp(-sqdist(tgt, src) / (2 * h * h)) * wj[None, :]
    W /= np.maximum(W.sum(1, keepdims=True), 1e-300)
    return W @ dz, W @ dx


def score(Zf, Xf, Zt, Xt, Xnow, scale):
    i, j = sub(len(Zf), NSUB), sub(len(Zt), NSUB, seed=1)
    out = {"energy": energy(Zf[i], Zt[j]), "sinkhorn_fixed": sinkhorn_div_fixed(Zf[i], Zt[j], scale),
           "mmd": mmd(Zf[i], Zt[j], scale)}
    pred, true = Xf.mean(0) - Xnow.mean(0), Xt.mean(0) - Xnow.mean(0)
    if np.allclose(pred, 0):
        out.update(pearson=float("nan"), p200_up=float("nan"), p200_down=float("nan"))
    else:
        out["pearson"] = float(np.corrcoef(pred, true)[0, 1])
        out["p200_up"] = len(set(np.argsort(-true)[:200]) & set(np.argsort(-pred)[:200])) / 200
        out["p200_down"] = len(set(np.argsort(true)[:200]) & set(np.argsort(pred)[:200])) / 200
    return out


def velocity_drift(d, st, stage, key):
    V = d[key][st == stage] / (1.0 + d["Ms"][st == stage])
    V[:, d["murk"].astype(bool)] = 0.0
    return V @ d["pca_comps"].T, V


def bt_energy(Zf, Ztrue):
    return energy(Zf[sub(len(Zf), NSUB)], Ztrue[sub(len(Ztrue), NSUB, seed=1)])


def main():
    src_dir, out_path = sys.argv[1], sys.argv[2]
    res = {"origins": {}}
    for s in range(2, 6):
        t0 = time.time()
        d = dict(np.load(os.path.join(src_dir, f"origin{s}.npz"), allow_pickle=True))
        st, Z, X = d["stage_past"], d["Z_past"], d["X_past"].astype(np.float32)
        Zt, Xt = d["Z_fut"], d["X_fut"].astype(np.float32)
        scale = fixed_scale(Z)
        pairs = [transition_pairs(Z, X, st, j) for j in range(s)]
        Zn, Xn = Z[st == s], X[st == s]
        Zb, Xb = Z[st == s - 1], X[st == s - 1]      # backtest: forecast stage s from s-1
        row = {}
        # multi-transition OT drift: backtest with transitions < s-1, forecast with transitions < s
        cand = []
        for rho in (0.5, 1.0):
            dzb, _ = drift_at(Zb, pairs, s - 1, rho)
            for lam in LAMBDAS:
                for noise in (0.0, 0.5):
                    Zf = Zb + lam * dzb + (RNG.normal(0, noise, Zb.shape) if noise else 0)
                    cand.append((bt_energy(Zf, Zn), rho, lam, noise))
        _, rho, lam, noise = min(cand)
        dz, dx = drift_at(Zn, pairs, s, rho)
        Zf = Zn + lam * dz + (RNG.normal(0, noise, Zn.shape) if noise else 0)
        row["ot_multi"] = {"hp": {"rho": rho, "lambda": lam, "noise": noise}, **score(Zf, Xn + lam * dx, Zt, Xt, Xn, scale)}
        # last-transition OT at full strength (the first run's ot without noise), rho -> 0 keeps transition s-1 only
        dz1, dx1 = drift_at(Zn, pairs, s, 1e-9)
        row["ot_last_full"] = {"hp": {}, **score(Zn + dz1, Xn + dx1, Zt, Xt, Xn, scale)}
        # velocity with scale and sign by backtest (scale 0 = persistence), MURK genes removed
        for key in ("v_dyn", "v_ss"):
            vzb, _ = velocity_drift(d, st, s - 1, key)
            cs = [(bt_energy(Zb, Zn), 0.0, 1)]
            for c in (0.25, 0.5, 1, 2, 4, 8):
                for sign in (1, -1):
                    cs.append((bt_energy(Zb + sign * c * vzb, Zn), c, sign))
            _, c, sign = min(cs)
            vz, vx = velocity_drift(d, st, s, key)
            row[f"vel_{key[2:]}_nomurk"] = {"hp": {"c": c, "sign": sign}, **score(Zn + sign * c * vz, Xn + sign * c * vx, Zt, Xt, Xn, scale)}
        row["persist"] = {"hp": {}, **score(Zn, Xn, Zt, Xt, Xn, scale)}
        row["meanshift"] = {"hp": {}, **score(Zn + (Zn.mean(0) - Zb.mean(0)), Xn + (Xn.mean(0) - Xb.mean(0)), Zt, Xt, Xn, scale)}
        half = np.random.default_rng(4).permutation(len(Zt))
        A, B = half[: len(Zt) // 2], half[len(Zt) // 2:]
        row["floor_halves"] = {"energy": energy(Zt[A][:NSUB], Zt[B][:NSUB]), "sinkhorn_fixed": sinkhorn_div_fixed(Zt[A][:NSUB], Zt[B][:NSUB], scale)}
        row["seconds"] = round(time.time() - t0, 1)
        res["origins"][s] = row
        print(s, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in row.items()}, default=float), flush=True)
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
