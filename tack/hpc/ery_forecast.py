"""Forecast the erythroid population of the next REAL embryonic stage from the past (strict rolling origin).

Input: ~/tack/out/ery/origin<s>.npz from ery_prep.py (everything fitted on stages <= s). Origin s forecasts stage
s+1 (days are 0.25 apart). Families (cells live in the origin's 30-d PCA space; gene space = 2000 HVG log values):
  persist        stage-s cells unchanged
  meanshift      stage-s cells + (mean_s - mean_{s-1})
  ot             entropic OT (own log-domain Sinkhorn) from stage s-1 to stage s; barycentric displacements of the
                 s-1 cells, kernel-regressed onto the stage-s cells (time-homogeneous drift), optional Brownian noise
                 (the Schroedinger-bridge reference); applied in PCA and gene space
  uot_growth     KL-unbalanced OT with source mass ~ exp(beta * proliferation): cells can grow, die or appear
                 (sink / source); drift as in ot, plus per-cell growth weights regressed onto stage-s cells
  vel_ss/vel_dyn RNA velocity (own steady-state / scVelo dynamical), log-scale v / (1 + Ms), scaled by c
  ot_vel_*       weighted average of the OT drift and the scaled velocity drift
Hyperparameters (noise, c, w, tau, beta) are chosen per origin by backtesting on the previous transition only
(predict stage s from stage s-1 with the drift fitted on s-2 -> s-1); origin 1 uses defaults.
Metrics against the true stage s+1 cells: Sinkhorn divergence and energy distance in PCA space (subsampled to 1500
cells, weights for uot_growth), and the gene-level change of the mean log expression (Pearson, P@200 up / down),
plus a sampling floor (two random halves of the true stage). Output ~/tack/out/ery/forecast.json.
"""
import json
import os
import time

import numpy as np

OUT = os.path.expanduser("~/tack/out/ery")
DT = 0.25
RNG = np.random.default_rng(0)
NSUB = 1500
_DRIFT = {}


# ---------------- OT (rectangular, log domain) ----------------

def lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def sinkhorn(a, b, C, eps, tau=None, iters=2000, tol=1e-7):
    """Entropic OT plan; tau = None balanced, else KL-relaxed marginals (unbalanced, Chizat et al. scaling)."""
    la, lb = np.log(a), np.log(b)
    f, g = np.zeros(len(a)), np.zeros(len(b))
    k = 1.0 if tau is None else tau / (tau + eps)
    for it in range(iters):
        f_new = k * eps * (la - lse((g[None, :] - C) / eps, 1))
        g_new = k * eps * (lb - lse((f_new[:, None] - C) / eps, 0))
        done = max(np.abs(f_new - f).max(), np.abs(g_new - g).max()) < tol
        f, g = f_new, g_new
        if done:
            break
    return np.exp((f[:, None] + g[None, :] - C) / eps)


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def sinkhorn_div(X, Y, wx=None, wy=None, eps_rel=0.05):
    wx = np.full(len(X), 1 / len(X)) if wx is None else wx / wx.sum()
    wy = np.full(len(Y), 1 / len(Y)) if wy is None else wy / wy.sum()
    scale = np.median(sqdist(X[:300], Y[:300]))
    eps = eps_rel * scale

    def ot(A, B, wa, wb):
        C = sqdist(A, B)
        P = sinkhorn(wa, wb, C, eps)
        return float((P * C).sum())
    return (ot(X, Y, wx, wy) - 0.5 * ot(X, X, wx, wx) - 0.5 * ot(Y, Y, wy, wy)) / scale


def energy(X, Y, wx=None, wy=None):
    wx = np.full(len(X), 1 / len(X)) if wx is None else wx / wx.sum()
    wy = np.full(len(Y), 1 / len(Y)) if wy is None else wy / wy.sum()
    d = lambda A, B: np.sqrt(sqdist(A, B))
    return float(2 * wx @ d(X, Y) @ wy - wx @ d(X, X) @ wx - wy @ d(Y, Y) @ wy)


def sub(n, k=NSUB, seed=0):
    r = np.random.default_rng(seed)
    return np.sort(r.choice(n, min(n, k), replace=False))


# ---------------- drift fields ----------------

def kernel_regress(src, vals, tgt, k=15):
    """Gaussian Nadaraya-Watson of vals (rows of src) at tgt; bandwidth = median k-NN distance among src."""
    d_ss = np.sqrt(sqdist(src[:min(len(src), 800)], src))
    h = np.median(np.sort(d_ss, 1)[:, min(k, d_ss.shape[1] - 1)]) + 1e-9
    W = np.exp(-sqdist(tgt, src) / (2 * h * h))
    W /= np.maximum(W.sum(1, keepdims=True), 1e-300)
    return W @ vals


def ot_coupling(Z0, Z1, eps_rel=0.05, tau=None, a=None):
    C = sqdist(Z0, Z1)
    scale = np.median(C)
    a = np.full(len(Z0), 1 / len(Z0)) if a is None else a / a.sum()
    b = np.full(len(Z1), 1 / len(Z1))
    return sinkhorn(a, b, C / scale, eps_rel, None if tau is None else tau), a


def ot_drift(Z0, X0, Z1, X1, tau=None, a=None):
    """Barycentric displacement of each source cell (PCA and gene space) and growth ratio (row mass / prior)."""
    P, a = ot_coupling(Z0, Z1, tau=tau, a=a)
    row = P.sum(1)
    Tz = (P @ Z1) / row[:, None]
    Tx = (P @ X1) / row[:, None]
    growth = row / a
    return Tz - Z0, Tx - X0, growth / growth.mean()


# ---------------- forecasts ----------------

def forecast(fam, hp, d, s, past_idx):
    """past_idx: (index of origin stage, index of previous stage) into the origin arrays."""
    st = d["stage_past"]
    i_now, i_prev = past_idx
    Zn, Xn = d["Z_past"][st == i_now], d["X_past"][st == i_now].astype(np.float32)
    Zp, Xp = d["Z_past"][st == i_prev], d["X_past"][st == i_prev].astype(np.float32)
    w = None
    if fam == "persist":
        Zf, Xf = Zn, Xn
    elif fam == "meanshift":
        Zf, Xf = Zn + (Zn.mean(0) - Zp.mean(0)), Xn + (Xn.mean(0) - Xp.mean(0))
    else:
        dz = dx = 0.0
        if fam in ("ot", "uot_growth") or fam.startswith("ot_vel"):
            a = None
            tau = None
            if fam == "uot_growth":
                tau = hp["tau"]
                a = np.exp(hp["beta"] * d["cc_past"][st == i_prev] * DT)
            key = (id(d), i_now, i_prev, tau, hp.get("beta") if fam == "uot_growth" else None)
            if key not in _DRIFT:  # one coupling serves every (c, w, noise) setting
                DZ, DX, gr = ot_drift(Zp, Xp, Zn, Xn, tau=tau, a=a)
                _DRIFT[key] = (kernel_regress(Zp, DZ, Zn), kernel_regress(Zp, DX, Zn),
                               np.maximum(kernel_regress(Zp, gr[:, None], Zn)[:, 0], 1e-6))
            dz, dx, gw = _DRIFT[key]
            if fam == "uot_growth":
                w = gw
        if fam.startswith("vel") or fam.startswith("ot_vel"):
            key = "v_ss" if fam.endswith("ss") else "v_dyn"
            V = d[key][st == i_now] / (1.0 + d["Ms"][st == i_now])
            vz, vx = V @ d["pca_comps"].T, V
            c = hp["c"]
            if fam.startswith("vel"):
                dz, dx = c * vz, c * vx
            else:
                ww = hp["w"]
                dz, dx = ww * dz + (1 - ww) * c * vz, ww * dx + (1 - ww) * c * vx
        Zf, Xf = Zn + dz, Xn + dx
        if hp.get("noise", 0) > 0:
            Zf = Zf + RNG.normal(0, hp["noise"], Zf.shape)
    return Zf, Xf, w


def score(Zf, Xf, w, Zt, Xt, Xnow):
    i, j = sub(len(Zf)), sub(len(Zt), seed=1)
    wi = None if w is None else w[i]
    out = {"sinkhorn_div": sinkhorn_div(Zf[i], Zt[j], wi), "energy": energy(Zf[i], Zt[j], wi)}
    mw = np.full(len(Xf), 1 / len(Xf)) if w is None else w / w.sum()
    pred = mw @ Xf - Xnow.mean(0)
    true = Xt.mean(0) - Xnow.mean(0)
    if np.allclose(pred, pred[0]):
        out.update(pearson=float("nan"), p200_up=float("nan"), p200_down=float("nan"))
    else:
        out["pearson"] = float(np.corrcoef(pred, true)[0, 1])
        k = 200
        out["p200_up"] = len(set(np.argsort(-true)[:k]) & set(np.argsort(-pred)[:k])) / k
        out["p200_down"] = len(set(np.argsort(true)[:k]) & set(np.argsort(pred)[:k])) / k
    return out


GRIDS = {
    "persist": [{}], "meanshift": [{}],
    "ot": [{"noise": n} for n in (0.0, 0.5, 1.0)],
    "uot_growth": [{"tau": t, "beta": b, "noise": 0.0} for t in (0.1, 1.0) for b in (0.0, 2.0, 5.0)],
    "vel_ss": [{"c": c} for c in (0.25, 0.5, 1, 2, 4, 8)],
    "vel_dyn": [{"c": c} for c in (0.25, 0.5, 1, 2, 4, 8)],
    "ot_vel_ss": [{"c": c, "w": w} for c in (0.5, 2, 8) for w in (0.25, 0.5, 0.75)],
    "ot_vel_dyn": [{"c": c, "w": w} for c in (0.5, 2, 8) for w in (0.25, 0.5, 0.75)],
}
DEFAULT = {"ot": {"noise": 0.0}, "uot_growth": {"tau": 1.0, "beta": 2.0, "noise": 0.0}, "vel_ss": {"c": 1},
           "vel_dyn": {"c": 1}, "ot_vel_ss": {"c": 2, "w": 0.5}, "ot_vel_dyn": {"c": 2, "w": 0.5}}


def main():
    res = {"origins": {}, "families": list(GRIDS)}
    for s in range(1, 6):
        t0 = time.time()
        d = dict(np.load(os.path.join(OUT, f"origin{s}.npz"), allow_pickle=True))
        Zt, Xt = d["Z_fut"], d["X_fut"].astype(np.float32)
        st = d["stage_past"]
        Xnow = d["X_past"][st == s].astype(np.float32)
        Ztrue_now = d["Z_past"][st == s]
        row = {"n_now": int((st == s).sum()), "n_true": int(len(Zt))}
        half = np.random.default_rng(4).permutation(len(Zt))
        A, B = half[: len(Zt) // 2], half[len(Zt) // 2:]
        row["floor"] = {"sinkhorn_div": sinkhorn_div(Zt[A][:NSUB], Zt[B][:NSUB]), "energy": energy(Zt[A][:NSUB], Zt[B][:NSUB])}
        for fam, grid in GRIDS.items():
            # backtest on the previous transition (predict stage s from s-1 using s-2 -> s-1), origin's own space
            if len(grid) == 1:
                hp = grid[0]
            elif s >= 2:
                bt = []
                for g in grid:
                    Zf, Xf, w = forecast(fam, g, d, s - 1, (s - 1, s - 2))
                    sc = score(Zf, Xf, w, Ztrue_now, d["X_past"][st == s].astype(np.float32), d["X_past"][st == s - 1].astype(np.float32))
                    bt.append(sc["sinkhorn_div"])
                hp = grid[int(np.argmin(bt))]
            else:
                hp = DEFAULT[fam]
            Zf, Xf, w = forecast(fam, hp, d, s, (s, s - 1))
            row[fam] = {"hp": hp, **score(Zf, Xf, w, Zt, Xt, Xnow)}
            print(s, fam, row[fam], flush=True)
        row["seconds"] = round(time.time() - t0, 1)
        res["origins"][s] = row
        with open(os.path.join(OUT, "forecast.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
