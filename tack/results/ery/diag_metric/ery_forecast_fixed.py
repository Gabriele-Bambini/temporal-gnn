"""Metric audit and re-score of the first erythroid forecasting run (tack/hpc/ery_forecast.py).

Base: verbatim copy ery_forecast_original.py (sha256 64351c57f109b3f8..., the version of job 716259). Forecast
families, grids, backtest design, subsampling and the order in which the noise generator is consumed are unchanged.
What changes:

1. Solver (numerics only). The same Sinkhorn updates and the same stopping rule (<= 2000 iterations, tol 1e-7 on the
   potentials, KL-relaxed exponent tau / (tau + eps)) are run in the kernel domain, u = (a / K v)^k, v = (b / K'u)^k,
   about 40x faster than the log domain. If the kernel would underflow (a row / column with min C / eps > 600) or a
   scaling stops being finite, that call falls back to the original log-domain code (counted in the output).
   Checked against the original in test_solver.py and end to end by reproducing forecast.json.
2. Metrics (true stage s+1 cells, origin-s PCA space, 1500-cell subsamples exactly as before):
   sd_orig   original sinkhorn_div, kept ONLY to reproduce the first run. FLAW: its scale is the median squared
             distance between the first 300 PREDICTED and first 300 true subsampled cells; it is the divisor and
             sets eps = 0.05 * scale, so a prediction that spreads the cells gets a larger scale, a blurrier OT and a
             smaller number. scale_orig is stored with every score as audit evidence.
   sd_fixed  same debiased quantity [OT(X,Y) - OT(X,X)/2 - OT(Y,Y)/2] / S_s with a FIXED per-origin scale S_s =
             median pairwise squared distance among PAST cells (stage <= s; fixed seeded subsample of <= 1500 past
             cells; off-diagonal pairs), eps = 0.05 * S_s. Self terms with the symmetric Sinkhorn update (converges;
             the alternating update of the original stops at the 2000-iteration cap on symmetric problems, the
             difference is ~1e-6 relative, measured).
   energy    unchanged (no scale).
   mmd       weighted V-statistic MMD^2, RBF kernel exp(-|x - y|^2 / (2 S_s)) (median heuristic: sigma = median
             pairwise distance among past cells, the same S_s).
   One S_s per origin, used for backtest, final and floor scores; it never depends on a prediction.
3. Hyperparameters. The backtest (predict stage s from s-1 with the drift fitted on s-2 -> s-1, origin s space) scores
   every grid point under all metrics. Mode A = the hp recorded in forecast.json (first run, chosen by sd_orig);
   Mode B = hp re-selected by argmin of energy / sd_fixed / mmd on the same backtest. The global generator (seed 0)
   is consumed in the original order (backtest grid points, then the final forecast with the recorded hp); a final
   forecast with any other hp draws its noise from a separate seeded generator.
4. Descriptive diagnostic (--spread, uses the true next stage, NOT a forecast): persistence cloud plus isotropic
   Gaussian noise of growing size, scored by every metric (does the metric reward spreading?).

Python only, no torch. Reads the origin<s>.npz files (--data), writes only into this folder.
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = (r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN"
        r"\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\ery")
DT = 0.25
RNG = np.random.default_rng(0)
NSUB = 1500
_DRIFT = {}
STATS = {"kernel_calls": 0, "log_fallbacks": 0, "sym_calls": 0, "sym_unconverged": 0}
UNDERFLOW = 600.0


# ---------------- OT (rectangular) ----------------

def lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def sinkhorn_log(a, b, C, eps, tau=None, iters=2000, tol=1e-7):
    """ORIGINAL log-domain solver (verbatim): entropic OT plan; tau = None balanced, else KL-relaxed marginals."""
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


def sinkhorn(a, b, C, eps, tau=None, iters=2000, tol=1e-7):
    """Same iteration and stopping rule as sinkhorn_log, in the kernel domain (u = e^{f/eps}, v = e^{g/eps})."""
    STATS["kernel_calls"] += 1
    Ce = np.asarray(C, np.float64) / eps
    if Ce.min(1).max() > UNDERFLOW or Ce.min(0).max() > UNDERFLOW:
        STATS["log_fallbacks"] += 1
        return sinkhorn_log(a, b, C, eps, tau, iters, tol)
    K = np.exp(-Ce)
    k = 1.0 if tau is None else tau / (tau + eps)
    f, g = np.zeros(len(a)), np.zeros(len(b))
    v = np.ones(len(b))
    with np.errstate(all="ignore"):
        for it in range(iters):
            u = a / (K @ v)
            if tau is not None:
                u = u ** k
            v = b / (K.T @ u)
            if tau is not None:
                v = v ** k
            if not (np.isfinite(u).all() and np.isfinite(v).all() and u.min() > 0 and v.min() > 0):
                STATS["log_fallbacks"] += 1
                return sinkhorn_log(a, b, C, eps, tau, iters, tol)
            f_new, g_new = eps * np.log(u), eps * np.log(v)
            done = max(np.abs(f_new - f).max(), np.abs(g_new - g).max()) < tol
            f, g = f_new, g_new
            if done:
                break
    return u[:, None] * K * v[None, :]


def sinkhorn_sym(a, C, eps, iters=2000, tol=1e-7):
    """Balanced symmetric problem (same cloud both sides): u <- sqrt(u * a / K u); plan u K u."""
    STATS["sym_calls"] += 1
    K = np.exp(-np.asarray(C, np.float64) / eps)  # diagonal of C is 0, so no row underflows
    u = np.ones(len(a))
    f = np.zeros(len(a))
    for it in range(iters):
        u = np.sqrt(u * a / (K @ u))
        f_new = eps * np.log(u)
        done = np.abs(f_new - f).max() < tol
        f = f_new
        if done:
            break
    else:
        STATS["sym_unconverged"] += 1
    return u[:, None] * K * u[None, :]


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def unif(n):
    return np.full(n, 1 / n)


def sinkhorn_div_orig(X, Y, wx=None, wy=None, eps_rel=0.05):
    """ORIGINAL (flawed) metric: scale from the PREDICTION's first 300 subsampled cells. Returns (value, scale)."""
    wx = unif(len(X)) if wx is None else wx / wx.sum()
    wy = unif(len(Y)) if wy is None else wy / wy.sum()
    scale = np.median(sqdist(X[:300], Y[:300]))
    eps = eps_rel * scale

    def ot(A, B, wa, wb):
        C = sqdist(A, B)
        P = sinkhorn(wa, wb, C, eps)
        return float((P * C).sum())
    return float((ot(X, Y, wx, wy) - 0.5 * ot(X, X, wx, wx) - 0.5 * ot(Y, Y, wy, wy)) / scale), float(scale)


def ot_cost(A, B, wa, wb, eps):
    C = sqdist(A, B)
    P = sinkhorn(wa, wb, C, eps)
    return float((P * C).sum())


def ot_self(A, wa, eps):
    C = sqdist(A, A)
    return float((sinkhorn_sym(wa, C, eps) * C).sum())


def sinkhorn_div_fixed(X, Y, wx, wy, S, yy, eps_rel=0.05):
    """Debiased Sinkhorn quantity with the fixed past-cell scale S (divisor and eps = eps_rel * S)."""
    eps = eps_rel * S
    return (ot_cost(X, Y, wx, wy, eps) - 0.5 * ot_self(X, wx, eps) - 0.5 * yy) / S


def rbf(A, B, S):
    return np.exp(-sqdist(A, B) / (2.0 * S))


def mmd2(X, Y, wx, wy, S, yy):
    return float(wx @ rbf(X, X, S) @ wx + yy - 2.0 * wx @ rbf(X, Y, S) @ wy)


def energy(X, Y, wx=None, wy=None):
    wx = np.full(len(X), 1 / len(X)) if wx is None else wx / wx.sum()
    wy = np.full(len(Y), 1 / len(Y)) if wy is None else wy / wy.sum()
    d = lambda A, B: np.sqrt(sqdist(A, B))
    return float(2 * wx @ d(X, Y) @ wy - wx @ d(X, X) @ wx - wy @ d(Y, Y) @ wy)


def sub(n, k=NSUB, seed=0):
    r = np.random.default_rng(seed)
    return np.sort(r.choice(n, min(n, k), replace=False))


def past_scale(Z_past, seed=7):
    """S_s: median squared distance between distinct past cells (fixed seeded subsample of <= NSUB cells)."""
    Zs = Z_past[sub(len(Z_past), seed=seed)]
    C = sqdist(Zs, Zs)
    iu = np.triu_indices(len(Zs), 1)
    return float(np.median(C[iu]))


# ---------------- drift fields (verbatim) ----------------

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


# ---------------- forecasts (verbatim except the noise generator argument) ----------------

def forecast(fam, hp, d, s, past_idx, rng=None):
    """past_idx: (index of origin stage, index of previous stage) into the origin arrays."""
    rng = RNG if rng is None else rng
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
            Zf = Zf + rng.normal(0, hp["noise"], Zf.shape)
    return Zf, Xf, w


class Target:
    """True cloud for one scoring problem: fixed subsample, its self terms under S (cached)."""

    def __init__(self, Zt, S, seed=1, idx=None):
        j = sub(len(Zt), seed=seed) if idx is None else idx
        self.Y = Zt[j]
        self.wy = unif(len(self.Y))
        self.S = S
        self.yy_ot = ot_self(self.Y, self.wy, 0.05 * S)
        self.yy_mmd = float(self.wy @ rbf(self.Y, self.Y, S) @ self.wy)


def score(Zf, Xf, w, tgt, Xt, Xnow, do_orig=True):
    i = sub(len(Zf))
    wi = None if w is None else w[i]
    X, Y = Zf[i], tgt.Y
    wx = unif(len(X)) if wi is None else wi / wi.sum()
    out = {}
    if do_orig:
        out["sd_orig"], out["scale_orig"] = sinkhorn_div_orig(X, Y, wi)
    out["energy"] = energy(X, Y, wi)
    out["sd_fixed"] = float(sinkhorn_div_fixed(X, Y, wx, tgt.wy, tgt.S, tgt.yy_ot))
    out["mmd"] = mmd2(X, Y, wx, tgt.wy, tgt.S, tgt.yy_mmd)
    out["spread"] = float(np.sqrt(((X - wx @ X) ** 2).sum(1) @ wx))  # rms distance to the (weighted) centroid
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
SELECTORS = ("sd_orig", "energy", "sd_fixed", "mmd")


def hkey(hp):
    return json.dumps(hp, sort_keys=True)


def grid_index(fam, hp):
    for gi, g in enumerate(GRIDS[fam]):
        if hkey(g) == hkey(hp):
            return gi
    return len(GRIDS[fam])


def load(data, s):
    return dict(np.load(os.path.join(data, f"origin{s}.npz"), allow_pickle=True))


def run_main(args):
    first = json.load(open(args.first))
    res = {"note": "see module docstring; Mode A = recorded hp (first run), Mode B = hp selected by each metric",
           "families": list(GRIDS), "origins": {}}
    out_path = os.path.join(args.out, "rescore.json")
    for s in args.origins:
        t0 = time.time()
        d = load(args.data, s)
        Zt, Xt = d["Z_fut"], d["X_fut"].astype(np.float32)
        st = d["stage_past"]
        Xnow = d["X_past"][st == s].astype(np.float32)
        Ztrue_now = d["Z_past"][st == s]
        S = past_scale(d["Z_past"])
        tgt_fut = Target(Zt, S)
        tgt_bt = Target(Ztrue_now, S) if s >= 2 else None
        rec = first["origins"][str(s)]
        row = {"n_now": int((st == s).sum()), "n_true": int(len(Zt)), "S_fixed": S, "families": {}}
        half = np.random.default_rng(4).permutation(len(Zt))
        A, B = half[: len(Zt) // 2], half[len(Zt) // 2:]
        ZA, ZB = Zt[A][:NSUB], Zt[B][:NSUB]
        tB = Target(ZB, S, idx=np.arange(len(ZB)))
        wA = unif(len(ZA))
        sdo, sco = sinkhorn_div_orig(ZA, ZB)
        row["floor"] = {"sd_orig": sdo, "scale_orig": sco, "energy": energy(ZA, ZB),
                        "sd_fixed": float(sinkhorn_div_fixed(ZA, ZB, wA, tB.wy, S, tB.yy_ot)),
                        "mmd": mmd2(ZA, ZB, wA, tB.wy, S, tB.yy_mmd),
                        "first_run": rec["floor"]}
        print(s, "S_fixed", round(S, 3), "floor", {k: v for k, v in row["floor"].items() if k != "first_run"},
              flush=True)
        for fi, (fam, grid) in enumerate(GRIDS.items()):
            fr = {"backtest": [], "sel": {}, "final": {}}
            rec_hp = rec[fam]["hp"]
            if len(grid) > 1 and s >= 2:
                for g in grid:
                    Zf, Xf, w = forecast(fam, g, d, s - 1, (s - 1, s - 2))  # global RNG, original order
                    sc = score(Zf, Xf, w, tgt_bt, d["X_past"][st == s].astype(np.float32),
                               d["X_past"][st == s - 1].astype(np.float32), do_orig=not args.no_orig)
                    fr["backtest"].append({"hp": g, **sc})
                for m in SELECTORS:
                    if m == "sd_orig" and args.no_orig:
                        continue
                    fr["sel"][m] = grid[int(np.argmin([b[m] for b in fr["backtest"]]))]
            else:
                hp0 = grid[0] if len(grid) == 1 else DEFAULT[fam]
                for m in SELECTORS:
                    fr["sel"][m] = hp0
            fr["sel"]["recorded"] = rec_hp
            fr["recorded_reproduced"] = (None if "sd_orig" not in fr["sel"]
                                         else hkey(fr["sel"]["sd_orig"]) == hkey(rec_hp))
            # final forecasts: the recorded hp first with the global generator (original order), then any other
            Zf, Xf, w = forecast(fam, rec_hp, d, s, (s, s - 1))
            fr["final"][hkey(rec_hp)] = score(Zf, Xf, w, tgt_fut, Xt, Xnow, do_orig=not args.no_orig)
            for m, hp in fr["sel"].items():
                k = hkey(hp)
                if k in fr["final"]:
                    continue
                rng = np.random.default_rng([s, fi, grid_index(fam, hp), 99])
                Zf, Xf, w = forecast(fam, hp, d, s, (s, s - 1), rng=rng)
                fr["final"][k] = score(Zf, Xf, w, tgt_fut, Xt, Xnow, do_orig=not args.no_orig)
            r0 = fr["final"][hkey(rec_hp)]
            fr["check_vs_first_run"] = {
                m: (abs(r0[m] - rec[fam][m2]) if np.isfinite(r0.get(m, np.nan)) and np.isfinite(rec[fam][m2])
                    else None)
                for m, m2 in (("energy", "energy"), ("pearson", "pearson"), ("p200_up", "p200_up"),
                              ("p200_down", "p200_down"), ("sd_orig", "sinkhorn_div")) if m in r0}
            row["families"][fam] = fr
            print(s, fam, "sel", {m: hkey(h) for m, h in fr["sel"].items()}, "repro", fr["recorded_reproduced"],
                  "check", {k: (None if v is None else float(f"{v:.2e}")) for k, v in fr["check_vs_first_run"].items()},
                  round(time.time() - t0, 1), "s", flush=True)
        row["seconds"] = round(time.time() - t0, 1)
        row["solver_stats"] = dict(STATS)
        res["origins"][str(s)] = row
        _DRIFT.clear()
        with open(out_path, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    print("done", STATS, flush=True)


def run_spread(args):
    """DESCRIPTIVE (looks at the true next stage, not a forecast): persistence + N(0, sigma^2) isotropic noise."""
    sigmas = (0.0, 0.5, 1.0, 2.0, 4.0)
    res = {"note": "descriptive: stage-s cells plus isotropic Gaussian noise vs the TRUE stage s+1; "
                   "not a forecast, not used for any selection", "sigmas": sigmas, "origins": {}}
    for s in args.origins:
        d = load(args.data, s)
        st = d["stage_past"]
        Zn, Zt = d["Z_past"][st == s], d["Z_fut"]
        S = past_scale(d["Z_past"])
        tgt = Target(Zt, S)
        rows = []
        for si, sg in enumerate(sigmas):
            Zf = Zn + np.random.default_rng([s, si, 5]).normal(0, sg, Zn.shape) if sg > 0 else Zn
            i = sub(len(Zf))
            X = Zf[i]
            wx = unif(len(X))
            sdo, sco = sinkhorn_div_orig(X, tgt.Y)
            rows.append({"sigma": sg, "sd_orig": sdo, "scale_orig": sco, "energy": energy(X, tgt.Y),
                         "sd_fixed": float(sinkhorn_div_fixed(X, tgt.Y, wx, tgt.wy, S, tgt.yy_ot)),
                         "mmd": mmd2(X, tgt.Y, wx, tgt.wy, S, tgt.yy_mmd)})
            print(s, rows[-1], flush=True)
        res["origins"][str(s)] = {"S_fixed": S, "rows": rows}
        with open(os.path.join(args.out, "spread_bias.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--first", default=os.path.join(DATA, "forecast.json"))
    ap.add_argument("--origins", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--no-orig", action="store_true", help="skip the (slow) original flawed metric")
    ap.add_argument("--spread", action="store_true", help="run only the descriptive spread-bias diagnostic")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run_spread(a) if a.spread else run_main(a)
