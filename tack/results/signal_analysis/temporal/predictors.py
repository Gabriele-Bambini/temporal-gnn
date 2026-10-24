"""Strict rolling-origin temporal forecasters for Dileo Task 2 under the three targets (orig, l1, linear).

At transition i (snapshot i -> i+1) every forecaster sees only snapshots 0..i (their mean, std, frac, cells).
Families (each a grid of configurations):
  trend         repeat the last observed change (target space)
  level_box     next mean = cell-weighted mean of the last k bins (pure reversion / denoising)
  lpoly_raw     weighted (n cells) local polynomial of degree 1/2 over the last W bins, raw mean, extrapolated 1 bin
  lpoly_l1      same on log1p(CP10k)
  holt_raw      Holt linear exponential smoothing (alpha, beta), raw mean
  holt_l1       same on log1p(CP10k)
  kalman_raw    local-linear-trend Kalman filter on the raw mean, R = (max(s^2 n/(n-1), mbar) + 1/n) / n (sampling noise of a
                bin mean, Poisson floor), Q = theta^2 (mbar^2 + mbar/50) (mbar = running cell-weighted mean, strict);
                theta_slope = 0 is the local-level filter
  kalman_l1     same on log1p(CP10k), R by the delta method, Q = theta^2
  ar_pooled     pooled OLS AR(p) on the target itself, fitted on past transitions only (all past / last 8)
  ar_strat      same, one AR per decile of the running mean CP10k (strict strata)
  obs_level, obs_kalman (orig only): forecast mean -> expected clipped target when bin t+1 is a finite sample of
                n_t cells from a zero-inflated law (the finite-cell observation model of task2_forecasters.ObsLevel)
Selection: per family, the configuration of transition i is the one with the best mean criterion
J = PCC + P@200 up + P@200 down (all genes) over the past transitions j in [i-8, i-1] (j >= 1); the default config
is used at i = 1. 'meta' selects among the family-selected sequences the same way. Hindsight-best fixed configs
are reported separately and flagged NOT strict.
Scoring: 20 random 25 % gene masks per transition (same for every model), mean over transitions i >= 1, mean and
sd over draws. Output: predictors.json.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, SNAPS, TRAIN_SPLIT, TARGETS, tgt, EPS, K  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEL_WINDOW = 8
N_DRAWS = 20
C_Q = 1 / 50


# ------------------------------------------------------------------ helpers

def run_mean(m, n):
    """Running cell-weighted mean over bins 0..b (strict), shape (G, B)."""
    w = n[None, :]
    return np.cumsum(m * w, 1) / np.cumsum(w, 1)


def lib(m):
    return np.maximum(m.sum(0), 1e-12)


def L_of(m):
    return np.log1p(1e4 * m / lib(m)[None, :])


def raw_to_targets(m, mhat_seq):
    """mhat_seq (T, G): forecast raw mean of bin i+1 made at i -> dict target -> (T, G)."""
    T = mhat_seq.shape[0]
    out = {t: np.zeros((T, m.shape[0]), np.float32) for t in TARGETS}
    for i in range(T):
        mh = np.maximum(mhat_seq[i], 0)
        for t in TARGETS:
            out[t][i] = tgt(t, m[:, i], mh)
    return out


def l1_to_targets(m, Lhat_seq):
    T = Lhat_seq.shape[0]
    Lobs = L_of(m)
    lb = lib(m)
    out = {t: np.zeros((T, m.shape[0]), np.float32) for t in TARGETS}
    for i in range(T):
        out["l1"][i] = Lhat_seq[i] - Lobs[:, i]
        mh = np.maximum(np.expm1(Lhat_seq[i]), 0) * lb[i] / 1e4
        out["linear"][i] = mh - m[:, i]
        out["orig"][i] = tgt("orig", m[:, i], mh)
    return out


class Truth:
    def __init__(self, Y):
        self.Y = Y
        self.up = [set(np.argsort(-y)[:K]) for y in Y]
        self.dn = [set(np.argsort(y)[:K]) for y in Y]
        self.yc = [y - y.mean() for y in Y]
        self.yn = [np.sqrt((c * c).sum()) for c in self.yc]

    def J(self, i, p):
        pc = p - p.mean()
        den = self.yn[i] * np.sqrt((pc * pc).sum())
        r = float((self.yc[i] * pc).sum() / den) if den > 0 else 0.0
        up = len(self.up[i] & set(np.argsort(-p)[:K])) / K
        dn = len(self.dn[i] & set(np.argsort(p)[:K])) / K
        return r, up, dn


def mask_score(Y, P, masks, start=1):
    """masks[d][i] -> per-draw transition means; returns mean, sd over draws and per-transition means."""
    T = len(Y)
    arr = np.zeros((len(masks), T, 3))
    for d, mk in enumerate(masks):
        for i in range(T):
            y, p = Y[i][mk[i]], P[i][mk[i]]
            yc, pc = y - y.mean(), p - p.mean()
            den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
            r = float((yc * pc).sum() / den) if den > 0 else np.nan     # constant prediction -> NaN (their nanmean)
            up = len(set(np.argsort(-y)[:K]) & set(np.argsort(-p)[:K])) / K
            dn = len(set(np.argsort(y)[:K]) & set(np.argsort(p)[:K])) / K
            arr[d, i] = (r, up, dn)
    with np.errstate(all="ignore"):
        per_draw = np.nanmean(arr[:, start:], 1)
        per_draw_all = np.nanmean(arr, 1)                              # all transitions incl. i = 0 (DistBridge convention)
    mean, sd = per_draw.mean(0), per_draw.std(0, ddof=1)
    ma = per_draw_all.mean(0)
    pt = np.nanmean(arr, 0)
    return {"pcc": float(mean[0]), "p200_up": float(mean[1]), "p200_down": float(mean[2]),
            "pcc_sd": float(sd[0]), "p200_up_sd": float(sd[1]), "p200_down_sd": float(sd[2]),
            "all_trans_pcc": float(ma[0]), "all_trans_p200_up": float(ma[1]), "all_trans_p200_down": float(ma[2]),
            "per_transition_from_i0": pt.round(4).tolist()}


# ------------------------------------------------------------------ forecasters (return (T, G) forecasts)

def fc_level_box(m, n, T, k):
    out = np.zeros((T, m.shape[0]))
    for i in range(T):
        lo = max(0, i - k + 1)
        w = n[lo:i + 1]
        out[i] = (m[:, lo:i + 1] * w).sum(1) / w.sum()
    return out


def fc_lpoly(X, n, T, deg, W):
    out = np.zeros((T, X.shape[0]))
    for i in range(T):
        lo = max(0, i - W + 1)
        npts = i - lo + 1
        d = min(deg, npts - 1)
        if d <= 0:
            w = n[lo:i + 1]
            out[i] = (X[:, lo:i + 1] * w).sum(1) / w.sum()
            continue
        x = np.arange(lo, i + 1, dtype=float) - i
        coef = np.polyfit(x, X[:, lo:i + 1].T, d, w=np.sqrt(n[lo:i + 1]))
        out[i] = np.polyval(coef, 1.0)
    return out


def fc_holt(X, T, alpha, beta):
    lvl = X[:, 0].copy()
    b = np.zeros_like(lvl)
    out = np.zeros((T, X.shape[0]))
    out[0] = lvl
    for t in range(1, T):
        new = alpha * X[:, t] + (1 - alpha) * (lvl + b)
        b = beta * (new - lvl) + (1 - beta) * b
        lvl = new
        out[t] = lvl + b
    return out


def kalman(X, R, Qs, T, th_l, th_b, return_innov=False):
    """Local-linear-trend (th_b > 0) or local-level (th_b == 0) filter, vectorised over genes.
    X, R, Qs: (G, B); forecast for bin t+1 made at t is level + slope."""
    G = X.shape[0]
    lvl = X[:, 0].copy()
    b = np.zeros(G)
    P11 = R[:, 0].copy()
    P12 = np.zeros(G)
    P22 = (th_b ** 2) * Qs[:, 0] * 10.0
    out = np.zeros((T, G))
    out[0] = lvl + b
    innov, Svar = [], []
    for t in range(1, T + 1 if return_innov else T):
        if t >= X.shape[1]:
            break
        ql, qb = th_l ** 2 * Qs[:, t], th_b ** 2 * Qs[:, t]
        lp, bp = lvl + b, b
        P11p = P11 + 2 * P12 + P22 + ql
        P12p = P12 + P22
        P22p = P22 + qb
        S = P11p + R[:, t]
        v = X[:, t] - lp
        innov.append(v)
        Svar.append(S)
        K1, K2 = P11p / S, P12p / S
        lvl, b = lp + K1 * v, bp + K2 * v
        P11, P12, P22 = (1 - K1) * P11p, (1 - K1) * P12p, P22p - K2 * P12p
        if t < T:
            out[t] = lvl + b
    if return_innov:
        return out, np.array(innov), np.array(Svar)
    return out


def kalman_inputs(m, s, n, space):
    nn = n[None, :]
    mbar = run_mean(m, n)
    s2u = s ** 2 * nn / np.maximum(nn - 1, 1)
    Rraw = (np.maximum(s2u, mbar) + 1.0 / nn) / nn
    if space == "raw":
        return m, Rraw, mbar ** 2 + C_Q * mbar
    lb = lib(m)[None, :]
    cbar = 1e4 * mbar / lb
    return L_of(m), (1e4 / lb) ** 2 * Rraw / (1 + cbar) ** 2, np.ones_like(m)


def fc_ar(Y, T, p, window, strata=None):
    """Y (T, G) target; returns (T, G) strict forecasts. strata: list over i of (G,) int labels or None."""
    G = Y.shape[1]
    out = np.zeros((T, G))
    for i in range(1, T):
        pe = min(p, i - 1)
        if pe < 1:
            out[i] = Y[i - 1]
            continue
        lo = max(pe, i - window) if window else pe
        rows = range(lo, i)
        lab = strata[i] if strata is not None else np.zeros(G, int)
        Xf = np.stack([Y[i - 1 - j] for j in range(pe)], 1)
        for q in np.unique(lab):
            g = lab == q
            A = np.concatenate([np.stack([Y[s - 1 - j, g] for j in range(pe)], 1) for s in rows])
            A = np.hstack([A, np.ones((len(A), 1))])
            yv = np.concatenate([Y[s, g] for s in rows])
            coef = np.linalg.lstsq(A, yv, rcond=None)[0]
            out[i, g] = Xf[g] @ coef[:-1] + coef[-1]
    return out


def strata_seq(m, n, T, nq=10):
    lab = []
    cp = 1e4 * m / lib(m)[None, :]
    rm = run_mean(cp, n)
    for i in range(T):
        r = rm[:, i]
        l = np.full(len(r), nq)                      # stratum nq = not yet expressed
        pos = r > 0
        rk = stats.rankdata(r[pos]) / pos.sum()
        l[pos] = np.minimum((rk * nq).astype(int), nq - 1)
        lab.append(l)
    return lab


def obs_expected(m_now, f, mpool, ex2, n, rng, draws=128):
    """E[clip(log(S/n + eps) - log(m_now + eps), +-5)], S = sum of n cells of a zero-inflated law (f, m, E c^2)."""
    f = np.clip(f, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(f > 0, mpool / np.where(f > 0, f, 1), 0.0)
        var = np.where(f > 0, np.maximum(ex2 / np.where(f > 0, f, 1) - mu ** 2, 0), 0.0)
    Kc = rng.binomial(int(n), f[:, None], size=(len(f), draws))
    tot = np.maximum(Kc * mu[:, None] + np.sqrt(Kc * var[:, None]) * rng.standard_normal((len(f), draws)), 0)
    d = np.log(tot / max(n, 1) + EPS) - np.log(np.maximum(m_now, 0)[:, None] + EPS)
    return np.clip(d, -5, 5).mean(1)


def fc_obs(m, s, f, n, T, mhat_seq, k_shape=4, seed=0):
    """Expected orig target given forecast mean mhat (law shape = cell-weighted mixture of the last k bins, its
    nonzero fraction rescaled so that the mean equals mhat)."""
    out = np.zeros((T, m.shape[0]))
    for i in range(T):
        lo = max(0, i - k_shape + 1)
        w = n[lo:i + 1] / n[lo:i + 1].sum()
        fm = (f[:, lo:i + 1] * w).sum(1)
        mm = (m[:, lo:i + 1] * w).sum(1)
        e2 = ((s[:, lo:i + 1] ** 2 + m[:, lo:i + 1] ** 2) * w).sum(1)
        mh = np.maximum(mhat_seq[i], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(mm > 0, mh / np.where(mm > 0, mm, 1), 0.0)
        fh, e2h = fm * r, e2 * r
        # cap f at 1: put the excess into the positive-part mean (scale E c^2 quadratically)
        over = fh > 1
        if over.any():
            scale = fh[over]
            fh[over] = 1.0
            e2h[over] = e2h[over] * scale
        newg = (mm == 0) & (mh > 0)                   # unseen in the shape window: one-count cells
        fh[newg], e2h[newg] = np.minimum(mh[newg], 1), mh[newg]
        out[i] = obs_expected(m[:, i], fh, mh, e2h, n[i], np.random.default_rng(seed * 1000 + i))
    return out


# ------------------------------------------------------------------ selection

def family_select(Jmat, default_idx):
    """Jmat (C, T): criterion of config c at transition j. Returns selected config per transition (strict)."""
    C, T = Jmat.shape
    sel = np.full(T, default_idx)
    for i in range(2, T):
        lo = max(1, i - SEL_WINDOW)
        sel[i] = int(np.argmax(Jmat[:, lo:i].mean(1)))
    return sel


def run_family(name, configs, make, truths, T, default_idx, keep_all=False):
    """configs: list of labels; make(c) -> dict target -> (T, G) forecasts. Returns per target: J matrix,
    selected sequence forecasts, selection indices, hindsight-best config."""
    res = {}
    Jm = {t: np.zeros((len(configs), T)) for t in truths}
    Jparts = {t: np.zeros((len(configs), T, 3)) for t in truths}
    store = {t: [] for t in truths}
    for c, lab in enumerate(configs):
        P = make(lab)
        for t in truths:
            if t not in P:
                continue
            for i in range(1, T):
                parts = truths[t].J(i, P[t][i])
                Jparts[t][c, i] = parts
                Jm[t][c, i] = sum(parts)
            store[t].append(P[t].astype(np.float32))
    for t in truths:
        if not store[t]:
            continue
        sel = family_select(Jm[t], default_idx)
        seq = np.stack([store[t][sel[i]][i] for i in range(T)])
        best = int(np.argmax(Jm[t][:, 1:].mean(1)))
        res[t] = {"J": Jm[t], "Jparts": Jparts[t], "sel": sel, "seq": seq, "best": best,
                  "best_seq": store[t][best], "configs": configs,
                  "all": store[t] if keep_all else None}
    return res


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    out = {}
    for name in SNAPS:
        d = load(name)
        m, s, f, n = d["m"], d["s"], d["f"], d["n"]
        T = TRAIN_SPLIT[name]
        G = m.shape[0]
        Y = {t: np.stack([tgt(t, m[:, i], m[:, i + 1]) for i in range(T)]) for t in TARGETS}
        truths = {t: Truth(Y[t]) for t in TARGETS}
        rng = np.random.default_rng(0)
        n_test = int(round(0.25 * G))
        masks = [[np.sort(rng.permutation(G)[:n_test]) for _ in range(T)] for _ in range(N_DRAWS)]
        Xl1 = L_of(m)
        fams = {}

        fams["trend"] = run_family("trend", ["last"], lambda c: {t: np.vstack([np.zeros((1, G)), Y[t][:-1]]) for t in TARGETS},
                                   truths, T, 0)
        ks = [2, 3, 4, 6, 8]
        fams["level_box"] = run_family("level_box", ks, lambda k: raw_to_targets(m, fc_level_box(m, n, T, k)), truths, T, 2)
        lp = [(1, 3), (1, 4), (1, 6), (1, 8), (2, 4), (2, 6), (2, 8)]
        fams["lpoly_raw"] = run_family("lpoly_raw", lp, lambda c: raw_to_targets(m, fc_lpoly(m, n, T, *c)), truths, T, 2)
        lpl = [(0, 2), (0, 4), (0, 6)] + lp
        fams["lpoly_l1"] = run_family("lpoly_l1", lpl, lambda c: l1_to_targets(m, fc_lpoly(Xl1, n, T, *c)), truths, T, 1)
        hg = [(a, b) for a in (0.2, 0.35, 0.5, 0.7, 0.9, 1.0) for b in (0.0, 0.1, 0.25, 0.5, 1.0) if not (a == 1.0 and b == 0.0)]
        fams["holt_raw"] = run_family("holt_raw", hg, lambda c: raw_to_targets(m, fc_holt(m, T, *c)), truths, T, hg.index((0.5, 0.1)))
        fams["holt_l1"] = run_family("holt_l1", hg, lambda c: l1_to_targets(m, fc_holt(Xl1, T, *c)), truths, T, hg.index((0.5, 0.1)))
        Xr, Rr, Qr = kalman_inputs(m, s, n, "raw")
        kg = [(a, b) for a in (0.05, 0.1, 0.2, 0.4, 0.8) for b in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)]
        fams["kalman_raw"] = run_family("kalman_raw", kg, lambda c: raw_to_targets(m, kalman(Xr, Rr, Qr, T, *c)), truths, T,
                                        kg.index((0.2, 0.0)))
        XL, RL, QL = kalman_inputs(m, s, n, "l1")
        kgl = [(a, b) for a in (0.02, 0.05, 0.1, 0.2, 0.4) for b in (0.0, 0.01, 0.03, 0.1, 0.3)]
        fams["kalman_l1"] = run_family("kalman_l1", kgl, lambda c: l1_to_targets(m, kalman(XL, RL, QL, T, *c)), truths, T,
                                       kgl.index((0.1, 0.0)))
        arg = [(p, w) for p in (1, 2, 3, 4, 5) for w in (0, 8)]
        fams["ar_pooled"] = run_family("ar_pooled", arg, lambda c: {t: fc_ar(Y[t], T, *c) for t in TARGETS}, truths, T, arg.index((2, 0)))
        lab = strata_seq(m, n, T)
        ars = [(p, 0) for p in (1, 2, 3, 5)]
        fams["ar_strat"] = run_family("ar_strat", ars, lambda c: {t: fc_ar(Y[t], T, *c, strata=lab) for t in TARGETS}, truths, T, 1)
        # finite-cell observation model on top of a forecast mean (orig only)
        tro = {"orig": truths["orig"]}
        fams["obs_level"] = run_family("obs_level", [2, 4, 6, 8],
                                       lambda k: {"orig": fc_obs(m, s, f, n, T, fc_level_box(m, n, T, k), k_shape=k)}, tro, T, 1)
        okg = [(a, b) for a in (0.1, 0.2, 0.4) for b in (0.0, 0.05, 0.2)]
        fams["obs_kalman"] = run_family("obs_kalman", okg,
                                        lambda c: {"orig": fc_obs(m, s, f, n, T, kalman(Xr, Rr, Qr, T, *c))}, tro, T, okg.index((0.2, 0.0)))
        print(f"{name}: families done {time.time() - t0:.0f}s", flush=True)

        # meta selection over family-selected sequences
        res = {"n_genes": G, "cells": n.astype(int).tolist(), "targets": {}}
        for t in TARGETS:
            fl = [k for k in fams if t in fams[k]]
            Jf = np.zeros((len(fl), T))
            for a, k in enumerate(fl):
                seq = fams[k][t]["seq"]
                for i in range(1, T):
                    Jf[a, i] = sum(truths[t].J(i, seq[i]))
            msel = family_select(Jf, fl.index("kalman_raw"))
            meta = np.stack([fams[fl[msel[i]]][t]["seq"][i] for i in range(T)])
            rt = {"families": {}, "meta": {}}
            for k in fl:
                F = fams[k][t]
                cfg = F["configs"]
                sc = mask_score(Y[t], F["seq"], masks)
                hb = mask_score(Y[t], F["best_seq"], masks)
                oracle = np.argmax(F["J"][:, 1:], 0)
                agree = float(np.mean(F["sel"][2:] == oracle[1:]))
                # regret: oracle-per-transition J minus selected J (all genes)
                regret = float(np.mean(F["J"][:, 1:].max(0) - F["J"][F["sel"][1:], np.arange(1, T)]))
                rt["families"][k] = {
                    "strict_selected": {kk: v for kk, v in sc.items()},
                    "selected_configs": [str(cfg[c]) for c in F["sel"]],
                    "hindsight_best_fixed_NOT_STRICT": {"config": str(cfg[F["best"]]), **{kk: v for kk, v in hb.items() if kk != "per_transition_from_i0"}},
                    "oracle_best_config_per_transition_NOT_STRICT": [str(cfg[c]) for c in oracle],
                    "selection_agrees_with_oracle": agree,
                    "mean_regret_J": regret,
                    "config_J_mean_allgenes": {str(c): float(F["J"][a, 1:].mean()) for a, c in enumerate(cfg)},
                }
            ms = mask_score(Y[t], meta, masks)
            rt["meta"] = {**ms, "selected_family": [fl[c] for c in msel]}
            res["targets"][t] = rt
            print(f"== {name} target={t}")
            for k in fl:
                x = rt["families"][k]["strict_selected"]
                hb = rt["families"][k]["hindsight_best_fixed_NOT_STRICT"]
                print(f"  {k:11s} RO  PCC {x['pcc']:+.3f}+-{x['pcc_sd']:.3f}  up {x['p200_up']:.3f}  dn {x['p200_down']:.3f}"
                      f"   | hindsight {hb['config']:>12s} PCC {hb['pcc']:+.3f} up {hb['p200_up']:.3f} dn {hb['p200_down']:.3f}"
                      f"   | agree {rt['families'][k]['selection_agrees_with_oracle']:.2f}")
            print(f"  META        RO  PCC {ms['pcc']:+.3f}+-{ms['pcc_sd']:.3f}  up {ms['p200_up']:.3f}  dn {ms['p200_down']:.3f}")
        # ---- drift of the best memory over pseudotime (oracle, all genes J)
        drift = {}
        for k, par in (("level_box", lambda c: c), ("lpoly_raw", lambda c: c[1]), ("ar_pooled", lambda c: c[0]),
                       ("kalman_raw", lambda c: c[0]), ("kalman_l1", lambda c: c[0]), ("holt_raw", lambda c: c[0])):
            dk = {}
            for t in TARGETS:
                if t not in fams[k]:
                    continue
                F = fams[k][t]
                orc = np.argmax(F["J"][:, 1:], 0)
                vals = np.array([par(F["configs"][c]) for c in orc], float)
                nn = n[1:T]
                dk[t] = {"oracle_param_per_transition": vals.tolist(),
                         "spearman_vs_cells_of_bin_t": float(stats.spearmanr(vals, nn).correlation) if vals.std() > 0 else None,
                         "spearman_vs_pseudotime": float(stats.spearmanr(vals, np.arange(1, T)).correlation) if vals.std() > 0 else None,
                         "frac_consecutive_equal": float(np.mean(vals[1:] == vals[:-1])),
                         "n_distinct": int(len(np.unique(vals)))}
            drift[k] = dk
        res["drift"] = drift
        # ---- Kalman innovations: is the series 'smooth curve + sampling noise'?
        diag = {}
        cp = (1e4 * m / lib(m)[None, :]).mean(1)
        live = m[:, :T + 1].max(1) > 0
        dec = np.full(G, -1)
        rk = stats.rankdata(cp[live]) / live.sum()
        dec[live] = np.minimum((rk * 10).astype(int), 9)
        for space, (Xs, Rs, Qs), grid in (("raw", (Xr, Rr, Qr), kg), ("l1", (XL, RL, QL), kgl)):
            ll = []
            for c in grid:
                _, v, S = kalman(Xs, Rs, Qs, T, *c, return_innov=True)
                z2 = v[1:, live] ** 2 / S[1:, live]
                ll.append(float(np.mean(-0.5 * (np.log(S[1:, live]) + z2))))
            cbest = grid[int(np.argmax(ll))]
            _, v, S = kalman(Xs, Rs, Qs, T, *cbest, return_innov=True)
            z = v[1:] / np.sqrt(S[1:])
            zl = z[:, live]
            ac1 = float((zl[1:] * zl[:-1]).sum() / np.sqrt((zl[1:] ** 2).sum() * (zl[:-1] ** 2).sum()))
            byd = {}
            for q in range(10):
                zq = z[:, dec == q]
                byd[q] = {"mean_z2": float(np.mean(zq ** 2)), "median_z2": float(np.median(zq ** 2)),
                          "lag1_acf": float((zq[1:] * zq[:-1]).sum() / np.sqrt((zq[1:] ** 2).sum() * (zq[:-1] ** 2).sum()))}
            diag[space] = {"ml_config": str(cbest), "loglik_by_config": dict(zip(map(str, grid), ll)),
                           "mean_z2": float(np.mean(zl ** 2)), "median_z2": float(np.median(zl ** 2)),
                           "frac_abs_z_gt_3": float(np.mean(np.abs(zl) > 3)), "lag1_acf_z": ac1, "by_decile": byd}
            print(f"  kalman {space}: ML config {cbest} mean z^2 {diag[space]['mean_z2']:.2f} median z^2 {diag[space]['median_z2']:.2f} "
                  f"|z|>3 {diag[space]['frac_abs_z_gt_3']:.3f} lag1 acf {ac1:+.3f}")
            print("     mean z2 by decile " + " ".join(f"{byd[q]['mean_z2']:.2f}" for q in range(10)))
            print("     lag1 acf by decile " + " ".join(f"{byd[q]['lag1_acf']:+.2f}" for q in range(10)))
        res["kalman_innovations"] = diag
        out[name] = res
        print(f"{name}: done {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(HERE, "predictors.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
