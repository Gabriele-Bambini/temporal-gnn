"""Strict rolling-origin forecasts of the next bin from cross-gene / population structure, vs per-gene forecasts.

Every config forecasts the next state L_hat(t+1) = log1p(CP10k) of all genes from snapshots <= t only; the targets
are then l1 = L_hat - L_t and linear = expm1(L_hat) * lib_hat / 1e4 - m_t, with lib_hat in {lib_t (lp), lib_t *
(lib_t / lib_t-1)^0.5 (lh), lib_t * lib_t / lib_t-1 (lt)} part of the config for linear.
Families
  pergene : level_k (cell-weighted CP mean of the last k bins), trendL_phi (L_t + phi (L_t - L_t-1)), linfit_w_phi
            (weighted linear fit of L over the last w bins, value at t + phi * slope), Kalman local-linear-trend per
            gene (observation noise kappa_g / n_b from the strict running per-cell variance, process noise q1, q2
            absolute, or kalmanS: q1, q2 = lambda, mu x the gene's strict mean squared past change of L).
  svd     : gene-centred, cell-weighted SVD of L over bins <= t; scores z_b = U_k^T (L_b - mu); score forecast
            persist / damped last change (0.5, 1) / weighted linear fit over 3 or 5 bins; map back with the residual
            zero (revert to the subspace), last (keep today's residual: pure factor change) or avg3 (residual
            smoothed over 3 bins). k = 'full' reproduces the per-gene trends exactly.
  nmf     : weighted-KL NMF of CP10k over bins <= t (nmf_kl.fit_subset), K archetypes, proportions h_b; forecast
            h_t+1 by persist / linear (0.5, 1) / log-linear (0.5, 1) / entropic OT-Markov (balanced, eps 0.05, 0.5,
            and eps 0.05 half step) / unbalanced OT-Markov (eps 0.05, tau 1 or 0.1): plan P between h_t-1 and h_t
            with cost = mean squared log-profile difference between archetypes (normalised by its mean off-diagonal),
            transition kernel P / h_t-1 applied again to h_t; map back h_hat H with residual zero / last / avg3.
Selection: per family (and sub-family) at each transition, the config with the best mean J over the last 8 past
transitions (all genes) -- common.select; at i <= 1 (no past transition with history) every family uses the
per-gene trend in the target formula (identical fallback). Oracles (NOT strict, ceilings): svd / nmf with the TRUE
scores / proportions of bin t+1 projected on the strict basis. Blends: z-scored pergene + factor forecast, weight
chosen by rolling origin. Outputs forecasts_<dataset>.json.
"""
import json
import sys
import time

import numpy as np

import common
import nmf_kl

DELTA = 1e-3


# ---------------------------------------------------------------- helpers
def wfit(y, t, w, at):
    """Weighted linear fit of rows of y (n x T) on times t, evaluated at `at`; returns (value, slope)."""
    w = w / w.sum()
    tm = (w * t).sum()
    ym = (y * w).sum(1)
    st = (w * (t - tm) ** 2).sum()
    slope = ((y - ym[:, None]) * (w * (t - tm))).sum(1) / st if st > 0 else np.zeros(y.shape[0])
    return ym + slope * (at - tm), slope


def omega(nc):
    return nc / (nc + 50.0)


def uot_plan(a, b, C, eps, tau=None, n_iter=2000):
    Kmat = np.exp(-C / eps)
    u = np.ones_like(a)
    v = np.ones_like(b)
    fi = 1.0 if tau is None else tau / (tau + eps)
    for _ in range(n_iter):
        u = (a / np.maximum(Kmat @ v, 1e-300)) ** fi
        v = (b / np.maximum(Kmat.T @ u, 1e-300)) ** fi
    return u[:, None] * Kmat * v[None, :]


# ---------------------------------------------------------------- families
class PerGene:
    name = "pergene"

    def __init__(self, d):
        self.d = d
        cfg = [("level", k) for k in (1, 2, 3, 4, 6, 8)]
        cfg += [("trendL", p) for p in (0.25, 0.5, 1.0)]
        cfg += [("linfit", w, p) for w in (3, 4, 6, 8) for p in (0.0, 0.5, 1.0)]
        cfg += [("kalman", q1, q2) for q1 in (1e-4, 1e-3, 1e-2) for q2 in (0.0, 1e-5, 1e-4, 1e-3)]
        cfg += [("kalmanS", lam, mu) for lam in (0.1, 0.3, 1.0) for mu in (0.0, 0.03, 0.1, 0.3)]
        self.configs = cfg
        self._kal = {}

    def kalman_all(self, q1, q2, scaled=False):
        key = (q1, q2, scaled)
        if key in self._kal:
            return self._kal[key]
        d = self.d
        pc = (1e4 / d.lib[None, :]) ** 2 * d.SD ** 2 / (1 + d.CP) ** 2      # per-cell variance in L units
        cum = np.cumsum(pc * d.nc[None, :], 1) / np.cumsum(d.nc)[None, :]   # strict running mean
        N = d.N
        l, b = d.L[:, 0].copy(), np.zeros(N)
        P11, P12, P22 = np.full(N, 1.0), np.zeros(N), np.full(N, 1e-2)
        fc = np.zeros((d.S, N))
        dL2 = np.zeros((N, d.S))
        dL2[:, 1:] = np.diff(d.L, axis=1) ** 2
        V = np.cumsum(dL2, 1) / np.maximum(np.arange(d.S), 1)[None, :]       # strict mean squared past change
        for s in range(d.S):
            if s > 0:                                                       # predict
                if scaled:                                                  # gene-specific process noise
                    Vs = np.maximum(V[:, s - 1] if s > 1 else V[:, 1] * 0 + 1e-4, 1e-8)
                    a1, a2 = q1 * Vs, q2 * Vs
                else:
                    a1, a2 = q1, q2
                l, b = l + b, b
                P11, P12, P22 = P11 + 2 * P12 + P22 + a1, P12 + P22, P22 + a2
            R = np.maximum(cum[:, s], 1e-6) / d.nc[s]
            Sv = P11 + R
            K1, K2 = P11 / Sv, P12 / Sv
            e = d.L[:, s] - l
            l, b = l + K1 * e, b + K2 * e
            P11, P12, P22 = (1 - K1) * P11, (1 - K1) * P12, P22 - K2 * P12
            fc[s] = l + b                                                   # forecast of s + 1 from <= s
        self._kal[key] = fc
        return fc

    def predict(self, c, i):
        d = self.d
        cfg = self.configs[c]
        if cfg[0] == "level":
            k = cfg[1]
            lo = max(0, i - k + 1)
            w = d.nc[lo:i + 1]
            return np.log1p((d.CP[:, lo:i + 1] * w).sum(1) / w.sum())
        if i == 0:
            return d.L[:, 0].copy()
        if cfg[0] == "trendL":
            return d.L[:, i] + cfg[1] * (d.L[:, i] - d.L[:, i - 1])
        if cfg[0] == "linfit":
            w_, p = cfg[1], cfg[2]
            lo = max(0, i - w_ + 1)
            t = np.arange(lo, i + 1, dtype=float)
            val, slope = wfit(d.L[:, lo:i + 1], t, omega(d.nc[lo:i + 1]), float(i))
            return val + p * slope
        if cfg[0] == "kalman":
            return self.kalman_all(cfg[1], cfg[2])[i]
        if cfg[0] == "kalmanS":
            return self.kalman_all(cfg[1], cfg[2], scaled=True)[i]
        raise ValueError(cfg)


class SVDFactor:
    name = "svd"
    KS = (1, 2, 3, 4, 6, 8, 12, "full")
    EXTRAP = ("persist", "tr0.5", "tr1", "lin3", "lin5")
    RESID = ("zero", "last", "avg3")

    def __init__(self, d):
        self.d = d
        cfg = []
        for k in self.KS:
            for e in self.EXTRAP:
                for r in self.RESID:
                    if e == "persist" and r == "last":
                        continue                       # = persistence, pred 0
                    if k == "full" and r != "zero":
                        continue                       # residual is 0 at full rank
                    cfg.append((k, e, r))
        self.configs = cfg
        self._cache = {}

    def basis(self, i):
        if i in self._cache:
            return self._cache[i]
        d = self.d
        X = d.L[:, :i + 1]
        w = omega(d.nc[:i + 1])
        mu = (X * w).sum(1) / w.sum()
        U, s, _ = np.linalg.svd((X - mu[:, None]) * np.sqrt(w)[None, :], full_matrices=False)
        r = max(1, min(i, int((s > 1e-8 * s[0]).sum())))     # rank of the centred matrix <= i
        U = U[:, :r]
        Z = U.T @ (X - mu[:, None])
        self._cache[i] = (mu, U, Z, w)
        return self._cache[i]

    def extrap(self, Z, w, e, i):
        if i == 0 or e == "persist":
            return Z[:, -1]
        if e.startswith("tr"):
            return Z[:, -1] + float(e[2:]) * (Z[:, -1] - Z[:, -2])
        wn = int(e[3:])
        lo = max(0, Z.shape[1] - wn)
        t = np.arange(lo, Z.shape[1], dtype=float)
        val, slope = wfit(Z[:, lo:], t, w[lo:], float(Z.shape[1]))   # evaluated at t + 1
        return val

    def predict(self, c, i, oracle=False):
        d = self.d
        k, e, r = self.configs[c]
        if i == 0:
            return d.L[:, 0].copy()
        mu, U, Z, w = self.basis(i)
        kk = U.shape[1] if k == "full" else min(k, U.shape[1])
        Uk, Zk = U[:, :kk], Z[:kk]
        X = d.L[:, :i + 1]
        if oracle:
            zh = Uk.T @ (d.L[:, i + 1] - mu)
        else:
            zh = self.extrap(Zk, w, e, i)
        base = mu + Uk @ zh
        if r == "zero" or k == "full":
            return base
        Rb = (X - mu[:, None]) - Uk @ Zk
        if r == "last":
            return base + Rb[:, -1]
        lo = max(0, i - 2)
        return base + (Rb[:, lo:] * w[lo:]).sum(1) / w[lo:].sum()


class NMFMix:
    name = "nmf"
    KS = (2, 3, 4, 6, 8)
    FC = ("persist", "lin0.5", "lin1", "loglin0.5", "loglin1", "ot0.05", "ot0.5", "ot0.05h", "uot0.05t1",
          "uot0.05t0.1")
    RESID = ("zero", "last", "avg3")

    def __init__(self, d):
        self.d = d
        self.configs = [(K, f, r) for K in self.KS for f in self.FC for r in self.RESID
                        if not (f == "persist" and r == "last")]
        self._fit = {}
        self.fit_seconds = 0.0

    def fit(self, i, K):
        Ke = min(K, i + 1)
        key = (i, Ke)
        if key not in self._fit:
            t0 = time.time()
            d = self.d
            X = d.CP[:, :i + 1].T
            om = d.nc[:i + 1] * d.lib[:i + 1] / 1e4
            W, H = nmf_kl.fit_subset(X, Ke, om)
            P = nmf_kl.to_props(W)
            LH = np.log1p(H)
            C = ((LH[:, None, :] - LH[None, :, :]) ** 2).mean(2)
            if Ke > 1:
                C = C / C[~np.eye(Ke, dtype=bool)].mean()
            Rb = d.L[:, :i + 1] - np.log1p((P @ H).T)
            self._fit[key] = (P, H.astype(np.float32), C, Rb.astype(np.float32))
            self.fit_seconds += time.time() - t0
        return self._fit[key]

    @staticmethod
    def forecast_props(P, C, f):
        h1 = P[-1]
        if P.shape[0] < 2 or f == "persist":
            return h1
        h0 = P[-2]
        if f.startswith("lin"):
            h = np.maximum(h1 + float(f[3:]) * (h1 - h0), 0)
        elif f.startswith("loglin"):
            h = h1 * ((h1 + DELTA) / (h0 + DELTA)) ** float(f[6:])
        else:
            a = (h0 + DELTA) / (h0 + DELTA).sum()
            b = (h1 + DELTA) / (h1 + DELTA).sum()
            if f.startswith("uot"):
                eps, tau = f[3:].split("t")
                Pl = uot_plan(a, b, C, float(eps), float(tau))
                phi = 1.0
            else:
                half = f.endswith("h")
                Pl = uot_plan(a, b, C, float(f[2:].rstrip("h")), None)
                phi = 0.5 if half else 1.0
            Tk = Pl / a[:, None]
            push = Tk.T @ h1
            h = np.maximum(h1 + phi * (push - h1), 0)
        s = h.sum()
        return h / s if s > 0 else h1

    def predict(self, c, i, oracle=False):
        d = self.d
        K, f, r = self.configs[c]
        if i == 0:
            return d.L[:, 0].copy()
        P, H, C, Rb = self.fit(i, K)
        H = H.astype(np.float64)
        if oracle:
            h = nmf_kl.to_props(nmf_kl.project(d.CP[:, i + 1][None, :], H))[0]
        else:
            h = self.forecast_props(P, C, f)
        Lh = np.log1p(h @ H)
        if r == "zero":
            return Lh
        if r == "last":
            return d.L[:, i] + Lh - np.log1p(P[-1] @ H)
        lo = max(0, i - 2)
        w = omega(d.nc[lo:i + 1])
        return Lh + (Rb[:, lo:].astype(np.float64) * w).sum(1) / w.sum()


# ---------------------------------------------------------------- mapping to targets
LIBS = ("lp", "lh", "lt")


def lib_hat(d, i, mode):
    if i == 0 or mode == "lp":
        return d.lib[i]
    r = d.lib[i] / d.lib[i - 1]
    return d.lib[i] * (r ** 0.5 if mode == "lh" else r)


def to_target(d, Lh, i, tname, lib_mode="lp"):
    if tname == "l1":
        return Lh - d.L[:, i]
    return np.expm1(Lh) * lib_hat(d, i, lib_mode) / 1e4 - d.M[:, i]


def baselines(d):
    out = {}
    for tname in ("l1", "linear"):
        tr = np.zeros((d.T, d.N))
        for i in range(1, d.T):
            tr[i] = common.target(d, tname, i - 1)
        out[(tname, "trend")] = tr
        out[(tname, "zero")] = np.zeros((d.T, d.N))
    lt = np.zeros((d.T, d.N))
    lo = np.zeros((d.T, d.N))
    for i in range(d.T):
        r_hat = d.lib[i] / d.lib[i - 1] if i > 0 else 1.0
        lt[i] = d.M[:, i] * (r_hat - 1) if i > 0 else 0
        lo[i] = d.M[:, i] * (d.lib[i + 1] / d.lib[i] - 1)
    out[("linear", "library_term_strict")] = lt
    out[("linear", "library_term_ORACLE")] = lo
    return out


# ---------------------------------------------------------------- driver
def j_matrix(d, fam, Y):
    """J (all genes) for every config x transition, l1 and linear x lib modes. Returns dict target -> C' x T."""
    C = len(fam.configs)
    Jl1 = np.full((C, d.T), np.nan)
    Jlin = np.full((C * len(LIBS), d.T), np.nan)
    for c in range(C):
        for i in range(1, d.T):
            Lh = fam.predict(c, i)
            Jl1[c, i] = common.J_of(Y["l1"][i], to_target(d, Lh, i, "l1"))
            for li, lm in enumerate(LIBS):
                Jlin[c * len(LIBS) + li, i] = common.J_of(Y["linear"][i], to_target(d, Lh, i, "linear", lm))
    return {"l1": Jl1, "linear": Jlin}


def cfg_label(fam, cc, tname):
    if tname == "l1":
        return str(fam.configs[cc])
    return str(fam.configs[cc // len(LIBS)]) + "+" + LIBS[cc % len(LIBS)]


def pred_of(d, fam, cc, i, tname, oracle=False):
    if tname == "l1":
        return to_target(d, fam.predict(cc, i, oracle) if oracle else fam.predict(cc, i), i, "l1")
    c, li = divmod(cc, len(LIBS))
    Lh = fam.predict(c, i, oracle) if oracle else fam.predict(c, i)
    return to_target(d, Lh, i, "linear", LIBS[li])


def selected_preds(d, fam, Jm, tname, subset, fallback):
    """Rolling-origin selection restricted to config rows `subset`; fallback (T x N) where no past exists."""
    sub = np.array(subset)
    sel = common.select(Jm[sub])
    P = np.zeros((d.T, d.N))
    labels = []
    for i in range(d.T):
        lo = max(1, i - common.WINDOW)
        if i - lo < 1:
            P[i] = fallback[i]
            labels.append("fallback_trend")
        else:
            cc = int(sub[sel[i]])
            P[i] = pred_of(d, fam, cc, i, tname)
            labels.append(cfg_label(fam, cc, tname))
    return P, labels


def hindsight(d, fam, Jm, tname, subset):
    """NOT strict: the single config with the best mean J over all transitions i >= 1 (all genes)."""
    sub = np.array(subset)
    cc = int(sub[np.argmax(np.nanmean(Jm[sub, 1:], axis=1))])
    P = np.zeros((d.T, d.N))
    for i in range(1, d.T):
        P[i] = pred_of(d, fam, cc, i, tname)
    return P, cfg_label(fam, cc, tname)


def rows_for(fam, tname, pred):
    """Config rows (in the J matrix of tname) whose config satisfies pred(config)."""
    m = 1 if tname == "l1" else len(LIBS)
    return [c * m + li for c, cfg in enumerate(fam.configs) if pred(cfg) for li in range(m)]


def zs(p):
    s = p.std()
    return (p - p.mean()) / s if s > 0 else p * 0


def blend(d, Y, A, B, tname, ws=(0.0, 0.25, 0.5, 0.75, 1.0)):
    preds = {w: np.stack([(1 - w) * zs(A[i]) + w * zs(B[i]) for i in range(d.T)]) for w in ws}
    Jm = np.array([[common.J_of(Y[i], preds[w][i]) if i >= 1 else np.nan for i in range(d.T)] for w in ws])
    sel = common.select(Jm, default=0)
    P = np.stack([preds[ws[sel[i]]][i] for i in range(d.T)])
    return P, [ws[s] for s in sel]


def partial_corr(Y, A, B, mk):
    """Mean over transitions i >= 1 and draws of the partial correlation corr(y, B | A) on the test genes."""
    vals = []
    for i in range(1, Y.shape[0]):
        for k in range(mk.shape[0]):
            t = mk[k, i]
            y, a, b = Y[i, t], A[i, t], B[i, t]
            if a.std() == 0 or b.std() == 0:
                continue
            r = np.corrcoef(np.stack([y, a, b]))
            ryb, rya, rab = r[0, 2], r[0, 1], r[1, 2]
            den = np.sqrt(max(1e-12, (1 - rya ** 2) * (1 - rab ** 2)))
            vals.append((ryb - rya * rab) / den)
    return float(np.mean(vals)) if vals else None


def main(name):
    t0 = time.time()
    d = common.Data(name)
    mk = common.masks(d)
    Y = {t: common.all_targets(d, t) for t in ("l1", "linear")}
    base = baselines(d)
    res = {"dataset": name, "n_genes": d.N, "n_transitions": d.T, "cells": [int(x) for x in d.nc],
           "protocol": __doc__, "methods": {}}
    M = res["methods"]

    def add(key, tname, P, extra=None):
        sc = common.test_scores(Y[tname], P, mk)
        M.setdefault(tname, {})[key] = common.report(sc, d)
        if extra:
            M[tname][key].update(extra)
        return sc

    scores = {t: {} for t in ("l1", "linear")}
    for (tname, key), P in base.items():
        scores[tname][key] = add(key, tname, P)
    fams = {"pergene": PerGene(d), "svd": SVDFactor(d), "nmf": NMFMix(d)}
    J = {}
    for fn, fam in fams.items():
        tf = time.time()
        J[fn] = j_matrix(d, fam, Y)
        print(name, fn, len(fam.configs), "configs", round(time.time() - tf, 1), "s", flush=True)

    sub_defs = {
        "pergene": {"pergene_all": lambda c: True,
                    "pergene_level_only": lambda c: c[0] == "level",
                    "pergene_kalman": lambda c: c[0] == "kalman",
                    "pergene_kalman_scaled": lambda c: c[0] == "kalmanS"},
        "svd": {"svd_all": lambda c: True,
                "svd_lowrank_only": lambda c: c[0] != "full",
                "svd_fullrank_pergene": lambda c: c[0] == "full",
                "svd_denoise_only": lambda c: c[1] == "persist",
                "svd_factor_change_only": lambda c: c[2] == "last" and c[0] != "full"},
        "nmf": {"nmf_all": lambda c: True,
                "nmf_persist_denoise": lambda c: c[1] == "persist",
                "nmf_linear_props": lambda c: c[1].startswith(("lin", "loglin")),
                "nmf_ot_props": lambda c: c[1].startswith(("ot", "uot")),
                "nmf_ot_balanced": lambda c: c[1].startswith("ot"),
                "nmf_uot": lambda c: c[1].startswith("uot"),
                "nmf_change_only": lambda c: c[2] == "last",
                "nmf_change_only_ot": lambda c: c[2] == "last" and c[1].startswith(("ot", "uot")),
                "nmf_change_only_linear": lambda c: c[2] == "last" and c[1].startswith(("lin", "loglin"))},
    }
    selP = {t: {} for t in ("l1", "linear")}
    for tname in ("l1", "linear"):
        fb = base[(tname, "trend")]
        for fn, fam in fams.items():
            for sn, pr in sub_defs[fn].items():
                rows = rows_for(fam, tname, pr)
                P, labels = selected_preds(d, fam, J[fn][tname], tname, rows, fb)
                selP[tname][sn] = P
                scores[tname][sn] = add(sn, tname, P, {"selected": labels})
                Ph, lab = hindsight(d, fam, J[fn][tname], tname, rows)
                add(sn + "__HINDSIGHT_not_strict", tname, Ph, {"config": lab})
            print(name, tname, fn, "selected", round(time.time() - t0, 1), "s", flush=True)
        # fixed-K NMF / fixed-k SVD views (selection over the other config axes only)
        for K in NMFMix.KS:
            rows = rows_for(fams["nmf"], tname, lambda c, K=K: c[0] == K)
            P, labels = selected_preds(d, fams["nmf"], J["nmf"][tname], tname, rows, fb)
            scores[tname][f"nmf_K{K}"] = add(f"nmf_K{K}", tname, P, {"selected": labels})
        for k in SVDFactor.KS:
            rows = rows_for(fams["svd"], tname, lambda c, k=k: c[0] == k)
            P, labels = selected_preds(d, fams["svd"], J["svd"][tname], tname, rows, fb)
            scores[tname][f"svd_k{k}"] = add(f"svd_k{k}", tname, P, {"selected": labels})
        # oracles (NOT strict): true next scores / proportions on the strict basis
        for fn, fam, grid, rs in (("svd", fams["svd"], (1, 2, 3, 4, 6, 8), ("zero", "last")),
                                  ("nmf", fams["nmf"], NMFMix.KS, ("zero", "last"))):
            for g in grid:
                for r in rs:
                    probe = (g, "tr1", r) if fn == "svd" else (g, "lin1", r)
                    c = fam.configs.index(probe)
                    cc = c if tname == "l1" else c * len(LIBS)   # lib = persist
                    P = np.zeros((d.T, d.N))
                    for i in range(1, d.T):
                        P[i] = pred_of(d, fam, cc, i, tname, oracle=True)
                    key = f"ORACLE_{fn}_{'k' if fn == 'svd' else 'K'}{g}_{r}"
                    scores[tname][key] = add(key, tname, P)
        # blends with the per-gene selection
        for other in ("svd_all", "nmf_all", "nmf_ot_props", "svd_factor_change_only", "nmf_change_only"):
            P, ws = blend(d, Y[tname], selP[tname]["pergene_all"], selP[tname][other], tname)
            key = f"blend_pergene+{other}"
            scores[tname][key] = add(key, tname, P, {"weights": ws})
        # paired comparisons and partial correlations
        comp = {}
        ref = {"trend": scores[tname]["trend"], "pergene_all": scores[tname]["pergene_all"]}
        for key, sc in scores[tname].items():
            if key in ("zero",) or key.startswith("ORACLE") or key == "library_term_ORACLE":
                continue
            comp[key] = {f"vs_{rn}": common.paired(sc, rs_) for rn, rs_ in ref.items() if rn != key}
        comp["nmf_ot_props_vs_nmf_linear_props"] = common.paired(scores[tname]["nmf_ot_props"],
                                                                 scores[tname]["nmf_linear_props"])
        comp["nmf_ot_props_vs_nmf_persist_denoise"] = common.paired(scores[tname]["nmf_ot_props"],
                                                                    scores[tname]["nmf_persist_denoise"])
        comp["nmf_change_only_ot_vs_linear"] = common.paired(scores[tname]["nmf_change_only_ot"],
                                                             scores[tname]["nmf_change_only_linear"])
        comp["svd_lowrank_vs_fullrank"] = common.paired(scores[tname]["svd_lowrank_only"],
                                                        scores[tname]["svd_fullrank_pergene"])
        res.setdefault("paired", {})[tname] = comp
        res.setdefault("partial_corr_given_pergene", {})[tname] = {
            o: partial_corr(Y[tname], selP[tname]["pergene_all"], selP[tname][o], mk)
            for o in ("svd_all", "nmf_all", "nmf_ot_props", "svd_factor_change_only", "nmf_change_only")}
        res.setdefault("partial_corr_given_trend", {})[tname] = {
            o: partial_corr(Y[tname], base[(tname, "trend")], selP[tname][o], mk)
            for o in ("pergene_all", "svd_all", "nmf_all", "nmf_ot_props")}
    res["nmf_fit_seconds"] = fams["nmf"].fit_seconds
    res["seconds"] = time.time() - t0
    with open(f"{common.OUT}/forecasts_{name}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    for tname in ("l1", "linear"):
        print(f"== {name} {tname} (i>=1: pearson / up / down | no last)")
        for key, r in M[tname].items():
            a, b = r["i>=1"], r["i>=1_no_last"]
            print(f"  {key:48s} {a['pearson']:.3f} {a['p_at_200_up']:.3f} {a['p_at_200_down']:.3f} | "
                  f"{b['pearson']:.3f} {b['p_at_200_up']:.3f} {b['p_at_200_down']:.3f}")
    print("seconds", round(res["seconds"], 1))


if __name__ == "__main__":
    main(sys.argv[1])
