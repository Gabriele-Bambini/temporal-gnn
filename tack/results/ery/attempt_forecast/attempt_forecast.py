"""Improved STRICT forecasts of the erythroid lineage (E7.0 .. E8.5), built on the three diagnoses
(diag_batch, diag_direction, diag_metric). Every forecast for origin s uses stages <= s only.

DESIGN (fixed in this docstring before any final score was computed; not switched afterwards)
  Metrics  sd_fixed = debiased Sinkhorn divergence with the fixed past-cell scale S_s (PRIMARY, as pre-registered
           after the fix), energy distance (secondary), MMD (extra), all in the origin-s PCA space against the true
           stage s+1, with exactly the scoring protocol of diag_metric/rescore.json (prediction subsample sub(n) seed 0,
           target Target(Z_fut, S) seed 1, S = past_scale(Z_past)). Gene-level PCC / P@200 of the mean change
           (secondary, as pre-registered) and, as a robustness column only, energy in the 2000-HVG gene space.
  hp       Every new family has a grid whose first point nests persistence (lambda = 0, a = 0, ...), so a tie goes to
           persistence. Selection by argmin of the metric being judged on a backtest:
           PRIMARY  'roll' = rolling origin: the forecast each grid point would have made at origin s-1 (all stages
                    <= s-1, origin s-1 PCA space), scored against the true stage s. This repeats the forecast on the
                    previous transition with the deployment geometry: the scored stage was NOT used to fit the PCA,
                    exactly as stage s+1 is not used at origin s. Reason: at origins 2-4 a stage projected on a past
                    PCA keeps its gene-space variance but loses about half of its PCA-space variance (probe below),
                    which the first-run backtest (target inside the PCA fit) cannot see. Origin 2's rolling backtest
                    falls on the degenerate origin 1 (43 -> 65 cells, 105 genes): origin-2 hp are weakly determined.
           SENSITIVITY 'same' = the first-run design: predict stage s from s-1 (drift fitted on s-2 -> s-1) in the
                    origin-s space, scored against stage s.
           DESCRIPTIVE 'hind' = best grid point in hindsight on the true stage s+1 (NOT a forecast, an upper bound).
           Old first-run families keep their diag_metric Mode-B hp (re-selected by the same metric on the 'same'
           backtest) and are re-run here with the same seeds (rng [s, fi, grid_index, 99] for non-recorded hp).
  Rule     Pre-registered (decisions.md 2026-10-24): a family 'beats' comparators if strictly lower in >= 3 of the
           4 origins 2-5 AND lower on the raw mean over 2-5 (verdicts.rule). Applied literally to the first-run
           families, and post hoc (labelled) to the new families.
  Probe    (descriptive, looked at true future stages, done before this design was fixed): RMS radius in the
           origin-s PCA of stage s vs stage s+1 = 6.94/4.52, 8.03/5.96, 6.76/5.82, 6.90/5.83 (origins 2-5), while the
           gene-space radius is 15.87/15.75, 19.82/19.34, 18.32/17.46, 17.56/15.19 and the PCA share of the variance
           falls 0.191 -> 0.082, 0.164 -> 0.095, 0.136 -> 0.111, 0.154 -> 0.147.

NEW FAMILIES (tags: OT = uses an OT coupling, VEL = uses RNA velocity, LEAK = uses the atlas cell-type labels, which
were assigned by clustering all stages together, a mild leak; every other family is strict without caveat)
  ms_damp      Z + lam (m_s - m_{s-1})                                  lam in {0,.25,.5,.75,1} (1 = meanshift)
  ms_smooth    Z + lam (m_s - m_{s-2}) / 2   (noise-robust trend; lag 1 when only two stages exist)
  ms_pool      Z + lam (pool-balanced mean shift: equal weight per embryo pool >= 20 cells)       [sample-aware]
  ot_damp      Z + lam * OT barycentric drift (balanced, kernel-regressed, as the first run)      [OT]
  uot_damp     Z + lam * unbalanced-OT drift, weights = growth (first-run uot_growth with lam)    [OT]
  uot_rw       Z unmoved, weights = (UOT growth)^lam                                              [OT]
  tilt_rw      Z unmoved, weights = (density ratio p_s / p_{s-1})^lam, kNN (k = 30) classifier
  comp_km      Z unmoved, weights = p_hat(cluster) / p_s(cluster), log-linear extrapolation of the
               composition, p_hat ~ p_s^(1+lam) / p_{s-1}^lam (pseudo 0.5 cell), k-means (k = 8, past cells)
  comp_atlas   same with the atlas cell types                                                     [LEAK]
  mk_km        composition from the type-level Markov matrix of the balanced OT plan s-1 -> s:
               p_hat = p_s + lam (p_s M - p_s), k-means clusters                                  [OT]
  mk_atlas     same with atlas cell types                                                         [OT, LEAK]
  contract     Z -> c + (1 - a)(Z - c), c = (weighted) centroid                a in {0,.1,.2,.3,.4,.5}
  xfit         contract with a = 1 - rho, rho^2 = PCA-space variance of stage s in a PCA refitted on stages <= s-1
               over its variance in the origin PCA (no hp; models the projection loss of a new stage)
  vel_ss_nm    Z + c V_ss @ comps',  V = v / (1 + Ms) with the MURK genes zeroed, c in {-4..8} incl. negative [VEL]
  vel_dyn_nm   same with the scVelo dynamical velocity                                             [VEL]
  pool_eq      weights: each embryo pool (>= 20 cells) gets equal mass, mixed with cell weights by g [sample-aware]
  desample     each pool's mean offset from the stage mean is removed by a factor kappa            [sample-aware]
  combinations (contraction a in {0,.15,.3,.45}): c_trend (contract + ms_smooth), c_tilt (contract + tilt_rw),
               c_ot (contract + ot_damp) [OT], c_uot (contract + uot_rw, tau 1 beta 2 fixed a priori) [OT],
               c_vel (contract + vel_ss_nm) [VEL], c_comp (contract + comp_km), comp_pool (comp_km + pool_eq)

Usage: python -I attempt_forecast.py [--data <folder with origin*.npz, obs.csv>] --origins <s ...>   (one process per
       origin can run in parallel), then the same with --pools for origins 2-5, then attempt_report.py.
Writes only into this folder: scores_o<s>.json (every grid point on the final and on the same-space backtest, first-run
baselines and their check against rescore.json), pools_o<s>.json (selected forecasts vs each target pool), logs.
Imports diag_metric/ery_forecast_fixed.py (metric + first-run families, unchanged) without editing it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
ERY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ERY, "diag_metric"))
import ery_forecast_fixed as F  # noqa: E402

DATA = (r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN"
        r"\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\ery")
STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
DT = 0.25
MIN_POOL = 20
K_NN = 30
K_MEANS = 8
PSEUDO = 0.5
OLD = ["persist", "meanshift", "ot", "uot_growth", "vel_ss", "vel_dyn", "ot_vel_ss", "ot_vel_dyn"]


def hkey(hp):
    return json.dumps(hp, sort_keys=True)


# ---------------------------------------------------------------- problem = (space, current stage, target)

class Problem:
    """Forecast stage i_now + 1 from stages <= i_now in the PCA space of origin npz d (all arrays strict)."""

    def __init__(self, d, smp_past, i_now, S, Zt, Xt, smp_t, km, name):
        st = d["stage_past"]
        self.d, self.name, self.i_now, self.S = d, name, i_now, S
        now, prev = st == i_now, st == i_now - 1
        self.now, self.prev = now, prev
        self.Zn, self.Xn = d["Z_past"][now], d["X_past"][now].astype(np.float32)
        self.Zp, self.Xp = d["Z_past"][prev], d["X_past"][prev].astype(np.float32)
        if i_now >= 2:
            pp = st == i_now - 2
            self.Zpp, self.Xpp = d["Z_past"][pp], d["X_past"][pp].astype(np.float32)
        else:
            self.Zpp = self.Xpp = None
        self.smp_n, self.smp_p = smp_past[now], smp_past[prev]
        self.ct_n, self.ct_p = d["celltype_past"][now].astype(str), d["celltype_past"][prev].astype(str)
        self.km_n, self.km_p = km[now], km[prev]
        self.cc_p = d["cc_past"][prev]
        self.tgt = F.Target(Zt, S)
        self.jt = F.sub(len(Zt), seed=1)
        self.Xt = Xt
        self.Xt_sub = Xt[self.jt]
        self.smp_t = smp_t
        self.cache = {}

    # ---- velocity (MURK genes zeroed before the projection)
    def vel(self, kind):
        key = ("vel", kind)
        if key not in self.cache:
            V = (self.d[kind][self.now] / (1.0 + self.d["Ms"][self.now])).astype(np.float32)
            V[:, self.d["murk"].astype(bool)] = 0.0
            self.cache[key] = (V @ self.d["pca_comps"].T.astype(np.float32), V)
        return self.cache[key]

    # ---- OT couplings s-1 -> s (same solver and settings as the first run)
    def ot(self, tau=None, beta=None):
        key = ("ot", tau, beta)
        if key not in self.cache:
            a = None if tau is None else np.exp(beta * self.cc_p * DT)
            P, a = F.ot_coupling(self.Zp, self.Zn, tau=tau, a=a)
            row = P.sum(1)
            DZ = (P @ self.Zn) / row[:, None] - self.Zp
            DX = (P @ self.Xn) / row[:, None] - self.Xp
            gr = row / a
            gr = gr / gr.mean()
            dz = F.kernel_regress(self.Zp, DZ, self.Zn)
            dx = F.kernel_regress(self.Zp, DX, self.Zn).astype(np.float32)
            gw = np.maximum(F.kernel_regress(self.Zp, gr[:, None], self.Zn)[:, 0], 1e-6)
            self.cache[key] = (P, dz, dx, gw)
        return self.cache[key]

    # ---- mean shifts
    def ms(self, kind):
        key = ("ms", kind)
        if key not in self.cache:
            if kind == "lag1":
                dz, dx = self.Zn.mean(0) - self.Zp.mean(0), self.Xn.mean(0) - self.Xp.mean(0)
            elif kind == "smooth":
                if self.Zpp is None:
                    dz, dx = self.Zn.mean(0) - self.Zp.mean(0), self.Xn.mean(0) - self.Xp.mean(0)
                else:
                    dz, dx = (self.Zn.mean(0) - self.Zpp.mean(0)) / 2, (self.Xn.mean(0) - self.Xpp.mean(0)) / 2
            else:  # pool-balanced
                dz = pool_mean(self.Zn, self.smp_n) - pool_mean(self.Zp, self.smp_p)
                dx = pool_mean(self.Xn, self.smp_n) - pool_mean(self.Xp, self.smp_p)
            self.cache[key] = (dz, dx.astype(np.float32))
        return self.cache[key]

    # ---- density ratio p_s / p_{s-1} at the stage-s cells (kNN classifier, Laplace smoothing)
    def ratio(self):
        if "ratio" not in self.cache:
            Zall = np.vstack([self.Zp, self.Zn])
            lab = np.r_[np.zeros(len(self.Zp)), np.ones(len(self.Zn))]
            k = min(K_NN, len(Zall) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Zall)
            _, ind = nn.kneighbors(self.Zn)
            ind = ind[:, 1:]  # drop the query cell itself (it is in Zall)
            q = (lab[ind].sum(1) + 1) / (k + 2)
            r = q / (1 - q) * len(self.Zp) / len(self.Zn)
            self.cache["ratio"] = np.clip(r, 0.05, 20.0)
        return self.cache["ratio"]

    # ---- projection loss of a new stage: refit PCA on stages <= i_now - 1, project stage i_now
    def rho(self):
        if "rho" not in self.cache:
            st = self.d["stage_past"]
            Xo = self.d["X_past"][st <= self.i_now - 1].astype(np.float64)
            mu = Xo.mean(0)
            Xc = Xo - mu
            w, V = np.linalg.eigh(Xc.T @ Xc)
            comps = V[:, ::-1][:, :self.d["pca_comps"].shape[0]].T
            Zo = (self.Xn.astype(np.float64) - mu) @ comps.T
            r_oos = ((Zo - Zo.mean(0)) ** 2).sum(1).mean()
            Zn = self.Zn.astype(np.float64)
            r_in = ((Zn - Zn.mean(0)) ** 2).sum(1).mean()
            self.cache["rho"] = float(min(1.0, np.sqrt(r_oos / r_in)))
        return self.cache["rho"]


def pool_mean(Z, smp):
    pools = [p for p in np.unique(smp) if (smp == p).sum() >= MIN_POOL]
    if not pools:
        return Z.mean(0)
    return np.mean([Z[smp == p].mean(0) for p in pools], axis=0)


# ---------------------------------------------------------------- building blocks

def norm_w(w):
    return w / w.sum()


def comp_weights_loglin(lab_n, lab_p, lam, base_w=None):
    """weights on stage-s cells so that the composition becomes p_s^(1+lam) / p_{s-1}^lam (normalised)."""
    types = np.unique(np.r_[lab_n, lab_p])
    nn = np.array([(lab_n == t).sum() for t in types], float)
    npv = np.array([(lab_p == t).sum() for t in types], float)
    pn = (nn + PSEUDO) / (nn.sum() + PSEUDO * len(types))
    pp = (npv + PSEUDO) / (npv.sum() + PSEUDO * len(types))
    lp = np.log(pn) + lam * (np.log(pn) - np.log(pp))
    return comp_to_cells(types, nn, np.exp(lp - lp.max()), lab_n, base_w)


def comp_to_cells(types, nn, phat, lab_n, base_w=None):
    phat = np.where(nn > 0, phat, 0.0)
    phat = phat / phat.sum()
    base_w = np.full(len(lab_n), 1.0 / len(lab_n)) if base_w is None else norm_w(base_w)
    cur = np.array([base_w[lab_n == t].sum() for t in types])
    fac = np.where(cur > 0, phat / np.maximum(cur, 1e-300), 0.0)
    idx = np.searchsorted(types, lab_n)
    return norm_w(base_w * fac[idx])


def comp_weights_markov(P, lab_p, lab_n, lam):
    types = np.unique(np.r_[lab_n, lab_p])
    ip, inn = np.searchsorted(types, lab_p), np.searchsorted(types, lab_n)
    K = len(types)
    T = np.zeros((K, K))
    Pt = np.zeros((K, P.shape[1]))
    np.add.at(Pt, ip, P)
    for l in range(K):
        T[:, l] = Pt[:, inn == l].sum(1)
    M = np.eye(K)
    rs = T.sum(1)
    M[rs > 0] = T[rs > 0] / rs[rs > 0, None]
    nn = np.array([(lab_n == t).sum() for t in types], float)
    pn = nn / nn.sum()
    phat = np.maximum(pn + lam * (pn @ M - pn), 0.0)
    return comp_to_cells(types, nn, phat, lab_n)


def pool_eq_weights(smp, g):
    n = len(smp)
    usable = [p for p in np.unique(smp) if (smp == p).sum() >= MIN_POOL]
    if len(usable) < 2 or g == 0:
        return np.full(n, 1.0 / n)
    tiny = ~np.isin(smp, usable)
    t = tiny.sum() / n
    w_pe = np.where(tiny, 1.0 / n, 0.0)
    for p in usable:
        m = smp == p
        w_pe[m] = (1 - t) / len(usable) / m.sum()
    return norm_w((1 - g) / n + g * w_pe)


def contract(Z, X, w, a):
    if a == 0:
        return Z, X
    ww = np.full(len(Z), 1.0 / len(Z)) if w is None else norm_w(w)
    cz, cx = ww @ Z, (ww @ X.astype(np.float64)).astype(np.float32)
    return cz + (1 - a) * (Z - cz), cx + (1 - a) * (X - cx)


def desample(P, Z, X, kappa):
    if kappa == 0:
        return Z, X
    Z, X = Z.copy(), X.copy()
    mz, mx = P.Zn.mean(0), P.Xn.mean(0)
    for p in np.unique(P.smp_n):
        m = P.smp_n == p
        if m.sum() < MIN_POOL:
            continue
        Z[m] -= kappa * (P.Zn[m].mean(0) - mz)
        X[m] -= kappa * (P.Xn[m].mean(0) - mx)
    return Z, X


# ---------------------------------------------------------------- families: name -> (grid, fn(P, hp) -> Z, X, w, tags)

def f_ms(kind):
    def fn(P, hp):
        dz, dx = P.ms(kind)
        return P.Zn + hp["lam"] * dz, P.Xn + hp["lam"] * dx, None
    return fn


def f_ot_damp(P, hp):
    if hp["lam"] == 0:
        return P.Zn, P.Xn, None
    _, dz, dx, _ = P.ot()
    return P.Zn + hp["lam"] * dz, P.Xn + hp["lam"] * dx, None


def f_uot_damp(P, hp):
    if hp["lam"] == 0:
        return P.Zn, P.Xn, None
    _, dz, dx, gw = P.ot(hp["tau"], hp["beta"])
    return P.Zn + hp["lam"] * dz, P.Xn + hp["lam"] * dx, gw


def f_uot_rw(P, hp):
    if hp["lam"] == 0:
        return P.Zn, P.Xn, None
    _, _, _, gw = P.ot(hp["tau"], hp["beta"])
    return P.Zn, P.Xn, gw ** hp["lam"]


def f_tilt(P, hp):
    if hp["lam"] == 0:
        return P.Zn, P.Xn, None
    return P.Zn, P.Xn, P.ratio() ** hp["lam"]


def f_comp(lab):
    def fn(P, hp):
        if hp["lam"] == 0:
            return P.Zn, P.Xn, None
        ln, lp = (P.km_n, P.km_p) if lab == "km" else (P.ct_n, P.ct_p)
        return P.Zn, P.Xn, comp_weights_loglin(ln, lp, hp["lam"])
    return fn


def f_mk(lab):
    def fn(P, hp):
        if hp["lam"] == 0:
            return P.Zn, P.Xn, None
        Pm = P.ot()[0]
        ln, lp = (P.km_n, P.km_p) if lab == "km" else (P.ct_n, P.ct_p)
        return P.Zn, P.Xn, comp_weights_markov(Pm, lp, ln, hp["lam"])
    return fn


def f_contract(P, hp):
    Z, X = contract(P.Zn, P.Xn, None, hp["a"])
    return Z, X, None


def f_xfit(P, hp):
    Z, X = contract(P.Zn, P.Xn, None, 1.0 - P.rho())
    return Z, X, None


def f_vel(kind):
    def fn(P, hp):
        if hp["c"] == 0:
            return P.Zn, P.Xn, None
        vz, vx = P.vel(kind)
        return P.Zn + hp["c"] * vz, P.Xn + hp["c"] * vx, None
    return fn


def f_pool_eq(P, hp):
    return P.Zn, P.Xn, (None if hp["g"] == 0 else pool_eq_weights(P.smp_n, hp["g"]))


def f_desample(P, hp):
    Z, X = desample(P, P.Zn, P.Xn, hp["kappa"])
    return Z, X, None


def combo(inner, inner_key):
    """contraction a applied after the inner family (around the weighted centroid of the inner forecast)."""
    def fn(P, hp):
        Z, X, w = inner(P, {k: v for k, v in hp.items() if k != "a"})
        Z, X = contract(Z, X, w, hp["a"])
        return Z, X, w
    return fn


def f_comp_pool(P, hp):
    base = pool_eq_weights(P.smp_n, hp["g"])
    if hp["lam"] == 0:
        return P.Zn, P.Xn, (None if hp["g"] == 0 else base)
    return P.Zn, P.Xn, comp_weights_loglin(P.km_n, P.km_p, hp["lam"], base_w=base)


LAM5 = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMW = [0.0, 0.5, 1.0, 1.5, 2.0]
A4 = [0.0, 0.15, 0.3, 0.45]
FAMILIES = {
    "ms_damp": ([{"lam": x} for x in LAM5], f_ms("lag1"), set()),
    "ms_smooth": ([{"lam": x} for x in LAM5], f_ms("smooth"), set()),
    "ms_pool": ([{"lam": x} for x in LAM5], f_ms("pool"), {"sample"}),
    "ot_damp": ([{"lam": x} for x in LAM5], f_ot_damp, {"OT"}),
    "uot_damp": ([{"lam": 0.0, "tau": 1.0, "beta": 0.0}] +
                 [{"lam": x, "tau": 1.0, "beta": b} for b in (0.0, 2.0) for x in (0.25, 0.5, 0.75, 1.0)],
                 f_uot_damp, {"OT"}),
    "uot_rw": ([{"lam": 0.0, "tau": 1.0, "beta": 0.0}] +
               [{"lam": x, "tau": t, "beta": b} for t in (0.1, 1.0) for b in (0.0, 2.0) for x in (0.5, 1.0, 2.0)],
               f_uot_rw, {"OT"}),
    "tilt_rw": ([{"lam": x} for x in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)], f_tilt, set()),
    "comp_km": ([{"lam": x} for x in LAMW], f_comp("km"), set()),
    "comp_atlas": ([{"lam": x} for x in LAMW], f_comp("atlas"), {"LEAK"}),
    "mk_km": ([{"lam": x} for x in LAMW], f_mk("km"), {"OT"}),
    "mk_atlas": ([{"lam": x} for x in LAMW], f_mk("atlas"), {"OT", "LEAK"}),
    "contract": ([{"a": x} for x in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)], f_contract, set()),
    "xfit": ([{}], f_xfit, set()),
    "vel_ss_nm": ([{"c": x} for x in (0.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0)], f_vel("v_ss"), {"VEL"}),
    "vel_dyn_nm": ([{"c": x} for x in (0.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0)], f_vel("v_dyn"),
                   {"VEL"}),
    "pool_eq": ([{"g": x} for x in (0.0, 0.5, 1.0)], f_pool_eq, {"sample"}),
    "desample": ([{"kappa": x} for x in (0.0, 0.25, 0.5, 0.75, 1.0)], f_desample, {"sample"}),
    "c_trend": ([{"a": a, "lam": x} for x in (0.0, 0.25, 0.5) for a in A4], combo(f_ms("smooth"), "lam"), set()),
    "c_tilt": ([{"a": a, "lam": x} for x in (0.0, 0.5, 1.0) for a in A4], combo(f_tilt, "lam"), set()),
    "c_ot": ([{"a": a, "lam": x} for x in (0.0, 0.25, 0.5) for a in A4], combo(f_ot_damp, "lam"), {"OT"}),
    "c_uot": ([{"a": a, "lam": x, "tau": 1.0, "beta": 2.0} for x in (0.0, 1.0, 2.0) for a in A4],
              combo(f_uot_rw, "lam"), {"OT"}),
    "c_vel": ([{"a": a, "c": x} for x in (0.0, -1.0, 1.0, 2.0) for a in A4], combo(f_vel("v_ss"), "c"), {"VEL"}),
    "c_comp": ([{"a": a, "lam": x} for x in (0.0, 0.5, 1.0, 1.5) for a in A4], combo(f_comp("km"), "lam"), set()),
    "comp_pool": ([{"lam": x, "g": g} for g in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0, 1.5)], f_comp_pool,
                  {"sample"}),
}


# ---------------------------------------------------------------- scoring (diag_metric protocol, plus extras)

def ot_full(A_, B_, wa, wb, eps, sym=False):
    """(transport cost <P,C>, full entropic value <P,C> + eps KL(P | a x b)) of the converged entropic plan."""
    C = F.sqdist(A_, B_)
    P = F.sinkhorn_sym(wa, C, eps) if sym else F.sinkhorn(wa, wb, C, eps)
    ab = wa[:, None] * wb[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(P > 0, P * np.log(P / ab), 0.0).sum() - P.sum() + ab.sum()
    return float((P * C).sum()), float((P * C).sum() + eps * kl)


def sd_full(X, wx, tgt):
    """textbook Sinkhorn divergence (full entropic values in all three terms), same eps and scale as sd_fixed.
    Added after the grid run as a CHECK (metric_check.py: sd_fixed falls when a forecast is merely contracted)."""
    eps = 0.05 * tgt.S
    if not hasattr(tgt, "yy_full"):
        tgt.yy_full = ot_full(tgt.Y, tgt.Y, tgt.wy, tgt.wy, eps, sym=True)[1]
    xy = ot_full(X, tgt.Y, wx, tgt.wy, eps)[1]
    xx = ot_full(X, X, wx, wx, eps, sym=True)[1]
    return (xy - xx / 2 - tgt.yy_full / 2) / tgt.S


def score(Zf, Xf, w, P, gene=True, tgt=None, Xt_sub=None, full=False):
    tgt = P.tgt if tgt is None else tgt
    i = F.sub(len(Zf))
    if w is not None and (w[i] <= 0).any():
        # zero-mass cells change no distance; dropping them avoids the slow log-domain fallback (added after the
        # grid run, which kept them: same values, only slower; used by the pool breakdown)
        i = i[w[i] > 0]
    wi = None if w is None else w[i]
    X = Zf[i]
    wx = F.unif(len(X)) if wi is None else wi / wi.sum()
    out = {"energy": F.energy(X, tgt.Y, wi),
           "sd_fixed": float(F.sinkhorn_div_fixed(X, tgt.Y, wx, tgt.wy, tgt.S, tgt.yy_ot)),
           "mmd": F.mmd2(X, tgt.Y, wx, tgt.wy, tgt.S, tgt.yy_mmd),
           "spread": float(np.sqrt(((X - wx @ X) ** 2).sum(1) @ wx)),
           "ess": float(1.0 / (wx ** 2).sum())}
    if full:
        out["sd_full"] = float(sd_full(X, wx, tgt))
    if gene:
        Xs = Xf[i].astype(np.float32)
        out["energy_gene"] = F.energy(Xs, (P.Xt_sub if Xt_sub is None else Xt_sub).astype(np.float32), wi)
        mw = np.full(len(Xf), 1 / len(Xf)) if w is None else w / w.sum()
        pred = mw @ Xf.astype(np.float64) - P.Xn.astype(np.float64).mean(0)
        true = P.Xt.astype(np.float64).mean(0) - P.Xn.astype(np.float64).mean(0)
        # a zero predicted change (persistence, pure contraction) has no correlation: the first run's persist PCC
        # was computed on float32 rounding noise of mean(X) - mean(X)
        if np.linalg.norm(pred) <= 1e-4 * np.linalg.norm(true):
            out.update(pearson=float("nan"), p200_up=float("nan"), p200_down=float("nan"))
        else:
            out["pearson"] = float(np.corrcoef(pred, true)[0, 1])
            out["p200_up"] = len(set(np.argsort(-true)[:200]) & set(np.argsort(-pred)[:200])) / 200
            out["p200_down"] = len(set(np.argsort(true)[:200]) & set(np.argsort(pred)[:200])) / 200
    return out


# ---------------------------------------------------------------- main

def origin_context(data, s, obs, codes):
    d = F.load(data, s)
    a, b = np.where(codes <= s)[0], np.where(codes == s + 1)[0]
    assert (d["celltype_past"].astype(str) == obs["celltype"].to_numpy()[a]).all()
    assert (d["celltype_fut"].astype(str) == obs["celltype"].to_numpy()[b]).all()
    smp_past, smp_fut = obs["sample"].to_numpy()[a], obs["sample"].to_numpy()[b]
    km = KMeans(n_clusters=K_MEANS, n_init=10, random_state=0).fit_predict(d["Z_past"].astype(np.float64))
    km = np.array([f"k{x}" for x in km])
    S = F.past_scale(d["Z_past"])
    return d, smp_past, smp_fut, km, S


def run_family_grid(P, out_p, gene):
    for fam, (grid, fn, tags) in FAMILIES.items():
        res = {}
        for hp in grid:
            t0 = time.time()
            Zf, Xf, w = fn(P, hp)
            res[hkey(hp)] = {**score(Zf, Xf, w, P, gene=gene), "sec": round(time.time() - t0, 2)}
        out_p[fam] = res


def load_obs(data):
    obs = pd.read_csv(os.path.join(data, "obs.csv"))
    obs["sample"] = obs["sample"].astype(str)
    obs["celltype"] = obs["celltype"].astype(str)
    codes = np.array([STAGES.index(x) for x in obs["stage"].astype(str)])
    return obs, codes


def build_final(data, s, obs, codes):
    d, smp_past, smp_fut, km, S = origin_context(data, s, obs, codes)
    Pf = Problem(d, smp_past, s, S, d["Z_fut"], d["X_fut"].astype(np.float32), smp_fut, km, f"final{s}")
    return d, smp_past, smp_fut, km, S, Pf


def run_grid(args, s):
    """All grid points of all new families on the final problem of origin s (+ same-space backtest + baselines)."""
    obs, codes = load_obs(args.data)
    rescore = json.load(open(os.path.join(ERY, "diag_metric", "rescore.json")))
    t0 = time.time()
    F._DRIFT.clear()
    d, smp_past, smp_fut, km, S, Pf = build_final(args.data, s, obs, codes)
    if str(s) in rescore["origins"]:
        assert abs(S - rescore["origins"][str(s)]["S_fixed"]) < 1e-9, (S, rescore["origins"][str(s)]["S_fixed"])
    st = d["stage_past"]
    out = {"origin": s, "meta": {"target": STAGES[s + 1], "now": STAGES[s], "n_now": int(len(Pf.Zn)),
                                 "n_target": int(len(Pf.Xt)), "S": S, "rho_xfit": Pf.rho(),
                                 "km_sizes_now": {k: int((Pf.km_n == k).sum()) for k in np.unique(km)},
                                 "pools_now": {p: int((Pf.smp_n == p).sum()) for p in np.unique(Pf.smp_n)},
                                 "pools_target": {p: int((smp_fut == p).sum()) for p in np.unique(smp_fut)}},
           "families": {f: {"grid": g, "tags": sorted(t)} for f, (g, _, t) in FAMILIES.items()},
           "final": {}, "same": {}, "baselines": {}, "check_vs_rescore": {}}
    run_family_grid(Pf, out["final"], gene=True)
    print(f"origin {s} final grid {time.time() - t0:.0f}s rho={Pf.rho():.3f}", flush=True)
    if s >= 2:
        Pb = Problem(d, smp_past, s - 1, S, d["Z_past"][st == s], d["X_past"][st == s].astype(np.float32),
                     smp_past[st == s], km, f"same{s}")
        out["meta"]["rho_xfit_same"] = Pb.rho()
        run_family_grid(Pb, out["same"], gene=False)
        print(f"origin {s} same-space backtest {time.time() - t0:.0f}s", flush=True)
        rr = rescore["origins"][str(s)]["families"]
        for fi, fam in enumerate(OLD):
            out["baselines"][fam], out["check_vs_rescore"][fam] = {}, {}
            for m in ("sd_fixed", "energy"):
                hp = rr[fam]["sel"][m]
                k = hkey(hp)
                if k in out["baselines"][fam]:
                    continue
                rng = np.random.default_rng([s, fi, F.grid_index(fam, hp), 99])
                Zf, Xf, w = F.forecast(fam, hp, d, s, (s, s - 1), rng=rng)
                r = score(Zf, Xf.astype(np.float32), w, Pf, gene=True)
                r["hp"] = hp
                r["noise_draw_differs_from_rescore"] = bool(hp.get("noise", 0) > 0 and
                                                            k == hkey(rr[fam]["sel"]["recorded"]))
                out["baselines"][fam][k] = r
                ref = rr[fam]["final"][k]
                chk = {mm: abs(r[mm] - ref[mm]) for mm in ("energy", "sd_fixed", "mmd")}
                chk["noise_draw_differs"] = r["noise_draw_differs_from_rescore"]
                out["check_vs_rescore"][fam][k] = chk
        mx = max(v for f in out["check_vs_rescore"].values() for h in f.values()
                 if not h["noise_draw_differs"] for kk, v in h.items() if kk != "noise_draw_differs")
        out["meta"]["max_abs_diff_vs_rescore_same_draws"] = mx
        print(f"origin {s} baselines {time.time() - t0:.0f}s max|diff| vs rescore (same draws) {mx:.2e}", flush=True)
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    out["meta"]["solver"] = dict(F.STATS)
    with open(os.path.join(HERE, f"scores_o{s}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


def load_scores(s):
    p = os.path.join(HERE, f"scores_o{s}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def selected_hps(fam, s, m, cur, prev):
    """hp chosen for origin s under metric m: 'roll' (primary), 'same' (first-run design), 'hind' (descriptive)."""
    grid = FAMILIES[fam][0]
    out = {}
    if prev is not None:
        out["roll"] = grid[int(np.argmin([prev["final"][fam][hkey(g)][m] for g in grid]))]
    if cur["same"]:
        out["same"] = grid[int(np.argmin([cur["same"][fam][hkey(g)][m] for g in grid]))]
    out["hind"] = grid[int(np.argmin([cur["final"][fam][hkey(g)][m] for g in grid]))]
    return out


def run_pools(args, s):
    """Evaluation breakdown (never used for selection): every selected forecast (roll / same / hind hp, by sd_fixed
    and by energy) scored against each target pool, and re-scored on the pooled target with the textbook Sinkhorn
    divergence sd_full as a check column."""
    obs, codes = load_obs(args.data)
    cur, prev = load_scores(s), load_scores(s - 1)
    t0 = time.time()
    F._DRIFT.clear()
    d, smp_past, smp_fut, km, S, Pf = build_final(args.data, s, obs, codes)
    tg = {}
    for p in np.unique(smp_fut):
        m = np.where(smp_fut == p)[0]
        if len(m) >= 100:
            tg[p] = F.Target(d["Z_fut"][m], S)
    out = {"origin": s, "target": STAGES[s + 1], "pools": {p: int((smp_fut == p).sum()) for p in tg},
           "batch": {p: str(obs.loc[obs["sample"] == p, "sequencing.batch"].iloc[0]) for p in tg},
           "scores": {}, "pooled_full": {}}
    for fam, (grid, fn, _) in FAMILIES.items():
        hps = {}
        for m in ("sd_fixed", "energy"):
            for des, hp in selected_hps(fam, s, m, cur, prev).items():
                hps[hkey(hp)] = hp
        out["scores"][fam], out["pooled_full"][fam] = {}, {}
        for k, hp in hps.items():
            Zf, Xf, w = fn(Pf, hp)
            out["scores"][fam][k] = {p: score(Zf, Xf, w, Pf, gene=False, tgt=t) for p, t in tg.items()}
            out["pooled_full"][fam][k] = score(Zf, Xf, w, Pf, gene=False, full=True)
    for fi, fam in enumerate(OLD):
        out["scores"]["OLD_" + fam], out["pooled_full"]["OLD_" + fam] = {}, {}
        for k, r in cur["baselines"][fam].items():
            hp = r["hp"]
            rng = np.random.default_rng([s, fi, F.grid_index(fam, hp), 99])
            Zf, Xf, w = F.forecast(fam, hp, d, s, (s, s - 1), rng=rng)
            out["scores"]["OLD_" + fam][k] = {p: score(Zf, Xf, w, Pf, gene=False, tgt=t)
                                              for p, t in tg.items()}
            out["pooled_full"]["OLD_" + fam][k] = score(Zf, Xf, w, Pf, gene=False, full=True)
    out["seconds"] = round(time.time() - t0, 1)
    print(f"origin {s} pools {out['pools']} {out['seconds']}s", flush=True)
    with open(os.path.join(HERE, f"pools_o{s}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--origins", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--pools", action="store_true", help="per-target-pool breakdown (needs scores_o<s-1>, o<s>)")
    a = ap.parse_args()
    for s in a.origins:
        run_pools(a, s) if a.pools else run_grid(a, s)


