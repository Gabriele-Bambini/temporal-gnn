"""INTERPOLATION of a held-out REAL embryonic stage (erythroid lineage): predict stage k from stages k-1 and k+1.

The classic OT / Schroedinger-bridge benchmark, on scVelo gastrulation_erythroid with real stages 0.25 days apart.
Input: interp<k>.npz, k = 2..5 (tack/hpc/ery_prep_interp.py): gene filter, normalisation, 2000 HVG, PCA 30, kNN
moments and velocities were fitted on stages k-1 and k+1 ONLY; the held stage k was only transformed. Every
prediction below uses only the fit cells (stages k-1, k+1) and their metadata; the held stage is used only to score.
Time: stages are equally spaced, so the held stage sits at t = 0.5 of the k-1 -> k+1 interval (0.5 days).

Methods (all in the interp file's 30-d PCA space; gene vectors follow the same construction):
  prev          stage k-1 cells unchanged                                   (baseline)
  next          stage k+1 cells unchanged                                   (baseline)
  union         50/50 mixture of the two stages (each stage half the mass)  (baseline)
  indep         midpoint (1-t) x + t y with x, y drawn INDEPENDENTLY       (control: OT pairing replaced by none)
  ot_pairs      entropic OT plan P between k-1 and k+1 (own Sinkhorn, cost C / median(C), eps_rel 0.05 as in
                ery_forecast.ot_coupling); McCann displacement interpolation by drawing n pairs (i, j) ~ P and
                placing (1-t) x_i + t y_j                                    (PRIMARY OT interpolant)
  ot_bary       barycentric map x + t (T(x) - x), T(x_i) = sum_j P_ij y_j / r_i, one point per k-1 cell (labelled:
                the entropic blur shrinks it)
  ot_bary_sym   same from both ends (k-1 cells forward, k+1 cells backward with the transposed map), 50/50
  sb            Schroedinger bridge: ot_pairs + Brownian-bridge noise N(0, t (1-t) eps I), eps = eps_rel * median(C)
                in absolute squared-distance units (variance exactly as specified by the task)
  sb_half       same with variance t (1-t) eps / 2: the reference consistent with cost |x - y|^2 (P ~ exp(-C/eps)
                is a Gaussian kernel of variance eps / 2 per dimension)
  uot           KL-unbalanced OT (tau 1.0, beta 2.0 = ery_forecast DEFAULT) with source mass ~ exp(beta * prolif *
                0.5 days); pairs drawn with time-t mass P_ij (u_i / r_i)^(1-t) (u_i = observed mass at k-1, r_i =
                plan row mass: the descendants of i grow geometrically along the path)
  vel_ss/vel_dyn  RNA velocity (own steady-state / scVelo dynamical), log scale V = v / (1 + Ms), projected on the
                PCA; k-1 cells pushed by (c / 2) V; c fitted STRICTLY on the k-1 -> k+1 pair (argmin over C_GRID of the
                distance between x_{k-1} + c V and the k+1 cells; c = 0 is in the grid, so velocity can collapse to
                prev). Fitted under sd_fixed (primary row) and under ED_U (reported).
  vel_*_back    (extra) k+1 cells pulled back by (c' / 2) V, c' fitted on k+1 -> k-1.
Proliferation score: Tirosh 2016 S + G2M genes (the list scVelo ships) among the interp file's HVGs, mean centred log
expression (centring on the fit cells) - strict. Gene list checked: it reproduces origin2.npz's cc score exactly
(there every kept gene is an HVG); against origin5's all-gene score the HVG-only score correlates 0.86-0.94.

Metrics against the held stage (subsampled to <= 1500 cells; predictions to n_pred = min(n_held, 1500)):
  sd_fixed  PRIMARY. Debiased Sinkhorn [OT(X,Y) - OT(X,X)/2 - OT(Y,Y)/2] / S, eps = 0.05 S, S = median pairwise
            squared distance among FIT cells (seeded subsample <= 1500): fixed per k, never depends on a prediction.
  ED_U      energy distance with the zero self-pairs excluded (U-statistic; weighted: within / (1 - sum w^2)).
            This is the one compared with the sample-level floors (diag_batch floors.csv, loso_ED_U / oracle_ED_U).
  ED_V      ery_forecast.energy (V-statistic), for continuity with forecast.json; comparable with loso_ED_V_fs only
            when the prediction really has n_pred points.
  ED_U_gene energy (U) in the 2000-HVG log space (robustness).
  gene      PCC / P@200 up / down of the mean change from k-1 (per gene). NOTE: for every balanced method (union,
            next, indep, ot_*, sb*) the predicted mean change is the same vector up to scale (by construction), so
            only uot and vel_* can differ; dev_pcc = PCC of the deviation of the mean from the linear midpoint
            (defined only when the prediction deviates at all).
  W1 / W2   exact Wasserstein-1 / -2 in PCA space by linear assignment (prediction resampled systematically to
            n_true equal points), the metric of the classic interpolation papers. ADDED AFTER a k = 3 timing run
            (1 rep) showed sd_fixed and ED_U ranking contracted vs spread clouds oppositely; not part of the decision
            rule below, reported next to it with the same rule as a tie-breaker. Biased by sample size (not debiased).
  comp_tv   (descriptive) total-variation distance of cell-type composition, both sides labelled by one 15-NN
            classifier fitted on the fit cells.
  per pool  ED_U against each held embryo pool (sample, >= 20 cells) separately, next to that pool's leave-one-
            sample-out reference ED_U(other pools of the held stage, pool) = how close a real replicate pool is.
Floors (DESCRIPTIVE, they use the held stage): diag_batch floors.csv (interp rows) + own sd_fixed / ED_U halves and
leave-one-sample-out.
Extras: (a) oracle-labelled sensitivity grid (eps_rel 0.01 / 0.2; uot tau 0.1 / 1.0 x beta 0 / 2 / 5): chosen by
looking at the held stage, so never used for a verdict; (b) batch-matched block: the same methods built only from
sequencing-batch-2 cells of k-1 and k+1, scored against the batch-2 cells of the held stage, next to the all-batch
methods scored against the same target (does batch mismatch limit the interpolation?).

DECISION RULE (builder-set before any result was seen; NOT a user pre-registration - the 2026-10-24 decisions row
covers the forecasting benchmark only). Primary metric sd_fixed, secondary ED_U, primary rows (DEFAULT
hyperparameters), mean over reps, held stages k = 2..5:
  R1 'OT interpolation helps'  ot_pairs beats ALL of prev, next, union in >= 3 of 4 held stages AND on the mean over k
                               (same test reported for sb, sb_half, uot, ot_bary, ot_bary_sym)
  R2 'the OT pairing matters'  ot_pairs beats indep by the same rule
  R3 'bridge noise helps'      sb (and sb_half) beats ot_pairs by the same rule
  R4 'growth helps'            uot beats ot_pairs by the same rule
  R5 'velocity helps'          vel_ss / vel_dyn beats BOTH prev and union by the same rule (also reported vs ot_pairs)
  Pool consistency             for each k: does the method beat each baseline in EVERY usable held pool (ED_U)?
k = 2 is weak (the k-1 side is 65 cells of E7.25); it is reported and counted, and flagged.

Python only (numpy, pandas, scikit-learn), no torch. Reads --data, writes only into --out.
Usage: python -I interp_bench.py --data <folder with interp*.npz, origin5.npz, obs.csv> --out <this folder>
       [--ks 2 3 4 5] [--reps 3] [--floors <diag_batch/floors.csv>]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier

STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
T = 0.5           # held stage position in the k-1 -> k+1 interval
DT_INT = 0.5      # days between k-1 and k+1
NSUB = 1500
MIN_POOL = 20
DEFAULT = {"eps_rel": 0.05, "tau": 1.0, "beta": 2.0}
C_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
UNDERFLOW = 600.0
S_GENES = ("Mcm5 Pcna Tyms Fen1 Mcm2 Mcm4 Rrm1 Ung Gins2 Mcm6 Cdca7 Dtl Prim1 Uhrf1 Mlf1ip Hells Rfc2 Rpa2 Nasp "
           "Rad51ap1 Gmnn Wdr76 Slbp Ccne2 Ubr7 Pold3 Msh2 Atad2 Rad51 Rrm2 Cdc45 Cdc6 Exo1 Tipin Dscc1 Blm Casp8ap2 "
           "Usp1 Clspn Pola1 Chaf1b Brip1 E2f8").split()
G2M_GENES = ("Hmgb2 Cdk1 Nusap1 Ube2c Birc5 Tpx2 Top2a Ndc80 Cks2 Nuf2 Cks1b Mki67 Tmpo Cenpf Tacc3 Fam64a Smc4 Ccnb2 "
             "Ckap2l Ckap2 Aurkb Bub1 Kif11 Anp32e Tubb4b Gtse1 Kif20b Hjurp Cdca3 Hn1 Cdc20 Ttk Cdc25c Kif2c Rangap1 "
             "Ncapd2 Dlgap5 Cdca2 Cdca8 Ect2 Kif23 Hmmr Aurka Psrc1 Anln Lbr Ckap5 Cenpe Ctcf Nek2 G2e3 Gas2l3 Cbx5 "
             "Cenpa").split()
STATS = {"kernel_calls": 0, "log_fallbacks": 0, "sym_calls": 0, "sym_unconverged": 0, "max_marg_err": 0.0}
LOGF = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOGF:
        with open(LOGF, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


# ---------------- OT (copied from diag_metric/ery_forecast_fixed.py: same updates and stopping rule) ----------------

def lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def sinkhorn_log(a, b, C, eps, tau=None, iters=2000, tol=1e-7):
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
    """Kernel-domain Sinkhorn (u = e^{f/eps}), log-domain fallback on underflow; tau = None balanced, else KL."""
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
    STATS["sym_calls"] += 1
    K = np.exp(-np.asarray(C, np.float64) / eps)
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
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def unif(n):
    return np.full(n, 1.0 / n)


def sub(n, k=NSUB, seed=0):
    r = np.random.default_rng(seed)
    return np.sort(r.choice(n, min(n, k), replace=False))


def med_scale(Z, seed=7):
    Zs = Z[sub(len(Z), seed=seed)]
    C = sqdist(Zs, Zs)
    return float(np.median(C[np.triu_indices(len(Zs), 1)]))


# ---------------- metrics ----------------

def ot_cost(A, B, wa, wb, eps):
    C = sqdist(A, B)
    P = sinkhorn(wa, wb, C, eps)
    return float((P * C).sum())


def ot_self(A, wa, eps):
    C = sqdist(A, A)
    return float((sinkhorn_sym(wa, C, eps) * C).sum())


def self_dist(A):
    D = np.sqrt(sqdist(A, A))
    np.fill_diagonal(D, 0.0)
    return D


class Target:
    """A scoring target (held stage subsample or one held pool) with cached self terms."""

    def __init__(self, Z, X, S, with_sd=True):
        self.Z, self.X = np.asarray(Z, np.float64), None if X is None else np.asarray(X, np.float64)
        self.w = unif(len(Z))
        self.S = S
        self.sw2 = float((self.w ** 2).sum())
        self.wdw = float(self.w @ self_dist(self.Z) @ self.w)
        self.wdw_g = None if X is None else float(self.w @ self_dist(self.X) @ self.w)
        self.yy = ot_self(self.Z, self.w, 0.05 * S) if with_sd else None


def ed_uv(P, wp, tgt, gene=False):
    """(ED_U, ED_V) of weighted points P against the target (PCA or gene space)."""
    Y = tgt.X if gene else tgt.Z
    wdw_t = tgt.wdw_g if gene else tgt.wdw
    P = np.asarray(P, np.float64)
    cross = float(wp @ np.sqrt(sqdist(P, Y)) @ tgt.w)
    xx = float(wp @ self_dist(P) @ wp)
    sx = float((wp ** 2).sum())
    ed_v = 2 * cross - xx - wdw_t
    ed_u = 2 * cross - xx / (1 - sx) - wdw_t / (1 - tgt.sw2)
    return ed_u, ed_v


def sd_fixed(P, wp, tgt):
    eps = 0.05 * tgt.S
    return (ot_cost(P, tgt.Z, wp, tgt.w, eps) - 0.5 * ot_self(np.asarray(P, np.float64), wp, eps) - 0.5 * tgt.yy) / tgt.S


def systematic(w, n):
    cs = np.cumsum(w / w.sum())
    return np.minimum(np.searchsorted(cs, (np.arange(n) + 0.5) / n), len(w) - 1)


def w_exact(P, wp, Y):
    """Exact W1 and W2 (linear assignment) after systematic resampling of the prediction to len(Y) equal points."""
    n = len(Y)
    if not (len(P) == n and np.allclose(wp, 1.0 / n)):
        P = np.asarray(P)[systematic(wp, n)]
    C = sqdist(P, Y)
    r, c = linear_sum_assignment(C)
    w2 = float(np.sqrt(C[r, c].mean()))
    D = np.sqrt(C)
    r, c = linear_sum_assignment(D)
    return float(D[r, c].mean()), w2


def basic_scores(P, wp, tgt):
    eu, ev = ed_uv(P, wp, tgt)
    w1, w2 = w_exact(P, wp, tgt.Z)
    return {"sd_fixed": sd_fixed(P, wp, tgt), "ED_U": eu, "ED_V": ev, "W1": w1, "W2": w2}


def gene_scores(gmean, ref_prev, ref_mid, mu_true):
    out = {}
    pred, true = gmean - ref_prev, mu_true - ref_prev
    if np.linalg.norm(pred) < 1e-9:
        out.update(pcc=np.nan, p200_up=np.nan, p200_down=np.nan)
    else:
        out["pcc"] = float(np.corrcoef(pred, true)[0, 1])
        out["p200_up"] = len(set(np.argsort(-true)[:200]) & set(np.argsort(-pred)[:200])) / 200
        out["p200_down"] = len(set(np.argsort(true)[:200]) & set(np.argsort(pred)[:200])) / 200
    dp, dt = gmean - ref_mid, mu_true - ref_mid
    out["dev_norm_ratio"] = float(np.linalg.norm(dp) / np.linalg.norm(dt))
    out["dev_pcc"] = float(np.corrcoef(dp, dt)[0, 1]) if np.linalg.norm(dp) > 1e-3 * np.linalg.norm(dt) else np.nan
    return out


# ---------------- interpolants ----------------

def plan(ZA, ZB, eps_rel, tau=None, a=None):
    C = sqdist(ZA, ZB)
    med = float(np.median(C))
    a = unif(len(ZA)) if a is None else a / a.sum()
    b = unif(len(ZB))
    P = sinkhorn(a, b, C / med, eps_rel, tau)
    if tau is None:
        err = max(np.abs(P.sum(1) / a - 1).max(), np.abs(P.sum(0) / b - 1).max())
        STATS["max_marg_err"] = max(STATS["max_marg_err"], float(err))
    return P, med


def draw_pairs(W, n, rng):
    p = W.ravel() / W.sum()
    flat = rng.choice(p.size, size=n, replace=True, p=p)
    return np.unravel_index(flat, W.shape)


def two_sided(PA, wA, PB, wB, n, rng):
    """50/50 mixture of two weighted point sets at size ~n (each side subsampled to <= n/2)."""
    ZA, XA = PA
    ZB, XB = PB
    mA = min(len(ZA), n // 2)
    mB = min(len(ZB), n - mA)
    iA = np.sort(rng.choice(len(ZA), mA, replace=False))
    iB = np.sort(rng.choice(len(ZB), mB, replace=False))
    wa, wb = wA[iA] / wA[iA].sum() * 0.5, wB[iB] / wB[iB].sum() * 0.5
    return (np.vstack([ZA[iA], ZB[iB]]), np.vstack([XA[iA], XB[iB]]), np.concatenate([wa, wb]))


def methods(ZA, XA, ZB, XB, ccA, comps, n, rng, eps_rel=DEFAULT["eps_rel"], tau=DEFAULT["tau"],
            beta=DEFAULT["beta"], which=None, VA=None, VB=None, cfit=None):
    """Return {name: (Z, X, w, gmean)}; X are gene vectors, gmean the exact expected gene mean of the method."""
    out = {}
    nA, nB = len(ZA), len(ZB)
    want = (lambda m: True) if which is None else (lambda m: m in which)
    muA, muB = XA.mean(0), XB.mean(0)
    if want("prev"):
        out["prev"] = (ZA, XA, unif(nA), muA)
    if want("next"):
        out["next"] = (ZB, XB, unif(nB), muB)
    if want("union"):
        Z, X, w = two_sided((ZA, XA), unif(nA), (ZB, XB), unif(nB), n, rng)
        out["union"] = (Z, X, w, 0.5 * muA + 0.5 * muB)
    if want("indep"):
        i, j = rng.integers(0, nA, n), rng.integers(0, nB, n)
        out["indep"] = ((1 - T) * ZA[i] + T * ZB[j], (1 - T) * XA[i] + T * XB[j], unif(n), (1 - T) * muA + T * muB)
    need_bal = any(want(m) for m in ("ot_pairs", "sb", "sb_half", "ot_bary", "ot_bary_sym"))
    if need_bal:
        P, med = plan(ZA, ZB, eps_rel)
        eps_abs = eps_rel * med
        r, c = P.sum(1), P.sum(0)
        gm = (1 - T) * (r @ XA) / r.sum() + T * (c @ XB) / c.sum()
        i, j = draw_pairs(P, n, rng)
        Zp, Xp = (1 - T) * ZA[i] + T * ZB[j], (1 - T) * XA[i] + T * XB[j]
        if want("ot_pairs"):
            out["ot_pairs"] = (Zp, Xp, unif(n), gm)
        for name, var in (("sb", T * (1 - T) * eps_abs), ("sb_half", T * (1 - T) * eps_abs / 2)):
            if want(name):
                noise = rng.normal(0.0, np.sqrt(var), Zp.shape)
                out[name] = (Zp + noise, Xp + noise @ comps, unif(n), gm)
        if want("ot_bary") or want("ot_bary_sym"):
            TzA, TxA = (P @ ZB) / r[:, None], (P @ XB) / r[:, None]
            fZ, fX = ZA + T * (TzA - ZA), XA + T * (TxA - XA)
            if want("ot_bary"):
                out["ot_bary"] = (fZ, fX, r / r.sum(), (r @ fX) / r.sum())
            if want("ot_bary_sym"):
                TzB, TxB = (P.T @ ZA) / c[:, None], (P.T @ XA) / c[:, None]
                bZ, bX = ZB + (1 - T) * (TzB - ZB), XB + (1 - T) * (TxB - XB)
                Z, X, w = two_sided((fZ, fX), r / r.sum(), (bZ, bX), c / c.sum(), n, rng)
                out["ot_bary_sym"] = (Z, X, w, 0.5 * (r @ fX) / r.sum() + 0.5 * (c @ bX) / c.sum())
        out["_plan"] = {"med": med, "eps_abs": eps_abs}
    if want("uot"):
        a = np.exp(beta * ccA * DT_INT)
        Pu, med = plan(ZA, ZB, eps_rel, tau=tau, a=a)
        ru = Pu.sum(1)
        u = np.full(nA, Pu.sum() / nA)          # observed mass at k-1 (uniform), same total as the plan
        W = Pu * ((u / ru) ** (1 - T))[:, None]
        i, j = draw_pairs(W, n, rng)
        wr, wc = W.sum(1), W.sum(0)
        gm = (1 - T) * (wr @ XA) / W.sum() + T * (wc @ XB) / W.sum()
        out["uot"] = ((1 - T) * ZA[i] + T * ZB[j], (1 - T) * XA[i] + T * XB[j], unif(n), gm)
        out["_uot"] = {"growth_cv": float(np.std(ru / u) / np.mean(ru / u)), "mass": float(Pu.sum()),
                       "col_mass_cv": float(np.std(Pu.sum(0)) / np.mean(Pu.sum(0)))}
    if VA is not None and cfit is not None:
        for vk in ("v_ss", "v_dyn"):
            for mode, cc_ in cfit[vk].items():
                nm = f"vel_{vk[2:]}" + ("" if mode == "sd" else "_cED")
                if want(nm):
                    vz, vx = VA[vk]
                    out[nm] = (ZA + cc_["fwd"] / 2 * vz, XA + cc_["fwd"] / 2 * vx, unif(nA),
                               (XA + cc_["fwd"] / 2 * vx).mean(0))
                nb = nm.replace(f"vel_{vk[2:]}", f"vel_{vk[2:]}_back")
                if want(nb) and VB is not None:
                    vz, vx = VB[vk]
                    out[nb] = (ZB - cc_["back"] / 2 * vz, XB - cc_["back"] / 2 * vx, unif(nB),
                               (XB - cc_["back"] / 2 * vx).mean(0))
    return out


def subsample_pred(Z, X, w, n, rng):
    if len(Z) <= n:
        return Z, X, w / w.sum()
    i = np.sort(rng.choice(len(Z), n, replace=False))
    return Z[i], X[i], w[i] / w[i].sum()


# ---------------- main ----------------

def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--floors", default=None)
    ap.add_argument("--no-extras", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    LOGF = os.path.join(args.out, "run.log")
    open(LOGF, "w").close()
    t_all = time.time()
    floors_path = args.floors or os.path.join(os.path.dirname(args.out.rstrip("\\/")), "diag_batch", "floors.csv")
    loso_path = os.path.join(os.path.dirname(floors_path), "loso.csv")
    floors = pd.read_csv(floors_path) if os.path.exists(floors_path) else None
    loso_ref = pd.read_csv(loso_path) if os.path.exists(loso_path) else None
    log("floors:", floors_path if floors is not None else "NOT FOUND")

    obs = pd.read_csv(os.path.join(args.data, "obs.csv"))
    obs_stage = np.array([STAGES.index(s) for s in obs["stage"].astype(str)])
    obs_sample = obs["sample"].astype(str).to_numpy()
    smp_batch = obs.groupby(obs["sample"].astype(str))["sequencing.batch"].first().astype(int).to_dict()
    # origin5 cc score (descriptive check of the gene list only; its gene filter saw the held stage)
    o5 = np.load(os.path.join(args.data, "origin5.npz"), allow_pickle=True)
    cc_o5 = np.empty(len(obs))
    cc_o5[obs_stage <= 5] = o5["cc_past"]
    cc_o5[obs_stage == 6] = o5["cc_fut"]
    assert (o5["celltype_fut"] == obs["celltype"].to_numpy()[obs_stage == 6]).all()
    del o5
    # gene-list check: in origin 2 every kept gene is an HVG, so our list must reproduce its cc score exactly
    o2 = np.load(os.path.join(args.data, "origin2.npz"), allow_pickle=True)
    m2 = np.array([g.capitalize() in set(S_GENES + G2M_GENES) for g in o2["genes"].astype(str)])
    X2 = o2["X_past"][:, m2].astype(np.float64)
    r2 = float(np.corrcoef((X2 - X2.mean(0)).mean(1), o2["cc_past"])[0, 1])
    log(f"gene-list check: origin2 (all kept genes are HVGs) {int(m2.sum())} cc genes, corr with its cc_past {r2:.6f}")
    del o2, X2

    res = {"config": {"T": T, "NSUB": NSUB, "DEFAULT": DEFAULT, "C_GRID": C_GRID, "reps": args.reps,
                      "ks": args.ks}, "gene_list_check_origin2_corr": r2, "k": {}}
    rows, pool_rows, sens_rows, bm_rows = [], [], [], []
    for k in args.ks:
        t0 = time.time()
        d = np.load(os.path.join(args.data, f"interp{k}.npz"), allow_pickle=True)
        st = d["stage_fit"]
        Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
        Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
        comps = d["pca_comps"].astype(np.float64)
        smp_fit, smp_held = d["sample_fit"].astype(str), d["sample_held"].astype(str)
        ct_fit, ct_held = d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
        # ---- label checks against obs.csv ----
        gfit = np.where((obs_stage == k - 1) | (obs_stage == k + 1))[0]
        gheld = np.where(obs_stage == k)[0]
        checks = {"fit_celltype": bool((obs["celltype"].to_numpy()[gfit] == ct_fit).all()),
                  "fit_sample": bool((obs_sample[gfit] == smp_fit).all()),
                  "fit_stage": bool((obs_stage[gfit] == st).all()),
                  "held_celltype": bool((obs["celltype"].to_numpy()[gheld] == ct_held).all()),
                  "held_sample": bool((obs_sample[gheld] == smp_held).all())}
        assert all(checks.values()), checks
        # ---- proliferation score (strict: HVG cc genes, centred on fit cells) ----
        genes = [g.capitalize() for g in d["genes"].astype(str)]
        ccset = set(S_GENES + G2M_GENES)
        ccm = np.array([g in ccset for g in genes])
        Xcc = d["X_fit"][:, ccm].astype(np.float64)
        cc_fit = (Xcc - Xcc.mean(0)).mean(1)
        r_cc = float(np.corrcoef(cc_fit, cc_o5[gfit])[0, 1])
        log(f"k={k} {STAGES[k-1]}+{STAGES[k+1]} -> {STAGES[k]}: n_fit {len(st)} n_held {len(Zh)}; "
            f"cc genes {int(ccm.sum())}, corr with origin5 cc {r_cc:.3f}; checks ok")
        A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
        S = med_scale(Zf)
        n_true = min(len(Zh), NSUB)
        n_pred = n_true
        it = sub(len(Zh), seed=1)
        tgt = Target(Zh[it], Xh[it], S)
        mu_true = Xh.astype(np.float64).mean(0)
        # pools of the held stage
        pools = [p for p in sorted(set(smp_held), key=lambda x: int(x)) if (smp_held == p).sum() >= MIN_POOL]
        ptg = {p: Target(Zh[smp_held == p], None, S, with_sd=False) for p in pools}
        # kNN classifier (descriptive composition)
        knn = KNeighborsClassifier(n_neighbors=15).fit(Zf, ct_fit)
        types = sorted(set(ct_fit) | set(ct_held))
        lab_held = knn.predict(Zh)
        comp_cls = np.array([(lab_held == t_).mean() for t_ in types])
        comp_true = np.array([(ct_held == t_).mean() for t_ in types])
        acc = float((lab_held == ct_held).mean())
        kres = {"n_fit_prev": int(len(A)), "n_fit_next": int(len(B)), "n_held": int(len(Zh)), "S_fit": S,
                "n_pred": n_pred, "cc_genes": int(ccm.sum()), "cc_corr_origin5": r_cc, "checks": checks,
                "pools": {p: {"n": int((smp_held == p).sum()), "batch": smp_batch[p]} for p in pools},
                "held_batches": {str(b): int(sum(smp_batch[s] == b for s in smp_held)) for b in (1, 2, 3)},
                "fit_batches": {STAGES[s_]: {str(b): int(sum(smp_batch[x] == b for x in smp_fit[st == s_]))
                                              for b in (1, 2, 3)} for s_ in (k - 1, k + 1)},
                "types": types, "comp_true": comp_true.tolist(), "comp_classifier": comp_cls.tolist(),
                "knn_accuracy_on_held": acc}
        # ---- floors (DESCRIPTIVE) ----
        fl = {}
        if floors is not None:
            for rep_ in ("pca", "genes"):
                fr = floors[(floors["space"] == f"interp{k}") & (floors["rep"] == rep_)]
                if len(fr):
                    fr = fr.iloc[0]
                    fl[rep_] = {c_: float(fr[c_]) for c_ in ("halves_ED_U", "halves_ED_V", "oracle_ED_U",
                                                             "oracle_ED_V_fs", "loso_ED_U_mean", "loso_ED_U_min",
                                                             "loso_ED_U_max", "loso_ED_V_fs_mean", "tau2")}
        rng_h = np.random.default_rng(4)
        perm = rng_h.permutation(len(Zh))
        hA, hB = perm[: len(Zh) // 2][:NSUB], perm[len(Zh) // 2:][:NSUB]
        tB = Target(Zh[hB], Xh[hB], S)
        fl["own_halves"] = dict(zip(("ED_U", "ED_V"), ed_uv(Zh[hA], unif(len(hA)), tB)))
        fl["own_halves"]["sd_fixed"] = sd_fixed(Zh[hA], unif(len(hA)), tB)
        m_ = min(len(hA), len(hB))
        fl["own_halves"]["W1"], fl["own_halves"]["W2"] = w_exact(Zh[hA[:m_]], unif(m_), Zh[hB[:m_]])
        fl["own_halves"]["n_each"] = int(m_)
        lo = {}
        for p in pools:
            rest = np.where(smp_held != p)[0]
            exact = len(rest) <= NSUB
            rest_w = rest[sub(len(rest), k=int((smp_held == p).sum()), seed=6)]
            rest = rest[sub(len(rest), seed=5)]
            pt = ptg[p]
            eu, ev = ed_uv(Zh[rest], unif(len(rest)), pt)
            ptg_sd = Target(Zh[smp_held == p], None, S)
            w1, w2 = w_exact(Zh[rest_w], unif(len(rest_w)), Zh[smp_held == p])
            lo[p] = {"ED_U": eu, "sd_fixed": sd_fixed(Zh[rest], unif(len(rest)), ptg_sd), "W1_pool_size": w1,
                     "W2_pool_size": w2, "exact": exact}
            if loso_ref is not None:
                ref = loso_ref[(loso_ref["space"] == f"interp{k}") & (loso_ref["rep"] == "pca") &
                               (loso_ref["sample"].astype(str) == p)]
                if len(ref):
                    lo[p]["ED_U_diag_batch"] = float(ref["ED_U"].iloc[0])
        fl["own_loso"] = lo
        fl["own_loso_mean"] = {m_: float(np.mean([v[m_] for v in lo.values()]))
                               for m_ in ("ED_U", "sd_fixed", "W1_pool_size", "W2_pool_size")}
        kres["floors"] = fl
        log(f"  floors: own halves {fl['own_halves']}, own loso mean {fl['own_loso_mean']}; diag_batch pca "
            f"{fl.get('pca')}")
        log("  loso check (own exact vs diag_batch):",
            {p: (round(v["ED_U"], 5), round(v.get("ED_U_diag_batch", np.nan), 5), v["exact"]) for p, v in lo.items()})

        # ---- velocity: c fitted strictly on k-1 -> k+1 (and k+1 -> k-1), rep-0 subsamples ----
        Vfull = {vk: d[vk].astype(np.float64) / (1.0 + d["Ms"].astype(np.float64)) for vk in ("v_ss", "v_dyn")}
        iA0, iB0 = sub(len(A), seed=1000), sub(len(B), seed=2000)
        cfit, cfit_curve = {}, {}
        tB_fit = Target(Zf[B[iB0]], None, S)
        tA_fit = Target(Zf[A[iA0]], None, S)
        for vk in ("v_ss", "v_dyn"):
            vzA = Vfull[vk][A[iA0]] @ comps.T
            vzB = Vfull[vk][B[iB0]] @ comps.T
            curve = {"fwd": {"sd": [], "ED_U": []}, "back": {"sd": [], "ED_U": []}}
            for c_ in C_GRID:
                Pf = Zf[A[iA0]] + c_ * vzA
                curve["fwd"]["sd"].append(sd_fixed(Pf, unif(len(Pf)), tB_fit))
                curve["fwd"]["ED_U"].append(ed_uv(Pf, unif(len(Pf)), tB_fit)[0])
                Pb = Zf[B[iB0]] - c_ * vzB
                curve["back"]["sd"].append(sd_fixed(Pb, unif(len(Pb)), tA_fit))
                curve["back"]["ED_U"].append(ed_uv(Pb, unif(len(Pb)), tA_fit)[0])
            cfit[vk] = {m_: {dr: C_GRID[int(np.argmin(curve[dr][m_]))] for dr in ("fwd", "back")}
                        for m_ in ("sd", "ED_U")}
            cfit_curve[vk] = curve
            vnorm_pca = np.linalg.norm(vzA, axis=1) / np.maximum(np.linalg.norm(Vfull[vk][A[iA0]], axis=1), 1e-12)
            cfit[vk]["_pca_share_median"] = float(np.median(vnorm_pca))
        kres["velocity_fit"] = {vk: {m_: cfit[vk][m_] for m_ in ("sd", "ED_U")} for vk in cfit}
        kres["velocity_fit_curves"] = cfit_curve
        kres["velocity_pca_share"] = {vk: cfit[vk]["_pca_share_median"] for vk in cfit}
        log(f"  velocity c (fwd/back, strict on k-1<->k+1): {kres['velocity_fit']}")
        cfit_use = {vk: {m_: cfit[vk][m_] for m_ in ("sd", "ED_U")} for vk in cfit}

        # ---- primary methods over reps ----
        per_rep = []
        for r in range(args.reps):
            rng = np.random.default_rng(1000 * k + r)
            iA, iB = sub(len(A), seed=1000 + r), sub(len(B), seed=2000 + r)
            ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
            VA = {vk: (Vfull[vk][A[iA]] @ comps.T, Vfull[vk][A[iA]]) for vk in Vfull}
            VB = {vk: (Vfull[vk][B[iB]] @ comps.T, Vfull[vk][B[iB]]) for vk in Vfull}
            M = methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng, VA=VA, VB=VB, cfit=cfit_use)
            info = {"plan": M.pop("_plan"), "uot": M.pop("_uot")}
            ref_prev, ref_mid = XA.mean(0), 0.5 * (XA.mean(0) + XB.mean(0))
            # sanity: t = 0 / t = 1 endpoints of the pair interpolant
            if r == 0:
                P0, _ = plan(ZA, ZB, DEFAULT["eps_rel"])
                i0, j0 = draw_pairs(P0, n_pred, np.random.default_rng(9))
                tA_ = Target(ZA, None, S, with_sd=False)
                tB_ = Target(ZB, None, S, with_sd=False)
                info["endpoint_check_ED_U"] = {"t0_vs_prev": ed_uv(ZA[i0], unif(n_pred), tA_)[0],
                                               "t1_vs_next": ed_uv(ZB[j0], unif(n_pred), tB_)[0]}
            rec = {"rep": r, **info, "m": {}}
            for name, (Z, X, w, gm) in M.items():
                Zs, Xs, ws = subsample_pred(Z, X, w, n_pred, rng)
                bs_ = basic_scores(Zs, ws, tgt)
                eug, _ = ed_uv(Xs, ws, tgt, gene=True)
                lab = knn.predict(Zs)
                comp = np.array([ws[lab == t_].sum() for t_ in types])
                mu_p = (ws @ Zs)
                spread = float(np.sqrt(ws @ ((Zs - mu_p) ** 2).sum(1)) /
                               np.sqrt(((tgt.Z - tgt.Z.mean(0)) ** 2).sum(1).mean()))
                g = gene_scores(gm, ref_prev, ref_mid, mu_true)
                pp = {p: ed_uv(Zs, ws, ptg[p])[0] for p in pools}
                rec["m"][name] = {**bs_, "ED_U_gene": eug, "n_points": int(len(Zs)),
                                  "comp_tv": float(0.5 * np.abs(comp - comp_cls).sum()), "comp": comp.tolist(),
                                  "spread_ratio": spread, **g, "pool_ED_U": pp}
            per_rep.append(rec)
            log(f"  rep {r} (sd_fixed/ED_U/W2): " + ", ".join(f"{m_} {v['sd_fixed']:.3f}/{v['ED_U']:.3f}/{v['W2']:.2f}"
                                                            for m_, v in rec["m"].items())
                + f"  [{time.time() - t0:.0f}s]")
        kres["reps"] = per_rep
        names = list(per_rep[0]["m"])
        for name in names:
            row = {"k": k, "held": STAGES[k], "method": name}
            for met in ("sd_fixed", "ED_U", "ED_V", "W1", "W2", "ED_U_gene", "comp_tv", "spread_ratio", "pcc", "p200_up",
                        "p200_down", "dev_pcc", "dev_norm_ratio"):
                vals = np.array([rr["m"][name][met] for rr in per_rep], float)
                row[met] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                row[met + "_sd"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else np.nan
            for p in pools:
                v = float(np.mean([rr["m"][name]["pool_ED_U"][p] for rr in per_rep]))
                row[f"pool_{p}"] = v
                pool_rows.append({"k": k, "held": STAGES[k], "method": name, "pool": p, "batch": smp_batch[p],
                                  "n": int((smp_held == p).sum()), "ED_U": v, "loso_ED_U": lo[p]["ED_U"],
                                  "ratio_to_loso": v / lo[p]["ED_U"]})
            rows.append(row)

        # ---- extras (1 rep) ----
        if not args.no_extras:
            rng = np.random.default_rng(77 + k)
            ZA, XA, ZB, XB = Zf[A[iA0]], Xf[A[iA0]].astype(np.float64), Zf[B[iB0]], Xf[B[iB0]].astype(np.float64)
            grid = [("eps_rel", e_, {"eps_rel": e_}, ["ot_pairs", "sb", "sb_half", "ot_bary"]) for e_ in (0.01, 0.2)]
            grid += [("uot", f"tau{t_}_beta{b_}", {"tau": t_, "beta": b_}, ["uot"])
                     for t_ in (0.1, 1.0) for b_ in (0.0, 2.0, 5.0)]
            for kind, lab_, kw, which in grid:
                M = methods(ZA, XA, ZB, XB, cc_fit[A[iA0]], comps, n_pred, rng, which=which, **kw)
                M.pop("_plan", None)
                ui = M.pop("_uot", None)
                for name, (Z, X, w, gm) in M.items():
                    Zs, Xs, ws = subsample_pred(Z, X, w, n_pred, rng)
                    sens_rows.append({"k": k, "held": STAGES[k], "grid": kind, "setting": lab_, "method": name,
                                      **basic_scores(Zs, ws, tgt),
                                      "growth_cv": None if ui is None else ui["growth_cv"],
                                      "oracle_labelled": True})
            # batch-matched block: batch-2 cells only
            b = 2
            Ab, Bb = A[np.array([smp_batch[x] == b for x in smp_fit[A]])], B[np.array([smp_batch[x] == b for x in smp_fit[B]])]
            hb = np.where(np.array([smp_batch[x] == b for x in smp_held]))[0]
            if len(Ab) >= 50 and len(Bb) >= 50 and len(hb) >= 50:
                ihb = hb[sub(len(hb), seed=3)]
                tgt_b = Target(Zh[ihb], Xh[ihb], S)
                nb = len(ihb)
                which = ["prev", "next", "union", "indep", "ot_pairs", "sb", "sb_half", "uot", "ot_bary_sym"]
                iAb, iBb = Ab[sub(len(Ab), seed=1100)], Bb[sub(len(Bb), seed=2100)]
                Mb = methods(Zf[iAb], Xf[iAb].astype(np.float64), Zf[iBb], Xf[iBb].astype(np.float64),
                             cc_fit[iAb], comps, nb, rng, which=which)
                Ma = methods(ZA, XA, ZB, XB, cc_fit[A[iA0]], comps, nb, rng, which=which)
                for tag, MM in (("batch2_fit", Mb), ("all_fit", Ma)):
                    MM.pop("_plan", None)
                    MM.pop("_uot", None)
                    for name, (Z, X, w, gm) in MM.items():
                        Zs, Xs, ws = subsample_pred(Z, X, w, nb, rng)
                        bm_rows.append({"k": k, "held": STAGES[k], "fit": tag, "method": name,
                                        "target": f"held batch {b} ({len(hb)} cells)",
                                        "n_prev_b": int(len(Ab)), "n_next_b": int(len(Bb)),
                                        **basic_scores(Zs, ws, tgt_b)})
        kres["seconds"] = round(time.time() - t0, 1)
        res["k"][str(k)] = kres
        res["stats"] = dict(STATS)
        log(f"  k={k} done in {kres['seconds']}s; solver {STATS}")
        dump(args.out, res, rows, pool_rows, sens_rows, bm_rows)

    res["verdicts"] = verdicts(pd.DataFrame(rows), pd.DataFrame(pool_rows), args.ks)
    res["seconds_total"] = round(time.time() - t_all, 1)
    dump(args.out, res, rows, pool_rows, sens_rows, bm_rows)
    log("VERDICTS:", json.dumps(res["verdicts"]["summary"], indent=1))
    log(f"total {res['seconds_total']}s")


def rule(tb, ks, a, bs):
    """a beats every b in bs at >= 3 of the ks (all of them if fewer than 4) and on the mean over ks."""
    va = tb.loc[a]
    wins = {}
    for k in ks:
        wins[k] = bool(all(va[k] < tb.loc[b_][k] for b_ in bs))
    mean_a = float(np.mean([va[k] for k in ks]))
    mean_ok = all(mean_a < float(np.mean([tb.loc[b_][k] for k in ks])) for b_ in bs)
    need = 3 if len(ks) >= 4 else len(ks)
    n_w = int(sum(wins.values()))
    return {"wins": {str(k): v for k, v in wins.items()}, "n_wins": n_w, "mean": mean_a,
            "means_b": {b_: float(np.mean([tb.loc[b_][k] for k in ks])) for b_ in bs},
            "mean_ok": bool(mean_ok), "pass": bool(n_w >= need and mean_ok)}


def verdicts(df, pdf, ks):
    out = {"rules": {}, "summary": {}, "pool_consistency": {}}
    # sd_fixed primary, ED_U secondary (decision rule); W2 / W1 reported with the same rule, labelled as added later
    tab = {met: df.pivot(index="method", columns="k", values=met) for met in ("sd_fixed", "ED_U", "W2", "W1")}
    base = ["prev", "next", "union"]
    tests = [("R1_ot_pairs", "ot_pairs", base), ("R1_sb", "sb", base), ("R1_sb_half", "sb_half", base),
             ("R1_uot", "uot", base), ("R1_ot_bary", "ot_bary", base), ("R1_ot_bary_sym", "ot_bary_sym", base),
             ("R2_pairing", "ot_pairs", ["indep"]), ("R3_sb_vs_ot", "sb", ["ot_pairs"]),
             ("R3_sb_half_vs_ot", "sb_half", ["ot_pairs"]), ("R4_growth", "uot", ["ot_pairs"]),
             ("R5_vel_ss", "vel_ss", ["prev", "union"]), ("R5_vel_dyn", "vel_dyn", ["prev", "union"]),
             ("R5x_vel_ss_vs_ot", "vel_ss", ["ot_pairs"]), ("R5x_vel_dyn_vs_ot", "vel_dyn", ["ot_pairs"])]
    for met, tb in tab.items():
        for name, a, bs in tests:
            if a in tb.index and all(b_ in tb.index for b_ in bs):
                r = rule(tb, ks, a, bs)
                out["rules"][f"{name}|{met}"] = r
                out["summary"][f"{name}|{met}"] = f"{'PASS' if r['pass'] else 'FAIL'} ({r['n_wins']}/{len(ks)}, mean {'ok' if r['mean_ok'] else 'no'})"
        best = {int(k): str(tb[k].idxmin()) for k in tb.columns}
        out["summary"][f"best_per_k|{met}"] = best
    for k in ks:
        sub_ = pdf[pdf["k"] == k]
        cons = {}
        for a in ("ot_pairs", "sb", "sb_half", "uot", "ot_bary_sym", "vel_ss", "vel_dyn"):
            per = {}
            for p in sorted(sub_["pool"].unique(), key=int):
                s = sub_[sub_["pool"] == p].set_index("method")["ED_U"]
                if a in s.index:
                    per[p] = bool(all(s[a] < s[b_] for b_ in base if b_ in s.index))
            cons[a] = {"pools_won": per, "all_pools": bool(all(per.values())) if per else None}
        best_pool = {p: str(sub_[sub_["pool"] == p].set_index("method")["ED_U"].idxmin())
                     for p in sorted(sub_["pool"].unique(), key=int)}
        out["pool_consistency"][str(k)] = {"vs_prev_next_union": cons, "best_method_per_pool": best_pool}
    return out


def dump(out, res, rows, pool_rows, sens_rows, bm_rows):
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.DataFrame(rows).to_csv(os.path.join(out, "table.csv"), index=False)
    pd.DataFrame(pool_rows).to_csv(os.path.join(out, "pools.csv"), index=False)
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(os.path.join(out, "sensitivity_oracle.csv"), index=False)
    if bm_rows:
        pd.DataFrame(bm_rows).to_csv(os.path.join(out, "batch_matched.csv"), index=False)


if __name__ == "__main__":
    main()
