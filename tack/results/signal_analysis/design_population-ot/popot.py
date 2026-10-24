"""Population-level OT on cell-state composition for Dileo & Sottoriva Task 2: a STRICT rolling-origin prototype.

Transition i maps snapshot i -> i+1 and reads snapshots 0..i only (gastrulation-human bins 2..31, pancreas-human
bins 0..14; first N genes of the node files, raw UMI bin statistics m, s, f, n cells).

Model, per transition i
 1. archetypes: weighted-KL NMF of the CP10k bin profiles of snapshots 0..i (crossgene nmf_kl.fit_subset through
    forecasts.NMFMix.fit; row weight n_b lib_b / 1e4; K in {3, 6, 8}, K_eff = min(K, i + 1)). Rows of W = UMI
    fractions P_b of the K archetypes; cost between archetypes C_kl = mean_g (log1p H_kg - log1p H_lg)^2 divided by
    its mean off-diagonal value (crossgene's cost).
 2. cell fractions: pi_bk = P_bk lib_b c_k / sum_k' (.), with c_k = 1 / (UMI per cell of archetype k) from a ridge
    fit of sum_k P_bk lib_b c_k = 1 over the bins (weights n_b, shrunk to 1 / mean library, clipped to [1/4, 4] x).
 3. archetype count laws: per gene and archetype the triplet A_k = (mean m_k, nonzero fraction f_k, second moment
    e_k = E[c^2]), fitted with pi fixed by a weighted-KL multiplicative loop on the three raw bin channels. For a
    mixture of cell populations the identities m_b = sum_k pi_bk m_k, f_b = sum_k pi_bk f_k, e_b = sum_k pi_bk e_k
    are exact, so a composition change moves the whole count law of every gene, zero fraction included.
 4. population forecast of pi_{i+1} (the only place OT / SB enters):
      persist | lin1 (pi_i + (pi_i - pi_{i-1})) | loglin1 (multiplicative)                     -- no OT
      ot0.05    balanced entropic OT plan between pi_{i-1} and pi_i (eps 0.05 on the normalised cost), its Markov
                kernel applied once more to pi_i                         (crossgene forecasts.NMFMix.forecast_props)
      uot0.05t1 unbalanced entropic OT (KL marginal relaxation tau = 1): growth of each archetype kept and
                extrapolated together with the transport                 (crossgene forecasts.NMFMix.forecast_props)
      sb3 / sbW static Schroedinger bridge: one time-homogeneous Markov kernel over archetypes = cell-weighted pool
                of the conditional entropic OT plans of the last 3 / 8 consecutive pairs, applied to pi_i (new here)
 5. map back by the mixture shift: T_hat = T_state + (pi_hat - pi_state) A for the three channels. T_state is the
    gene's own law pooled over the last k bins (box k: cell-weighted mixture of the triplets, k in {1, 2, 4}; pi_state
    pooled with the same weights) or mixavg3 (pi_i A + the residual T_b - pi_b A averaged over 3 bins with weights
    n / (n + 50)). Optional per-gene residual trend (box1 only): + beta [(T_i - T_{i-1}) - (pi_i - pi_{i-1}) A],
    beta in {0.5, 1}; lin1 with beta = 1 is exactly the raw-mean per-gene trend, so the two-pathway model nests it.
    Fix-up: m >= 0, 0 <= f <= min(1, m) (f = 1 - exp(-m) if m > 0 and f = 0), e >= max(m, m^2 / f).
 6. observation / target: orig -> dist_bridge.expected_target(T_hat, m_i, n_i) (bin i+1 = n_i cells drawn from the
    forecast law: zero-crossings analytic, the rest Monte Carlo, 256 draws, numpy generator keyed by transition and
    config); l1 -> plug-in log1p(1e4 m_hat / sum m_hat) - log1p(CP10k_i); linear -> m_hat - m_i.
 7. selection: rolling origin per transition, the config with the best mean dist_bridge.criterion (Pearson + tie-aware
    expected P@K up + down, K = 200 / n_test x N, all genes, targets rebuilt from bins <= i) over the last 8 past
    transitions j >= 1; ties within 1e-3 go to non-OT forecasters, then to the earlier config. No past transition
    (i <= 1): orig uses box4 persist (= level + observation model), l1 / linear the repeat-last-change trend.
Comparators run here: repeat-last-change trend (all targets); level + observation model (box4 persist; and RO over
box2 / box4 persist); RO groups restricted to non-OT / OT forecasters / one forecaster at a time (isolates the OT
increment, paired Wilcoxon over transitions); ORACLE next composition (bin i+1 projected on the strict archetypes,
NOT strict, ceiling of any composition forecaster). Reproduction check: crossgene 'nmf_linear_props' on l1 with
their own pipeline. Diagnostics: turn-on / turn-off AUC on genes at zero / nonzero in bin i, composition forecast
error (total variation against the oracle projection).
Scores: crossgene common.masks (10 draws of a random 25 % of genes per transition, seed 20261022), Pearson and
P@200 up / down with np.argsort (their metric); 'i>=1' = mean over transitions 1..T-1 (NaN Pearson -> 0); 'all' =
nanmean over all transitions including i = 0 (the convention of the CONTEXT DistBridge numbers).
Usage: python popot.py <dataset>   ->   popot_<dataset>.json
"""
import json
import os
import sys
import time
import zlib

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crossgene"))
import common  # noqa: E402  (crossgene shared loading / masks / metrics; read-only import)

sys.path.insert(0, common.HPC)
import dist_bridge  # noqa: E402  (observation model and selection criterion; numpy only)
import forecasts  # noqa: E402  (crossgene NMFMix: strict NMF fits and proportion forecasters)
import nmf_kl  # noqa: E402

KS = (3, 6, 8)
FC = ("persist", "lin1", "loglin1", "ot0.05", "uot0.05t1", "sb3", "sbW")
OT = ("ot0.05", "uot0.05t1", "sb3", "sbW")
EPS_SB = 0.05
TINY = 1e-10
TIE = 1e-3
TARGETS = ("orig", "l1", "linear")
STATES = (("box1", 0.0), ("box2", 0.0), ("box4", 0.0), ("mixavg3", 0.0), ("box1", 0.5), ("box1", 1.0))
ORIG_STATES = (("box2", 0.0), ("box4", 0.0))
DIST_BRIDGE_CONTEXT = {"gastrulation-human": {"orig": [0.585, 0.406, 0.541]},
                       "pancreas-human": {"orig": [0.488, 0.301, 0.442], "l1": [0.10, 0.17, 0.21],
                                          "l1_without_OT": [-0.05, 0.10, 0.15], "linear": [0.30, 0.25, 0.30],
                                          "linear_without_OT": [-0.14, 0.06, 0.06],
                                          "trend_context_l1": [0.15, 0.21, 0.21], "trend_context_linear": [0.40, 0.27, 0.35]}}


def omega(n):
    return np.asarray(n, float) / (np.asarray(n, float) + 50.0)


def fixed_w_kl(W, X, w, A0, n_iter=150):
    """A >= 0 (K x G) minimising sum_b w_b KL(X_b | (W A)_b) with W (B x K) fixed: multiplicative updates."""
    A = np.maximum(np.asarray(A0, np.float64), 1e-8)
    Xw = w[:, None] * X
    den = np.maximum((W * w[:, None]).sum(0)[:, None], TINY)
    for _ in range(n_iter):
        A *= (W.T @ (Xw / np.maximum(W @ A, TINY))) / den
    A[:, X.max(0) <= 0] = 0.0
    return A


def ot_plan(a, b, C, eps):
    return forecasts.uot_plan(a, b, C, eps, None)


def sb_forecast(pi, C, L, n):
    """Static Schroedinger-bridge kernel: cell-weighted pool of the conditional entropic OT plans of the last L
    consecutive pairs (rows of pi = proportions of bins 0..i), applied to pi_i."""
    i = pi.shape[0] - 1
    if i < 1:
        return pi[-1]
    K = pi.shape[1]
    num, den = np.zeros((K, K)), np.zeros(K)
    for j in range(max(1, i - L + 1), i + 1):
        a = (pi[j - 1] + forecasts.DELTA) / (pi[j - 1] + forecasts.DELTA).sum()
        b = (pi[j] + forecasts.DELTA) / (pi[j] + forecasts.DELTA).sum()
        w = float(omega(min(n[j - 1], n[j])))
        num += w * ot_plan(a, b, C, EPS_SB)
        den += w * a
    Q = num / den[:, None]
    h = np.maximum(pi[-1] @ Q, 0)
    return h / h.sum()


def fixup(T):
    m = np.maximum(T[0], 0.0)
    f = np.clip(T[1], 0.0, 1.0)
    f = np.where(m <= 0, 0.0, f)
    f = np.where((m > 0) & (f <= 1e-9), 1 - np.exp(-m), f)
    f = np.minimum(f, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.maximum(T[2], np.maximum(m, np.where(f > 0, m ** 2 / np.where(f > 0, f, 1), 0.0)))
    return np.stack([m, f, e])


class PopOT:
    def __init__(self, d):
        self.d = d
        self.fam = forecasts.NMFMix(d)          # shares the strict NMF fit cache with the reproduction check
        self.T3 = np.stack([d.M, d.F, d.SD ** 2 + d.M ** 2])     # 3 x N x S raw channels (m, f, e = E[c^2])
        self._mix, self._fc = {}, {}
        self.fit_seconds = 0.0

    def mix(self, i, K):
        Ke = min(K, i + 1)
        key = (i, Ke)
        if key in self._mix:
            return self._mix[key]
        t0 = time.time()
        d = self.d
        P, H, C, _ = self.fam.fit(i, K)
        P, H = np.asarray(P, np.float64), np.asarray(H, np.float64)
        n, lib = d.nc[:i + 1], d.lib[:i + 1]
        X = P * lib[:, None]                                       # = pi_bk * ell_k (UMI per cell from k)
        c0 = n.sum() / (n * lib).sum()
        G = X.T @ (n[:, None] * X)
        lam = 0.1 * np.trace(G) / Ke
        c = np.linalg.solve(G + lam * np.eye(Ke), X.T @ n + lam * c0)
        c = np.clip(c, 0.25 * c0, 4 * c0)
        pi = X * c[None, :]
        pi = pi / pi.sum(1, keepdims=True)
        ell = 1.0 / c
        m0 = ell[:, None] * H / 1e4
        Am = fixed_w_kl(pi, d.M[:, :i + 1].T, n, m0)
        Af = np.minimum(fixed_w_kl(pi, d.F[:, :i + 1].T, n, 1 - np.exp(-Am)), 1.0)
        Ae = fixed_w_kl(pi, self.T3[2][:, :i + 1].T, n, Am + 2 * Am ** 2)
        res = {"pi": pi, "C": C, "A": np.stack([Am, Af, Ae]), "H": H, "c": c, "ell": ell, "s_b": X @ c}
        self._mix[key] = res
        self.fit_seconds += time.time() - t0
        return res

    def forecast(self, i, K, fc, oracle=False):
        key = (i, min(K, i + 1), fc, oracle)
        if key in self._fc:
            return self._fc[key]
        mx = self.mix(i, K)
        pi, C = mx["pi"], mx["C"]
        if oracle:                                                 # NOT strict: true bin i+1 on strict archetypes
            Pn = nmf_kl.to_props(nmf_kl.project(self.d.CP[:, i + 1][None, :], mx["H"]))[0]
            x = Pn * self.d.lib[i + 1] * mx["c"]
            h = x / x.sum()
        elif fc in ("sb3", "sbW"):
            h = sb_forecast(pi, C, 3 if fc == "sb3" else common.WINDOW, self.d.nc[:i + 1])
        else:
            h = forecasts.NMFMix.forecast_props(pi, C, fc)
        self._fc[key] = h
        return h

    def law(self, i, cfg, oracle=False):
        """Forecast triplet (3 x N) of bin i+1 from snapshots <= i (the oracle variant reads bin i+1's composition)."""
        state, K, fc, beta = cfg
        d = self.d
        T3 = self.T3[:, :, :i + 1]
        if state.startswith("box"):
            k = int(state[3:])
            lo = max(0, i - k + 1)
            w = d.nc[lo:i + 1]
            Ts = (T3[:, :, lo:] * w).sum(-1) / w.sum()
        if i == 0 or (fc == "persist" and state.startswith("box") and beta == 0 and not oracle):
            if state.startswith("box"):
                return fixup(Ts)
        mx = self.mix(i, K)
        pi, A = mx["pi"], mx["A"]
        pih = self.forecast(i, K, fc, oracle)
        if state.startswith("box"):
            ps = (pi[lo:] * w[:, None]).sum(0) / w.sum()
        else:                                                      # mixavg3: denoised current law + smoothed residual
            lo = max(0, i - 2)
            w = omega(d.nc[lo:i + 1])
            fit = np.einsum("bk,ckn->cnb", pi[lo:], A)
            Ts = fit[:, :, -1] + ((T3[:, :, lo:] - fit) * w).sum(-1) / w.sum()
            ps = pi[i]
        Th = Ts + np.einsum("k,ckn->cn", pih - ps, A)
        if beta > 0 and i >= 1:
            Th = Th + beta * ((T3[:, :, i] - T3[:, :, i - 1]) - np.einsum("k,ckn->cn", pi[i] - pi[i - 1], A))
        return fixup(Th)

    def target_pred(self, i, Th, tname, seed_key):
        d = self.d
        if tname == "orig":
            rng = np.random.default_rng([zlib.crc32(d.name.encode()), i, zlib.crc32(seed_key.encode())])
            return dist_bridge.expected_target(Th.T.copy(), d.M[:, i], d.nc[i], rng).astype(np.float64)
        if tname == "l1":
            mh = Th[0]
            return np.log1p(1e4 * mh / max(mh.sum(), 1e-12)) - d.L[:, i]
        return Th[0] - d.M[:, i]


def build_configs():
    cfgs = []
    for state, beta in STATES:
        for fc in FC:
            if fc == "persist" and state.startswith("box") and beta == 0:
                cfgs.append((state, 0, fc, beta))                  # independent of K
                continue
            for K in KS:
                cfgs.append((state, K, fc, beta))
    return cfgs


def label(cfg):
    s, K, fc, b = cfg
    return f"{s}{'' if b == 0 else f'+beta{b:g}'}|K{K}|{fc}"


def auc(score, lab):
    lab = np.asarray(lab, bool)
    n1, n0 = lab.sum(), (~lab).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def select(Jm, rows, cfgs, i, default):
    """Rolling-origin choice among `rows` at transition i (past transitions j in [max(1, i-8), i-1])."""
    lo = max(1, i - common.WINDOW)
    if i - lo < 1:
        return default
    sc = np.array([np.nanmean(Jm[r, lo:i]) for r in rows])
    best = np.nanmax(sc)
    cand = [r for r, s in zip(rows, sc) if s >= best - TIE]
    return sorted(cand, key=lambda r: (cfgs[r][2] in OT, rows.index(r)))[0]


def summ(per_t):
    x = np.where(np.isnan(per_t), 0.0, per_t)
    with np.errstate(all="ignore"):
        allm = [float(np.nanmean(per_t[:, 0])), float(np.nanmean(per_t[:, 1])), float(np.nanmean(per_t[:, 2]))]
    return {"i>=1": [round(float(v), 4) for v in x[1:].mean(0)], "all": [round(v, 4) for v in allm],
            "i>=1_no_last": [round(float(v), 4) for v in x[1:-1].mean(0)]}


def main(name):
    t0 = time.time()
    d = common.Data(name)
    mk = common.masks(d)
    n_test = mk.shape[2]
    q = 200.0 / n_test
    Y = {t: common.all_targets(d, t) for t in TARGETS}
    mdl = PopOT(d)
    cfgs = build_configs()
    C = len(cfgs)
    orig_rows = [r for r, c in enumerate(cfgs) if (c[0], c[3]) in ORIG_STATES]
    rows_t = {"orig": orig_rows, "l1": list(range(C)), "linear": list(range(C))}
    J = {t: np.full((C, d.T), np.nan) for t in TARGETS}
    S = {t: np.full((C, d.T, 3), np.nan) for t in TARGETS}
    aucs = {k: np.full((C, d.T), np.nan) for k in ("on", "off")}
    hist_auc = {"on_frac_past_bins_expressed": []}
    print(name, "N", d.N, "T", d.T, "configs", C, "orig configs", len(orig_rows), flush=True)
    for i in range(d.T):
        zero_now = d.M[:, i] <= 0
        on_lab = d.M[:, i + 1][zero_now] > 0
        off_lab = d.M[:, i + 1][~zero_now] <= 0
        frac_past = (d.M[:, :i + 1] > 0).mean(1)
        for r, cfg in enumerate(cfgs):
            need = [t for t in TARGETS if r in rows_t[t]]
            if not need:
                continue
            Th = mdl.law(i, cfg)
            for t in need:
                p = mdl.target_pred(i, Th, t, label(cfg))
                J[t][r, i] = dist_bridge.criterion(Y[t][i], p, q) * 3
                S[t][r, i] = common.test_scores(Y[t][i:i + 1], p[None, :], mk[:, i:i + 1])[0]
            if r in orig_rows and i >= 1:
                aucs["on"][r, i] = auc(Th[1][zero_now], on_lab)
                aucs["off"][r, i] = auc(-Th[1][~zero_now], off_lab)
        if i >= 1:
            hist_auc["on_frac_past_bins_expressed"].append(auc(frac_past[zero_now], on_lab))
        print(name, "transition", i, round(time.time() - t0, 1), "s", flush=True)

    # ---------------- comparators
    trend = {t: np.zeros((d.T, d.N)) for t in TARGETS}
    for t in TARGETS:
        for i in range(1, d.T):
            trend[t][i] = Y[t][i - 1]
    S_trend = {t: common.test_scores(Y[t], trend[t], mk) for t in TARGETS}
    J_trend = {t: np.array([dist_bridge.criterion(Y[t][i], trend[t][i], q) * 3 if i >= 1 else np.nan
                            for i in range(d.T)]) for t in TARGETS}

    # ---------------- groups (rolling-origin selection within a subset)
    def rows_where(t, pred):
        return [r for r in rows_t[t] if pred(cfgs[r])]

    groups = {
        "full": lambda c: True,
        "noOT": lambda c: c[2] not in OT,
        "OT_only": lambda c: c[2] in OT,
        "persist_only(level_obs_RO)": lambda c: c[2] == "persist" and c[0].startswith("box") and c[3] == 0,
        "composition_only(box_states,beta0)": lambda c: c[0].startswith("box") and c[3] == 0,
        "two_pathway(box1+beta)": lambda c: c[0] == "box1" and c[3] > 0,
        "mixavg3_denoised": lambda c: c[0] == "mixavg3",
    }
    for fc in FC:
        groups[f"fc={fc}"] = (lambda c, fc=fc: c[2] == fc)
    default_orig = cfgs.index(("box4", 0, "persist", 0.0))
    out = {"dataset": name, "n_genes": d.N, "n_transitions": d.T, "n_test": int(n_test),
           "cells": [int(round(x)) for x in d.nc], "protocol": __doc__, "n_configs": {t: len(rows_t[t]) for t in TARGETS},
           "targets": {}}
    for t in TARGETS:
        res = {"comparators": {}, "groups": {}, "paired": {}, "fixed_configs_hindsight_NOT_strict": {}}
        res["comparators"]["trend_repeat_last_change"] = summ(S_trend[t])
        if t == "orig":
            res["comparators"]["level_obs_box4_persist"] = summ(S[t][default_orig])
        else:
            res["comparators"]["level_box4_persist_plugin"] = summ(S[t][cfgs.index(("box4", 0, "persist", 0.0))])
        g_scores, g_sel = {}, {}
        for gname, pred in groups.items():
            rows = rows_where(t, pred)
            if not rows:
                continue
            sc = np.zeros((d.T, 3))
            sel = []
            for i in range(d.T):
                dflt = default_orig if t == "orig" else None
                r = select(J[t], rows, cfgs, i, dflt)
                if r is None:                                      # l1 / linear without past: trend fallback
                    sc[i] = S_trend[t][i]
                    sel.append("fallback_trend")
                else:
                    sc[i] = S[t][r, i]
                    sel.append(label(cfgs[r]))
            g_scores[gname], g_sel[gname] = sc, sel
            entry = summ(sc)
            fcs = [s.split("|")[-1] for s in sel[1:] if s != "fallback_trend"]
            entry["share_transitions_OT_selected"] = round(float(np.mean([f in OT for f in fcs])) if fcs else 0.0, 3)
            entry["selected"] = sel
            if t == "orig":
                ra = [aucs["on"][cfgs.index(_unlabel(s, cfgs))][i] if s != "fallback_trend" else np.nan
                      for i, s in enumerate(sel)]
                rf = [aucs["off"][cfgs.index(_unlabel(s, cfgs))][i] if s != "fallback_trend" else np.nan
                      for i, s in enumerate(sel)]
                entry["turn_on_auc_mean_i>=1"] = round(float(np.nanmean(ra[1:])), 4)
                entry["turn_off_auc_mean_i>=1"] = round(float(np.nanmean(rf[1:])), 4)
            entry["per_transition"] = [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in sc]
            res["groups"][gname] = entry
        # paired tests (per transition, i >= 1): what does OT add on top of the same pipeline?
        pairs = [("full", "noOT"), ("OT_only", "noOT"), ("fc=ot0.05", "fc=lin1"), ("fc=ot0.05", "fc=persist"),
                 ("fc=uot0.05t1", "fc=loglin1"), ("fc=sb3", "fc=persist"), ("fc=sbW", "fc=persist"),
                 ("fc=sb3", "fc=lin1"), ("full", "persist_only(level_obs_RO)")]
        for a, b in pairs:
            if a in g_scores and b in g_scores:
                res["paired"][f"{a} - {b}"] = common.paired(g_scores[a], g_scores[b])
        for a in ("full", "noOT", "two_pathway(box1+beta)", "mixavg3_denoised", "composition_only(box_states,beta0)"):
            if a in g_scores:
                res["paired"][f"{a} - trend"] = common.paired(g_scores[a], S_trend[t])
        # hindsight-best fixed config per forecaster (NOT strict, for reference only)
        for fc in FC:
            rows = rows_where(t, lambda c, fc=fc: c[2] == fc)
            if rows:
                best = max(rows, key=lambda r: np.nanmean(J[t][r, 1:]))
                res["fixed_configs_hindsight_NOT_strict"][fc] = {"config": label(cfgs[best]), **summ(S[t][best])}
        out["targets"][t] = res
        print(name, t, "groups done", round(time.time() - t0, 1), "s", flush=True)

    # ---------------- ORACLE next composition (NOT strict): ceiling of any composition forecaster
    orc = {}
    ostates = {"orig": [("box4", 0.0)], "l1": [("box1", 0.0), ("box4", 0.0), ("mixavg3", 0.0)],
               "linear": [("box1", 0.0), ("box4", 0.0), ("mixavg3", 0.0)]}
    for t in TARGETS:
        orc[t] = {}
        for st, beta in ostates[t]:
            for K in KS:
                sc = np.full((d.T, 3), np.nan)
                for i in range(1, d.T):
                    Th = mdl.law(i, (st, K, "persist", beta), oracle=True)
                    p = mdl.target_pred(i, Th, t, f"oracle|{st}|{K}")
                    sc[i] = common.test_scores(Y[t][i:i + 1], p[None, :], mk[:, i:i + 1])[0]
                sc[0] = S_trend[t][0] if t != "orig" else S[t][default_orig][0]
                orc[t][f"ORACLE_props|{st}|K{K}"] = summ(sc)
        out["targets"][t]["oracle_next_composition_NOT_strict"] = orc[t]
    print(name, "oracles done", round(time.time() - t0, 1), "s", flush=True)

    # ---------------- composition diagnostics (strict basis): forecast error of pi against the oracle projection
    comp = {}
    for K in KS:
        comp[f"K{K}"] = {}
        for fc in FC:
            tv = []
            for i in range(2, d.T):
                tv.append(0.5 * np.abs(mdl.forecast(i, K, fc) - mdl.forecast(i, K, "persist", oracle=True)).sum())
            comp[f"K{K}"][fc] = round(float(np.mean(tv)), 4)
        mx = mdl.mix(d.T - 1, K)
        comp[f"K{K}"]["archetype_UMI_per_cell_final_fit"] = [round(float(x), 1) for x in mx["ell"]]
        comp[f"K{K}"]["bin_scale_s_b_range"] = [round(float(mx["s_b"].min()), 3), round(float(mx["s_b"].max()), 3)]
        comp[f"K{K}"]["proportions_final_fit"] = [[round(float(x), 3) for x in row] for row in mx["pi"]]
    out["composition_forecast_tv_error_vs_oracle_projection_i>=2"] = comp
    out["turn_on_auc_history_fraction_of_past_bins_expressed"] = round(float(np.nanmean(hist_auc["on_frac_past_bins_expressed"])), 4)

    # ---------------- reproduction check: crossgene nmf_linear_props on l1 with their own pipeline
    fam = mdl.fam
    sub = [c for c, cfg in enumerate(fam.configs) if cfg[1].startswith(("lin", "loglin"))]
    Jx = np.full((len(fam.configs), d.T), np.nan)
    for c in sub:
        for i in range(1, d.T):
            Jx[c, i] = common.J_of(Y["l1"][i], forecasts.to_target(d, fam.predict(c, i), i, "l1"))
    P, _ = forecasts.selected_preds(d, fam, Jx, "l1", sub, trend["l1"])
    out["reproduction_crossgene_nmf_linear_props_l1"] = summ(common.test_scores(Y["l1"], P, mk))
    out["context_numbers_quoted_not_rerun"] = DIST_BRIDGE_CONTEXT.get(name, {})
    out["seconds"] = round(time.time() - t0, 1)
    out["nmf_and_law_fit_seconds"] = round(mdl.fit_seconds + fam.fit_seconds, 1)
    np.savez_compressed(os.path.join(HERE, f"popot_{name}_scores.npz"), labels=np.array([label(c) for c in cfgs]),
                        **{f"S_{t}": S[t] for t in TARGETS}, **{f"J_{t}": J[t] for t in TARGETS},
                        **{f"S_trend_{t}": S_trend[t] for t in TARGETS}, auc_on=aucs["on"], auc_off=aucs["off"])
    with open(os.path.join(HERE, f"popot_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(name, "done", out["seconds"], "s", flush=True)


def _unlabel(s, cfgs):
    for c in cfgs:
        if label(c) == s:
            return c
    raise KeyError(s)


if __name__ == "__main__":
    main(sys.argv[1])
