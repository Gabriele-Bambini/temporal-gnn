"""NASS: Noise-Aware Stacked State-space forecaster for Dileo Task 2 (strict, rolling origin, numpy only).

Design (each block answers one measured finding of the signal analysis):
  1. Noise-aware denoisers of today's level, per gene, in the target's base scale (raw counts per cell for orig and
     linear; CP10k for l1): cell-weighted box pools (k = 2, 4, 8); local-level Kalman filters whose observation
     noise is KNOWN from the bin stats, R = (max(s^2, running mean) + 1/n) / (n - 1) (population sd, so Var(mean) =
     s^2/(n-1)), with signal variance either theta^2 (zbar^2 + zbar/50) (theta = 0.1, 0.3) or empirical Bayes
     (moment estimate from past first differences minus their known noise, shrunk to its expression-class median).
     Features: reversion moves T(today -> denoised today), and the denoised last change (one-bin momentum).
  2. Own history: last three observed changes y_{i-1..i-3}.
  3. SNR gate: log relative standard error of today's mean, nonzero fraction, and lag-1 x SNR, reversion x SNR.
  4. Cross-gene denoiser: today's log1p(CP10k) projected on the rank-3 / rank-8 SVD subspace of bins <= i
     (cell-count weights n/(n+50), gene-centred), mapped back to the base scale; feature T(today -> projection).
  5. orig only, finite-cell observation head: expected clipped target when bin i+1 is n_i cells drawn from the box-4
     law or the Kalman(0.2) law (dist_bridge.expected_target), P(all zero) of the box-4 law, zero-today indicator.
  6. optional: GRN (bin-i GRNBoost2 graph, importance-weighted, row-normalised: regulators' last change and
     reversion move, has-regulator) and the OT/SB increment (ot_feature.py).
Learner: features z-scored within each transition (uses only that transition's own features), coefficients by 5
expression classes of m_i (0, <0.03, <0.3, <3, >=3 counts/cell), ridge fitted on past transitions j = 1..i-1 with
the target standardised per transition (each past transition counts equally, as Pearson does) and recency weights
0.5 ** ((i-1-j)/h); h in {0.5,1,2,4,inf}, ridge in {1e-3, 1e-1}, small-bin down-weighting on/off, chosen at every
transition by rolling origin on past transitions only (J = Pearson + tie-aware P@200 up + down, all genes, window
8; i = 2 uses the default; earlier transitions use the fallback below). Labels of transition j need bins j+1 <= i.
Fallback (identical for every model): i = 0 -> finite-cell expectation of bin 0's own law (orig, l1) or 0 (linear);
i = 1 -> orig: box-4 + observation head, l1/linear: repeat last change.
Comparators in this script, same masks: trend (repeat last change), level (orig: box-4 + observation head, which
reproduces DistBridge; l1/linear: box-4 reversion), regime (lags + box-4/8 reversion by class: the variance analyst's
recency-weighted filter), plus ablations.
Scoring: 20 fixed random 25 % gene masks per transition (fresh per transition, seed 20261022, the same for every
model), Pearson and P@200 up/down (their np.argsort) plus tie-aware P@200; mean over transitions, mean and sd over
masks. Ranges: all (nanmean incl. i = 0, the CONTEXT DistBridge convention), i>=2, i>=3, i>=2 without transitions
touching a bin < 100 cells.
Output: results.json, nass.log.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import sparse, stats

import dsdata as D
from dsdata import DB

HS = (0.5, 1.0, 2.0, 4.0, np.inf)
LAMS = (1e-3, 1e-1)
NWS = (0, 1)
CONFIGS = [(h, lam, nw) for h in HS for lam in LAMS for nw in NWS]
DEFAULT = CONFIGS.index((1.0, 1e-3, 0))
WINDOW = 8
NU_EB = 4.0
N_CLS = 5


def level_class(m):
    lv = np.where(m > 0, np.log10(np.where(m > 0, m, 1)), -9)
    return np.digitize(lv, [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])


def zs(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


# ------------------------------------------------------------------ denoisers (causal)
class Base:
    """Base-scale levels Z (G, B), known observation variance R, running means, for one target family."""

    def __init__(self, d, tg):
        self.tg = tg
        if tg == "l1":
            self.Z, self.kap = d.CP, 1e4 / d.lib
        else:
            self.Z, self.kap = d.m, np.ones(d.B)
        n = d.n
        w = np.cumsum(n)
        self.zbar = np.cumsum(self.Z * n[None, :], 1) / w[None, :]          # strict running mean (bins <= b)
        self.mbar = np.cumsum(d.m * n[None, :], 1) / w[None, :]
        k = self.kap[None, :]
        self.R = (np.maximum(k ** 2 * d.s ** 2, k * self.zbar) + k ** 2 / n[None, :]) / np.maximum(n - 1, 1)[None, :]
        self.n = n

    def box(self, b, k):
        lo = max(0, b - k + 1)
        w = self.n[lo:b + 1]
        return (self.Z[:, lo:b + 1] * w).sum(1) / w.sum()

    def q_fixed(self, theta):
        return lambda b: theta ** 2 * (self.zbar[:, b] ** 2 + self.kap[b] * self.zbar[:, b] / 50)

    def q_eb(self):
        Z, R = self.Z, self.R
        dz2 = np.diff(Z, axis=1) ** 2 - R[:, :-1] - R[:, 1:]                    # excess of each past difference
        cum = np.cumsum(dz2, 1)

        def q(b):                                                              # uses differences k < b (bins <= b)
            nd = b
            raw = cum[:, b - 1] / nd
            scale = self.zbar[:, b] ** 2 + self.kap[b] * self.zbar[:, b] / 50 + 1e-12
            rel = raw / scale
            cls = level_class(self.mbar[:, b])
            prior_rel = np.zeros_like(rel)
            for c in range(N_CLS):
                sel = cls == c
                if sel.any():
                    prior_rel[sel] = max(float(np.median(rel[sel])), 1e-4)
            prior = prior_rel * scale
            return np.maximum((nd * raw + NU_EB * prior) / (nd + NU_EB), 0.05 * prior)
        return q

    def kalman(self, qfun):
        Z, R = self.Z, self.R
        G, B = Z.shape
        out = np.zeros((G, B))
        x, P = Z[:, 0].copy(), R[:, 0].copy()
        out[:, 0] = x
        for b in range(1, B):
            Pm = P + qfun(b)
            S = Pm + R[:, b]
            K = np.where(S > 0, Pm / np.where(S > 0, S, 1), 1.0)
            x = x + K * (Z[:, b] - x)
            P = (1 - K) * Pm
            out[:, b] = x
        return out


def svd_project(d, i, k):
    """Today's log1p(CP10k) projected on the rank-k subspace of bins <= i (returns None at full rank)."""
    Lb = d.L[:, :i + 1]
    w = d.n[:i + 1] / (d.n[:i + 1] + 50.0)
    mu = (Lb * w).sum(1) / w.sum()
    U, s, _ = np.linalg.svd((Lb - mu[:, None]) * np.sqrt(w)[None, :], full_matrices=False)
    rank = int((s > 1e-8 * s[0]).sum()) if s[0] > 0 else 0
    if k >= rank:
        return None
    Uk = U[:, :k]
    return mu + Uk @ (Uk.T @ (d.L[:, i] - mu))


def l1_fn(m_now, Lh):
    Lt = max(float(np.maximum(m_now, 0).sum()), 1e-12)
    return lambda m0, s: (np.log1p(1e4 * s / Lh) - np.log1p(1e4 * np.maximum(m0, 0) / Lt)).astype(np.float32)


# ------------------------------------------------------------------ features
GROUPS = {
    "lag": ["lag1", "lag2", "lag3"],
    "rev": ["rev_box2", "rev_box4", "rev_box8"],
    "kal": ["rev_kal1", "rev_kal3", "rev_eb", "mom_eb"],
    "snr": ["logse", "frac", "lag1_x_se", "rev4_x_se"],
    "cg": ["rev_svd3", "rev_svd8"],
    "obs": ["obs_box4", "obs_kal", "p0", "zero_now"],
    "grn": ["grn_lag1", "grn_rev4", "has_reg"],
    "grn_null": ["grnnull_lag1", "grnnull_rev4"],
    "ot": ["ot_inc"],
}


def level_null(u, lvl, rng):
    """Regulators permuted among regulators of the same expression decile (the graph analyst's level-matched null,
    re-implemented): keeps each target's in-degree, importances and the level profile of its regulators."""
    regs = np.unique(u)
    dec = np.searchsorted(np.quantile(lvl[regs], np.linspace(0.1, 0.9, 9)), lvl[regs], side="right")
    mapping = np.arange(len(lvl))
    for k in np.unique(dec):
        grp = regs[dec == k]
        mapping[grp] = rng.permutation(grp)
    return mapping[u]


def _rownorm(A):
    s = np.asarray(A.sum(1)).ravel()
    return sparse.diags(np.where(s > 0, 1.0 / np.where(s > 0, s, 1), 0.0)) @ A, (s > 0).astype(float)


def load_graph(d):
    """Per snapshot bin b: (row-normalised W[target, regulator] of the bin-b GRNBoost2 edges, has-regulator flag,
    level-matched null W)."""
    z = np.load(os.path.join(D.EDGE_CACHE, f"{d.name}_edges.npz"))
    u, i, ts, w = z["u"], z["i"], z["ts"], z["w"]
    order = np.argsort(ts, kind="stable")
    ts_s = ts[order]
    rng = np.random.default_rng(13)
    out = {}
    for j, b in enumerate(d.bins):
        lo, hi = np.searchsorted(ts_s, b), np.searchsorted(ts_s, b, side="right")
        sel = order[lo:hi]
        ww = w[sel].astype(np.float64)
        W, has = _rownorm(sparse.csr_matrix((ww, (i[sel], u[sel])), shape=(d.G, d.G)))
        un = level_null(u[sel], d.L[:, j], rng)
        Wn, _ = _rownorm(sparse.csr_matrix((ww, (i[sel], un)), shape=(d.G, d.G)))
        out[b] = (W, has, Wn)
    return out


def build_features(d, tg, Y, graph, ot):
    """Dict name -> (T, G) raw features (strict: row i uses bins <= i) and the per-transition classes."""
    bs = Base(d, tg)
    kal1 = bs.kalman(bs.q_fixed(0.1))
    kal3 = bs.kalman(bs.q_fixed(0.3))
    keb = bs.kalman(bs.q_eb())
    T, G = d.T, d.G
    F = {k: np.zeros((T, G)) for g in GROUPS.values() for k in g}
    if tg == "orig":
        trip = [DB.triplet(d.X[b]) for b in range(d.B)]
        box4law = DB.box_pool(trip, list(d.n), 4)
        kallaw = DB.kalman_pool(trip, list(d.n), 0.2)
    key = 1 + (d.name == "pancreas-human")
    for i in range(T):
        Zi = bs.Z[:, i]
        for k in (1, 2, 3):
            if i - k >= 0:
                F[f"lag{k}"][i] = Y[i - k]
        for k in (2, 4, 8):
            F[f"rev_box{k}"][i] = D.tfun(tg, Zi, bs.box(i, k))
        F["rev_kal1"][i] = D.tfun(tg, Zi, kal1[:, i])
        F["rev_kal3"][i] = D.tfun(tg, Zi, kal3[:, i])
        F["rev_eb"][i] = D.tfun(tg, Zi, keb[:, i])
        if i >= 1:
            F["mom_eb"][i] = D.tfun(tg, keb[:, i - 1], keb[:, i])
        se = np.log(np.sqrt(bs.R[:, i]) / (Zi + bs.kap[i] / d.n[i]) + 1e-6)
        F["logse"][i] = se
        F["frac"][i] = d.f[:, i]
        F["lag1_x_se"][i] = zs(F["lag1"][i]) * zs(se)
        F["rev4_x_se"][i] = zs(F["rev_box4"][i]) * zs(se)
        for k in (3, 8):
            Lh = svd_project(d, i, k)
            if Lh is not None:
                c = np.maximum(np.expm1(Lh), 0)
                Zh = c if tg == "l1" else c * d.lib[i] / 1e4
                F[f"rev_svd{k}"][i] = D.tfun(tg, Zi, Zh)
        if tg == "orig":
            m_now = d.m[:, i]
            F["obs_box4"][i] = DB.expected_target(box4law[i], m_now, d.n[i], np.random.default_rng([key, i, 0]))
            F["obs_kal"][i] = DB.expected_target(kallaw[i], m_now, d.n[i], np.random.default_rng([key, i, 1]))
            F["p0"][i] = (1 - np.clip(box4law[i][:, 1], 0, 1)) ** max(int(round(d.n[i])), 1)
            F["zero_now"][i] = (m_now == 0).astype(float)
        if graph is not None:
            W, has, Wn = graph[d.bins[i]]
            F["grn_lag1"][i] = W @ F["lag1"][i]
            F["grn_rev4"][i] = W @ F["rev_box4"][i]
            F["has_reg"][i] = has
            F["grnnull_lag1"][i] = Wn @ F["lag1"][i]
            F["grnnull_rev4"][i] = Wn @ F["rev_box4"][i]
        if ot is not None and tg in ("orig", "l1"):
            F["ot_inc"][i] = ot[f"inc_{tg}"][i]
    cls = np.stack([level_class(d.m[:, i]) for i in range(T)])
    return F, cls


def fallback(d, tg, Y, F):
    """Predictions for i = 0 and i = 1 shared by every model."""
    out = {}
    if tg == "orig":
        out[0], out[1] = F["obs_box4"][0].copy(), F["obs_box4"][1].copy()
    else:
        if tg == "l1":
            H0 = DB.triplet(d.X[0])
            out[0] = DB.expected_value(H0, d.m[:, 0], d.n[0], np.random.default_rng([9, 0]),
                                       l1_fn(d.m[:, 0], max(float(H0[:, 0].sum()), 1e-12))).astype(np.float64)
        else:
            out[0] = np.zeros(d.G)
        out[1] = Y[0].copy()
    return out


# ------------------------------------------------------------------ learner
class Stacker:
    """Per-transition Gram matrices of the class-interacted design, so any feature subset / weighting is cheap."""

    def __init__(self, F, cls, Y, names, standardise=True):
        self.names = names
        T, G = Y.shape
        self.T, self.G = T, G
        self.X, self.Gm, self.bv = [], [], []
        for i in range(T):
            raw = np.column_stack([F[k][i] for k in names])
            Fi = np.column_stack([zs(c) for c in raw.T]) if standardise else raw
            X = np.hstack([Fi * (cls[i] == c)[:, None] for c in range(N_CLS)] +
                          [(cls[i] == c)[:, None].astype(float) for c in range(N_CLS)])
            y = Y[i]
            y = zs(y) if standardise else y
            self.X.append(X.astype(np.float32))
            self.Gm.append(X.T @ X)
            self.bv.append(X.T @ y)
        p = len(names)
        self.col = {k: [c * p + j for c in range(N_CLS)] for j, k in enumerate(names)}
        self.dummies = list(range(N_CLS * p, N_CLS * p + N_CLS))

    def cols(self, subset):
        return sorted(sum((self.col[k] for k in subset), []) + self.dummies)

    def predict_all(self, i, subset, n_cells, Gs, bs):
        """Predictions of transition i for every config in CONFIGS (columns), trained on transitions 1..i-1."""
        idx = self.cols(subset)
        betas = []
        for (h, lam, nw) in CONFIGS:
            j = np.arange(1, i)
            w = np.ones(len(j)) if not np.isfinite(h) else 0.5 ** ((i - 1 - j) / h)
            if nw:
                nm = np.minimum(n_cells[j], n_cells[j + 1])
                w = w * nm / (nm + 100.0)
            A = np.tensordot(w, Gs[1:i], 1)
            A = A + lam * max(float(np.mean(np.diag(A))), 1e-12) * np.eye(len(idx))   # scale-free ridge
            b = np.tensordot(w, bs[1:i], 1)
            betas.append(np.linalg.solve(A, b))
        return self.X[i][:, idx].astype(np.float64) @ np.column_stack(betas)

    def grams(self, subset):
        idx = self.cols(subset)
        return np.stack([g[np.ix_(idx, idx)] for g in self.Gm]), np.stack([b[idx] for b in self.bv])


def run_learned(st, subsets, Y, d, fb, extra=None, cfg_idx=None):
    """Rolling-origin over (subset, config). subsets: list of feature-name lists (several = model selection too);
    extra: optional fixed strict predictions (lists over transitions, e.g. the trend) that compete in the selection;
    cfg_idx: restrict the config grid (a single index = fixed hyperparameters, no selection)."""
    T = d.T
    cfg_idx = list(range(len(CONFIGS))) if cfg_idx is None else list(cfg_idx)
    cand = [(s, c) for s in range(len(subsets)) for c in cfg_idx]
    P = {k: [None] * T for k in cand}
    J = {k: np.full(T, np.nan) for k in cand}
    for e, pe in enumerate(extra or []):
        k = (-1 - e, DEFAULT)
        cand.append(k)
        P[k] = pe
        J[k] = np.array([D.criterion(Y[i], pe[i], d.n_test) if i >= 2 else np.nan for i in range(T)])
    for s, sub in enumerate(subsets):
        Gs, bs = st.grams(sub)
        for i in range(2, T):
            Pi = st.predict_all(i, sub, d.n, Gs, bs)
            for c in cfg_idx:
                P[(s, c)][i] = Pi[:, c]
                J[(s, c)][i] = D.criterion(Y[i], Pi[:, c], d.n_test)
    pred, chosen = [fb[0], fb[1]], [None, None]
    default = (0, DEFAULT if DEFAULT in cfg_idx else cfg_idx[0])
    for i in range(2, T):
        if i == 2:
            k = default
        else:
            lo = max(2, i - WINDOW)
            scores = [np.mean(J[k][lo:i]) for k in cand]
            k = cand[int(np.argmax(scores))]
        pred.append(P[k][i])
        chosen.append([int(k[0]), str(CONFIGS[k[1]][0]), CONFIGS[k[1]][1], CONFIGS[k[1]][2]])
    return pred, chosen


def partial_corr(y, x, z):
    D_ = np.column_stack([np.ones(len(y)), z])
    ry = y - D_ @ np.linalg.lstsq(D_, y, rcond=None)[0]
    rx = x - D_ @ np.linalg.lstsq(D_, x, rcond=None)[0]
    den = np.sqrt((ry ** 2).sum() * (rx ** 2).sum())
    return float((ry * rx).sum() / den) if den > 0 else np.nan


def paired(Sa, Sb, idx):
    """Per-transition (mask-averaged) PCC and J = PCC + up + down, a - b, Wilcoxon over transitions idx."""
    out = {}
    a, b = np.nanmean(Sa[:, idx], 0), np.nanmean(Sb[:, idx], 0)
    for name, va, vb in (("pcc", a[:, 0], b[:, 0]), ("J", a[:, :3].sum(1), b[:, :3].sum(1))):
        dlt = np.nan_to_num(va) - np.nan_to_num(vb)
        nz = dlt[np.abs(dlt) > 1e-12]
        out[name] = {"mean_delta": round(float(dlt.mean()), 4), "better": int((dlt > 1e-12).sum()),
                     "worse": int((dlt < -1e-12).sum()),
                     "wilcoxon_p": round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None}
    return out


def main(names):
    res = {}
    log = open(os.path.join(D.OUT, "nass.log"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    say(f"==== run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for name in names:
        t0 = time.time()
        d = D.Data(name)
        graph = load_graph(d)
        otf = os.path.join(D.OUT, f"_ot_{name}.npz")
        ot = dict(np.load(otf)) if os.path.exists(otf) else None
        mk = D.masks(d.G, d.T)
        ranges = {"all": list(range(d.T)), "i>=2": list(range(2, d.T)), "i>=3": list(range(3, d.T)),
                  "i>=2_bins>=100": [i for i in range(2, d.T) if min(d.n[i], d.n[i + 1]) >= 100]}
        res[name] = {"n_cells": d.n.round().tolist(), "n_test": d.n_test, "ranges": {k: v for k, v in ranges.items()},
                     "ot_feature_present": ot is not None, "targets": {}}
        for tg in D.TARGETS:
            Y = d.Y(tg)
            F, cls = build_features(d, tg, Y, graph, ot)
            fb = fallback(d, tg, Y, F)
            G_ = GROUPS
            full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"] + (G_["obs"] if tg == "orig" else [])
            allnames = full + G_["grn"] + G_["grn_null"] + (G_["ot"] if (ot is not None and tg != "linear") else [])
            st = Stacker(F, cls, Y, allnames)
            regime = G_["lag"] + ["rev_box4", "rev_box8"]
            st_raw = Stacker(F, cls, Y, regime, standardise=False)
            minus = lambda grp: [k for k in full if k not in G_[grp]]
            variants = {
                "regime": [regime],
                "full": [full],
                "full-cg": [minus("cg")],
                "full-snr": [minus("snr")],
                "full-kal": [minus("kal")],
                "full+grn": [full + G_["grn"]],
                "full+grn_null": [full + G_["grn_null"] + ["has_reg"]],
                "auto(full|full-cg)": [full, minus("cg")],
            }
            if tg == "orig":
                variants["full-obs"] = [minus("obs")]
                variants["regime+obs"] = [regime + G_["obs"]]
            if ot is not None and tg != "linear":
                variants["full+ot"] = [full + G_["ot"]]
                variants["full+grn+ot"] = [full + G_["grn"] + G_["ot"]]
            preds, chosen = {}, {}
            # comparators
            preds["trend"] = [fb[0]] + [Y[i - 1] for i in range(1, d.T)]
            if tg == "orig":
                preds["level(box4+obs)"] = [F["obs_box4"][i] for i in range(d.T)]
            else:
                preds["level(box4)"] = [fb[0]] + [F["rev_box4"][i] for i in range(1, d.T)]
            p_raw, c_raw = run_learned(st_raw, [regime], Y, d, fb)
            preds["regime_unstd"], chosen["regime_unstd"] = p_raw, c_raw
            for vname, subs in variants.items():
                preds[vname], chosen[vname] = run_learned(st, subs, Y, d, fb)
            # fixed fast-adapting gain (h = 0.5, ridge 1e-3, no small-bin weighting): NOT selected by rolling origin;
            # h = 0.5 was the best half-life of the variance analyst's regime filter on these same data, so these rows
            # carry a hyperparameter informed by an earlier look at the test period (optimistic, reported separately)
            fx = [CONFIGS.index((0.5, 1e-3, 0))]
            small = [CONFIGS.index((h, 1e-3, 0)) for h in (0.5, 1.0, 2.0)]
            for vname, sub in (("regime[h=0.5 fixed]", regime), ("full[h=0.5 fixed]", full),
                               ("full+grn[h=0.5 fixed]", full + G_["grn"])):
                preds[vname], chosen[vname] = run_learned(st, [sub], Y, d, fb, cfg_idx=fx)
            preds["full[RO h in 0.5,1,2]"], chosen["full[RO h in 0.5,1,2]"] = run_learned(st, [full], Y, d, fb,
                                                                                         cfg_idx=small)
            # the trend competes with the stacker in the same rolling-origin selection (strict meta-selection)
            preds["auto(full|full-cg|trend)"], chosen["auto(full|full-cg|trend)"] =                 run_learned(st, [full, minus("cg")], Y, d, fb, extra=[preds["trend"]])
            S = {k: D.score(Y, v, mk) for k, v in preds.items()}
            tab = {k: {r: D.summarize(S[k], idx) for r, idx in ranges.items()} for k in S}
            per_t = {k: np.round(np.nanmean(S[k], 0)[:, :3], 4).tolist() for k in S}
            pr = {}
            idx2 = ranges["i>=2"]
            for a, b in (("full", "regime"), ("full", "trend"), ("auto(full|full-cg)", "regime"),
                         ("auto(full|full-cg)", "trend"), ("full", "full-cg"), ("full", "full-snr"),
                         ("full", "full-kal"), ("full+grn", "full"), ("full+grn_null", "full"),
                         ("full+grn", "full+grn_null"), ("full+ot", "full"),
                         ("auto(full|full-cg|trend)", "trend"), ("auto(full|full-cg|trend)", "regime"), ("full", "full-obs"),
                         ("regime", "regime_unstd"), ("full", "level(box4+obs)"),
                         ("full[h=0.5 fixed]", "regime[h=0.5 fixed]"), ("full[h=0.5 fixed]", "trend"),
                         ("full+grn[h=0.5 fixed]", "full[h=0.5 fixed]"), ("full[h=0.5 fixed]", "full"),
                         ("full[RO h in 0.5,1,2]", "full"), ("full[h=0.5 fixed]", "level(box4+obs)")):
                if a in S and b in S:
                    pr[f"{a} vs {b}"] = paired(S[a], S[b], idx2)
            # partial correlation of candidate add-on features with the truth, given the full model's prediction
            pc = {}
            for feat in ("ot_inc", "grn_lag1", "grnnull_lag1", "rev_svd3", "rev_svd8"):
                if feat in F and np.any(F[feat]):
                    v = [partial_corr(Y[i], F[feat][i], np.column_stack([preds["full"][i]])) for i in idx2]
                    pc[feat] = {"mean": round(float(np.nanmean(v)), 4), "n_pos": int(np.sum(np.array(v) > 0)),
                                "n": len(v)}
            res[name]["targets"][tg] = {"table": tab, "per_transition_pcc_up_down": per_t, "paired": pr,
                                        "partial_corr_given_full": pc, "chosen": chosen}
            say(f"== {name} {tg}  ({time.time() - t0:.0f}s)")
            for k in S:
                r2, ra = tab[k]["i>=2"], tab[k]["all"]
                rb = tab[k]["i>=2_bins>=100"]
                say(f"  {k:22s} i>=2 {r2['pcc']:.3f}/{r2['p200_up']:.3f}/{r2['p200_down']:.3f} (sd {r2['pcc_sd']:.3f})"
                    f" | all {ra['pcc']:.3f}/{ra['p200_up']:.3f}/{ra['p200_down']:.3f}"
                    f" | i>=2 big {rb['pcc']:.3f}/{rb['p200_up']:.3f}/{rb['p200_down']:.3f}")
            for k, v in pr.items():
                say(f"  paired {k:34s} dPCC {v['pcc']['mean_delta']:+.3f} ({v['pcc']['better']}/{v['pcc']['worse']},"
                    f" p={v['pcc']['wilcoxon_p']}) dJ {v['J']['mean_delta']:+.3f} ({v['J']['better']}/{v['J']['worse']},"
                    f" p={v['J']['wilcoxon_p']})")
            for k, v in pc.items():
                say(f"  partial corr | full: {k:10s} {v['mean']:+.4f} ({v['n_pos']}/{v['n']} > 0)")
        out = os.path.join(D.OUT, "results.json")
        prev = json.load(open(out)) if os.path.exists(out) else {}
        prev.update(res)
        json.dump(prev, open(out, "w"), indent=1)
    log.close()


if __name__ == "__main__":
    main(sys.argv[1:] or list(D.SNAPS))
