"""C13 (parts A, B): low-rank structure of the l1 change and factor-space vs per-gene local-linear-trend filters.

A. Share of y_i in the top-3 left-singular subspace of the past change matrix [y_0 .. y_{i-1}] (G x i, uncentred;
   also gene-centred columns): ||U3' y_i||^2 / ||y_i||^2. For i <= 3 the past span has rank <= 3, so the top-3
   share equals the full-past-span share (trivial); reported as NaN there and the mean is over i >= 4.
   Also the full-span share (all i past changes), as reference; chance level for a random vector = rank / G.
B. Strict filters on L = log1p(CP10k), forecast y_i = E[L_{i+1} | bins <= i] - L_i:
   - pergene_LL  : per-gene local level (slope fixed 0), known noise R;
   - pergene_LLT : per-gene local linear trend (level + slope), known noise R;
   - svdkf_chg_k : LLT run on the k factor scores a_b = U' L_b (b <= i), U = top-k left singular vectors of the past
                   change matrix [y_0..y_{i-1}]; score noise = sum_g U_gk^2 R_gb (diagonal); forecast U (l + b - a_i);
   - svdkf_lvl_k : same with U from the cell-weighted gene-centred SVD of L over bins <= i (NASS svd_project basis).
   R = delta-method noise of L (plain s^2/(n-1), the definition that reproduces the CONTEXT noise budget) plus a
   one-count floor kap^2 / (n (n-1)) / (1 + CP)^2 so zero-count genes are not treated as exact.
   Process noise: q_level = a v, q_slope = b v, v = empirical-Bayes excess variance of past changes (mean over past
   transitions of y^2 - R_j - R_{j+1}, shrunk to its expression-class median with 4 pseudo-transitions, floored at
   5 % of the prior); for factors v_k = max(mean(c^2 - Rs_j - Rs_{j+1}), 0.05 mean(c^2)) on past score changes c.
   Grid a in {0.3, 1}, b in {0, 0.03, 0.1, 0.3, 1} (LL: b = 0 only); k in {3, 8}. Config chosen at every transition
   by rolling origin: mean J (Pearson + tie-aware P@K up + down, K = 200 N / n_test, all genes) over past
   transitions max(1, i-8) .. i-1 (their labels use bins <= i); i = 1 uses the default (a = 1, b = 0.1).
   'svdkf_auto': k and basis also selected by the same rolling origin.
Scoring: CONTEXT masks (default_rng(1000 s + i), 3 seeds), Pearson / tie-aware P@200 up / down, mean over i >= 1 and
i >= 2; plus NASS's 20 masks (seed 20261022) at i >= 2 for comparison with NASS rows. Forecasts saved to
c13_forecasts_<name>.npz for the NASS ablation (c13_nass_ablation.py).
"""
import json
import os
import sys

import numpy as np

import common as C

sys.path.insert(0, C.DBS)
import dsdata as D  # noqa: E402  (numpy; metrics, masks, criterion)

A_GRID = (0.3, 1.0)
B_GRID = (0.0, 0.03, 0.1, 0.3, 1.0)
CFG = [(a, b) for a in A_GRID for b in B_GRID]
DEFAULT = CFG.index((1.0, 0.1))
WINDOW = 8
NU = 4.0


def lvl_class(cp):
    c = np.full(len(cp), 4)
    c[cp <= 10] = 3
    c[cp <= 1] = 2
    c[cp <= 0.1] = 1
    c[cp <= 0] = 0
    return c


def llt(Z, R, ql, qb, p0b):
    """Local linear trend Kalman filter, vectorised over rows. Z, R: (G, b+1). Returns filtered level, slope at last bin."""
    G = Z.shape[0]
    lv, sl = Z[:, 0].copy(), np.zeros(G)
    Pll, Plb, Pbb = R[:, 0].copy(), np.zeros(G), p0b.copy()
    for t in range(1, Z.shape[1]):
        lv = lv + sl
        Pll_ = Pll + 2 * Plb + Pbb + ql
        Plb_ = Plb + Pbb
        Pbb_ = Pbb + qb
        S = Pll_ + R[:, t]
        Kl, Kb = Pll_ / S, Plb_ / S
        e = Z[:, t] - lv
        lv, sl = lv + Kl * e, sl + Kb * e
        Pll, Plb, Pbb = (1 - Kl) * Pll_, (1 - Kl) * Plb_, Pbb_ - Kb * Plb_
    return lv, sl


def eb_v(Yp, Rs, cls):
    nd = Yp.shape[1]
    raw = (Yp ** 2 - Rs).mean(1)
    prior = np.zeros(len(raw))
    for c in np.unique(cls):
        sel = cls == c
        prior[sel] = max(float(np.median(raw[sel])), 1e-6)
    return np.maximum((nd * raw + NU * prior) / (nd + NU), 0.05 * prior)


def lvl_basis(d, i, k):
    Lb = d.L[:, :i + 1]
    w = d.n[:i + 1] / (d.n[:i + 1] + 50.0)
    mu = (Lb * w).sum(1) / w.sum()
    U, s, _ = np.linalg.svd((Lb - mu[:, None]) * np.sqrt(w)[None, :], full_matrices=False)
    r = int((s > 1e-8 * s[0]).sum())
    return U[:, :min(k, r)]


def chg_basis(Y, i, k):
    U, s, _ = np.linalg.svd(Y[:i].T, full_matrices=False)
    r = int((s > 1e-8 * s[0]).sum())
    return U[:, :min(k, r)]


def part_a(d, Y):
    out = {"share3": [], "share3_centred": [], "share_full": [], "rank_full": []}
    for i in range(1, d.T):
        Ui = chg_basis(Y, i, i)
        y = Y[i]
        full = float(((Ui.T @ y) ** 2).sum() / (y ** 2).sum())
        if i >= 4:
            U3 = Ui[:, :3]
            s3 = float(((U3.T @ y) ** 2).sum() / (y ** 2).sum())
            Yc = Y[:i] - Y[:i].mean(1, keepdims=True)
            Uc, sc, _ = np.linalg.svd(Yc.T, full_matrices=False)
            yc = y - y.mean()
            s3c = float(((Uc[:, :3].T @ yc) ** 2).sum() / (yc ** 2).sum())
        else:
            s3 = s3c = float("nan")
        out["share3"].append(s3)
        out["share3_centred"].append(s3c)
        out["share_full"].append(full)
        out["rank_full"].append(Ui.shape[1])
    return out


def forecasts(d, Y, R):
    """Dict model -> list over transitions of per-config forecast arrays (G, n_cfg); i = 0 -> None."""
    G, T = d.G, d.T
    models = ["pergene_LLT"] + [f"svdkf_{b}_{k}" for b in ("chg", "lvl") for k in (3, 8)]
    P = {m: [None] * T for m in models}
    for i in range(1, T):
        Rs = R[:, :i] + R[:, 1:i + 1]
        v = eb_v(Y[:i].T, Rs.T if False else Rs, lvl_class(d.CP[:, i]))
        Z, Ri = d.L[:, :i + 1], R[:, :i + 1]
        cols = []
        for a, b in CFG:
            lv, sl = llt(Z, Ri, a * v, b * v, v)
            cols.append(lv + sl - Z[:, i])
        P["pergene_LLT"][i] = np.column_stack(cols)
        for basis in ("chg", "lvl"):
            for k in (3, 8):
                U = chg_basis(Y, i, k) if basis == "chg" else lvl_basis(d, i, k)
                Zs = U.T @ Z
                Rsc = (U ** 2).T @ Ri
                c = np.diff(Zs, axis=1)
                ex = (c ** 2 - Rsc[:, :-1] - Rsc[:, 1:]).mean(1)
                vk = np.maximum(ex, 0.05 * (c ** 2).mean(1))
                cols = []
                for a, b in CFG:
                    lv, sl = llt(Zs, Rsc, a * vk, b * vk, vk)
                    cols.append(U @ (lv + sl - Zs[:, i]))
                P[f"svdkf_{basis}_{k}"][i] = np.column_stack(cols)
    return P


def ro_select(Y, cand, n_test, T):
    """cand: dict key -> list over i of forecast (G,) (None at i = 0). Rolling-origin selection by mean J."""
    keys = list(cand)
    J = {k: np.array([D.criterion(Y[i], cand[k][i], n_test) if cand[k][i] is not None else np.nan
                      for i in range(T)]) for k in keys}
    pred, chosen = [np.zeros(Y.shape[1])], [None]
    for i in range(1, T):
        if i == 1:
            k = keys[0]
        else:
            lo = max(1, i - WINDOW)
            k = keys[int(np.argmax([np.nanmean(J[k][lo:i]) for k in keys]))]
        pred.append(cand[k][i])
        chosen.append(k)
    return pred, chosen


def score_ctx(Y, P, mk):
    T = len(Y)
    rows = np.full((3, T, 3), np.nan)
    for s in range(3):
        for i in range(1, T):
            t = mk[s, i]
            m = C.dileo_rescore.metrics(Y[i][t], P[i][t])
            rows[s, i] = (m["pearson"], m["p_ta_up"], m["p_ta_down"])
    return rows


def summ(rows, lo):
    v = np.nanmean(rows[:, lo:], 1).mean(0)
    return [round(float(x), 4) for x in v]


def main():
    res = {}
    for name in C.NAMES:
        d = C.Data(name)
        Y = d.Y("l1")
        T = d.T
        out = {"partA": part_a(d, Y)}
        A = out["partA"]
        out["partA_mean_share3_i>=4"] = round(float(np.nanmean(A["share3"])), 4)
        out["partA_mean_share3c_i>=4"] = round(float(np.nanmean(A["share3_centred"])), 4)
        out["partA_mean_sharefull_i>=4"] = round(float(np.mean(A["share_full"][3:])), 4)
        print(f"== {name} A: share3 per i>=1 {np.round(A['share3'], 3).tolist()}")
        print(f"   mean share3 (i>=4) {out['partA_mean_share3_i>=4']} centred {out['partA_mean_share3c_i>=4']} "
              f"full-past-span (i>=4) {out['partA_mean_sharefull_i>=4']}")
        R = d.R_L(False) + d.kap[None, :] ** 2 / (d.n * np.maximum(d.n - 1, 1))[None, :] / (1 + d.CP) ** 2
        F = forecasts(d, Y, R)
        cand_models = {}
        # per-gene LL: b = 0 configs only; LLT: all configs; factor models: all configs
        cand_models["pergene_LL"] = {CFG[c]: [None] + [F["pergene_LLT"][i][:, c] for i in range(1, T)]
                                     for c in range(len(CFG)) if CFG[c][1] == 0.0}
        for m in F:
            cand_models[m] = {CFG[c]: [None] + [F[m][i][:, c] for i in range(1, T)] for c in range(len(CFG))}
        preds, chosen = {}, {}
        for m, cand in cand_models.items():
            # put the default first so i = 1 uses it
            keys = sorted(cand, key=lambda k: k != CFG[DEFAULT] if CFG[DEFAULT] in cand else k != (1.0, 0.0))
            preds[m], chosen[m] = ro_select(Y, {k: cand[k] for k in keys}, d.n_test, T)
        auto = {}
        for m in [x for x in F if x.startswith("svdkf")]:
            for k in cand_models[m]:
                auto[(m, k)] = cand_models[m][k]
        keys = sorted(auto, key=lambda k: k != ("svdkf_chg_3", CFG[DEFAULT]))
        preds["svdkf_auto"], chosen["svdkf_auto"] = ro_select(Y, {k: auto[k] for k in keys}, d.n_test, T)
        preds["trend"] = [np.zeros(d.G)] + [Y[i - 1] for i in range(1, T)]
        mk = d.masks()
        nmk = D.masks(d.G, T)
        tab = {}
        for m, P in preds.items():
            rows = score_ctx(Y, P, mk)
            Sn = D.score(Y, [None] + P[1:], nmk)
            tab[m] = {"ctx_i>=1": summ(rows, 1), "ctx_i>=2": summ(rows, 2),
                      "nass_masks_i>=2_pcc": round(float(np.mean(np.nanmean(Sn[:, 2:, 0], 1))), 4),
                      "per_t_pcc": np.round(np.nanmean(rows[:, :, 0], 0), 3).tolist()}
            print(f"  {m:14s} ctx i>=1 {tab[m]['ctx_i>=1']}  i>=2 {tab[m]['ctx_i>=2']}  nass-masks i>=2 PCC {tab[m]['nass_masks_i>=2_pcc']}")
        out["filters"] = tab
        out["chosen"] = {m: [str(c) for c in v] for m, v in chosen.items()}
        res[name] = out
        np.savez(os.path.join(C.OUT, f"c13_forecasts_{name}.npz"),
                 **{m: np.stack([np.zeros(d.G) if p is None else p for p in P]) for m, P in preds.items()})
        json.dump(res, open(os.path.join(C.OUT, "c13_factor.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
