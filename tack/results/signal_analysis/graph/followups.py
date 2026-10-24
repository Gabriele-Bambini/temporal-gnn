"""Follow-ups on what a strict forecaster can latch onto (all features from bins <= a; rolling-origin fits).

F1. Lag-1 persistence by expression quintile: per transition, corr(y_prev, y) inside each quintile of
    log1p(CP10k_a) among expressed genes (and zero_a), averaged over transitions; plus the sampling-noise share of
    the bin-a level, r_a = (s_a^2 / n_a) / (s_a^2 / n_a + m_a^2) (a per-gene noise-to-signal proxy).
F2. Noise-aware weighting: strict rolling-origin ridge (as grn_signal.analysis_B, z-scored per transition, fitted
    on transitions < i) with own features vs own + interactions of the last change and of the reversion move with
    the gene's log level and its log standard error se_a = sqrt(s_a^2 / n_a) / (m_a + 1/n_a) (the noise of the bin-a
    mean relative to its size). Tests whether letting the trend / reversion weight depend on noise pays.
F3. Library factor for the linear target: features m_a * sign(last log library change) and m_a * last log library
    change (strict), added to own.
Scored with their metrics on 25 % test masks (5 draws).
Output: followups.json.
"""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge

import common

DRAWS = 5


def zscore(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0.0


def cp10k(m):
    return 1e4 * m / max(m.sum(), 1e-12)


def main():
    out = {}
    for name in common.SNAPS:
        d = common.load_node(name)
        N = d["m"].shape[0]
        snaps = common.SNAPS[name]
        trans = common.transitions(name)
        n, lib = d["n"], d["m"].sum(0)
        res = {}
        for T in common.TARGETS:
            Y = [common.target(T, d["m"][:, a], d["m"][:, b]) for a, b in trans]
            F, groups = {}, {}
            for ti in range(1, len(trans)):
                a = trans[ti][0]
                hb = [s for s in snaps if s <= a]
                p = hb[-2]
                lo = max(0, len(hb) - 4)
                w = n[hb[lo:]]
                pool4 = (d["m"][:, hb[lo:]] * w).sum(1) / w.sum()
                m, s = d["m"][:, a], d["s"][:, a]
                lvl = np.log1p(cp10k(m))
                se = np.log(np.sqrt(s ** 2 / n[a]) / (m + 1.0 / n[a]) + 1e-6)
                o_last, o_rev = Y[ti - 1], common.target(T, m, pool4)
                llr = np.log(lib[a] / lib[p])
                F[ti] = {"o_last": o_last, "o_rev": o_rev, "o_lvl": lvl, "o_f": d["f"][:, a],
                         "o_zero": (m == 0).astype(float),
                         "x_last_lvl": zscore(o_last) * zscore(lvl), "x_rev_lvl": zscore(o_rev) * zscore(lvl),
                         "x_last_se": zscore(o_last) * zscore(se), "x_rev_se": zscore(o_rev) * zscore(se), "se": se,
                         "lib_sign_m": m * np.sign(llr), "lib_llr_m": m * llr}
                ex = m > 0
                q = np.quantile(lvl[ex], [0.2, 0.4, 0.6, 0.8])
                g = np.full(N, -1); g[ex] = np.searchsorted(q, lvl[ex], side="right")
                groups[ti] = g
            # F1
            per_g = {}
            for ti in F:
                for k in range(-1, 5):
                    sel = groups[ti] == k
                    lab = "zero_a" if k < 0 else f"Q{k + 1}"
                    if sel.sum() > 50 and np.std(Y[ti][sel]) > 0 and np.std(F[ti]["o_last"][sel]) > 0:
                        per_g.setdefault(lab, []).append(common.pearson(Y[ti][sel], F[ti]["o_last"][sel]))
            f1 = {lab: float(np.mean(v)) for lab, v in per_g.items()}
            # noise share of the level by quintile (pooled)
            ns = {}
            for ti in F:
                a = trans[ti][0]
                m, s = d["m"][:, a], d["s"][:, a]
                r = (s ** 2 / n[a]) / (s ** 2 / n[a] + m ** 2 + 1e-12)
                for k in range(0, 5):
                    sel = groups[ti] == k
                    ns.setdefault(f"Q{k + 1}", []).append(float(np.median(r[sel])))
            f1_noise = {k: float(np.mean(v)) for k, v in ns.items()}
            # F2 / F3
            own = ["o_last", "o_rev", "o_lvl", "o_f", "o_zero"]
            sets = {"own": own, "own+noise_interactions": own + ["se", "x_last_lvl", "x_rev_lvl", "x_last_se", "x_rev_se"]}
            if T == "linear":
                sets["own+library"] = own + ["lib_sign_m", "lib_llr_m"]
            valid = sorted(F)
            evals = valid[1:]
            mk = common.masks(N, len(evals), DRAWS, seed=11)
            f2 = {}
            for sname, cols in sets.items():
                Xs = {t: np.column_stack([zscore(F[t][c]) for c in cols]) for t in valid}
                ys = {t: zscore(Y[t]) for t in valid}
                P, coefs = [], []
                for t in evals:
                    tr = [u for u in valid if u < t]
                    mdl = Ridge(alpha=10.0).fit(np.vstack([Xs[u] for u in tr]), np.concatenate([ys[u] for u in tr]))
                    P.append(mdl.predict(Xs[t])); coefs.append(mdl.coef_)
                sc = common.score_masked([Y[t] for t in evals], P, mk)
                sc["mean_coef"] = dict(zip(cols, np.mean(coefs, 0).round(4).tolist()))
                f2[sname] = sc
            res[T] = {"F1_lag1_corr_by_quintile": f1, "F1_median_noise_share_of_level": f1_noise, "F2_F3_strict_ridge": f2,
                      "eval_transitions": [f"{trans[t][0]}->{trans[t][1]}" for t in evals]}
            print(name, T, "F1", {k: round(v, 3) for k, v in f1.items()}, "noise", {k: round(v, 3) for k, v in f1_noise.items()})
            for sname, sc in f2.items():
                print(f"   {sname:26s} pcc={sc['pcc']:.3f} up={sc['p200_up']:.3f} down={sc['p200_down']:.3f}", flush=True)
        out[name] = res
    with open(os.path.join(common.OUT, "followups.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
