"""Post-hoc checks on top of variance_decomp.py (needs results_raw.json).

1. Per-transition-normalised tables (each transition's target sum of squares = 1, then averaged, which is what a
   per-transition Pearson sees): share of the target SS by mean level and the signal fraction (1 - bootstrap noise
   / observed SS, both bins resampled from the smoothed laws) by level; effective number of genes carrying the
   target, N_eff = (sum yc^2)^2 / sum yc^4, and the number of genes that hold 50 % / 80 % of the SS.
2. Persistence of the noise-free change: corr(true change i, true change i+1) = (cov_obs + V_noise(shared bin)) /
   sqrt(S_i S_{i+1}), S = V_tot - V_noise(both bins); and the second-difference check Var(y_{i+1} - y_i) observed vs
   the noise-only prediction V_n(a) + 4 V_n(b) + V_n(b+1) (ratio < 1 means the noise model is too large).
3. Ceiling of per-gene univariate history: OLS of y_i on lags y_{i-1..i-3}, deviation of m_a from the cell-weighted
   level of the last 4 / 8 bins (in target units), and for orig the finite-cell expected target of the level law;
   (A) oracle coefficients fitted on y_i itself, (B) strict: coefficients fitted on transitions 1..i-1 only;
   both plain and with the coefficients varying by 5 expression-level classes of m_a.
4. Linear: one scalar on m_a with oracle sign and scale (|corr(y, m_a)|).
Output: results_posthoc.json.
"""
import json
import os
import time

import numpy as np

import variance_decomp as V

OUT = V.OUT
TARGETS = V.TARGETS


def level_class(m):
    return np.digitize(np.where(m > 0, np.log10(np.where(m > 0, m, 1)), -9), [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def apply(coef, X):
    return coef[0] + X @ coef[1:]


def interact(X, cls, n_cls=5):
    return np.column_stack([X * (cls == c)[:, None] for c in range(n_cls)] + [(cls == c).astype(float) for c in range(1, n_cls)])


def main():
    raw = json.load(open(os.path.join(OUT, "results_raw.json")))
    out = {}
    lv_edges = [0, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, np.inf]
    for name in V.SNAPS:
        t0 = time.time()
        m, s, f, n = V.load(name)
        G, nbin = m.shape
        Hs = [V.H_of(m[:, j], s[:, j] * np.sqrt(n[j] / max(n[j] - 1, 1)), f[:, j]) for j in range(nbin)]
        Hsm = [V.mix([Hs[k] for k in range(max(0, j - 1), min(nbin, j + 2))],
                     [n[k] for k in range(max(0, j - 1), min(nbin, j + 2))]) for j in range(nbin)]
        n_tr = nbin - 1
        Y = {tg: np.stack([V.T(tg, m[:, i], m[:, i + 1]) for i in range(n_tr)], 1) for tg in TARGETS}
        rng = np.random.default_rng(7 + (name == "pancreas-human"))
        rng_mask = np.random.default_rng(99)
        n_test = int(round(0.25 * G))
        masks = [[np.sort(rng_mask.permutation(G)[:n_test]) for _ in range(V.N_MASK)] for _ in range(n_tr)]
        # bootstrap draws of every bin's mean from its smoothed law (shared across targets)
        Dm = [V.sample_means(Hsm[j], n[j], 64, rng).astype(np.float32) for j in range(nbin)]
        res = {}
        for tg in TARGETS:
            raw_tr = raw[name]["transitions"]
            # ---- 1. per-transition normalised level tables and concentration
            shares, sigf, neff, n50, n80 = [], [], [], [], []
            for i in range(n_tr):
                y = Y[tg][:, i]
                yc = y - y.mean()
                ss = yc ** 2
                tot = ss.sum()
                yb = V.T(tg, Dm[i], Dm[i + 1])
                vn = yb.var(1)
                lev = 0.5 * (m[:, i] + m[:, i + 1])
                row_s, row_f = [], []
                for lo, hi in zip(lv_edges[:-1], lv_edges[1:]):
                    sel = (lev > lo) & (lev <= hi) if lo > 0 else (lev <= hi)
                    row_s.append(ss[sel].sum() / tot)
                    row_f.append(1 - vn[sel].sum() / ss[sel].sum() if ss[sel].sum() > 0 else np.nan)
                shares.append(row_s)
                sigf.append(row_f)
                neff.append(tot ** 2 / (ss ** 2).sum())
                cs = np.cumsum(np.sort(ss)[::-1]) / tot
                n50.append(int(np.searchsorted(cs, 0.5) + 1))
                n80.append(int(np.searchsorted(cs, 0.8) + 1))
            shares, sigf = np.array(shares), np.array(sigf)
            tab = []
            for j, (lo, hi) in enumerate(zip(lv_edges[:-1], lv_edges[1:])):
                tab.append({"level_range": [lo, hi], "mean_share_of_target_ss": float(shares[:, j].mean()),
                            "median_signal_fraction": float(np.nanmedian(sigf[:, j]))})
            r = {"by_level_per_transition_normalised": tab,
                 "N_eff_genes_median": float(np.median(neff)), "N_eff_genes_per_transition": [round(x, 1) for x in neff],
                 "genes_for_50pct_ss_median": float(np.median(n50)), "genes_for_80pct_ss_median": float(np.median(n80))}
            # ---- 2. persistence of the noise-free change and second-difference check
            rho, rho_noise_only, sd_ratio = [], [], []
            for i in range(n_tr - 1):
                t1, t2 = raw_tr[i][tg], raw_tr[i + 1][tg]
                V1, V2 = t1["V_tot"], t2["V_tot"]
                vb = 0.5 * (t1["noise_share_next_smooth"] * V1 + t2["noise_share_now_smooth"] * V2)
                S1 = V1 * (1 - t1["noise_share_both_smooth"])
                S2 = V2 * (1 - t2["noise_share_both_smooth"])
                y1, y2 = Y[tg][:, i], Y[tg][:, i + 1]
                cov = float(np.mean((y1 - y1.mean()) * (y2 - y2.mean())))
                rho_noise_only.append(-vb / np.sqrt(V1 * V2))
                rho.append((cov + vb) / np.sqrt(S1 * S2) if S1 > 0 and S2 > 0 else np.nan)
                va = t1["noise_share_now_smooth"] * V1
                vc = t2["noise_share_next_smooth"] * V2
                sd_ratio.append(float(np.var(y2 - y1) / (va + 4 * vb + vc)))
            r["true_change_persistence_median"] = float(np.nanmedian(rho))
            r["true_change_persistence_per_pair"] = [None if not np.isfinite(x) else round(float(x), 3) for x in rho]
            r["lag1_corr_predicted_by_noise_only_median"] = float(np.median(rho_noise_only))
            r["lag1_corr_observed_median"] = float(np.median([raw_tr[i]["lag1_corr"][tg] for i in range(n_tr - 1)]))
            r["second_diff_obs_over_noise_median"] = float(np.median(sd_ratio))
            r["second_diff_obs_over_noise_per_pair"] = [round(x, 3) for x in sd_ratio]
            # ---- 3. per-gene univariate history regressions
            feats = []
            for i in range(n_tr):
                a = i
                cols = [Y[tg][:, i - k] if i - k >= 0 else np.zeros(G) for k in (1, 2, 3)]
                for K in (4, 8):
                    ks = list(range(max(0, a - K + 1), a + 1))
                    lev = V.mix([Hs[k] for k in ks], [n[k] for k in ks])
                    cols.append(V.T(tg, m[:, a], lev[:, 0]))
                    if tg == "orig" and K == 4:
                        cols.append(V.T(tg, m[:, a][:, None], V.sample_means(lev, n[a], 96, rng)).mean(1))
                feats.append(np.column_stack(cols))
            cls = [level_class(m[:, i]) for i in range(n_tr)]
            pred = {"oracle_coef_plain": {}, "oracle_coef_by_level": {}, "strict_coef_plain": {}, "strict_coef_by_level": {}}
            for i in range(1, n_tr):
                y = Y[tg][:, i]
                X, Xi = feats[i], interact(feats[i], cls[i])
                pred["oracle_coef_plain"][i] = apply(ols(X, y), X)
                pred["oracle_coef_by_level"][i] = apply(ols(Xi, y), Xi)
                if i >= 2:
                    Xp = np.vstack([feats[j] for j in range(1, i)])
                    Xpi = np.vstack([interact(feats[j], cls[j]) for j in range(1, i)])
                    yp = np.concatenate([Y[tg][:, j] for j in range(1, i)])
                    pred["strict_coef_plain"][i] = apply(ols(Xp, yp), X)
                    pred["strict_coef_by_level"][i] = apply(ols(Xpi, yp), Xi)
            hist = {}
            for k, P in pred.items():
                rows = [V.score(Y[tg][:, i], p, masks[i]) for i, p in P.items()]
                hist[k] = {"n_transitions": len(rows), "first_i": int(min(P)),
                           **{q: float(np.mean([rw[q] for rw in rows])) for q in ("pearson_nan0", "p_at_200_up", "p_at_200_down")}}
            # same-transition-set reference: strict level-obs and trend from results_raw on i >= 2
            for k in ("strict_level_obs_na", "strict_trend1", "oracle_next_interp1_nb"):
                rows = [raw_tr[i][tg]["pred"][k] for i in range(2, n_tr)]
                hist[f"ref_{k}_i>=2"] = {q: float(np.mean([rw[q] for rw in rows])) for q in ("pearson_nan0", "p_at_200_up", "p_at_200_down")}
            r["per_gene_history"] = hist
            if tg == "linear":
                r["oracle_abs_corr_y_m_a_mean"] = float(np.mean([abs(raw_tr[i][tg]["pred"]["descr_y_on_m_a"]["pearson_all_genes"]) for i in range(n_tr)]))
            res[tg] = r
            print(name, tg, f"{time.time() - t0:.0f}s", flush=True)
        out[name] = res
    with open(os.path.join(OUT, "results_posthoc.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for name, res in out.items():
        print(f"\n===== {name}")
        for tg, r in res.items():
            print(f" -- {tg}: N_eff median {r['N_eff_genes_median']:.0f}, genes for 50% SS {r['genes_for_50pct_ss_median']:.0f}, 80% {r['genes_for_80pct_ss_median']:.0f}")
            print("   by level (per-transition normalised):", [(t["level_range"], round(t["mean_share_of_target_ss"], 3), round(t["median_signal_fraction"], 2)) for t in r["by_level_per_transition_normalised"]])
            print(f"   true-change persistence median {r['true_change_persistence_median']:.3f}; lag1 obs {r['lag1_corr_observed_median']:.3f} vs noise-only {r['lag1_corr_predicted_by_noise_only_median']:.3f}; 2nd diff obs/noise {r['second_diff_obs_over_noise_median']:.2f}")
            for k, h in r["per_gene_history"].items():
                print(f"   {k:40s} r={h['pearson_nan0']:.3f} up={h['p_at_200_up']:.3f} dn={h['p_at_200_down']:.3f}")
            if "oracle_abs_corr_y_m_a_mean" in r:
                print(f"   |corr(y, m_a)| mean {r['oracle_abs_corr_y_m_a_mean']:.3f}")


if __name__ == "__main__":
    main()
