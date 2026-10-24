"""What carries the Task-2 target variance (Dileo & Sottoriva '-human' builds), per target and dataset.

Per transition t -> t+1 (snapshot bins a -> b) and target T in {orig, l1, linear}, over all N genes:
 (a) zero-crossings: genes split into BZ (0 -> 0), UP (0 -> >0), DN (>0 -> 0), EX (>0 -> >0); share of the
     target sum of squares (around the transition mean) per group, R^2 of the 4-group mean (descriptive), tie
     blocks at the +/-5 clip of orig and the P@200 ceilings they impose (p_max of dileo_rescore.metrics);
     expected number of DN / UP crossings from pure finite sampling, (1 - f)^n_b, under a law of bin b that does
     not use bin b (interpolated from bins a and b+1, non-causal) and under the plug-in law of bin b.
 (b) global factor: library L = sum_g m and cell count n per bin; oracles that know L_b and/or n_b; descriptive
     R^2 of y on m_a (linear) and of a binned function of the current level (m_a, f_a); the same function learned
     strictly on past transitions; k-scalar oracle (k = 1, 3, 10 coefficients fitted on y itself) on gene loadings
     learned strictly from past changes (SVD of the past target columns).
 (c) finite-sample noise: s02_std is the population sd (ddof = 0, checked: single-cell genes at n = 2, 3, 7, 9 give
     s/m = sqrt(n - 1) exactly), so Var(bin mean) = s^2 / (n - 1). Linear target: analytic noise s_a^2/(n_a-1) +
     s_b^2/(n_b-1). All targets: parametric bootstrap of the bin means (zero-inflated count law matching m, f,
     E[c^2]; positives c = 1 + E, E negative binomial or Poisson; sum of K positives drawn exactly), with the law of
     each bin taken either as its own plug-in law or as the n-weighted mixture of bins b-1, b, b+1 (smoothed,
     non-causal; the plug-in law has zero noise for genes with m = 0, the smoothed one blurs fast trends). Noise of
     the next bin only (irreducible for any forecaster), of the current bin only (recoverable by denoising), both.
     Pearson ceiling = sqrt(1 - V_noise_next / V_tot); model-world ceilings for Pearson and P@200: truth = smoothed
     law of bin b, prediction = E[T], scored against fresh draws on 25 % gene masks.
     Model-free check: lag-1 covariance of consecutive linear changes vs -mean_g s_b^2/(n_b - 1).
     SNR per gene by mean level and by n.
 (d) smooth trend: strict predictors (repeat last change, mean of last 3 changes, level-k law + finite-cell
     observation with n_a, linear extrapolation of the level) and non-causal oracles (bin b interpolated from its
     neighbours a and b+1 or a-1, a, b+1, b+2 without bin b itself, observed with the true n_b).
Metrics: dileo_rescore.metrics on 10 random 25 % gene masks per transition (same masks for every predictor),
averaged over masks, then over transitions (all, and i >= 1 for strict predictors). Output: results.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import dileo_protocol  # noqa: E402  (numpy only; patch() is never called, so torch is never imported)
import dileo_rescore  # noqa: E402

GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
OUT = os.path.join(ROOT, "tack", "results", "signal_analysis", "variance")
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
TARGETS = ("orig", "l1", "linear")
B_DRAWS = 128          # bootstrap draws per law
N_EXP = 112            # draws used for expectations; the rest are fresh 'truth' draws for model-world ceilings
N_MASK = 10
LEVEL_K = 4
EPS = 1e-8


# ---------------------------------------------------------------- data
def load(name):
    node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"))
    n_genes = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    node = node.iloc[:n_genes]
    sn = SNAPS[name]
    g = lambda s: node[[f"t{b:03d}_{s}" for b in sn]].to_numpy(np.float64)
    m, s, f, tot = g("s00_mean"), g("s02_std"), g("s03_frac_nonzero"), g("s04_sum")
    n = np.array([np.median(tot[m[:, j] > 0, j] / m[m[:, j] > 0, j]) for j in range(len(sn))])
    n = np.round(n).astype(int)
    return m, s, f, n


def H_of(m, s, f):
    """Law triplet (m, f, E[c^2]) per gene."""
    return np.stack([m, f, s ** 2 + m ** 2], 1)


def mix(Hs, w):
    w = np.asarray(w, float)
    return sum(wi * H for wi, H in zip(w, Hs)) / w.sum()


# ---------------------------------------------------------------- targets (vectorised over draws on axis 1)
def T(name, m0, m1):
    m0 = np.maximum(m0, 0)
    m1 = np.maximum(m1, 0)
    if name == "orig":
        return np.clip(np.log(m1 + EPS) - np.log(m0 + EPS), -5.0, 5.0)
    if name == "l1":
        c0 = 1e4 * m0 / np.maximum(m0.sum(0, keepdims=True), 1e-12)
        c1 = 1e4 * m1 / np.maximum(m1.sum(0, keepdims=True), 1e-12)
        return np.log1p(c1) - np.log1p(c0)
    return m1 - m0


# ---------------------------------------------------------------- sampling bin means from a count law
def sample_means(H, n, B, rng):
    """(G, B) means of n iid cells from the zero-inflated law H = (m, f, E[c^2]); positives c = 1 + E."""
    m, f, e = H[:, 0], np.clip(H[:, 1], 0, 1), H[:, 2]
    G = len(m)
    pos = f > 0
    mu = np.where(pos, m / np.where(pos, f, 1), 0.0)
    var = np.where(pos, np.maximum(e / np.where(pos, f, 1) - mu ** 2, 0.0), 0.0)
    a = np.maximum(mu - 1.0, 0.0)               # mean of E
    K = rng.binomial(n, f[:, None], size=(G, B))
    tot = K.astype(np.float64)
    nb = pos & (a > 1e-9) & (var > a * (1 + 1e-6))
    po = pos & (a > 1e-9) & ~nb
    if nb.any():
        r = a[nb] ** 2 / (var[nb] - a[nb])
        p = r / (r + a[nb])
        Kn = K[nb]
        draw = rng.negative_binomial(np.where(Kn > 0, Kn * r[:, None], 1.0), np.broadcast_to(p[:, None], Kn.shape))
        tot[nb] += np.where(Kn > 0, draw, 0)
    if po.any():
        tot[po] += rng.poisson(K[po] * a[po][:, None])
    return tot / max(n, 1)


# ---------------------------------------------------------------- scoring helpers
def score(y, p, masks):
    rows = [dileo_rescore.metrics(y[mk], p[mk]) for mk in masks]
    keys = ("pearson_nan0", "p_at_200_up", "p_at_200_down", "p_ta_up", "p_ta_down", "p_max_up", "p_max_down")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def corr(y, p):
    if np.std(p) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def group_r2(y, labels):
    yc = y - y.mean()
    tot = (yc ** 2).sum()
    if tot == 0:
        return 0.0
    within = 0.0
    for u in np.unique(labels):
        sel = labels == u
        within += ((y[sel] - y[sel].mean()) ** 2).sum()
    return float(1 - within / tot)


EDGES = np.linspace(-3.5, 3.5, 36)


def level_bins(m, f):
    """Labels for the 'function of the current level' map: 0 = not expressed, else log10 m bin x f tercile."""
    lab = np.zeros(len(m), int)
    pos = m > 0
    lb = np.clip(np.digitize(np.log10(np.where(pos, m, 1)), EDGES), 1, len(EDGES))
    fb = np.digitize(f, [0.05, 0.3])
    lab[pos] = (lb[pos] * 3 + fb[pos]) + 1
    return lab


def main():
    rng_mask = np.random.default_rng(20261022)
    results = {"settings": dict(B_DRAWS=B_DRAWS, N_EXP=N_EXP, N_MASK=N_MASK, LEVEL_K=LEVEL_K,
                                std_ddof=0, sampling_var="s^2/(n-1)")}
    for name in SNAPS:
        t0 = time.time()
        m, s, f, n = load(name)
        G, nb_bins = m.shape
        L = m.sum(0)
        Hs = [H_of(m[:, j], s[:, j] * np.sqrt(n[j] / max(n[j] - 1, 1)), f[:, j]) for j in range(nb_bins)]  # unbiased var
        Hsm = [mix([Hs[k] for k in range(max(0, j - 1), min(nb_bins, j + 2))],
                   [n[k] for k in range(max(0, j - 1), min(nb_bins, j + 2))]) for j in range(nb_bins)]
        n_tr = nb_bins - 1
        Y = {tg: np.stack([dileo_protocol.TARGET_NP[tg](m[:, i].astype(np.float32), m[:, i + 1].astype(np.float32))
                           .astype(np.float64) for i in range(n_tr)], 1) for tg in TARGETS}
        for tg in TARGETS:  # my vectorised T reproduces their numpy target
            Ymine = np.stack([T(tg, m[:, i], m[:, i + 1]) for i in range(n_tr)], 1)
            assert np.allclose(Ymine, Y[tg], atol=1e-3 * (1 + np.abs(Y[tg]).max() * (tg == "linear"))), tg
        n_test = int(round(0.25 * G))
        res_ds = {"n_genes": G, "cells_per_bin": n.tolist(), "library_per_bin": L.round(1).tolist(),
                  "transitions": []}
        snr_acc = {tg: [] for tg in TARGETS}
        lin_noise_analytic = []
        for i in range(n_tr):
            a, b = i, i + 1
            rng = np.random.default_rng(1000 * (name == "pancreas-human") + i)
            masks = [np.sort(rng_mask.permutation(G)[:n_test]) for _ in range(N_MASK)]
            tr = {"i": i, "bins": f"{SNAPS[name][a]}->{SNAPS[name][b]}", "n_a": int(n[a]), "n_b": int(n[b]),
                  "log_L_ratio": float(np.log(L[b] / L[a]))}
            ma, mb = m[:, a], m[:, b]
            grp = np.where((ma == 0) & (mb == 0), 0, np.where(ma == 0, 1, np.where(mb == 0, 2, 3)))
            tr["group_frac"] = {k: float((grp == j).mean()) for j, k in enumerate(("BZ", "UP", "DN", "EX"))}
            # laws for bin b that do not use bin b (non-causal interpolations)
            H_int1 = Hs[a] if b + 1 >= nb_bins else mix([Hs[a], Hs[b + 1]], [1, 1])
            nbrs = [k for k in (a - 1, a, b + 1, b + 2) if 0 <= k < nb_bins]
            H_int2 = mix([Hs[k] for k in nbrs], [1] * len(nbrs))
            # strict level law (cell-weighted mixture of the last LEVEL_K bins <= a)
            ks = list(range(max(0, a - LEVEL_K + 1), a + 1))
            H_lev = mix([Hs[k] for k in ks], [n[k] for k in ks])
            # expected DN / UP crossings from finite sampling alone
            for lab, H in (("interp_noB", H_int1), ("plugin_B", Hs[b]), ("smooth_B", Hsm[b])):
                p0 = (1 - np.clip(H[:, 1], 0, 1)) ** n[b]
                tr[f"exp_DN_{lab}"] = float(p0[ma > 0].sum())
                tr[f"exp_UP_{lab}"] = float((1 - p0)[ma == 0].sum())
            tr["obs_DN"], tr["obs_UP"] = int((grp == 2).sum()), int((grp == 1).sum())
            # draws
            D = {
                "next_plug": sample_means(Hs[b], n[b], B_DRAWS, rng),
                "next_smooth": sample_means(Hsm[b], n[b], B_DRAWS, rng),
                "now_smooth": sample_means(Hsm[a], n[a], B_DRAWS, rng),
                "int1_nb": sample_means(H_int1, n[b], N_EXP, rng),
                "int2_nb": sample_means(H_int2, n[b], N_EXP, rng),
                "lev_na": sample_means(H_lev, n[a], N_EXP, rng),
                "lev_nb": sample_means(H_lev, n[b], N_EXP, rng),
            }
            lev_mean = H_lev[:, 0]
            scale = L[b] / lev_mean.sum()                  # oracle library: rescale the level law to L_b
            H_levL = H_lev * np.array([scale, 1.0, scale ** 2])
            D["levL_nb"] = sample_means(H_levL, n[b], N_EXP, rng)
            # linear analytic noise
            va = s[:, a] ** 2 / max(n[a] - 1, 1)
            vb = s[:, b] ** 2 / max(n[b] - 1, 1)
            for tg in TARGETS:
                y = Y[tg][:, i]
                yc = y - y.mean()
                Vt = float(yc.var())
                ss = (yc ** 2)
                st = {"V_tot": Vt}
                st["ss_share"] = {k: float(ss[grp == j].sum() / ss.sum()) if ss.sum() > 0 else 0.0
                                  for j, k in enumerate(("BZ", "UP", "DN", "EX"))}
                st["R2_zc_groups"] = group_r2(y, grp)
                if tg == "orig":
                    st["n_at_+5"] = int((y >= 5 - 1e-4).sum())
                    st["n_at_-5"] = int((y <= -5 + 1e-4).sum())
                    st["n_at_0"] = int((y == 0).sum())
                # noise
                ynp = T(tg, ma[:, None], D["next_plug"])
                yns = T(tg, ma[:, None], D["next_smooth"])
                yno = T(tg, D["now_smooth"], mb[:, None])
                ybo = T(tg, D["now_smooth"], D["next_smooth"])
                for lab, Yd in (("next_plugin", ynp), ("next_smooth", yns), ("now_smooth", yno), ("both_smooth", ybo)):
                    st[f"noise_share_{lab}"] = float(Yd.var(1).mean() / Vt) if Vt > 0 else float("nan")
                if tg == "linear":
                    st["noise_share_analytic_next"] = float(vb.mean() / Vt)
                    st["noise_share_analytic_now"] = float(va.mean() / Vt)
                    lin_noise_analytic.append((i, va + vb))
                for lab in ("next_plugin", "next_smooth"):
                    st[f"pearson_ceiling_{lab}"] = float(np.sqrt(max(0.0, 1 - st[f"noise_share_{lab}"])))
                # model-world ceiling: truth = smoothed law of b, prediction E[T], fresh draws scored
                Ey = yns[:, :N_EXP].mean(1)
                mw = [score(yns[:, N_EXP + d], Ey, masks[d % N_MASK:d % N_MASK + 1]) for d in range(B_DRAWS - N_EXP)]
                st["modelworld_ceiling"] = {k: float(np.mean([r[k] for r in mw])) for k in mw[0]}
                # per-gene snr material: noise variance (both bins, smoothed laws) and level
                snr_acc[tg].append((i, y, ybo.var(1), 0.5 * (ma + mb), min(n[a], n[b])))
                # predictors
                P = {}
                P["oracle_zc_groups"] = np.array([y[grp == j].mean() if (grp == j).any() else 0 for j in range(4)])[grp]
                lab_lv = level_bins(ma, f[:, a])
                gm = pd.Series(y).groupby(lab_lv).transform("mean").to_numpy()
                P["descr_level_fn"] = gm
                P["oracle_scalar_L"] = (L[b] / L[a] - 1) * ma if tg == "linear" else \
                    T(tg, ma, ma * L[b] / L[a])
                P["oracle_next_interp1_nb"] = T(tg, ma[:, None], D["int1_nb"]).mean(1)
                P["oracle_next_interp2_nb"] = T(tg, ma[:, None], D["int2_nb"]).mean(1)
                P["oracle_next_smoothB_point"] = T(tg, ma, Hsm[b][:, 0])  # includes bin b: leaky upper bound
                P["strict_level_obs_na"] = T(tg, ma[:, None], D["lev_na"]).mean(1)
                P["oracle_n_level_obs_nb"] = T(tg, ma[:, None], D["lev_nb"]).mean(1)
                P["oracle_nL_level_obs"] = T(tg, ma[:, None], D["levL_nb"]).mean(1)
                P["strict_level_point"] = T(tg, ma, lev_mean)
                if i >= 1:
                    P["strict_trend1"] = Y[tg][:, i - 1]
                    P["strict_trend3"] = Y[tg][:, max(0, i - 3):i].mean(1)
                    # linear extrapolation of the level (cell-weighted last-2 levels), observed with n_a
                    ks2 = list(range(max(0, a - LEVEL_K), a))
                    H_prev = mix([Hs[k] for k in ks2], [n[k] for k in ks2])
                    ext = np.maximum(2 * lev_mean - H_prev[:, 0], 0)
                    P["strict_extrap_point"] = T(tg, ma, ext)
                    # strict level function learned on past transitions
                    past_lab = np.concatenate([level_bins(m[:, j], f[:, j]) for j in range(i)])
                    past_y = np.concatenate([Y[tg][:, j] for j in range(i)])
                    mp = pd.Series(past_y).groupby(past_lab).mean()
                    P["strict_level_fn"] = pd.Series(lab_lv).map(mp).fillna(past_y.mean()).to_numpy()
                    # low rank: loadings from past target columns (strict), k coefficients fitted on y (oracle)
                    Yp = Y[tg][:, :i] - Y[tg][:, :i].mean(0)
                    U, S, _ = np.linalg.svd(Yp, full_matrices=False)
                    for k in (1, 3, 10):
                        if k <= i:
                            Uk = U[:, :k]
                            P[f"oracle_k{k}_scalars_pastloadings"] = y.mean() + Uk @ (Uk.T @ yc)
                    Uk = U[:, :min(3, i)]
                    P["strict_lowrank3_trend"] = Uk @ (Uk.T @ (Y[tg][:, i - 1] - Y[tg][:, i - 1].mean()))
                # leave-one-out loadings (all other transitions, non-causal)
                Yo = np.delete(Y[tg], i, 1)
                Yo = Yo - Yo.mean(0)
                U2, _, _ = np.linalg.svd(Yo, full_matrices=False)
                for k in (1, 3, 10):
                    Uk = U2[:, :k]
                    P[f"oracle_k{k}_scalars_looloadings"] = y.mean() + Uk @ (Uk.T @ yc)
                if tg == "linear":
                    P["descr_y_on_m_a"] = ma.copy()
                st["pred"] = {}
                for k, p in P.items():
                    sc = score(y, p, masks)
                    sc["pearson_all_genes"] = corr(y, p)
                    st["pred"][k] = sc
                tr[tg] = st
            # model-free lag-1 check (linear): cov(y_i, y_{i+1}) vs -mean s_b^2/(n_b-1)
            if i + 1 < n_tr:
                y1, y2 = Y["linear"][:, i], Y["linear"][:, i + 1]
                tr["lag1_linear"] = {"cov_obs": float(np.mean((y1 - y1.mean()) * (y2 - y2.mean()))),
                                     "cov_noise_pred": float(-vb.mean()),
                                     "var_y_i": float(y1.var()), "var_y_next": float(y2.var())}
                tr["lag1_corr"] = {tg: corr(Y[tg][:, i], Y[tg][:, i + 1]) for tg in TARGETS}
            res_ds["transitions"].append(tr)
            print(f"{name} transition {i} done ({time.time() - t0:.0f}s)", flush=True)
        # ---------------- SNR by level and by n
        lv_edges = [0, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, np.inf]
        n_edges = [0, 100, 300, np.inf]
        snr = {}
        for tg in TARGETS:
            tab = {}
            ys =np.concatenate([r[1] - r[1].mean() for r in snr_acc[tg]])
            vs = np.concatenate([r[2] for r in snr_acc[tg]])
            ls = np.concatenate([r[3] for r in snr_acc[tg]])
            ns = np.concatenate([np.full(len(r[1]), r[4]) for r in snr_acc[tg]])
            tot_ss = (ys ** 2).sum()
            by_level = []
            for lo, hi in zip(lv_edges[:-1], lv_edges[1:]):
                sel = (ls > lo) & (ls <= hi) if lo > 0 else (ls <= hi)
                if sel.sum() == 0:
                    continue
                sig = (ys[sel] ** 2).sum()
                by_level.append({"level_range": [lo, hi], "frac_gene_transitions": float(sel.mean()),
                                 "share_of_target_ss": float(sig / tot_ss),
                                 "signal_fraction": float(1 - vs[sel].sum() / sig) if sig > 0 else None,
                                 "frac_abs_z_gt2": float(np.mean(ys[sel] ** 2 > 4 * np.maximum(vs[sel], 1e-300)))})
            tab["by_level"] = by_level
            by_n = []
            for lo, hi in zip(n_edges[:-1], n_edges[1:]):
                sel = (ns >= lo) & (ns < hi)
                if sel.sum() == 0:
                    continue
                sig = (ys[sel] ** 2).sum()
                by_n.append({"n_min_range": [lo, hi], "n_transitions": int(sel.sum() // G),
                             "signal_fraction": float(1 - vs[sel].sum() / sig) if sig > 0 else None})
            tab["by_n"] = by_n
            snr[tg] = tab
        # linear analytic SNR by level (bins themselves give s)
        ys = np.concatenate([Y["linear"][:, i] - Y["linear"][:, i].mean() for i, _ in lin_noise_analytic])
        vs = np.concatenate([v for _, v in lin_noise_analytic])
        ls = np.concatenate([0.5 * (m[:, i] + m[:, i + 1]) for i, _ in lin_noise_analytic])
        snr["linear_analytic_by_level"] = []
        for lo, hi in zip(lv_edges[:-1], lv_edges[1:]):
            sel = (ls > lo) & (ls <= hi) if lo > 0 else (ls <= hi)
            sig = (ys[sel] ** 2).sum()
            snr["linear_analytic_by_level"].append({"level_range": [lo, hi],
                                                    "signal_fraction": float(1 - vs[sel].sum() / sig) if sig > 0 else None})
        res_ds["snr"] = snr
        results[name] = res_ds
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "results_raw.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
