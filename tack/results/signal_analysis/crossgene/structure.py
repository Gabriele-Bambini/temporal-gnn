"""Descriptive + strict structure of the genes x bins signal (no forecasting here).

Per dataset:
 1. noise model: per gene and bin, sampling variance of the bin mean, v_lin = sigma^2 / n (raw scale) and its
    delta-method image on L = log1p(CP10k): v_l1 = (1e4 / lib)^2 sigma^2 / n / (1 + CP)^2, with sigma^2 and CP
    pooled over a +/-2 bin window (so a gene that happens to be zero in a bin still gets a noise level).
    Calibration: across transitions, regress Var_genes(Delta_t) on the predicted noise mean(v_t + v_t+1).
 2. noise budget of each next-bin change Delta_t (l1 and linear): share of Var_genes(Delta) that is noise of bin t
    (predictable by denoising, enters with a minus sign), noise of bin t+1 (unpredictable by any strict model) and
    the remainder (population change); ceiling PCC_max = sqrt(1 - mean v_t+1 / Var Delta); noise-corrected lag-1
    correlation of the signal part (how much of the change persists to the next transition).
 3. SVD spectrum of the gene-centred L (full matrix, descriptive) and of every prefix of bins <= t (strict), with a
    heteroscedastic noise edge (top singular value of simulated noise matrices with the same v_l1).
 4. share of Delta_t (l1, gene-centred) in the top-k subspace: strict (left singular vectors of the centred bins
    <= t), strict change subspace (SVD of the past changes Delta_0..Delta_t-1), oracle (all bins, leaks t+1);
    raw energy share, the share expected from noise alone, and the noise-corrected signal share.
 5. linear target: exact split m_t+1 - m_t = (lib_t+1 / 1e4)(CP_t+1 - CP_t) + m_t (r - 1), r = lib_t+1 / lib_t;
    share of Var(Delta) carried by the library term, corr(m_t, Delta), dominance of the top-1 % genes.
 6. NMF (weighted KL, nmf_kl.py) on all bins: K chosen by masked-entry cross-validation (10 % of gene x bin entries of the
    4000 highest-count genes held out, 3 folds, held-out Poisson deviance); proportions per bin, archetype marker genes, archetype costs.
Output: structure.json in this folder.
"""
import json
import os
import time

import numpy as np

import common
import nmf_kl

RNG = np.random.default_rng(7)


def pooled(X, w, half=2):
    """Cell-weighted mean over a +/-half window of bins, per gene (descriptive smoothing for noise levels)."""
    S = X.shape[1]
    out = np.zeros_like(X)
    for s in range(S):
        lo, hi = max(0, s - half), min(S, s + half + 1)
        out[:, s] = (X[:, lo:hi] * w[lo:hi]).sum(1) / w[lo:hi].sum()
    return out


def noise_vars(d):
    sig2 = pooled(d.SD ** 2, d.nc)
    cp = pooled(d.CP, d.nc)
    v_lin = sig2 / d.nc[None, :]
    v_l1 = (1e4 / d.lib[None, :]) ** 2 * v_lin / (1 + cp) ** 2
    return v_l1, v_lin


def noise_budget(d, v, tname):
    rows = []
    Y = common.all_targets(d, tname)
    for i in range(d.T):
        y = Y[i]
        tot = float(y.var())
        a, b = float(v[:, i].mean()), float(v[:, i + 1].mean())
        rows.append({"i": i, "bins": f"{d.bins[i]}->{d.bins[i + 1]}", "cells": [int(d.nc[i]), int(d.nc[i + 1])],
                     "var_delta": tot, "share_noise_t": a / tot, "share_noise_t1": b / tot,
                     "share_signal": 1 - (a + b) / tot, "pcc_ceiling_strict": float(np.sqrt(max(0.0, 1 - b / tot)))})
    # calibration: Var(Delta_t) = c + beta * predicted noise
    tot = np.array([r["var_delta"] for r in rows])
    pn = np.array([v[:, i].mean() + v[:, i + 1].mean() for i in range(d.T)])
    A = np.stack([np.ones_like(pn), pn], 1)
    coef = np.linalg.lstsq(A, tot, rcond=None)[0]
    # signal persistence: noise-corrected lag-1 correlation across genes
    pers = []
    for i in range(1, d.T):
        c = float(np.cov(Y[i], Y[i - 1])[0, 1])
        c_sig = c + float(v[:, i].mean())
        s1 = tot[i] - pn[i]
        s0 = tot[i - 1] - pn[i - 1]
        raw = float(np.corrcoef(Y[i], Y[i - 1])[0, 1])
        pers.append({"i": i, "raw_lag1_corr": raw, "noise_cov_share": float(-v[:, i].mean() / c) if c != 0 else None,
                     "signal_lag1_corr": float(c_sig / np.sqrt(s1 * s0)) if s1 > 0 and s0 > 0 else None,
                     "c_sig": c_sig, "s1": float(s1), "s0": float(s0)})
    agg = lambda pp: float(sum(p["c_sig"] for p in pp) / np.sqrt(max(1e-30, sum(max(p["s1"], 0) for p in pp) *
                                                                    sum(max(p["s0"], 0) for p in pp))))
    def avg(key, first=1, rr=rows):
        return float(np.mean([r[key] for r in rr[first:] if r[key] is not None]))
    return {"per_transition": rows,
            "mean_i>=1": {k: avg(k) for k in ("share_noise_t", "share_noise_t1", "share_signal", "pcc_ceiling_strict")},
            "mean_i>=1_no_last": {k: float(np.mean([r[k] for r in rows[1:-1]])) for k in
                                  ("share_noise_t", "share_noise_t1", "share_signal", "pcc_ceiling_strict")},
            "calibration_var_delta_on_predicted_noise": {"intercept": float(coef[0]), "slope": float(coef[1]),
                                                         "corr": float(np.corrcoef(pn, tot)[0, 1])},
            "signal_persistence": pers,
            "mean_raw_lag1_corr": float(np.mean([p["raw_lag1_corr"] for p in pers])),
            "pooled_signal_lag1_corr": agg(pers), "pooled_signal_lag1_corr_no_last": agg(pers[:-1]),
            "median_signal_lag1_corr": float(np.median([p["signal_lag1_corr"] for p in pers
                                                         if p["signal_lag1_corr"] is not None]))}


def noise_edge(v, n_sim=3):
    """Top singular value^2 of gene-centred Gaussian noise matrices with variances v (N x B)."""
    tops = []
    for _ in range(n_sim):
        E = RNG.standard_normal(v.shape) * np.sqrt(v)
        E = E - E.mean(1, keepdims=True)
        tops.append(np.linalg.svd(E, compute_uv=False)[0] ** 2)
    return float(np.mean(tops))


def spectrum(X, v):
    Xc = X - X.mean(1, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    edge = noise_edge(v)
    noise_tot = float((v * (1 - 1 / X.shape[1])).sum())
    return {"sv2_share": [round(float(x), 4) for x in (lam / lam.sum())[:12]],
            "cum_share_k1_5_10": [float(lam[:k].sum() / lam.sum()) for k in (1, 2, 3, 5, 10)],
            "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "n_above_noise_edge": int((lam > edge).sum()),
            "noise_share_of_total_energy": noise_tot / float(lam.sum())}


def topk_share(d, v_l1, ks=(1, 2, 3, 5, 8)):
    Lc_all = d.L - d.L.mean(1, keepdims=True)
    U_or = np.linalg.svd(Lc_all, full_matrices=False)[0]
    rows = []
    for i in range(2, d.T):
        dl = d.L[:, i + 1] - d.L[:, i]
        dl = dl - dl.mean()
        Dn = v_l1[:, i] + v_l1[:, i + 1]
        X = d.L[:, :i + 1]
        Xl = d.L[:, :i]                      # bins <= t-1: basis independent of the noise of bins t and t+1
        U_st = np.linalg.svd(X - X.mean(1, keepdims=True), full_matrices=False)[0]
        U_lag = np.linalg.svd(Xl - Xl.mean(1, keepdims=True), full_matrices=False)[0]
        U_dlag = np.linalg.svd(np.diff(Xl, axis=1), full_matrices=False)[0]
        row = {"i": i}
        for lab, U in (("strict_level_bins_le_t", U_st), ("strict_level_bins_le_t-1", U_lag),
                       ("strict_change_bins_le_t-1", U_dlag), ("oracle_all_bins", U_or)):
            for k in ks:
                kk = {"strict_level_bins_le_t": min(k, i), "strict_level_bins_le_t-1": min(k, i - 1),
                      "strict_change_bins_le_t-1": min(k, i - 1), "oracle_all_bins": min(k, d.S - 1)}[lab]
                Uk = U[:, :kk]
                e = float(((Uk.T @ dl) ** 2).sum() / (dl ** 2).sum())
                en = float((Uk ** 2 * Dn[:, None]).sum() / (dl ** 2).sum())
                sig = ((dl ** 2).sum() - Dn.sum())
                es = float((((Uk.T @ dl) ** 2).sum() - (Uk ** 2 * Dn[:, None]).sum()) / sig) if sig > 0 else None
                row[f"{lab}_k{k}"] = {"raw": round(e, 4), "noise_expected": round(en, 4),
                                      "signal_share": None if es is None else round(es, 4),
                                      "_num": float(((Uk.T @ dl) ** 2).sum() - (Uk ** 2 * Dn[:, None]).sum()),
                                      "_den": float(sig), "_tot": float((dl ** 2).sum()),
                                      "_proj": float(((Uk.T @ dl) ** 2).sum())}
        rows.append(row)
    summ = {}
    for key in rows[0]:
        if key == "i":
            continue
        for f in ("raw", "noise_expected", "signal_share"):
            vals = [r[key][f] for r in rows if r[key][f] is not None]
            summ.setdefault(key, {})[f] = round(float(np.mean(vals)), 4) if vals else None
            vals2 = [r[key][f] for r in rows[:-1] if r[key][f] is not None]
            summ[key][f + "_no_last"] = round(float(np.mean(vals2)), 4) if vals2 else None
        for suf, rr in (("", rows), ("_no_last", rows[:-1])):
            summ[key]["pooled_raw" + suf] = round(sum(r[key]["_proj"] for r in rr) / sum(r[key]["_tot"] for r in rr), 4)
            summ[key]["pooled_signal_share" + suf] = round(sum(r[key]["_num"] for r in rr) /
                                                          max(1e-30, sum(max(r[key]["_den"], 0) for r in rr)), 4)
        summ[key].pop("signal_share", None)
        summ[key].pop("signal_share_no_last", None)
    total_sig = [sum(max(r[k]["_den"], 0) for r in rows) / sum(r[k]["_tot"] for r in rows) for k in ["oracle_all_bins_k1"]]
    return {"mean_over_i>=1": summ, "pooled_signal_share_of_delta_energy": round(total_sig[0], 4),
            "per_transition": rows,
            "note": "transitions i >= 2. signal_share = (projected energy - expected projected noise) / (total energy "
                    "- total noise); valid ONLY for the bases built from bins <= t-1 (independent of the noise of bins "
                    "t, t+1); bases that contain bin t (strict_level_bins_le_t) or t and t+1 (oracle) also capture "
                    "the bins' own noise, so for them only 'raw' is meaningful (signal_share can exceed 1). Pooled = "
                    "energy-weighted over transitions (per-transition ratios are unstable at high noise share)."}


def library_term(d):
    Y = common.all_targets(d, "linear")
    rows = []
    for i in range(d.T):
        y = Y[i]
        r = d.lib[i + 1] / d.lib[i]
        lt = d.M[:, i] * (r - 1)
        ct = d.lib[i + 1] / 1e4 * (d.CP[:, i + 1] - d.CP[:, i])
        assert np.allclose(lt + ct, y, atol=1e-3)
        top = np.argsort(-d.M[:, i])[: max(1, d.N // 100)]
        yc = y - y.mean()
        rows.append({"i": i, "lib_ratio": float(r), "corr_m_t_delta": float(np.corrcoef(d.M[:, i], y)[0, 1]),
                     "var_share_library_term": float(lt.var() / y.var()),
                     "var_share_composition_term": float(ct.var() / y.var()),
                     "share_of_sumsq_top1pct_genes": float((yc[top] ** 2).sum() / (yc ** 2).sum()),
                     "lib_ratio_sign_same_as_previous": None if i == 0 else
                     bool(np.sign(r - 1) == np.sign(d.lib[i] / d.lib[i - 1] - 1))})
    f = lambda k: float(np.mean([x[k] for x in rows[1:]]))
    return {"per_transition": rows, "mean_i>=1": {k: f(k) for k in ("corr_m_t_delta", "var_share_library_term",
                                                                     "var_share_composition_term",
                                                                     "share_of_sumsq_top1pct_genes")},
            "lib_sign_persistence_i>=1": float(np.mean([x["lib_ratio_sign_same_as_previous"] for x in rows[1:]])),
            "lib_per_bin": [round(float(x), 1) for x in d.lib]}


def nmf_cv(d, Ks=(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18), folds=3, frac=0.1):
    Ks = [k for k in Ks if k <= d.S - 1]
    X = d.CP.T.copy()
    om = d.nc * d.lib / 1e4
    top = np.argsort(-(X * om[:, None]).sum(0))[:4000]   # the genes the forecasting fits use for W
    X = X[:, top]
    B, G = X.shape
    res = {}
    for K in Ks:
        dev = []
        for f in range(folds):
            rng = np.random.default_rng(100 + f)
            hold = rng.random((B, G)) < frac
            W, H = nmf_kl.fit(X, K, om, mask=(~hold).astype(float), n_iter=1000, tol=1e-6)
            WH = W @ H
            Om = om[:, None] * hold
            dev.append(2 * nmf_kl.kl_loss(X, WH, Om) / hold.sum())
        res[K] = float(np.mean(dev))
    best = min(res, key=res.get)
    # one-standard-error-style rule: smallest K within 1 % of the best deviance
    k1 = min(k for k, v in res.items() if v <= res[best] * 1.01)
    return res, best, k1


def nmf_describe(d, K):
    X = d.CP.T.copy()
    om = d.nc * d.lib / 1e4
    W, H = nmf_kl.fit_subset(X, K, om)
    P = nmf_kl.to_props(W)
    LH = np.log1p(H)
    C = ((LH[:, None, :] - LH[None, :, :]) ** 2).mean(2)
    markers = {}
    for k in range(K):
        spec = np.log1p(H[k]) - np.log1p(np.delete(H, k, 0).max(0))
        markers[f"arch{k}"] = [str(g) for g in d.genes[np.argsort(-spec)[:12]]]
    # reconstruction quality of each bin in L space
    Lhat = np.log1p(P @ H)
    rec = [float(np.corrcoef(Lhat[s], d.L[:, s])[0, 1]) for s in range(d.S)]
    return {"K": K, "proportions_per_bin": [[round(float(x), 3) for x in row] for row in P],
            "row_sums_before_norm": [round(float(x), 3) for x in W.sum(1)],
            "archetype_cost_meanSqLogDiff": [[round(float(x), 4) for x in r] for r in C],
            "markers": markers, "recon_corr_L_per_bin": [round(x, 4) for x in rec]}


def main():
    out = {}
    for name in common.SNAPS:
        t0 = time.time()
        d = common.Data(name)
        v_l1, v_lin = noise_vars(d)
        R = {"n_genes": d.N, "bins": d.bins, "cells": [int(x) for x in d.nc]}
        R["noise_budget_l1"] = noise_budget(d, v_l1, "l1")
        R["noise_budget_linear"] = noise_budget(d, v_lin, "linear")
        R["spectrum_full"] = spectrum(d.L, v_l1)
        R["spectrum_full_cellweighted_bins"] = spectrum(d.L * np.sqrt(d.nc / d.nc.mean())[None, :],
                                                        v_l1 * (d.nc / d.nc.mean())[None, :])
        R["spectrum_prefix"] = {int(t): spectrum(d.L[:, :t + 1], v_l1[:, :t + 1]) for t in range(2, d.S, 3)}
        R["delta_in_topk"] = topk_share(d, v_l1)
        R["library_term_linear"] = library_term(d)
        cv, best, k1 = nmf_cv(d)
        R["nmf_cv"] = {"heldout_poisson_deviance_per_entry": cv, "best_K": best, "K_within_1pct": k1}
        R["nmf_all_bins"] = nmf_describe(d, k1)
        R["seconds"] = time.time() - t0
        out[name] = R
        print(name, "done", round(R["seconds"], 1), "s", flush=True)
        print(" noise budget l1", R["noise_budget_l1"]["mean_i>=1"], R["noise_budget_l1"]["calibration_var_delta_on_predicted_noise"])
        print(" noise budget lin", R["noise_budget_linear"]["mean_i>=1"], R["noise_budget_linear"]["calibration_var_delta_on_predicted_noise"])
        for t in ("l1", "linear"):
            nb = R["noise_budget_" + t]
            print(" persistence", t, nb["mean_raw_lag1_corr"], nb["pooled_signal_lag1_corr"], nb["pooled_signal_lag1_corr_no_last"])
        print(" spectrum", R["spectrum_full"])
        print(" topk signal share of delta energy", R["delta_in_topk"]["pooled_signal_share_of_delta_energy"])
        for k, v in R["delta_in_topk"]["mean_over_i>=1"].items():
            print("   ", k, v["pooled_raw"], v["pooled_signal_share"], v["pooled_signal_share_no_last"])
        print(" lib", R["library_term_linear"]["mean_i>=1"], R["library_term_linear"]["lib_sign_persistence_i>=1"])
        print(" nmf cv", cv, best, k1)
    with open(os.path.join(common.OUT, "structure.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
