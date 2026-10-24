"""Descriptive temporal structure of the per-gene changes (no prediction; uses all bins).

1. Noise budget per transition: Delta = dmu + e_{t+1} - e_t with e = sampling error of a bin mean,
   var(e_g,t) = s^2/(n-1) (s02_std is the population sd, ddof 0). Pooled over genes:
     V      = sum_g (Delta_g - mean)^2           (cross-gene variance that Pearson sees)
     N_next = sum_g var(e_g,t+1),  N_now = sum_g var(e_g,t)
     ceiling PCC for ANY predictor that knows the true curve = sqrt(1 - N_next / V)
     signal share (true change of the curve) = 1 - (N_now + N_next) / V; reversion share = N_now / V.
   linear: exact; l1: delta method var(log1p c) = (1e4/sum m)^2 var(e) / (1 + c)^2 (library noise ignored);
   orig: not computed (clip + log(m + 1e-8) not linearisable).
2. Autocorrelation of consecutive changes, lags 1..5:
   cross-sectional  rho_L = mean_i corr_genes(Delta_i, Delta_{i+L})  (= PCC of 'repeat the change of L steps ago');
   within-gene      pooled autocorrelation of the gene-centred series, by expression decile;
   noise-corrected  lag 1: (cov + N_shared) / sqrt((V_a - N_a0 - N_a1)(V_b - N_b0 - N_b1)), lag >= 2: cov / sqrt(signal vars)
   by bin size: lag-1 rho of each pair vs n of the shared bin.
   orig also on 'no zero-crossing' genes (m > 0 in all bins of the pair).
3. Resolution test: merge consecutive bins pairwise (exact pooled mean / second moment / frac) and recompute.
Output: noise_acf.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, SNAPS, TRAIN_SPLIT, TARGETS, tgt, level_space  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAGS = (1, 2, 3, 4, 5)


def noise_var(target, m, s, n):
    """Sampling variance of the level X_t (genes x bins) whose difference is the target (linear, l1)."""
    ve = s ** 2 / np.maximum(n - 1, 1)[None, :]
    if target == "linear":
        return ve
    if target == "l1":
        lib = m.sum(0, keepdims=True)
        c = 1e4 * m / lib
        return (1e4 / lib) ** 2 * ve / (1 + c) ** 2
    return None


def deltas(target, m, ntr):
    return np.stack([tgt(target, m[:, i], m[:, i + 1]) for i in range(ntr)])


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def analyse(m, s, n, ntr, expr_rank, label):
    out = {"label": label, "cells": n.astype(int).tolist()}
    live = m[:, :ntr + 1].max(1) > 0
    dec = np.full(m.shape[0], -1)
    ranks = stats.rankdata(expr_rank[live]) / live.sum()
    dec[live] = np.minimum((ranks * 10).astype(int), 9)
    decl = dec[live]
    for target in TARGETS:
        D = deltas(target, m, ntr)[:, live]                      # (T, G_live)
        r = {}
        r["xs_rho"] = {L: float(np.nanmean([corr(D[i], D[i + L]) for i in range(ntr - L)])) for L in LAGS if ntr - L >= 2}
        pairs1 = [corr(D[i], D[i + 1]) for i in range(ntr - 1)]
        r["xs_rho1_per_pair"] = np.round(pairs1, 4).tolist()
        nsh = n[1:ntr]                                            # shared bin of pair (i, i+1) is snapshot i+1
        r["xs_rho1_vs_shared_n_spearman"] = float(stats.spearmanr(pairs1, nsh).correlation)
        terc = np.quantile(nsh, [1 / 3, 2 / 3])
        r["xs_rho1_by_shared_n_tercile"] = {
            f"n<={terc[0]:.0f}": float(np.mean([p for p, k in zip(pairs1, nsh) if k <= terc[0]])),
            "mid": float(np.mean([p for p, k in zip(pairs1, nsh) if terc[0] < k <= terc[1]])),
            f"n>{terc[1]:.0f}": float(np.mean([p for p, k in zip(pairs1, nsh) if k > terc[1]]))}
        Dc = D - D.mean(0, keepdims=True)
        wg = {}
        for q in range(10):
            x = Dc[:, decl == q]
            v0 = (x * x).sum()
            wg[q] = {L: float((x[:-L] * x[L:]).sum() / v0) for L in LAGS if ntr - L >= 2}
        r["within_gene_rho_by_decile"] = wg
        r["within_gene_rho_all"] = {L: float((Dc[:-L] * Dc[L:]).sum() / (Dc * Dc).sum()) for L in LAGS if ntr - L >= 2}
        # cross-sectional lag-1 by decile (what a stratified 'repeat last change' sees)
        r["xs_rho1_by_decile"] = {q: float(np.nanmean([corr(D[i, decl == q], D[i + 1, decl == q]) for i in range(ntr - 1)]))
                                  for q in range(10)}
        NV = noise_var(target, m, s, n)
        if NV is not None:
            NV = NV[live]
            V = np.array([((D[i] - D[i].mean()) ** 2).sum() for i in range(ntr)])
            Nn = np.array([NV[:, i].sum() for i in range(ntr + 1)])
            ceil = np.sqrt(np.clip(1 - Nn[1:ntr + 1] / V, 0, 1))
            r["noise_budget"] = {
                "ceiling_pcc_per_transition": ceil.round(4).tolist(),
                "ceiling_pcc_mean_i>=1": float(ceil[1:].mean()),
                "signal_share_per_transition": (1 - (Nn[:ntr] + Nn[1:ntr + 1]) / V).round(4).tolist(),
                "signal_share_mean": float(np.mean(1 - (Nn[:ntr] + Nn[1:ntr + 1]) / V)),
                "reversion_share_mean": float(np.mean(Nn[:ntr] / V)),
                "next_noise_share_mean": float(np.mean(Nn[1:ntr + 1] / V)),
            }
            bd = {}
            for q in range(10):
                g = decl == q
                Vq = sum(((D[i, g] - D[i, g].mean()) ** 2).sum() for i in range(ntr))
                N0 = sum(NV[g, i].sum() for i in range(ntr))
                N1 = sum(NV[g, i + 1].sum() for i in range(ntr))
                bd[q] = {"signal_share": float(1 - (N0 + N1) / Vq), "ceiling_pcc": float(np.sqrt(max(0, 1 - N1 / Vq))),
                         "share_of_total_V": float(Vq / V.sum()),
                         "mean_cp10k": float(np.mean(expr_rank[live][g]))}
            r["noise_budget"]["by_decile"] = bd
            w = np.array([((D[i] - D[i].mean()) ** 2) for i in range(ntr)])
            r["effective_n_genes_mean"] = float(np.mean([(wi.sum() ** 2) / (wi ** 2).sum() for wi in w]))
            r["top1pct_share_of_V"] = float(np.mean([np.sort(wi)[::-1][:max(1, len(wi) // 100)].sum() / wi.sum() for wi in w]))
            sv = V - Nn[:ntr] - Nn[1:ntr + 1]
            nc = {}
            for L in LAGS:
                if ntr - L < 2:
                    continue
                cs, va, vb = 0.0, 0.0, 0.0
                for i in range(ntr - L):
                    if sv[i] > 0 and sv[i + L] > 0:
                        a, b = D[i] - D[i].mean(), D[i + L] - D[i + L].mean()
                        cs += (a * b).sum() + (Nn[i + 1] if L == 1 else 0.0)
                        va += sv[i]
                        vb += sv[i + L]
                nc[L] = float(cs / np.sqrt(va * vb)) if va > 0 else np.nan
            r["noise_corrected_xs_rho"] = nc   # ratio of sums over pairs whose signal variance is positive
            # PCC reachable by components: perfect denoising of bin t only, true curve change only, both
            r["noise_budget"]["pcc_denoise_only_mean"] = float(np.mean(np.sqrt(np.clip(Nn[1:ntr] / V[1:], 0, 1))))
            r["noise_budget"]["pcc_true_change_only_mean"] = float(np.mean(np.sqrt(np.clip(sv[1:] / V[1:], 0, 1))))
            # validity of the plug-in noise: var(second difference) / its pure-noise value (>= 1 if noise model right)
            X = level_space(target, m)[live][:, :ntr + 1]
            D2 = X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
            N2 = NV[:, 2:ntr + 1] + 4 * NV[:, 1:ntr] + NV[:, :ntr - 1]
            R2 = ((D2 - D2.mean(0)) ** 2).sum(0) / N2.sum(0)
            r["noise_budget"]["R2_second_diff_over_noise_per_triple"] = R2.round(3).tolist()
            r["noise_budget"]["R2_median"] = float(np.median(R2))
            r["pure_noise_expected_rho1"] = float(np.mean([-Nn[i + 1] / np.sqrt(V[i] * V[i + 1]) for i in range(ntr - 1)]))
        if target == "orig":
            Dall = deltas(target, m, ntr)
            vals = []
            for i in range(ntr - 1):
                g = (m[:, i] > 0) & (m[:, i + 1] > 0) & (m[:, i + 2] > 0)
                vals.append(corr(Dall[i, g], Dall[i + 1, g]))
            r["xs_rho1_no_zero_crossing"] = float(np.mean(vals))
            r["clip_mass_mean"] = float(np.mean(np.abs(Dall[:, live]) >= 5 - 1e-4))
            r["clip_mass_mean_all_genes"] = float(np.mean(np.abs(Dall) >= 5 - 1e-4))
            vals = []
            for i in range(ntr - 1):
                zc = (np.abs(Dall[i]) >= 5 - 1e-4) | (np.abs(Dall[i + 1]) >= 5 - 1e-4)
                vals.append(float(np.mean(zc[live])))
            r["frac_pairs_with_atom"] = float(np.mean(vals))
        out[target] = r
    return out


def merge_pairs(m, s, f, n):
    k = (m.shape[1] // 2) * 2
    M, S2, F, N = [], [], [], []
    for a in range(0, k, 2):
        na, nb = n[a], n[a + 1]
        mm = (na * m[:, a] + nb * m[:, a + 1]) / (na + nb)
        e2 = (na * (s[:, a] ** 2 + m[:, a] ** 2) + nb * (s[:, a + 1] ** 2 + m[:, a + 1] ** 2)) / (na + nb)
        M.append(mm)
        S2.append(np.maximum(e2 - mm ** 2, 0))
        F.append((na * f[:, a] + nb * f[:, a + 1]) / (na + nb))
        N.append(na + nb)
    return np.stack(M, 1), np.sqrt(np.stack(S2, 1)), np.stack(F, 1), np.array(N, float)


def main():
    res = {}
    for name in SNAPS:
        d = load(name)
        m, s, f, n = d["m"], d["s"], d["f"], d["n"]
        ntr = TRAIN_SPLIT[name]
        cp = 1e4 * m / m.sum(0, keepdims=True)
        expr = cp.mean(1)
        res[name] = {"native": analyse(m, s, n, ntr, expr, "native bins")}
        M, Sm, F, N = merge_pairs(m, s, f, n)
        res[name]["merged_pairs"] = analyse(M, Sm, N, M.shape[1] - 1, expr, "bins merged pairwise")
        for key in ("native", "merged_pairs"):
            r = res[name][key]
            print(f"== {name} [{key}] cells={r['cells']}")
            for t in TARGETS:
                x = r[t]
                line = f"  {t:6s} xs_rho " + " ".join(f"L{L}={v:+.3f}" for L, v in x["xs_rho"].items())
                line += " | within-gene " + " ".join(f"L{L}={v:+.3f}" for L, v in x["within_gene_rho_all"].items())
                print(line)
                print("         xs_rho1 by decile " + " ".join(f"{v:+.2f}" for v in x["xs_rho1_by_decile"].values()))
                print("         within-gene rho1 by decile " + " ".join(f"{v[1]:+.2f}" for v in x["within_gene_rho_by_decile"].values()))
                if "noise_budget" in x:
                    nb = x["noise_budget"]
                    print(f"         ceiling PCC {nb['ceiling_pcc_mean_i>=1']:.3f} signal share {nb['signal_share_mean']:.3f} "
                          f"reversion share {nb['reversion_share_mean']:.3f} next-noise share {nb['next_noise_share_mean']:.3f} "
                          f"pure-noise rho1 {x['pure_noise_expected_rho1']:+.3f} | noise-corrected rho "
                          + " ".join(f"L{L}={v:+.3f}" for L, v in x["noise_corrected_xs_rho"].items())
                          + f" | eff genes {x['effective_n_genes_mean']:.0f} top1% {x['top1pct_share_of_V']:.2f}")
                    print(f"         PCC denoise-only {nb['pcc_denoise_only_mean']:.3f} true-change-only {nb['pcc_true_change_only_mean']:.3f} "
                          f"R2 median {nb['R2_median']:.2f}")
                    print("         signal share by decile " + " ".join(f"{v['signal_share']:+.2f}" for v in nb["by_decile"].values()))
                    print("         ceiling by decile      " + " ".join(f"{v['ceiling_pcc']:.2f}" for v in nb["by_decile"].values()))
                    print("         V share by decile      " + " ".join(f"{v['share_of_total_V']:.2f}" for v in nb["by_decile"].values()))
                if t == "orig":
                    print(f"         rho1 no-zero-crossing {x['xs_rho1_no_zero_crossing']:+.3f} clip mass(live) {x['clip_mass_mean']:.3f} "
                          f"pairs w/ atom {x['frac_pairs_with_atom']:.3f}")
                print(f"         rho1 vs shared-n spearman {x['xs_rho1_vs_shared_n_spearman']:+.2f} terciles "
                      + str({k: round(v, 3) for k, v in x['xs_rho1_by_shared_n_tercile'].items()}))
    with open(os.path.join(HERE, "noise_acf.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
