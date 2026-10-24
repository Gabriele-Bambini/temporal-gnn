"""Probe beyond per-gene time series (strict): does sharing strength ACROSS genes recover the current true level?

The noise budget (noise_acf.py) says that knowing the current true level mu_t would let a predictor reach PCC ~0.65
on l1, while every per-gene temporal filter (predictors.py, kalman_ml.py) stays at 0.16-0.28: the bottleneck is
estimating mu_t from a noisy, one-sided own history. If gene programs are low-rank in pseudotime, the sampling noise
of bin t (independent across genes) is mostly orthogonal to the few gene-loading directions and can be removed.
Model at transition i (bins 0..i only):
  Z = (L_g,b - mean_b L_g) / sigma_g  on b <= i, columns weighted by sqrt(n_b); sigma_g = rms delta-method noise sd
      of gene g over bins <= i (floor at the 10th percentile of positive values)
  rank-r SVD of Z -> reconstruction mu_hat (unscaled, mean added back)
  forecast L_{i+1} = mu_hat_i + phi (mu_hat_i - mu_hat_{i-1}); l1 target = forecast - L_i (observed)
  linear / orig: forecast mapped back to the raw mean (library of bin i); orig also through the finite-cell
  observation model (predictors.fc_obs)
Grid r in {1, 2, 3, 5, 8}, phi in {0, 0.5, 1}; rolling-origin selection as in predictors.py. Output: lowrank_probe.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, SNAPS, TRAIN_SPLIT, TARGETS, tgt  # noqa: E402
import predictors as pr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RANKS = (1, 2, 3, 5, 8)
PHIS = (0.0, 0.5, 1.0)


def lowrank_levels(L, RL, n, i, r):
    """Denoised levels of bins i-1 and i (G,) each, from bins 0..i."""
    X = L[:, :i + 1]
    mu = (X * n[None, :i + 1]).sum(1) / n[:i + 1].sum()
    sd = np.sqrt(RL[:, :i + 1].mean(1))
    pos = sd[sd > 0]
    floor = np.quantile(pos, 0.1) if len(pos) else 1.0
    sd = np.maximum(sd, floor)
    w = np.sqrt(n[:i + 1])[None, :]
    Z = (X - mu[:, None]) / sd[:, None] * w
    re = min(r, i)                                     # at most i components for i + 1 columns (centred)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    Zr = (U[:, :re] * S[:re]) @ Vt[:re]
    Xr = Zr / w * sd[:, None] + mu[:, None]
    return Xr[:, i - 1], Xr[:, i], float((S[:re] ** 2).sum() / (S ** 2).sum())


def main():
    out = {}
    for name in SNAPS:
        d = load(name)
        m, s, f, n = d["m"], d["s"], d["f"], d["n"]
        T = TRAIN_SPLIT[name]
        G = m.shape[0]
        Y = {t: np.stack([tgt(t, m[:, i], m[:, i + 1]) for i in range(T)]) for t in TARGETS}
        truths = {t: pr.Truth(Y[t]) for t in TARGETS}
        rng = np.random.default_rng(0)
        n_test = int(round(0.25 * G))
        masks = [[np.sort(rng.permutation(G)[:n_test]) for _ in range(T)] for _ in range(pr.N_DRAWS)]
        L, RL, _ = pr.kalman_inputs(m, s, n, "l1")
        Lobs = pr.L_of(m)
        cache = {}
        varexp = {}
        for r in RANKS:
            prev, cur = np.zeros((T, G)), np.zeros((T, G))
            prev[0] = cur[0] = Lobs[:, 0]
            for i in range(1, T):
                prev[i], cur[i], ve = lowrank_levels(L, RL, n, i, r)
                varexp.setdefault(r, []).append(ve)
            cache[r] = (prev, cur)
        configs = [(r, p) for r in RANKS for p in PHIS]

        def make(c):
            r, p = c
            prev, cur = cache[r]
            return pr.l1_to_targets(m, cur + p * (cur - prev))
        fam = pr.run_family("lowrank", configs, make, truths, T, configs.index((3, 0.5)), keep_all=True)
        res = {"variance_explained_by_rank_mean": {r: float(np.mean(v)) for r, v in varexp.items()}}
        for t in TARGETS:
            F = fam[t]
            res[t] = {"strict_RO": pr.mask_score(Y[t], F["seq"], masks),
                      "RO_selected": [str(configs[c]) for c in F["sel"]],
                      "per_config_i>=1": {str(c): {k: v for k, v in pr.mask_score(Y[t], F["all"][a], masks).items() if k != "per_transition_from_i0"}
                                          for a, c in enumerate(configs) if c[1] in (0.0, 0.5)}}
        # orig through the observation model: raw mean of the selected (orig criterion) low-rank forecast
        sel = fam["orig"]["sel"]
        lb = pr.lib(m)
        mh = np.stack([np.maximum(np.expm1(cache[configs[sel[i]][0]][1][i] + configs[sel[i]][1] *
                                           (cache[configs[sel[i]][0]][1][i] - cache[configs[sel[i]][0]][0][i])), 0) * lb[i] / 1e4
                       for i in range(T)])
        res["orig"]["obs_strict_RO"] = pr.mask_score(Y["orig"], pr.fc_obs(m, s, f, n, T, mh), masks)
        # fixed r = 3 / phi = 0 (pure low-rank denoising) through the obs model
        prev, cur = cache[3]
        mh3 = np.maximum(np.expm1(cur), 0) * lb[:T, None] / 1e4
        res["orig"]["obs_r3_phi0"] = pr.mask_score(Y["orig"], pr.fc_obs(m, s, f, n, T, mh3), masks)
        out[name] = res
        print(f"== {name}  variance explained by rank: " + " ".join(f"r{r}={v:.2f}" for r, v in res["variance_explained_by_rank_mean"].items()))
        for t in TARGETS:
            x = res[t]["strict_RO"]
            print(f"   {t:6s} lowrank RO  i>=1 PCC {x['pcc']:+.3f}+-{x['pcc_sd']:.3f} up {x['p200_up']:.3f} dn {x['p200_down']:.3f}"
                  f" | all {x['all_trans_pcc']:+.3f} {x['all_trans_p200_up']:.3f} {x['all_trans_p200_down']:.3f}")
            print("          per config (r, phi) PCC: " + " ".join(f"{k}:{v['pcc']:+.2f}" for k, v in res[t]["per_config_i>=1"].items()))
        for k in ("obs_strict_RO", "obs_r3_phi0"):
            x = res["orig"][k]
            print(f"   orig   {k:12s} i>=1 PCC {x['pcc']:+.3f} up {x['p200_up']:.3f} dn {x['p200_down']:.3f}"
                  f" | all {x['all_trans_pcc']:+.3f} {x['all_trans_p200_up']:.3f} {x['all_trans_p200_down']:.3f}")
    with open(os.path.join(HERE, "lowrank_probe.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
