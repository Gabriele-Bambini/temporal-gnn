"""Precision study of the label clock (design study, not a test of real data).

Question: with a PERFECT decoder (true theta, true posteriors), how wide is the label-clock band, and what narrows
it? World = SELF of the simulation recovery (truth = frozen decoder of job 679647, c = c_A = 17.86 h/unit), same
real cells (T, group, depth, detection alpha). Only the decoder-oracle arm is scored, so no Chronocell fit is needed.

Configurations:
  base      : 150 genes, real score cells
  genes4    : 600 virtual genes (theta rows resampled with replacement, levels jittered by lognormal sd 0.2)
  cells2    : score cells duplicated within groups (twice the cells, fresh counts)
  fastgenes : gamma x 3 for every gene (mRNA half-lives / 3)
  r4        : + four early groups at D3 + 5/10/15/20 h (1,000 cells each, alpha and depth resampled from D3)
  delta     : posteriors replaced by the TRUE (lineage, time) of each cell (upper bound: no time blur)
  combo     : genes4 + cells2 + r4
For each: label profile S(c) on the 41-point grid; bands from (i) bootstrap of groups (the frozen rule) and
(ii) bootstrap of cells within groups; per-group argmax; whether c_U = 24 (x1.34) and c_E (x3.3) are excluded.

Usage: python precision_study.py [--quick] [--configs base,genes4,...]
"""
import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

import scifate2_clock_probe as probe  # noqa: E402
import sim_recovery as sr  # noqa: E402
from label_model import new_fraction  # noqa: E402

CONFIGS = ["base", "genes4", "cells2", "fastgenes", "r4", "delta", "combo"]


def per_cell_ll(model, theta, tau, q, new_s, tot_s, a_s, amb_g):
    """Per-read mean log-likelihood of each cell for every c of the grid (alpha x 1), shape (n, nC)."""
    t = np.linspace(tau[0], tau[-1], probe.M)
    Y = model.get_Y_hat(theta, t, tau, probe.TOPO, {})
    beta, gamma = theta[:, -2], theta[:, -1]
    old_s = tot_s - new_s
    read_tot = np.maximum(tot_s.sum(1), 1)
    out = np.empty((len(a_s), len(probe.C_GRID)))
    for k, c in enumerate(probe.C_GRID):
        h = probe.L_PULSE / c
        Y0 = model.get_Y_hat(theta, np.clip(t - h, tau[0], None), tau, probe.TOPO, {})
        pi = new_fraction(Y[..., 0], Y[..., 1], Y0[..., 0], Y0[..., 1], beta, gamma, h, amb_g)
        P = np.clip(a_s[:, None] * np.einsum("nlm,lmp->np", q, pi), 1e-6, 1 - 1e-6)
        out[:, k] = (new_s * np.log(P) + old_s * np.log1p(-P)).sum(1) / read_tot
    return out


def bands(ll, groups, rng, n_boot):
    ug = np.unique(groups)
    idx_g = [np.where(groups == g)[0] for g in ug]
    G = np.stack([ll[i].mean(0) for i in idx_g])  # (n_groups, nC)
    S = G.mean(0)
    am = int(np.argmax(S))
    per_group = [float(probe.C_GRID[int(np.argmax(G[j]))]) for j in range(len(ug))]
    bg = np.argmax(G[rng.integers(0, len(ug), size=(n_boot, len(ug)))].mean(1), axis=1)
    bc = np.empty(n_boot, int)
    for b in range(n_boot):
        Gb = np.stack([ll[rng.choice(i, size=len(i), replace=True)].mean(0) for i in idx_g])
        bc[b] = int(np.argmax(Gb.mean(0)))
    pct = lambda v: [float(probe.C_GRID[int(np.floor(np.percentile(v, 2.5)))]),
                     float(probe.C_GRID[int(np.ceil(np.percentile(v, 97.5)))])]
    lc = np.log(probe.C_GRID)
    curv = float(np.gradient(np.gradient(S, lc), lc)[am]) if 0 < am < len(S) - 1 else None
    return dict(c_label=float(probe.C_GRID[am]), band_groups=pct(bg), band_cells=pct(bc),
                per_group_argmax=dict(zip([str(g) for g in ug], per_group)), curvature_logc=curv,
                S_profile=S.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()
    from Chronocell.models import two_species_ss as model

    out = os.path.expanduser(f"~/tack/out/precision_study{'_quick' if a.quick else ''}")
    os.makedirs(out, exist_ok=True)
    T0 = time.time()
    real = json.load(open(os.path.join(sr.OUT_REAL, "clock_probe.json")))
    dec = np.load(os.path.join(sr.OUT_REAL, "decoder.npz"), allow_pickle=True)
    theta_fit, tau = dec["theta"], dec["tau"]
    th_age = real["clocks"]["age"]["theta"]
    c_true = real["clocks"]["c_A"]
    obs, tot, new, U, S, genes, amb = probe.load(False)
    groups = (obs["timepoint"].astype(str) + "_" + obs["rep"].astype(str)).to_numpy()
    Tcell = obs["timepoint"].astype(str).map(probe.GROUP_T).to_numpy(float)
    keep, alpha, _ = probe.qc(obs, tot, genes)
    fit, score = probe.split(groups, keep, 1000)
    r, gidx, _ = probe.depth_and_genes(U, S, fit, genes, amb, 150)
    del tot, new, U, S
    amb150 = np.nan_to_num(amb[gidx], nan=0.0)
    base_T, base_g, base_a, base_r = Tcell[score], groups[score], alpha[score], r[score]
    if a.quick:
        sub = np.random.default_rng(3).random(len(base_T)) < 0.2
        base_T, base_g, base_a, base_r = base_T[sub], base_g[sub], base_a[sub], base_r[sub]
    n_boot = 100 if a.quick else 1000
    rep = dict(c_true=c_true, c_U=24.0, c_E=real["clocks"]["c_E"], quick=a.quick, configs={})
    t_grid = np.linspace(tau[0], tau[-1], probe.M)

    for cfg in a.configs.split(","):
        tc = time.time()
        rng = np.random.default_rng(1000 + CONFIGS.index(cfg))
        theta = theta_fit.copy()
        ambg = amb150.copy()
        Tc, gc, ac, rc = base_T.copy(), base_g.copy(), base_a.copy(), base_r.copy()
        if cfg in ("genes4", "combo"):
            pick = rng.integers(0, len(theta_fit), size=4 * len(theta_fit))
            theta = theta_fit[pick].copy()
            n_states = theta.shape[1] - 2
            theta[:, :n_states] *= np.exp(rng.normal(0, 0.2, size=(len(pick), 1)))
            ambg = amb150[pick]
        if cfg == "fastgenes":
            theta[:, -1] *= 3.0
        if cfg in ("cells2", "combo"):
            Tc, gc, ac, rc = (np.concatenate([x, x]) for x in (Tc, gc, ac, rc))
        if cfg in ("r4", "combo"):
            d3 = np.where(base_T == 72.0)[0]
            for hrs in (5, 10, 15, 20):
                pick = rng.choice(d3, size=250 if a.quick else 1000, replace=True)
                Tc = np.concatenate([Tc, np.full(len(pick), 72.0 + hrs)])
                gc = np.concatenate([gc, np.full(len(pick), f"D3+{hrs}h_r4")])
                ac = np.concatenate([ac, base_a[pick]])
                rc = np.concatenate([rc, base_r[pick]])
        # simulate (states from the age prior, counts, labels) with the true model of this configuration
        Ub = np.ones(len(theta))
        rs, rl = np.random.default_rng(100), np.random.default_rng(201)
        th = np.array(th_age, float)
        tf = np.linspace(tau[0], tau[-1], sr.FINE * probe.M)
        Uc, Sc, totc, labels = sr.simulate(model, theta, Ub, tau, th, [c_true], Tc, rc, ac, ambg, rs, rl)
        newc = labels[0]
        if cfg == "delta":
            # replay the generating stream to recover (lineage, fine time), mapped to the analysis grid
            rs2 = np.random.default_rng(100)
            lin = rs2.integers(0, probe.TOPO.shape[0], size=len(Tc))
            tix = np.empty(len(Tc), int)
            th2 = th.copy()
            th2[1] = np.log(c_true)
            for T in np.unique(Tc):
                sel = np.where(Tc == T)[0]
                qq = np.maximum(probe.age_prior(T, th2, tf), 0)
                tix[sel] = rs2.choice(len(tf), size=len(sel), p=qq / qq.sum())
            m_idx = np.clip(np.rint(tf[tix] / (t_grid[1] - t_grid[0])).astype(int), 0, probe.M - 1)
            q = np.zeros((len(Tc), probe.TOPO.shape[0], probe.M))
            q[np.arange(len(Tc)), lin, m_idx] = 1.0
        else:
            X = np.stack([Uc, Sc], axis=-1)
            lf = model.get_logL(X, theta, t_grid, tau, probe.TOPO, {"r": rc})
            q = np.exp(lf - logsumexp(lf, axis=(1, 2), keepdims=True))
            del lf
        ll = per_cell_ll(model, theta, tau, q, newc, totc, np.clip(ac, 1e-6, None), ambg)
        res = bands(ll, gc, np.random.default_rng(7), n_boot)
        for key in ("band_groups", "band_cells"):
            lo, hi = res[key]
            res[f"{key}_ratio"] = hi / lo
            res[f"{key}_contains_truth"] = bool(lo / sr.RSTEP <= c_true <= hi * sr.RSTEP)
            res[f"{key}_excludes_cU"] = not (lo / sr.RSTEP <= 24.0 <= hi * sr.RSTEP)
            res[f"{key}_excludes_cE"] = not (lo / sr.RSTEP <= rep["c_E"] <= hi * sr.RSTEP)
        res.update(n_cells=int(len(Tc)), n_genes=int(len(theta)), n_groups=int(len(np.unique(gc))),
                   seconds=time.time() - tc)
        rep["configs"][cfg] = res
        with open(os.path.join(out, "precision.json"), "w") as fh:
            json.dump(rep, fh, indent=1)
        print(cfg, json.dumps({k: v for k, v in res.items() if k not in ("S_profile", "per_group_argmax")}),
              flush=True)
    rep["seconds"] = time.time() - T0
    with open(os.path.join(out, "precision.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print("DONE", rep["seconds"], flush=True)


if __name__ == "__main__":
    main()
