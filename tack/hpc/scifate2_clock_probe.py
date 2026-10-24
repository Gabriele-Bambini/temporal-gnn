"""sci-FATE2 'do two independent clocks agree?' (adapter probe, NOT TACK).

Implements tack/prereg/2026-10-02-scifate2-clock-adapter-prereg.md (v2, sha256 10d74859...180b) literally.
Chronocell (commit a827f89) is the external decoder engine, called through its public fit and its model
functions get_logL / get_Y_hat; everything else (QC, gene rules, priors, label model, clocks, decision) is ours.

Usage: python scifate2_clock_probe.py --out ~/tack/out/scifate2_clock [--quick] [--procs 8]
"""
import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import chrono_common as cc  # noqa: E402  (adds Chronocell to sys.path, Tirosh list)
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from label_model import new_fraction  # noqa: E402

D = os.path.expanduser("~/tack/data/scifate2/")
GROUP_T = {"D3": 72.0, "D5": 120.0, "D6": 144.0, "D7": 168.0, "D8": 192.0}
REPS = ["r1", "r2", "r3"]
TOPO = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
TAU = (0.0, 2.5, 5.0)
M = 100
L_PULSE = 2.0
C_GRID = np.geomspace(3.0, 300.0, 41)
ALPHA_SCALES = (0.8, 1.0, 1.25)
HALF_LIFE_EXT = 7.1  # h, Sharova et al. 2009
EPS = 1e-300


# ----------------------------------------------------------------------------- data

def load(quick):
    import anndata

    C = anndata.read_h5ad(D + "GSE236512_processed_data_counting.h5ad")
    keep = C.obs["timepoint"].astype(str).isin(GROUP_T) & C.obs["rep"].astype(str).isin(REPS)
    keep = keep.to_numpy()
    obs = C.obs[keep].copy()
    tot = sp.csr_matrix(C.layers["total"][keep])
    new = sp.csr_matrix(C.layers["new"][keep])
    genes = np.asarray(C.var_names)
    del C
    Sf = anndata.read_h5ad(D + "GSE236512_processed_data_splicing.h5ad")
    U = sp.csr_matrix(Sf.layers["unspliced"][keep])
    S = sp.csr_matrix(Sf.layers["spliced"][keep])
    del Sf
    amb = np.load(os.path.expanduser("~/tack/out/ambiguous_share_per_gene.npy"))
    return obs, tot, new, U, S, genes, amb


def qc(obs, tot, genes):
    n_umi = np.asarray(tot.sum(1)).ravel()
    n_gen = np.asarray((tot > 0).sum(1)).ravel()
    mt = np.char.startswith(genes.astype(str), "mt-")
    mt_frac = np.asarray(tot[:, mt].sum(1)).ravel() / np.maximum(n_umi, 1)
    alpha = obs["total_TC_alpha"].to_numpy(float)
    pc = obs["p_c_TC"].to_numpy(float)
    ok = ~np.isnan(alpha) & ~np.isnan(pc)
    lo_u, hi_u = np.percentile(n_umi, [10, 90])
    lo_g, hi_g = np.percentile(n_gen, [10, 90])
    keep = (ok & (n_umi >= lo_u) & (n_umi <= hi_u) & (n_gen >= lo_g) & (n_gen <= hi_g)
            & (mt_frac < np.percentile(mt_frac, 90)) & (alpha > np.nanpercentile(alpha, 10)))
    a_lo, a_hi = np.percentile(alpha[keep], [1, 99])
    alpha_w = np.clip(alpha, a_lo, a_hi)
    info = dict(n_before=int(len(obs)), n_after=int(keep.sum()), umi_range=[float(lo_u), float(hi_u)],
                genes_range=[float(lo_g), float(hi_g)], alpha_winsor=[float(a_lo), float(a_hi)])
    return keep, alpha_w, info


def split(groups, keep, cap, seed=1):
    rng = np.random.default_rng(seed)
    fit = np.zeros(len(groups), bool)
    score = np.zeros(len(groups), bool)
    for g in np.unique(groups):
        idx = np.where((groups == g) & keep)[0]
        idx = rng.permutation(idx)
        half = len(idx) // 2
        fit[idx[:half][:cap]] = True
        score[idx[half:]] = True
    return fit, score


def col_mean_var(X):
    mu = np.asarray(X.mean(0)).ravel()
    ex2 = np.asarray(X.multiply(X).mean(0)).ravel()
    return mu, ex2 - mu**2


def norm_cov_offdiag_sparse(X):
    """Notebook estimate sum_{j!=k} cov_jk/(m_j m_k) / (p(p-1)) in O(nnz); ddof=1 as np.cov."""
    n, p = X.shape
    m, v = col_mean_var(X)
    v = v * n / (n - 1)
    rowsum = np.asarray(X @ (1.0 / m)).ravel()
    total = np.var(rowsum, ddof=1)
    diag = np.sum(v / m**2)
    return (total - diag) / ((p - 1) * p)


def depth_and_genes(U, S, fit, genes, amb, p_max, a=1.0):
    Uf, Sf = U[fit], S[fit]
    X = (Uf + Sf).tocsc()
    both = (np.asarray(Uf.mean(0)).ravel() > 0) & (np.asarray(Sf.mean(0)).ravel() > 0)
    Xb = X[:, both]
    sp_ = norm_cov_offdiag_sparse(Xb)
    xm, xv = col_mean_var(Xb)
    for _ in range(5):
        pmask = (xm > 0.01) & (xv / (a * xm + sp_ * xm**2) < 1.2)
        sp_ = norm_cov_offdiag_sparse(Xb[:, pmask])
    pois_idx = np.where(both)[0][pmask]
    depth_all = np.asarray((U + S)[:, pois_idx].mean(1)).ravel()
    r = depth_all / depth_all[fit].mean()
    um, uv = col_mean_var(Uf)
    sm, sv = col_mean_var(Sf)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_ratio = uv / (a * um + sp_ * um**2)
        s_ratio = sv / (a * sm + sp_ * sm**2)
        sel = ((um > 0.02) & (sm > 0.1) & (u_ratio > 1.2) & (s_ratio > 1.2)
               & (np.abs(np.log(sm / um)) < 4))
    g = genes.astype(str)
    sel &= ~np.isin(g, list(cc.CELL_CYCLE)) & ~np.char.startswith(g, "mt-") & (amb <= 0.3) & ~np.isnan(amb)
    idx = np.where(sel)[0]
    n_pass = int(len(idx))
    idx = idx[np.argsort(-s_ratio[idx])][:p_max]
    return r, idx, dict(sp=float(sp_), n_poissonian=int(pmask.sum()), n_genes_pass=n_pass, n_genes=int(len(idx)))


# ----------------------------------------------------------------------------- decoder

def fit_one(args):
    X, r, seed, epoch = args
    from Chronocell.inference import Trajectory

    t0 = time.time()
    tr = Trajectory(TOPO, TAU, model="two_species_ss", verbose=0)
    tr = tr.fit(X, m=M, n_init=1, params={"r": r}, perm_theta=False, epoch=epoch, parallel=False,
                n_threads=1, seed=seed, norm_Q=False)
    return dict(seed=seed, theta=tr.theta.copy(), tau=np.asarray(tr.tau).copy(), elbo=float(tr.elbos[0][-1]),
                seconds=time.time() - t0)


def grid_loglik(model, X, theta, tau, r):
    t = np.linspace(tau[0], tau[-1], M)
    return model.get_logL(X, theta, t, tau, TOPO, {"r": r}), t  # (n, L, m)


def dense3(U, S, idx, rows):
    X = np.zeros((int(rows.sum()), len(idx), 2))
    X[:, :, 0] = U[rows][:, idx].toarray()
    X[:, :, 1] = S[rows][:, idx].toarray()
    return X


# ----------------------------------------------------------------------------- priors (lineage-collapsed)

def age_prior(T, th, t):
    z, log_c, g, w0, w1 = th
    T0 = 48.0 + 36.0 / (1.0 + np.exp(-z))
    c = np.exp(log_c)
    A = T - T0
    w = 1.0 / (1.0 + np.exp(-(w0 + w1 * (T - 72.0) / 24.0)))
    q = np.zeros_like(t)
    if A <= 0:
        q[0] = 1.0
        return q
    d = t[1] - t[0]
    lo = np.concatenate([[0.0], t[1:] - d / 2]) * c
    hi = np.concatenate([t[:-1] + d / 2, [np.inf]]) * c
    lo, hi = np.minimum(lo, A), np.minimum(hi, A)
    width = np.maximum(hi - lo, 0.0)
    if abs(g) < 1e-9:
        mass = width
    else:
        ref = A if g > 0 else 0.0
        mass = np.exp(g * (lo - ref)) * np.expm1(g * width) / g
    mass = np.maximum(mass, 0.0)
    if mass.sum() <= 0:
        mass[0] = 1.0
    q = (1 - w) * mass / mass.sum()
    q[0] += w
    return q


def gauss_prior(T, th, t):
    mu0, s, log_sd = th
    mu = mu0 + s * (T - 72.0) / 24.0
    return np.exp(-0.5 * ((t - mu) / np.exp(log_sd)) ** 2)


def fit_prior(kind, Ms, Es, Ts, t, starts):
    f = age_prior if kind == "age" else gauss_prior
    n = sum(len(m) for m in Ms)

    def obj(x):
        tot = 0.0
        for Mi, Ei, T in zip(Ms, Es, Ts):
            q = np.maximum(f(T, x, t), 1e-12)
            q = q / q.sum()
            tot += np.sum(Mi + np.log(Ei @ q))
        return -tot / n

    best = None
    for x0 in starts:
        res = minimize(obj, np.asarray(x0, float), method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-4, fatol=1e-8))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    from multiprocessing import Pool

    from Chronocell.models import two_species_ss as model

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    rep = dict(prereg_sha256="10d748593a6d2e15699a3473dfec5a678a833a7128e994a97f203701ad2c180b",
               quick=a.quick, note="adapter probe on Chronocell; NOT TACK")
    T_start = time.time()

    def save():
        with open(os.path.join(out, "clock_probe.json"), "w") as fh:
            json.dump(rep, fh, indent=1)

    obs, tot, new, U, S, genes, amb = load(a.quick)
    groups = (obs["timepoint"].astype(str) + "_" + obs["rep"].astype(str)).to_numpy()
    Tcell = obs["timepoint"].astype(str).map(GROUP_T).to_numpy(float)
    keep, alpha, rep["qc"] = qc(obs, tot, genes)
    cap = 200 if a.quick else 1000
    fit, score = split(groups, keep, cap)
    if a.quick:  # smaller score set for the smoke run
        score &= np.random.default_rng(3).random(len(score)) < 0.25
    rep["cells"] = dict(fit=int(fit.sum()), score=int(score.sum()),
                        per_group={g: [int((fit & (groups == g)).sum()), int((score & (groups == g)).sum())]
                                   for g in np.unique(groups)})
    r, gidx, rep["genes"] = depth_and_genes(U, S, fit, genes, amb, 60 if a.quick else 150)
    rep["genes"]["names"] = genes[gidx].astype(str).tolist()
    print("cells", json.dumps(rep["cells"]), "genes", rep["genes"]["n_genes"], flush=True)
    save()
    if rep["genes"]["n_genes"] < (20 if a.quick else 50):
        rep["verdict"] = "STOP: fewer than 50 genes"
        save()
        return

    # frozen decoder: 8 random initialisations, keep the best ELBO
    Xf = dense3(U, S, gidx, fit)
    n_init, epoch = (2, 5) if a.quick else (8, 100)
    t0 = time.time()
    with Pool(min(a.procs, n_init)) as pool:
        fits = pool.map(fit_one, [(Xf, r[fit], s, epoch) for s in range(n_init)], chunksize=1)
    # Amendment A1 (declared before the full run): direction check from total counts and collection times only.
    from scipy.stats import spearmanr

    rhos = []
    for f in fits:
        lf, tt = grid_loglik(model, Xf, f["theta"], f["tau"], r[fit])
        post = np.exp(lf - logsumexp(lf, axis=(1, 2), keepdims=True))
        rhos.append(float(spearmanr(post.sum(1) @ tt, Tcell[fit]).correlation))
        del lf, post
    forward = [f for f, rho in zip(fits, rhos) if rho > 0]
    rep["decoder"] = dict(elbos=[f["elbo"] for f in fits], spearman_t_vs_T=rhos, seconds=time.time() - t0,
                          mean_fit_seconds=float(np.mean([f["seconds"] for f in fits])))
    if not forward:
        rep["verdict"] = "UNINFORMATIVE (amendment A1: no initialisation has process time increasing with T)"
        save()
        print("VERDICT", rep["verdict"], flush=True)
        return
    best = max(forward, key=lambda d: d["elbo"])
    theta, tau = best["theta"], best["tau"]
    rep["decoder"].update(best_seed=best["seed"], best_overall_seed=max(fits, key=lambda d: d["elbo"])["seed"],
                          tau=tau.tolist())
    beta_u, gamma_u = theta[:, -2], theta[:, -1]
    np.savez_compressed(os.path.join(out, "decoder.npz"), theta=theta, tau=tau, genes=genes[gidx])
    print("decoder", json.dumps(rep["decoder"]), flush=True)
    save()

    # lineage-collapsed fit-cell likelihood for the collection-time clocks
    logL_f, t = grid_loglik(model, Xf, theta, tau, r[fit])
    Lbar = logsumexp(logL_f, axis=1) - np.log(TOPO.shape[0])
    Mi = Lbar.max(1)
    Ei = np.exp(Lbar - Mi[:, None])
    Tfit = Tcell[fit]
    Ts = sorted(GROUP_T.values())
    Ms = [Mi[Tfit == T] for T in Ts]
    Es = [Ei[Tfit == T] for T in Ts]
    logit = lambda p: np.log(p / (1 - p))
    age_starts = [[logit((T0 - 48) / 36), np.log(c), 0.0, w0, 0.0]
                  for T0 in (60.0, 72.0) for c in (12.0, 24.0, 48.0) for w0 in (-2.0, 2.0)]
    gauss_starts = [[m0, s, np.log(sd)] for m0 in (0.0, 1.0) for s in (0.5, 1.0) for sd in (0.5, 1.5)]
    if a.quick:
        age_starts, gauss_starts = age_starts[:2], gauss_starts[:2]
    th_a, ll_a = fit_prior("age", Ms, Es, Ts, t, age_starts)
    th_g, ll_g = fit_prior("gauss", Ms, Es, Ts, t, gauss_starts)
    c_A = float(np.exp(th_a[1]))
    c_G = float(24.0 / th_g[1]) if th_g[1] > 0 else None
    c_U = 24.0
    c_E = float(np.median(gamma_u) * HALF_LIFE_EXT / np.log(2))
    T0_A = float(48.0 + 36.0 / (1.0 + np.exp(-th_a[0])))
    rep["clocks"] = dict(c_A=c_A, c_G=c_G, c_U=c_U, c_E=c_E,
                         age=dict(theta=th_a.tolist(), T0=T0_A, train_ll=ll_a),
                         gauss=dict(theta=th_g.tolist(), train_ll=ll_g))
    print("clocks", json.dumps(rep["clocks"]), flush=True)
    save()

    # label clock: uniform-prior posteriors of score cells, held-out new / total
    Xs = dense3(U, S, gidx, score)
    logL_s, _ = grid_loglik(model, Xs, theta, tau, r[score])
    q = np.exp(logL_s - logsumexp(logL_s, axis=(1, 2), keepdims=True))  # (n, L, m), uniform prior
    del logL_s
    new_s = new[score][:, gidx].toarray()
    tot_s = tot[score][:, gidx].toarray()
    old_s = tot_s - new_s
    read_tot = tot_s.sum(1)
    ok_cell = read_tot > 0
    a_s = alpha[score]
    g_s = groups[score]
    ug = np.unique(g_s)
    Y = model.get_Y_hat(theta, t, tau, TOPO, {"r": r[fit]})  # (L, m, p, 2)
    amb_g = amb[gidx]

    def group_scores(c, scale):
        h = L_PULSE / c
        Y0 = model.get_Y_hat(theta, np.clip(t - h, tau[0], None), tau, TOPO, {"r": r[fit]})
        pi = new_fraction(Y[..., 0], Y[..., 1], Y0[..., 0], Y0[..., 1], beta_u, gamma_u, h, amb_g)  # (L, m, p)
        pi_cell = np.einsum("nlm,lmp->np", q, pi)
        P = np.clip(scale * a_s[:, None] * pi_cell, 1e-6, 1 - 1e-6)
        ll = (new_s * np.log(P) + old_s * np.log1p(-P)).sum(1) / np.maximum(read_tot, 1)
        return np.array([ll[ok_cell & (g_s == g)].mean() for g in ug])

    prof = {sc: np.stack([group_scores(c, sc) for c in C_GRID], axis=1) for sc in ALPHA_SCALES}  # (G, nC)
    rng = np.random.default_rng(7)
    n_boot = 200 if a.quick else 2000
    boot_idx = rng.integers(0, len(ug), size=(n_boot, len(ug)))
    bands, argmax = {}, {}
    for sc, G in prof.items():
        Sc = G.mean(0)
        argmax[sc] = int(np.argmax(Sc))
        bmax = np.argmax(G[boot_idx].mean(1), axis=1)
        bands[sc] = [float(C_GRID[int(np.percentile(bmax, 2.5))]), float(C_GRID[int(np.percentile(bmax, 97.5))])]
    lo = min(b[0] for b in bands.values())
    hi = max(b[1] for b in bands.values())
    c_label = float(C_GRID[argmax[1.0]])

    def S_at(c, sc=1.0):
        return float(group_scores(c, sc).mean()) if c is not None else None

    rep["label_clock"] = dict(
        c_label=c_label, band=[lo, hi], band_by_alpha_scale={str(k): v for k, v in bands.items()},
        argmax_by_alpha_scale={str(k): float(C_GRID[v]) for k, v in argmax.items()},
        S_profile={str(k): v.mean(0).tolist() for k, v in prof.items()}, c_grid=C_GRID.tolist(),
        S_at_candidates=dict(c_A=S_at(c_A), c_G=S_at(c_G), c_U=S_at(c_U), c_E=S_at(c_E), c_label=S_at(c_label)),
        n_groups=int(len(ug)))

    # decision rule (pre-registration section 7)
    inside = lambda c: (c is not None) and (lo <= c <= hi)
    gate1 = 0 < argmax[1.0] < len(C_GRID) - 1
    med_unspliced_half_life_A = float(c_A * np.log(2) / np.median(beta_u))
    gate2 = med_unspliced_half_life_A < 1.0
    others = {"c_U": c_U, "c_E": c_E, "c_G": c_G}
    n_out = sum(1 for c in others.values() if c is not None and not inside(c))
    if not gate1:
        verdict = "UNINFORMATIVE (gate 1: label profile peaks at the grid edge)"
    elif not gate2:
        verdict = "UNINFORMATIVE (gate 2: implied unspliced half-life >= 1 h)"
    elif inside(c_A) and not inside(c_U) and not inside(c_E):
        verdict = "POSITIVE" + (" and POSITIVE vs VeloVAE-like" if (c_G is not None and not inside(c_G)) else
                                " (c_G inside the band or undefined)")
    elif inside(c_A) and all(inside(c) for c in others.values() if c is not None):
        verdict = "UNINFORMATIVE (band contains all candidates)"
    elif (not inside(c_A)) and n_out >= 2:
        verdict = "NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"
    rep["gates"] = dict(gate1_peak_inside=bool(gate1), gate2_unspliced_half_life_h_at_cA=med_unspliced_half_life_A,
                        gate2_pass=bool(gate2))
    rep["rates_per_hour"] = {
        name: dict(beta_median=float(np.median(beta_u / c)), gamma_median=float(np.median(gamma_u / c)),
                   mRNA_half_life_median_h=float(np.log(2) / np.median(gamma_u / c)))
        for name, c in (("c_label", c_label), ("c_A", c_A))}
    rep["verdict"] = verdict
    rep["total_seconds"] = time.time() - T_start
    np.savez_compressed(os.path.join(out, "label_profiles.npz"), c_grid=C_GRID, groups=ug,
                        **{f"G_{k}": v for k, v in prof.items()})
    save()
    print("label", json.dumps(dict(c_label=c_label, band=[lo, hi], S_at=rep["label_clock"]["S_at_candidates"])),
          flush=True)
    print("VERDICT", verdict, flush=True)


if __name__ == "__main__":
    main()
