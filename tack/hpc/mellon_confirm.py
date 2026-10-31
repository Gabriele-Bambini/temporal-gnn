"""Pre-registered test of a time-continuous cell-state density model (Mellon, Otto et al. 2023, external method behind
an adapter) against the 50/50 mix (tack/prereg/2026-10-31-mellon-prereg.md).

Signal: a held-out stage looks like a mixture of states already present at its neighbours, with proportions that
change. Mellon models log cell-state density as a Gaussian process over state and time; its time-sensitive version
interpolates density between measured time points. Prediction for held stage k: the cells of stages k-1 and k+1 (the
mix's own cells) reweighted by w(x) = rho(x, t_k) / (rho(x, t_{k-1}) / 2 + rho(x, t_{k+1}) / 2).

Per held stage k (strict: stage k never enters any fit):
  1. cells = atlas_prep.load / zf_prep.load (seed 0, the frozen products' cells and order); preprocessing
     atlas_prep.transform with fit = every stage except k (gene filter, normalisation, 2000 HVG, PCA 30)
  2. embedding: primary 'dm25' = Palantir-style diffusion map (own code: kNN 30, adaptive Gaussian kernel with
     sigma_i = distance to the 10th neighbour, symmetrised, row-normalised; 25 non-trivial eigenvectors scaled
     lambda / (1 - lambda)), as in the Mellon paper's gastrulation analysis (25 components, Palantir);
     descriptive 'pc30' = the 30 PCs directly
  3. mellon.TimeSensitiveDensityEstimator(normalize_per_time_point=True), defaults otherwise, fitted on all cells of
     every stage except k with their real times (atlas days 6.5 + 0.25 i; zebrafish hpf 18 + 2 i)
  4. log density at the cells of k-1 and k+1 for t_{k-1}, t_k, t_{k+1}; w as above, divided by its mean under the mix
     (mean over k-1 cells / 2 + mean over k+1 cells / 2), clipped to [0.05, 20] (the frozen tilt_rw clip)
  5. the frozen interpolation loop (atlas_confirm.run_interp seeds and subsamples, interp<k>.npz PCA space, ED_U);
     'mellon' = subsampled k-1 / k+1 cells with weights (1/2) w / n_side; control 'mix_full' = same with w = 1
Parity gate: the interp<k>.npz fit cells must be the loaded stage k-1 / k+1 cells (same samples, same order) and the
frozen rivals must reproduce the frozen confirm json (max |ED_U diff| <= 1e-9).
Usage (in ~/envs/mellon): python mellon_confirm.py --dataset atlas|zf --out <json> --frozen <confirm json>
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_confirm as AC  # noqa: E402
import interp_bench_frozen as IB  # noqa: E402

KNN, KA, N_DM = 30, 10, 25
CLIP = (0.05, 20.0)
FROZEN_METHODS = ("prev", "next", "union", "indep", "ot_pairs")
DATA = {"atlas": {"out": "~/tack/out/atlas", "ks": list(range(1, 8)), "time": lambda i: 6.5 + 0.25 * i},
        "zf": {"out": "~/tack/out/zf", "ks": list(range(1, 10)), "time": lambda i: 18.0 + 2.0 * i}}


def diffusion_map(Z, n_comp=N_DM, knn=KNN, ka=KA):
    """Palantir-style multiscale diffusion components (own implementation)."""
    nn = NearestNeighbors(n_neighbors=knn).fit(Z)
    dist, ind = nn.kneighbors(Z)
    sigma = np.maximum(dist[:, ka - 1], 1e-12)
    n = len(Z)
    rows = np.repeat(np.arange(n), knn)
    vals = np.exp(-(dist.ravel() / sigma[rows]) ** 2)
    W = sp.csr_matrix((vals, (rows, ind.ravel())), shape=(n, n))
    W = W + W.T
    d = np.asarray(W.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    lam, U = eigsh(Dm @ W @ Dm, k=n_comp + 11, which="LA")
    o = np.argsort(-lam)
    lam, U = lam[o], U[:, o]
    psi = (Dm @ U)
    psi = psi / np.linalg.norm(psi, axis=0)
    keep = np.where(lam < 1 - 1e-6)[0][:n_comp]          # eigenvalue 1 = trivial / disconnected-component indicators
    lam, psi = lam[keep], psi[:, keep]
    return psi * (lam / (1 - lam))[None, :]


def weights(lr_prev, lr_mid, lr_next, side_a):
    """Clipped density-ratio weights for the pooled neighbour cells (side_a: True for stage k-1 cells)."""
    m = np.maximum(np.maximum(lr_prev, lr_next), lr_mid)
    w = np.exp(lr_mid - m) / (0.5 * np.exp(lr_prev - m) + 0.5 * np.exp(lr_next - m))
    w = w / (0.5 * w[side_a].mean() + 0.5 * w[~side_a].mean())
    return np.clip(w, *CLIP)


def weighted_pred(ZA, XA, ZB, XB, wA, wB):
    v = np.concatenate([0.5 * wA / len(wA), 0.5 * wB / len(wB)])
    return np.vstack([ZA, ZB]), np.vstack([XA, XB]), v / v.sum()


def mellon_logdens(E, times, rows, tq):
    import mellon
    m = mellon.TimeSensitiveDensityEstimator(normalize_per_time_point=True)
    # amendment 2026-10-31 (crash fix, zebrafish k = 2): duplicated cells give zero nearest-neighbour distances and
    # Mellon's length-scale heuristic fails; a fixed jitter of 1e-6 standard deviations separates them
    E = E + np.random.default_rng(0).normal(0.0, 1e-6, E.shape) * E.std(0)[None, :]
    m.fit(E, times)
    return {t: np.asarray(m.predict(E[rows], float(t))) for t in tq}, {"ls": float(m.ls), "ls_time": float(m.ls_time)}


def run_k(name, X, stage, smp, genes, cc_genes, k, AP):
    cfg = DATA[name]
    t_of = cfg["time"]
    fit = np.where(stage != k)[0]
    held = np.where(stage == k)[0]
    r = AP.transform(X, fit, held, genes, cc_genes)
    Zfit, st_fit = r["Z_fit"].astype(np.float64), stage[fit]
    nb = np.where((st_fit == k - 1) | (st_fit == k + 1))[0]          # rows of the neighbour cells inside fit
    side_a = st_fit[nb] == k - 1
    times = np.array([t_of(i) for i in st_fit])
    tq = (t_of(k - 1), t_of(k), t_of(k + 1))
    out = {"n_fit": int(len(fit)), "embed": {}}
    W = {}
    for emb in ("dm25", "pc30"):
        t0 = time.time()
        E = diffusion_map(Zfit) if emb == "dm25" else Zfit
        lr, hp = mellon_logdens(E, times, nb, tq)
        w = weights(lr[tq[0]], lr[tq[1]], lr[tq[2]], side_a)
        W[emb] = w
        gap = np.abs(lr[tq[0]] - lr[tq[2]])
        out["embed"][emb] = {**hp, "sec": round(time.time() - t0, 1), "ess_frac": float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                             "clipped_frac": float(np.mean((w <= CLIP[0]) | (w >= CLIP[1]))),
                             "weight_on_prev": float(0.5 * w[side_a].mean() / (0.5 * w[side_a].mean() + 0.5 * w[~side_a].mean())),
                             "corr_logw_gap": float(np.corrcoef(np.log(w), gap)[0, 1])}
    return out, W, fit[nb], side_a


def interp_k(data, k, W, reps=3):
    """Frozen interpolation loop (atlas_confirm.run_interp) plus 'mellon_<emb>' and 'mix_full'."""
    d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
    comps = d["pca_comps"].astype(np.float64)
    smp_held = d["sample_held"].astype(str)
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in set(IB.S_GENES + IB.G2M_GENES) for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    cc_fit = (Xcc - Xcc.mean(0)).mean(1)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], Xh[it], S, with_sd=False)
    pools = [p for p in sorted(set(smp_held)) if (smp_held == p).sum() >= AC.MIN_POOL]
    ptg = {p: IB.Target(Zh[smp_held == p], None, S, with_sd=False) for p in pools}
    # W[emb] is ordered as the fit rows of interp<k>.npz (stage k-1 and k+1 cells in loaded order)
    out = {"pools": pools, "reps": []}
    for r in range(reps):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        M.pop("_plan", None)
        M.pop("_uot", None)
        rec = {}
        for name, (Z, X, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
            if name in FROZEN_METHODS:
                rec[name] = {"ED_U": IB.ed_uv(Zs, ws, tgt)[0], "ED_U_gene": IB.ed_uv(Xs, ws, tgt, gene=True)[0],
                             "pool_ED_U": {p: IB.ed_uv(Zs, ws, ptg[p])[0] for p in pools}}
        mr = np.random.default_rng(555000 + 1000 * k + r)
        arms = {"mix_full": (np.ones(len(iA)), np.ones(len(iB)))}
        for emb, w in W.items():
            arms[f"mellon_{emb}"] = (w[A[iA]], w[B[iB]])
        for name, (wA, wB) in arms.items():
            Z, X, v = weighted_pred(ZA, XA, ZB, XB, wA, wB)
            Zs, Xs, vs = IB.subsample_pred(Z, X, v, n_pred, mr)
            rec[name] = {"ED_U": IB.ed_uv(Zs, vs, tgt)[0], "ED_U_gene": IB.ed_uv(Xs, vs, tgt, gene=True)[0],
                         "pool_ED_U": {p: IB.ed_uv(Zs, vs, ptg[p])[0] for p in pools}}
        out["reps"].append(rec)
    names = list(out["reps"][0])
    out["mean"] = {m: {key: float(np.mean([rp[m][key] for rp in out["reps"]])) for key in ("ED_U", "ED_U_gene")}
                   for m in names}
    out["pool_mean"] = {m: {p: float(np.mean([rp[m]["pool_ED_U"][p] for rp in out["reps"]])) for p in pools}
                        for m in names}
    return out, d


def beats(units, a, b):
    n = len(units)
    need = int(np.ceil(2 * n / 3))
    w = sum(u[a] < u[b] for u in units.values())
    mean_ok = bool(np.mean([u[a] for u in units.values()]) < np.mean([u[b] for u in units.values()]))
    return {"wins": int(w), "of": n, "need": need, "mean_ok": mean_ok, "pass": bool(w >= need and mean_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATA), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--ks", type=int, nargs="*")
    args = ap.parse_args()
    logging.getLogger("mellon").setLevel(logging.WARNING)
    import importlib

    import anndata as ad
    import pandas as pd
    import atlas_prep as AP
    loader = AP if args.dataset == "atlas" else importlib.import_module("zf_prep")
    X, obs, genes = loader.load()
    if args.dataset == "atlas":
        stage = np.array([AP.STAGES.index(s) for s in obs["stage"]])
        smp = obs["sample"].to_numpy().astype(str)
    else:
        stage = np.array([loader.TPS.index(t) for t in obs["timepoint"]])
        smp = obs["embryo"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc_genes = set(str(g).capitalize() for g in list(s_g) + list(g2m))
    data = os.path.expanduser(DATA[args.dataset]["out"])
    fz = json.load(open(args.frozen))
    t0 = time.time()
    res = {"config": vars(args), "interp": {}, "parity": {}}
    diffs = []
    for k in (args.ks or DATA[args.dataset]["ks"]):
        info, W, nb_idx, side_a = run_k(args.dataset, X, stage, smp, genes, cc_genes, k, AP)
        r, d = interp_k(data, k, W)
        same = bool(len(nb_idx) == len(d["sample_fit"]) and np.array_equal(smp[nb_idx], d["sample_fit"].astype(str))
                    and np.array_equal(stage[nb_idx], d["stage_fit"]))
        diffs += [abs(r["mean"][m]["ED_U"] - fz["interp"][str(k)]["mean"][m]["ED_U"]) for m in FROZEN_METHODS]
        res["parity"][k] = same
        res["interp"][k] = {**r, **info}
        print("interp", k, {m: round(v["ED_U"], 3) for m, v in r["mean"].items()},
              {e: {x: round(y, 3) for x, y in v.items()} for e, v in info["embed"].items()}, "cells_aligned", same,
              f"[{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(args.out, "w"), indent=1, default=float)
    units = {k: {m: v["mean"][m]["ED_U"] for m in v["mean"]} for k, v in res["interp"].items()}
    res["parity_max_abs_diff"] = float(max(diffs))
    res["parity_cells_aligned"] = all(res["parity"].values())
    res["verdicts"] = {"H1_mellon_vs_union": beats(units, "mellon_dm25", "union"),
                       "H1b_mellon_vs_mix_full": beats(units, "mellon_dm25", "mix_full"),
                       "desc_pc30_vs_union": beats(units, "mellon_pc30", "union"),
                       "desc_mix_full_vs_union": beats(units, "mix_full", "union")}
    res["seconds"] = round(time.time() - t0, 1)
    print("PARITY", res["parity_max_abs_diff"], res["parity_cells_aligned"], flush=True)
    print("VERDICTS", json.dumps(res["verdicts"]), flush=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
