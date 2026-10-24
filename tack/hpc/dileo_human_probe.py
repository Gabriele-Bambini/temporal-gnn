"""Probe: how were Dileo's '-human' gastrulation files and the shared pseudotime built?

Part 1 (expression + gene mapping, binning-free): per node gene, the number of nonzero cells over all bins
(sum_b frac_nonzero_b * n_b, an integer that does not depend on normalisation) and the overall mean
(sum_b s04_sum_b / 9815) are compared with candidate mouse matrices (X, spliced, spliced+unspliced) and
normalisations, matching genes by upper-cased mouse symbol and, for the rest, by nonzero-count fingerprint.
Part 2 (pseudotime): scVelo velocity pseudotime and latent time on the reproduced mouse pipeline
(min_shared_counts=20, moments 30 pcs / 30 neighbours on the stored X_pca), scored against the 67-point
cumulative cell-count profile implied by the 35-bin mouse and 32-bin human node files (same pseudotime).
Writes ~/tack/out/dileo_bins/human_probe.json; every step is wrapped so one failure does not stop the rest.
"""
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvelo as scv

H5 = os.path.expanduser("~/tack/data/erythroid/erythroid_lineage.h5ad")
GRN = os.path.expanduser("~/tgl-grn/GRN")
OUT = os.path.expanduser("~/tack/out/dileo_bins")
REP = {}


def log(key, value):
    REP[key] = value
    print(key, value if isinstance(value, str) else json.dumps(value)[:3000], flush=True)
    with open(os.path.join(OUT, "human_probe.json"), "w") as fh:
        json.dump(REP, fh, indent=1)


def node_stats(name):
    d = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"))
    nb = len([c for c in d.columns if c.endswith("_s00_mean")])
    m = d[[f"t{b:03d}_s00_mean" for b in range(nb)]].to_numpy(float)
    f = d[[f"t{b:03d}_s03_frac_nonzero" for b in range(nb)]].to_numpy(float)
    s = d[[f"t{b:03d}_s04_sum" for b in range(nb)]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = s / m
    n = np.array([int(round(np.nanmedian(r[m[:, b] > 0, b]))) for b in range(nb)])
    return d["gene"].astype(str).to_numpy(), n, (f * n).sum(1), s.sum(1) / n.sum()


def cdf_points():
    pts = []
    for name, nb in (("gastrulation", 35), ("gastrulation-human", 32)):
        _, n, _, _ = node_stats(name)
        pts += [((k + 1) / nb, int(c)) for k, c in enumerate(np.cumsum(n))]
    return sorted(pts)


def cdf_l1(pt, pts):
    pt = np.asarray(pt, float)
    pt = (pt - np.nanmin(pt)) / (np.nanmax(pt) - np.nanmin(pt))
    return int(sum(abs(int((pt < x).sum()) - c) for x, c in pts[:-1]))


def step(name, fn):
    try:
        t = time.time()
        fn()
        log(f"{name}_seconds", round(time.time() - t, 1))
    except Exception:  # noqa: BLE001
        log(f"{name}_error", traceback.format_exc()[-2500:])


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = sc.read_h5ad(H5)
    log("var_columns", list(raw.var.columns))
    log("obs_columns", list(raw.obs.columns))
    X = raw.X.tocsr() if sp.issparse(raw.X) else sp.csr_matrix(raw.X)
    S = raw.layers["spliced"].tocsr()
    U = raw.layers["unspliced"].tocsr()
    log("X_info", dict(dtype=str(X.dtype), max=float(X.max()), integer=bool(np.allclose(X.data, np.round(X.data))),
                       equals_spliced=bool((X != S).nnz == 0), row_sums_head=np.asarray(X.sum(1)).ravel()[:5].tolist()))
    genes_h, n_h, nnz_h, mean_h = node_stats("gastrulation-human")
    log("human_node", dict(n_genes=len(genes_h), n_cells=int(n_h.sum()), nnz_integer=bool(np.allclose(nnz_h, np.round(nnz_h), atol=1e-3)),
                           numeric_ids=int(sum(g.isdigit() for g in genes_h))))
    mouse = np.asarray(raw.var_names).astype(str)
    upper = pd.Series(np.arange(len(mouse)), index=pd.Index([g.upper() for g in mouse]))
    upper = upper[~upper.index.duplicated(keep=False)]
    hit = np.array([g in upper.index for g in genes_h])
    idx = np.array([upper[g] if h else -1 for g, h in zip(genes_h, hit)])
    log("symbol_match", dict(matched=int(hit.sum()), of=len(genes_h)))

    mats = {"X": X, "spliced": S, "spliced_plus_unspliced": (S + U).tocsr()}
    best = {}

    def part1():
        for key, M in mats.items():
            nnz = np.diff(M.tocsc().indptr)
            ok = np.abs(nnz[idx[hit]] - nnz_h[hit]) < 0.5
            best[key] = float(ok.mean())
            log(f"nnz_match_{key}", dict(frac_symbol_genes_equal=float(ok.mean())))
        key = max(best, key=best.get)
        M = mats[key]
        nnz = np.diff(M.tocsc().indptr)
        # fingerprint the genes that failed the symbol route: unique nonzero counts
        fail = np.where(~hit | (np.abs(nnz[np.maximum(idx, 0)] - nnz_h) >= 0.5))[0]
        cnt = pd.Series(nnz).value_counts()
        uniq = [int(np.round(nnz_h[i])) for i in fail if cnt.get(int(np.round(nnz_h[i])), 0) == 1]
        log("fingerprint_unmatched", dict(n_unmatched=len(fail), unique_count_candidates=len(uniq),
                                          examples=[str(genes_h[i]) for i in fail[:15]]))
        tot_all = np.asarray(M.sum(1)).ravel()
        sel = idx[hit]
        cands = {}
        for norm in ("none", "total_1e4", "total_median", "total_1e4_mapped_genes", "total_median_mapped_genes"):
            if norm == "none":
                scale = np.ones(M.shape[0])
            else:
                tot = tot_all if "mapped" not in norm else np.asarray(M[:, sel].sum(1)).ravel()
                target = 1e4 if "1e4" in norm else np.median(tot)
                scale = target / np.maximum(tot, 1e-12)
            N = sp.diags(scale) @ M[:, sel]
            for lg in (False, True):
                V = N.copy()
                if lg:
                    V.data = np.log1p(V.data)
                m = np.asarray(V.mean(0)).ravel()
                rel = np.abs(m - mean_h[hit]) / np.maximum(np.abs(mean_h[hit]), 1e-12)
                cands[f"{norm}{'_log1p' if lg else ''}"] = dict(median_rel=float(np.median(rel)), p90_rel=float(np.quantile(rel, 0.9)))
        log(f"normalisation_{key}", cands)

    step("part1", part1)

    pts = cdf_points()
    log("cdf_points", len(pts))
    ad = raw.copy()
    scv.pp.filter_and_normalize(ad, min_shared_counts=20)
    scv.pp.moments(ad, n_pcs=30, n_neighbors=30)

    def vel():
        scv.tl.velocity(ad, mode="stochastic")
        scv.tl.velocity_graph(ad, n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
        scv.tl.velocity_pseudotime(ad)
        log("velocity_pseudotime_l1", cdf_l1(ad.obs["velocity_pseudotime"].to_numpy(float), pts))

    def dpt_on_velocity_graph():
        roots = ad.obs.get("root_cells")
        log("root_cells_present", roots is not None)
        sc.tl.diffmap(ad)
        res = {}
        for r in np.argsort(-ad.obs["root_cells"].to_numpy(float))[:20] if roots is not None else []:
            ad.uns["iroot"] = int(r)
            sc.tl.dpt(ad)
            res[int(r)] = cdf_l1(ad.obs["dpt_pseudotime"].to_numpy(float), pts)
        log("dpt_from_velocity_roots_l1", res)

    def latent():
        scv.tl.recover_dynamics(ad, n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
        scv.tl.latent_time(ad)
        log("latent_time_l1", cdf_l1(ad.obs["latent_time"].to_numpy(float), pts))
        np.save(os.path.join(OUT, "latent_time.npy"), ad.obs["latent_time"].to_numpy(float))

    step("velocity", vel)
    if "velocity_pseudotime" in ad.obs:
        np.save(os.path.join(OUT, "velocity_pseudotime.npy"), ad.obs["velocity_pseudotime"].to_numpy(float))
    step("dpt_velocity_roots", dpt_on_velocity_graph)
    step("latent_time", latent)
    log("done", True)


if __name__ == "__main__":
    main()
