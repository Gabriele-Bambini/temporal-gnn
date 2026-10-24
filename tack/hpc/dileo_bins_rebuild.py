"""Gate: rebuild Dileo & Sottoriva's 35 gastrulation pseudotime bins from the scVelo erythroid cells.

Their paper states: diffusion pseudotime (DPT) scaled to [0, 1], equal-width bins, GRNs on the scVelo 'Ms' layer.
The gene filter, neighbour graph and root cell are not recorded, so they are searched and checked against
Dileo's own gastrulation_node.csv:
  A. genes: scVelo filter_genes(min_shared_counts=k) must give exactly the 6,345 genes of the node file;
  B. Ms: per-gene mean of Ms over all cells vs sum_b(s04_sum) / 9815 (does not depend on the binning);
  C. bins: DPT from EVERY cell as root, on several graphs, must reproduce the 35 cell counts per bin
     (count_b = s04_sum / s00_mean, the same for every gene); velocity pseudotime only if DPT fails;
  D. per-bin per-gene Ms means (and stds) vs s00_mean (s02_std), for every Ms variant.
GATE_PASS needs A, an exact C and D within 1e-3 relative error. Writes ~/tack/out/dileo_bins/gate.json and, on
an exact C, gastrulation_rebuilt.h5ad (X = Ms of the best variant, obs 'dileo_bin' and 'dpt_pseudotime').
"""
import json
import os
import time

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy.sparse.csgraph import connected_components

H5 = os.path.expanduser("~/tack/data/erythroid/erythroid_lineage.h5ad")
NODE = os.path.expanduser("~/tgl-grn/GRN/gastrulation_node.csv")
OUT = os.path.expanduser("~/tack/out/dileo_bins")
NB = 35
KS = (5, 10, 15, 20, 25, 30, 40, 50)
# (name, recompute PCA, n_neighbors, n_pcs); scVelo's tutorial default is the first one
VARIANTS = (("s30_30", False, 30, 30), ("s15_30", False, 15, 30), ("s30_50", False, 30, 50),
            ("r30_30", True, 30, 30), ("r15_30", True, 15, 30))
REL_TOL = 1e-3
REP = {}


def log(key, value):
    REP[key] = value
    print(key, value if isinstance(value, str) else json.dumps(value), flush=True)
    with open(os.path.join(OUT, "gate.json"), "w") as fh:
        json.dump(REP, fh, indent=1)


def to_bins(pt):
    b = np.full(pt.shape, -1)
    ok = np.isfinite(pt)
    b[ok] = np.minimum((pt[ok] * NB).astype(int), NB - 1)
    return b


def dpt_l1_all_roots(ad, n_dcs, target, chunk=128):
    """L1 distance between the bin counts and the target, with every cell as DPT root.

    Same distance as scanpy Neighbors._get_dpt_row (eigenvalues < 0.9994, weights l/(1-l), inf across graph
    components) and the same scaling as DPT._set_pseudotime (divide by the max finite distance)."""
    ev = np.asarray(ad.uns["diffmap_evals"])[:n_dcs]
    keep = ev < 0.9994
    E = np.asarray(ad.obsm["X_diffmap"])[:, :n_dcs][:, keep].astype(np.float64) * (ev[keep] / (1 - ev[keep]))
    _, comp = connected_components(ad.obsp["connectivities"])
    n = E.shape[0]
    l1 = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))
        d = np.sqrt(((E[idx, None, :] - E[None, :, :]) ** 2).sum(-1))
        d[comp[idx][:, None] != comp[None, :]] = np.inf
        fin = np.isfinite(d)
        mx = np.maximum(np.where(fin, d, -np.inf).max(1, keepdims=True), 1e-300)
        b = np.minimum((np.where(fin, d, 0.0) / mx * NB).astype(np.int64), NB - 1)
        b = b + NB * np.arange(len(idx))[:, None]
        cnt = np.bincount(b[fin], minlength=NB * len(idx)).reshape(len(idx), NB)
        l1[idx] = np.abs(cnt - target[None, :]).sum(1)
    return l1


def rel_err(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-8)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    node = pd.read_csv(NODE)
    genes_d = node["gene"].astype(str).to_numpy()
    mean_d = node[[f"t{b:03d}_s00_mean" for b in range(NB)]].to_numpy(float)
    std_d = node[[f"t{b:03d}_s02_std" for b in range(NB)]].to_numpy(float)
    sum_d = node[[f"t{b:03d}_s04_sum" for b in range(NB)]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sum_d / mean_d
    target = np.array([int(round(np.nanmedian(ratio[mean_d[:, b] > 0, b]))) for b in range(NB)])
    log("target_counts", target.tolist())
    overall_d = sum_d.sum(1) / target.sum()

    raw = sc.read_h5ad(H5)
    log("h5ad", dict(n_obs=int(raw.n_obs), n_vars=int(raw.n_vars), layers=list(raw.layers.keys()),
                     obsm={k: list(np.shape(v)) for k, v in raw.obsm.items()}, uns=list(raw.uns.keys())))

    # A. gene filter
    gene_res, k_exact = {}, None
    for k in KS:
        ad = raw.copy()
        scv.pp.filter_genes(ad, min_shared_counts=k)
        g = np.asarray(ad.var_names).astype(str)
        gene_res[k] = dict(n=int(len(g)), in_dileo=int(np.isin(g, genes_d).sum()),
                           same_order=bool(len(g) == len(genes_d) and (g == genes_d).all()))
        if k_exact is None and gene_res[k]["in_dileo"] == len(g) == len(genes_d):
            k_exact = k
        del ad
    log("A_gene_filter", {str(k): v for k, v in gene_res.items()})
    log("A_pass", k_exact is not None)
    k_use = k_exact if k_exact is not None else min(
        KS, key=lambda k: abs(gene_res[k]["n"] - len(genes_d)) + gene_res[k]["n"] - gene_res[k]["in_dileo"])
    log("A_k_used", int(k_use))

    # B. normalisation + Ms variants, checked against the binning-free overall mean
    base = raw
    del raw
    scv.pp.filter_and_normalize(base, min_shared_counts=k_use)
    present = set(np.asarray(base.var_names).astype(str))
    mask_d = np.array([g in present for g in genes_d])
    base = base[:, genes_d[mask_d]].copy()
    variants, bres = {}, {}
    for name, recompute, n_nb, n_pcs in VARIANTS:
        ad = base.copy()
        if recompute and "X_pca" in ad.obsm:
            del ad.obsm["X_pca"]
        if not recompute and "X_pca" in ad.obsm and ad.obsm["X_pca"].shape[1] < n_pcs:
            bres[name] = "skipped: stored X_pca too narrow"
            continue
        scv.pp.moments(ad, n_pcs=n_pcs, n_neighbors=n_nb)
        r = rel_err(np.asarray(ad.layers["Ms"]).mean(0), overall_d[mask_d])
        bres[name] = dict(median_rel=float(np.median(r)), p99_rel=float(np.quantile(r, 0.99)), max_rel=float(r.max()))
        variants[name] = ad
        log("B_moments", bres)
    ad = base.copy()
    sc.pp.neighbors(ad)  # scanpy default graph, DPT only
    variants["sc_default"] = ad

    # C. DPT from every root on every graph
    search = []
    for name, ad in variants.items():
        sc.tl.diffmap(ad, n_comps=15)
        for n_dcs in (10, 15):
            l1 = dpt_l1_all_roots(ad, n_dcs, target)
            top = np.argsort(l1, kind="stable")[:5]
            search.append(dict(graph=name, n_dcs=n_dcs, best_l1=int(l1[top[0]]), n_exact=int((l1 == 0).sum()),
                               top_roots=[int(i) for i in top], top_l1=[int(l1[i]) for i in top],
                               top_celltype=[str(ad.obs["celltype"].iloc[i]) if "celltype" in ad.obs else "" for i in top]))
            log("C_dpt_search", search)
    best = min(search, key=lambda d: d["best_l1"])
    ad = variants[best["graph"]]
    ad.uns["iroot"] = best["top_roots"][0]
    sc.tl.dpt(ad, n_dcs=best["n_dcs"])
    pt = ad.obs["dpt_pseudotime"].to_numpy(float)
    b = to_bins(pt)
    cnt = np.bincount(b[b >= 0], minlength=NB)
    log("C_best", dict(graph=best["graph"], n_dcs=best["n_dcs"], root=best["top_roots"][0],
                       l1_scanpy=int(np.abs(cnt - target).sum()), counts=cnt.tolist()))
    method = "dpt"
    if not (cnt == target).all():
        try:
            v = variants["s30_30"]
            scv.tl.velocity(v, mode="stochastic")
            scv.tl.velocity_graph(v)
            scv.tl.velocity_pseudotime(v)
            ptv = v.obs["velocity_pseudotime"].to_numpy(float)
            ptv = (ptv - np.nanmin(ptv)) / (np.nanmax(ptv) - np.nanmin(ptv))
            bv = to_bins(ptv)
            cv = np.bincount(bv[bv >= 0], minlength=NB)
            log("C_velocity_pseudotime", dict(l1=int(np.abs(cv - target).sum()), counts=cv.tolist()))
            if np.abs(cv - target).sum() < np.abs(cnt - target).sum():
                pt, b, cnt, method = ptv, bv, cv, "velocity_pseudotime"
        except Exception as e:  # noqa: BLE001
            log("C_velocity_pseudotime", f"failed: {e!r}")
    exact = bool((cnt == target).all())
    log("C_pass", exact)
    log("C_method", method)

    # D. per-bin Ms statistics vs Dileo's node file
    dres = {}
    for name, v in variants.items():
        if "Ms" not in v.layers:
            continue
        Ms = np.asarray(v.layers["Ms"])
        em, es0, es1 = [], [], []
        for k in range(NB):
            sel = b == k
            if not sel.any():
                em.append(None), es0.append(None), es1.append(None)
                continue
            em.append(float(rel_err(Ms[sel].mean(0), mean_d[mask_d, k]).max()))
            if sel.sum() > 1:
                es0.append(float(rel_err(Ms[sel].std(0), std_d[mask_d, k]).max()))
                es1.append(float(rel_err(Ms[sel].std(0, ddof=1), std_d[mask_d, k]).max()))
            else:
                es0.append(None), es1.append(None)
        dres[name] = dict(max_rel_mean=max(e for e in em if e is not None),
                          max_rel_std_ddof0=max((e for e in es0 if e is not None), default=None),
                          max_rel_std_ddof1=max((e for e in es1 if e is not None), default=None),
                          per_bin_mean=em)
        log("D_bin_stats", dres)
    best_ms = min(dres, key=lambda n: dres[n]["max_rel_mean"])
    log("D_best_variant", best_ms)
    log("D_bin_mid", [float(pt[b == k].mean()) if (b == k).any() else None for k in range(NB)])
    passed = bool(k_exact is not None and exact and dres[best_ms]["max_rel_mean"] < REL_TOL)
    log("GATE_PASS", passed)

    if exact:
        v = variants[best_ms]
        obs = v.obs[[c for c in ("celltype", "stage") if c in v.obs]].copy()
        obs["dpt_pseudotime"] = pt
        obs["dileo_bin"] = b
        out = anndata.AnnData(X=np.asarray(v.layers["Ms"], dtype=np.float32), obs=obs, var=pd.DataFrame(index=v.var_names))
        out.write_h5ad(os.path.join(OUT, "gastrulation_rebuilt.h5ad"))
    np.save(os.path.join(OUT, "bin_of_cell.npy"), b)
    np.save(os.path.join(OUT, "pseudotime.npy"), pt)
    log("seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
