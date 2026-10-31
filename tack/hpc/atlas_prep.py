"""Strict data products for the CONFIRMATION study on the Pijuan-Sala 2019 mouse gastrulation atlas.

Confirms, on independent cells, the two OT signals found post hoc on the erythroid lineage (decisions.md 2026-10-24):
interpolation of a held-out stage (ot_pairs, uot) and forecasting by reweighting (uot_rw). Pre-registration:
tack/prereg/2026-10-25-atlas-confirmation-prereg.md.

Cells: ~/rafm_data/processed/pijuan_sala2019_atlas.npz (raw counts, 108857 cells, 9 stages E6.5 .. E8.5, 0.25 days
apart) joined with ~/rafm_data/pijuan_sala2019/meta.tab.gz. Kept: annotated, not doublet, not stripped, and NOT in
the blood lineage (Haematoendothelial progenitors, Blood progenitors 1/2, Erythroid1/2/3: the erythroid study used
them). At most 3000 cells per stage, drawn per stage proportionally to each sample's size (seed 0).
For every product the gene filter (detected in >= 10 fit cells), the normalisation (per-cell total counts scaled to
the fit-cell median, log1p), 2000 HVGs (scanpy, seurat flavour), PCA (30) and the proliferation score (Tirosh 2016 S +
G2M genes shipped with scVelo, mean centred log expression, centred on fit cells) are computed from the FIT cells only:
  origin<s>.npz, s = 1..7: fit = stages <= s; stage s+1 is only transformed (fields as ery_prep.py, without velocity)
  interp<k>.npz, k = 1..7: fit = stages k-1 and k+1; stage k is only transformed (fields as ery_prep_interp.py)
The atlas cell types are kept for descriptive use only (never as a model input). Output: ~/tack/out/atlas/.
"""
import importlib
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

NPZ = os.path.expanduser("~/rafm_data/processed/pijuan_sala2019_atlas.npz")
META = os.path.expanduser("~/rafm_data/pijuan_sala2019/meta.tab.gz")
OUT = os.path.expanduser("~/tack/out/atlas")
STAGES = ["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
BLOOD = {"Haematoendothelial progenitors", "Blood progenitors 1", "Blood progenitors 2",
         "Erythroid1", "Erythroid2", "Erythroid3"}
MAX_PER_STAGE, N_HVG, N_PC, MIN_CELLS = 3000, 2000, 30, 10


def load():
    z = np.load(NPZ, allow_pickle=True)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    obs = pd.DataFrame({"cell": z["cell"].astype(str), "stage": z["stage"].astype(str), "sample": z["sample"].astype(str)})
    meta = pd.read_csv(META, sep="\t", usecols=["cell", "celltype", "doublet", "stripped"])
    obs = obs.merge(meta, on="cell", how="left")
    keep = (obs["celltype"].notna() & ~obs["celltype"].isin(BLOOD) & (obs["doublet"].astype(str) == "False")
            & (obs["stripped"].astype(str) == "False") & obs["stage"].isin(STAGES)).to_numpy()
    rng = np.random.default_rng(0)
    chosen = []
    for st in STAGES:
        idx = np.where(keep & (obs["stage"] == st).to_numpy())[0]
        if len(idx) > MAX_PER_STAGE:
            smp = obs["sample"].to_numpy()[idx]
            take = []
            for s in np.unique(smp):
                ii = idx[smp == s]
                k = int(round(MAX_PER_STAGE * len(ii) / len(idx)))
                take.append(rng.choice(ii, min(k, len(ii)), replace=False))
            idx = np.sort(np.concatenate(take))
        chosen.append(idx)
    chosen = np.concatenate(chosen)
    return X[chosen], obs.iloc[chosen].reset_index(drop=True), z["gene_symbols"].astype(str)


def transform(X, fit, other, genes, cc_genes):
    """Fit-cell-only preprocessing; returns HVG log expression, PCA scores and proliferation for fit and other cells."""
    det = np.asarray((X[fit] > 0).sum(0)).ravel()
    keep = det >= MIN_CELLS
    size = np.asarray(X.sum(1)).ravel()
    Xn = (sp.diags(np.median(size[fit]) / np.maximum(size, 1.0)) @ X[:, keep]).tocsr()
    Xn.data = np.log1p(Xn.data)
    p = ad.AnnData(X=Xn[fit].copy())
    sc.pp.highly_variable_genes(p, n_top_genes=N_HVG, flavor="seurat")
    hv = p.var["highly_variable"].to_numpy()
    Xf = np.asarray(Xn[fit][:, hv].todense(), np.float32)
    Xo = np.asarray(Xn[other][:, hv].todense(), np.float32)
    mu = Xf.mean(0)
    _, _, Vt = np.linalg.svd(Xf - mu, full_matrices=False)
    comps = Vt[:N_PC]
    ccm = np.array([g.capitalize() in cc_genes for g in genes[keep]])
    Xcc = np.asarray(Xn[:, ccm].todense())
    cc = (Xcc - Xcc[fit].mean(0)).mean(1)
    return dict(genes=genes[keep][hv], X_fit=Xf, X_other=Xo, mu=mu, comps=comps,
                Z_fit=((Xf - mu) @ comps.T).astype(np.float32), Z_other=((Xo - mu) @ comps.T).astype(np.float32),
                cc_fit=cc[fit], cc_other=cc[other], n_cc=int(ccm.sum()), n_kept=int(keep.sum()))


def main():
    os.makedirs(OUT, exist_ok=True)
    X, obs, genes = load()
    stage = np.array([STAGES.index(s) for s in obs["stage"]])
    smp, ct = obs["sample"].to_numpy(), obs["celltype"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc_genes = set(str(g).capitalize() for g in list(s_g) + list(g2m))
    meta = {"stages": STAGES, "n_cells_per_stage": np.bincount(stage, minlength=9).tolist(),
            "n_samples_per_stage": [int(len(np.unique(smp[stage == k]))) for k in range(9)], "origins": {}, "interp": {}}
    print(meta, flush=True)
    for s in range(1, 8):
        t0 = time.time()
        fit, fut = np.where(stage <= s)[0], np.where(stage == s + 1)[0]
        r = transform(X, fit, fut, genes, cc_genes)
        np.savez_compressed(os.path.join(OUT, f"origin{s}.npz"), genes=r["genes"], stage_past=stage[fit], stage_fut=stage[fut],
                            sample_past=smp[fit], sample_fut=smp[fut], celltype_past=ct[fit], celltype_fut=ct[fut],
                            X_past=r["X_fit"].astype(np.float16), X_fut=r["X_other"].astype(np.float16), pca_mean=r["mu"],
                            pca_comps=r["comps"], Z_past=r["Z_fit"], Z_fut=r["Z_other"], cc_past=r["cc_fit"], cc_fut=r["cc_other"])
        meta["origins"][s] = {"n_past": len(fit), "n_fut": len(fut), "n_kept": r["n_kept"], "n_cc": r["n_cc"], "sec": round(time.time() - t0, 1)}
        print("origin", s, meta["origins"][s], flush=True)
    for k in range(1, 8):
        t0 = time.time()
        fit, held = np.where((stage == k - 1) | (stage == k + 1))[0], np.where(stage == k)[0]
        r = transform(X, fit, held, genes, cc_genes)
        np.savez_compressed(os.path.join(OUT, f"interp{k}.npz"), genes=r["genes"], stage_fit=stage[fit], sample_fit=smp[fit],
                            sample_held=smp[held], celltype_fit=ct[fit], celltype_held=ct[held],
                            X_fit=r["X_fit"].astype(np.float16), X_held=r["X_other"].astype(np.float16), pca_mean=r["mu"],
                            pca_comps=r["comps"], Z_fit=r["Z_fit"], Z_held=r["Z_other"], cc_fit=r["cc_fit"])
        meta["interp"][k] = {"n_fit": len(fit), "n_held": len(held), "n_kept": r["n_kept"], "sec": round(time.time() - t0, 1)}
        print("interp", k, meta["interp"][k], flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
