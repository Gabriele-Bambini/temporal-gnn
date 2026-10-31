"""Strict data products for the zebrafish replication (Saunders et al. 2023, GEO GSE202639 reference;
pre-registration tack/prereg/2026-10-27-zebrafish-replication-prereg.md).

Cells: experiment GAP14 only (one sequencing batch, 48 individually hashed embryos per timepoint, 18..38 hpf every 2 h,
28 C), control embryos only (gene_target starting with 'ctrl-'). Embryo = biological unit ('sample' field below).
At most 3000 cells per timepoint, drawn per timepoint proportionally to each embryo's cell count (seed 0).
Counts: ~/rafm_data/saunders2023/csc (from zf_convert.R). Preprocessing = atlas_prep.transform (fit cells only: gene
filter >= 10 fit cells, size normalisation to the fit median, log1p, 2000 seurat HVGs, PCA 30, S/G2M score).
  origin<s>.npz, s = 1..9: fit = timepoints <= s; timepoint s+1 only transformed (fields as atlas_prep)
  interp<k>.npz, k = 1..9: fit = timepoints k-1 and k+1; k only transformed (fields as atlas_prep)
Timepoint index i <-> 18 + 2 i hpf. Output: ~/tack/out/zf/.
"""
import importlib
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_prep as AP  # noqa: E402

D = os.path.expanduser("~/rafm_data/saunders2023")
OUT = os.path.expanduser("~/tack/out/zf")
TPS = [18.0 + 2 * i for i in range(11)]
STAGES = [f"{int(t)}hpf" for t in TPS]
MAX_PER_STAGE = 3000


def load():
    c = os.path.join(D, "csc")
    ng, nc = [int(x) for x in open(os.path.join(c, "zf_dims.txt")).read().split()]
    i = np.fromfile(os.path.join(c, "zf_i.int32"), np.int32)
    p = np.fromfile(os.path.join(c, "zf_p.float64"), np.float64).astype(np.int64)
    x = np.fromfile(os.path.join(c, "zf_x.float32"), np.float32)
    X = sp.csc_matrix((x, i, p), shape=(ng, nc)).T.tocsr()          # cells x genes
    genes = np.array(open(os.path.join(c, "zf_genes.txt")).read().split("\n")[:ng])
    cells = np.array(open(os.path.join(c, "zf_cells.txt")).read().split("\n")[:nc])
    meta = pd.read_csv(os.path.join(D, "GSE202639_reference_cell_metadata.csv.gz"), low_memory=False,
                       usecols=["cell", "timepoint", "expt", "embryo", "gene_target", "cell_type_broad"])
    gm = pd.read_csv(os.path.join(D, "GSE202639_reference_gene_metadata.csv.gz"))
    sym = dict(zip(gm["id"].astype(str), gm["gene_short_name"].astype(str)))
    genes = np.array([sym.get(g, g) for g in genes])
    meta = meta.set_index("cell").reindex(cells)
    keep = ((meta["expt"] == "GAP14") & meta["gene_target"].astype(str).str.startswith("ctrl-")
            & meta["timepoint"].isin(TPS)).to_numpy()
    rng = np.random.default_rng(0)
    chosen = []
    for t in TPS:
        idx = np.where(keep & (meta["timepoint"] == t).to_numpy())[0]
        if len(idx) > MAX_PER_STAGE:
            emb = meta["embryo"].to_numpy()[idx]
            take = []
            for e in np.unique(emb):
                ii = idx[emb == e]
                k = int(round(MAX_PER_STAGE * len(ii) / len(idx)))
                take.append(rng.choice(ii, min(k, len(ii)), replace=False))
            idx = np.sort(np.concatenate(take))
        chosen.append(idx)
    chosen = np.concatenate(chosen)
    obs = meta.iloc[chosen].reset_index()
    return X[chosen], obs, genes


def main():
    os.makedirs(OUT, exist_ok=True)
    X, obs, genes = load()
    stage = np.array([TPS.index(t) for t in obs["timepoint"]])
    smp, ct = obs["embryo"].astype(str).to_numpy(), obs["cell_type_broad"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc_genes = set(str(g).capitalize() for g in list(s_g) + list(g2m))
    meta = {"stages": STAGES, "n_cells_per_stage": np.bincount(stage, minlength=11).tolist(),
            "n_embryos_per_stage": [int(len(np.unique(smp[stage == k]))) for k in range(11)],
            "n_cc_genes_total": int(sum(g.capitalize() in cc_genes for g in genes)), "origins": {}, "interp": {}}
    print(meta, flush=True)
    for s in range(1, 10):
        t0 = time.time()
        fit, fut = np.where(stage <= s)[0], np.where(stage == s + 1)[0]
        r = AP.transform(X, fit, fut, genes, cc_genes)
        np.savez_compressed(os.path.join(OUT, f"origin{s}.npz"), genes=r["genes"], stage_past=stage[fit], stage_fut=stage[fut],
                            sample_past=smp[fit], sample_fut=smp[fut], celltype_past=ct[fit], celltype_fut=ct[fut],
                            X_past=r["X_fit"].astype(np.float16), X_fut=r["X_other"].astype(np.float16), pca_mean=r["mu"],
                            pca_comps=r["comps"], Z_past=r["Z_fit"], Z_fut=r["Z_other"], cc_past=r["cc_fit"], cc_fut=r["cc_other"])
        meta["origins"][s] = {"n_past": len(fit), "n_fut": len(fut), "n_kept": r["n_kept"], "n_cc": r["n_cc"], "sec": round(time.time() - t0, 1)}
        print("origin", s, meta["origins"][s], flush=True)
    for k in range(1, 10):
        t0 = time.time()
        fit, held = np.where((stage == k - 1) | (stage == k + 1))[0], np.where(stage == k)[0]
        r = AP.transform(X, fit, held, genes, cc_genes)
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
