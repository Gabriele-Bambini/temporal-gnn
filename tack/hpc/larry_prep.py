"""Strict data products for the LARRY test (Weinreb et al. 2020, GEO GSM4185642 in vitro; pre-registration
tack/prereg/2026-10-29-larry-prereg.md).

Cells: in vitro LARRY hematopoiesis, days 2, 4, 6 (index 0, 1, 2); the two starting populations (Lin-Kit+Sca1+ and
Lin-Kit+Sca1-) are separate cultures and get separate products. Unit = culture well. Counts = GEO normed_counts
(cells x genes); preprocessing = atlas_prep.transform on the fit cells only.
  ~/tack/out/larry/<pop>/interp1.npz : fit = day 2 + day 6 (<= 3000 cells per day, proportional to wells, seed 0),
                                       held = day 4 (<= 3000); same fields as atlas_prep interp products.
  ~/tack/out/larry/<pop>/couple_<a><b>.npz, (a, b) = (2, 4), (4, 6): cells carrying exactly one clone barcode,
                                       <= 3000 per day (seed 0), PCA fitted on both days; fields Z_a, Z_b, clone_a,
                                       clone_b, celltype_a, celltype_b, well_a, well_b.
"""
import gzip
import importlib
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_prep as AP  # noqa: E402

D = os.path.expanduser("~/rafm_data/weinreb2020")
OUT = os.path.expanduser("~/tack/out/larry")
P = "GSM4185642_stateFate_inVitro_"
DAYS = [2.0, 4.0, 6.0]
POPS = {"sca1pos": "Lin-Kit+Sca1+", "sca1neg": "Lin-Kit+Sca1-"}
MAX = 3000


def load():
    t0 = time.time()
    X = scipy.io.mmread(os.path.join(D, P + "normed_counts.mtx.gz")).tocsr().astype(np.float32)
    print("counts", X.shape, X.nnz, f"{time.time() - t0:.0f}s", flush=True)
    genes = np.array([g.strip() for g in gzip.open(os.path.join(D, P + "gene_names.txt.gz"), "rt")])
    if len(genes) < X.shape[1]:
        genes = np.r_[genes, [f"unnamed_{i}" for i in range(X.shape[1] - len(genes))]]
    meta = pd.read_csv(os.path.join(D, P + "metadata.txt.gz"), sep="\t")
    C = scipy.io.mmread(os.path.join(D, P + "clone_matrix.mtx.gz")).tocsr()
    nclo = np.diff(C.indptr)
    clone = np.full(C.shape[0], -1)
    one = np.where(nclo == 1)[0]
    clone[one] = C.indices[C.indptr[one]]
    return X, genes[: X.shape[1]], meta, clone


def sample(idx, wells, n, rng):
    if len(idx) <= n:
        return np.sort(idx)
    take = []
    for w in np.unique(wells[idx]):
        ii = idx[wells[idx] == w]
        take.append(rng.choice(ii, min(int(round(n * len(ii) / len(idx))), len(ii)), replace=False))
    return np.sort(np.concatenate(take))


def main():
    X, genes, meta, clone = load()
    day = meta["Time point"].to_numpy()
    pop = meta["Starting population"].astype(str).to_numpy()
    well = meta["Well"].astype(str).to_numpy()
    ct = meta["Cell type annotation"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc = set(str(g).capitalize() for g in list(s_g) + list(g2m))
    summary = {}
    for key, name in POPS.items():
        os.makedirs(os.path.join(OUT, key), exist_ok=True)
        rng = np.random.default_rng(0)
        sel = {d: sample(np.where((pop == name) & (day == d))[0], well, MAX, rng) for d in DAYS}
        fit = np.concatenate([sel[2.0], sel[6.0]])
        held = sel[4.0]
        r = AP.transform(X, fit, held, genes, cc)
        stg = np.r_[np.zeros(len(sel[2.0]), int), np.full(len(sel[6.0]), 2)]
        np.savez_compressed(os.path.join(OUT, key, "interp1.npz"), genes=r["genes"], stage_fit=stg,
                            sample_fit=well[fit], sample_held=well[held], celltype_fit=ct[fit], celltype_held=ct[held],
                            X_fit=r["X_fit"].astype(np.float16), X_held=r["X_other"].astype(np.float16),
                            pca_mean=r["mu"], pca_comps=r["comps"], Z_fit=r["Z_fit"], Z_held=r["Z_other"], cc_fit=r["cc_fit"])
        summary[key] = {"n": {str(d): int(len(sel[d])) for d in DAYS}, "n_cc": r["n_cc"]}
        for a, b in ((2.0, 4.0), (4.0, 6.0)):
            ia = sample(np.where((pop == name) & (day == a) & (clone >= 0))[0], well, MAX, rng)
            ib = sample(np.where((pop == name) & (day == b) & (clone >= 0))[0], well, MAX, rng)
            both = np.concatenate([ia, ib])
            rr = AP.transform(X, both, ib[:1], genes, cc)
            Z = rr["Z_fit"]
            shared = len(set(clone[ia]) & set(clone[ib]))
            np.savez_compressed(os.path.join(OUT, key, f"couple_{int(a)}{int(b)}.npz"), Z_a=Z[: len(ia)], Z_b=Z[len(ia):],
                                clone_a=clone[ia], clone_b=clone[ib], celltype_a=ct[ia], celltype_b=ct[ib],
                                well_a=well[ia], well_b=well[ib])
            summary[key][f"couple_{int(a)}{int(b)}"] = {"n_a": int(len(ia)), "n_b": int(len(ib)), "shared_clones": shared}
        print(key, summary[key], flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
