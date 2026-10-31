"""Strict data products for the LARRY cytokine-perturbation coupling test (Weinreb et al. 2020, GEO GSM4185644;
pre-registration tack/prereg/2026-10-30-larry-cytokine-prereg.md).

Design (read before freezing, larry_cyto_design.py): Lin-Kit+ cells barcoded and grown in the full cocktail, sampled at
day 2 (libraries d2_1-6), then split into Epo / G-csf / M-csf / SCF / full cocktail and sampled at day 4 and day 6 (SCF
day 4 only). Unit = consecutive transition within a condition: day 2 -> day 4 of condition c (source = the common day-2
sample) and day 4 -> day 6 of the same condition, kept if >= MIN_SHARED clones are shared (single-clone cells). Source
<= MAX_A cells, target <= MAX_B cells (proportional to libraries, seed 0). Counts = GEO normed_counts; PCA by
atlas_prep.transform fitted on source + target (no clone information used).
  ~/tack/out/larry_cyto/<condition>_<a><b>.npz: Z_a, Z_b, clone_a, clone_b, celltype_a, celltype_b, library_a, library_b
  ~/tack/out/larry_cyto/meta.json: units, sizes, shared clones
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
import larry_prep as LP  # noqa: E402

D = os.path.expanduser("~/rafm_data/weinreb2020_cyto")
P = "GSM4185644_stateFate_cytokinePerturbation_"
OUT = os.path.expanduser("~/tack/out/larry_cyto")
CONDS = {"Epo": "Epo", "G-csf": "Gcsf", "M-csf": "Mcsf", "SCF": "SCF", "full cocktail": "full"}
SOURCE_D2 = "full cocktail"
MIN_SHARED = 50
MAX_A, MAX_B = 9000, 6000


def load():
    """As larry_vivo_prep.load (orientation-safe counts and clone matrix), for the cytokine files."""
    t0 = time.time()
    X = scipy.io.mmread(os.path.join(D, P + "normed_counts.mtx.gz")).tocsr().astype(np.float32)
    print("counts", X.shape, X.nnz, f"{time.time() - t0:.0f}s", flush=True)
    meta = pd.read_csv(os.path.join(D, P + "metadata.txt.gz"), sep="\t")
    if X.shape[0] != len(meta):
        X = X.T.tocsr()
    genes = np.array([g.strip() for g in gzip.open(os.path.join(D, P + "gene_names.txt.gz"), "rt")])
    if len(genes) < X.shape[1]:
        genes = np.r_[genes, [f"unnamed_{i}" for i in range(X.shape[1] - len(genes))]]
    C = scipy.io.mmread(os.path.join(D, P + "clone_matrix.mtx.gz"))
    C = (C if C.shape[0] == len(meta) else C.T).tocsr()
    C.eliminate_zeros()
    nclo = np.diff(C.indptr)
    clone = np.full(C.shape[0], -1)
    one = np.where(nclo == 1)[0]
    clone[one] = C.indices[C.indptr[one]]
    print("clone matrix", C.shape, "single-clone cells", len(one), flush=True)
    return X, genes[: X.shape[1]], meta, clone


def transitions(day, cond, ok):
    """(name, source index, target index) of every consecutive within-condition transition, before the size rule."""
    out = []
    d2 = np.where(ok & (day == 2) & (cond == SOURCE_D2))[0]
    for c, short in CONDS.items():
        d4 = np.where(ok & (day == 4) & (cond == c))[0]
        d6 = np.where(ok & (day == 6) & (cond == c))[0]
        if len(d4):
            out.append((f"{short}_24", d2, d4))
        if len(d4) and len(d6):
            out.append((f"{short}_46", d4, d6))
    return out


def main():
    X, genes, meta, clone = load()
    day = meta["Time point"].to_numpy()
    cond = meta["Cytokine condition"].astype(str).to_numpy()
    lib = meta["Library"].astype(str).to_numpy()
    ct = meta["Cell type annotation"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc = set(str(g).capitalize() for g in list(s_g) + list(g2m))
    os.makedirs(OUT, exist_ok=True)
    summary = {"min_shared": MIN_SHARED, "units": {}, "excluded": {}}
    for name, src, tgt in transitions(day, cond, clone >= 0):
        shared = len(set(clone[src]) & set(clone[tgt]))
        if shared < MIN_SHARED:
            summary["excluded"][name] = shared
            continue
        rng = np.random.default_rng(0)
        ia = LP.sample(src, lib, MAX_A, rng)
        ib = LP.sample(tgt, lib, MAX_B, rng)
        both = np.concatenate([ia, ib])
        Z = AP.transform(X, both, ib[:1], genes, cc)["Z_fit"]
        np.savez_compressed(os.path.join(OUT, f"{name}.npz"), Z_a=Z[: len(ia)], Z_b=Z[len(ia):],
                            clone_a=clone[ia], clone_b=clone[ib], celltype_a=ct[ia], celltype_b=ct[ib],
                            library_a=lib[ia], library_b=lib[ib])
        summary["units"][name] = {"n_a": int(len(ia)), "n_b": int(len(ib)), "shared_clones_all": shared,
                                  "shared_clones_sampled": len(set(clone[ia]) & set(clone[ib]))}
        print(name, summary["units"][name], flush=True)
    print("excluded", summary["excluded"], flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
