"""Strict data products for the LARRY in vivo coupling test (Weinreb et al. 2020, GEO GSM4185643; pre-registration
tack/prereg/2026-10-30-larry-invivo-prereg.md).

Design (read before freezing, larry_vivo_design.py): barcoded Lin-Kit+Sca1hi cells, cohorts A and B; each cohort has a
day-2 in vitro sample (libraries d2A_* / d2B_*) and recipient mice A1-A5 / B1-B5 sampled at day 9 (A1-A3, B1-B3) or
day 16 (A4, A5, B4, B5). Unit = transition day-2 cohort -> recipient mouse of the same cohort with >= MIN_SHARED clones
shared between the two (single-clone cells, Sca1hi). Cells: exactly one clone barcode; source = all day-2 cells of the
cohort (<= MAX_A), target = the mouse's cells (<= MAX_B, proportional to libraries, seed 0). Counts = GEO normed_counts;
PCA by atlas_prep.transform fitted on source + target (no clone information used).
  ~/tack/out/larry_vivo/<cohort>_<mouse>.npz: Z_a, Z_b, clone_a, clone_b, celltype_a, celltype_b, library_a, library_b
  ~/tack/out/larry_vivo/meta.json: units, sizes, shared clones
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

D = os.path.expanduser("~/rafm_data/weinreb2020_invivo")
P = "GSM4185643_stateFate_inVivo_"
OUT = os.path.expanduser("~/tack/out/larry_vivo")
POP = "Lin-Kit+Sca1hi"
MIN_SHARED = 50
MAX_A, MAX_B = 9000, 6000


def load():
    """As larry_prep.load, but orientation-safe: the in vivo clone matrix may be stored clones x cells."""
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


def main():
    X, genes, meta, clone = load()
    day = meta["Time point"].to_numpy()
    pop = meta["Starting population"].astype(str).to_numpy()
    lib = meta["Library"].astype(str).to_numpy()
    mouse = meta["Recipient mouse"].astype(str).to_numpy()
    ct = meta["Cell type annotation"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc = set(str(g).capitalize() for g in list(s_g) + list(g2m))
    os.makedirs(OUT, exist_ok=True)
    ok = (pop == POP) & (clone >= 0)
    summary = {"min_shared": MIN_SHARED, "units": {}, "excluded": {}}
    for coh in ("A", "B"):
        src = np.where(ok & (day == 2) & np.char.startswith(lib.astype(str), f"d2{coh}_"))[0]
        for m in sorted(set(mouse[ok & (day > 2)])):
            if not m.startswith(coh):
                continue
            tgt = np.where(ok & (mouse == m))[0]
            shared = len(set(clone[src]) & set(clone[tgt]))
            name = f"{coh}_{m}"
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
            summary["units"][name] = {"day_b": int(day[tgt[0]]), "n_a": int(len(ia)), "n_b": int(len(ib)),
                                      "shared_clones_all": shared,
                                      "shared_clones_sampled": len(set(clone[ia]) & set(clone[ib]))}
            print(name, summary["units"][name], flush=True)
    print("excluded", summary["excluded"], flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
