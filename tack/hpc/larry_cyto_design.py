"""Design summary + clone-label integrity check of LARRY cytokine perturbation (Weinreb et al. 2020, GEO GSM4185644,
~/rafm_data/weinreb2020_cyto), read BEFORE any pre-registration: metadata and clone matrix only (no expression).
Prints cells per time point x cytokine condition / library, clonally labelled cells, clones shared between time points
(the units a coupling test can use). Integrity (the in vivo GEO matrix failed it): same-type sister enrichment inside
each time point x condition (each cell uniform over the other cells of its annotated type; > 1 = sisters share fate,
i.e. clone labels carry cell-level information) vs a type-permutation null. In vitro reference: 3.2-4.3.
Usage: python -I larry_cyto_design.py
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from larry_vivo_typecheck import same_type_enrichment  # noqa: E402

D = os.path.expanduser("~/rafm_data/weinreb2020_cyto")
P = "GSM4185644_stateFate_cytokinePerturbation_"
MIN_CELLS = 200


def main():
    meta = pd.read_csv(os.path.join(D, P + "metadata.txt.gz"), sep="\t")
    print("cells", len(meta), "columns", list(meta.columns))
    day = meta["Time point"].to_numpy()
    cond = meta["Cytokine condition"].astype(str).to_numpy()
    lib = meta["Library"].astype(str).to_numpy()
    ct = meta["Cell type annotation"].astype(str).to_numpy()
    print(pd.crosstab(meta["Cytokine condition"], meta["Time point"], margins=True))
    print(meta.groupby(["Time point", "Cytokine condition", "Library"]).size().to_string())
    print(pd.crosstab(meta["Cell type annotation"], meta["Time point"]))
    C = scipy.io.mmread(os.path.join(D, P + "clone_matrix.mtx.gz"))
    C = (C if C.shape[0] == len(meta) else C.T).tocsr()
    C.eliminate_zeros()
    nclo = np.diff(C.indptr)
    print("clone matrix (cells x clones)", C.shape, "cells by number of barcodes",
          pd.Series(nclo).value_counts().sort_index().head(5).to_dict())
    clone = np.full(C.shape[0], -1)
    one = np.where(nclo == 1)[0]
    clone[one] = C.indices[C.indptr[one]]
    ok = clone >= 0
    print("single-clone cells per time point x condition")
    print(pd.crosstab(cond[ok], day[ok]))
    groups = sorted({(int(d), c) for d, c in zip(day[ok], cond[ok])})
    sets = {g: set(clone[ok & (day == g[0]) & (cond == g[1])]) for g in groups}
    print("clones shared between (time point, condition) groups")
    for a in groups:
        for b in groups:
            if a[0] < b[0]:
                s = sets[a] & sets[b]
                if s:
                    na = int((ok & (day == a[0]) & (cond == a[1]) & np.isin(clone, list(s))).sum())
                    nb = int((ok & (day == b[0]) & (cond == b[1]) & np.isin(clone, list(s))).sum())
                    print(f"  {a} -> {b}: shared clones {len(s)}, cells in shared clones {na} / {nb}")
    print("integrity: same-type sister enrichment inside each group (type-permutation null)")
    rng = np.random.default_rng(0)
    for g in groups:
        i = np.where(ok & (day == g[0]) & (cond == g[1]))[0]
        if len(i) < MIN_CELLS:
            continue
        e = same_type_enrichment(clone[i], ct[i])
        null = [same_type_enrichment(clone[i], rng.permutation(ct[i])) for _ in range(5)]
        print(f"  {g}: n {len(i)} clones {len(set(clone[i]))} enrichment {e:.2f} "
              f"(null {np.mean(null):.2f} +- {np.std(null):.2f})", flush=True)
    print("integrity per library")
    for L in sorted(set(lib[ok])):
        i = np.where(ok & (lib == L))[0]
        if len(i) >= MIN_CELLS:
            print(f"  {L}: n {len(i)} enrichment {same_type_enrichment(clone[i], ct[i]):.2f}", flush=True)


if __name__ == "__main__":
    main()
