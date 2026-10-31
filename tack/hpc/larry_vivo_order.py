"""EXPLORATORY data-integrity check: the GEO in vivo clone matrix (GSM4185643) read in metadata order gives sisters NO
shared fate inside a mouse (same-type enrichment 1.00 = permutation null), while Weinreb et al. 2020 report sisters in
separate mice sharing the dominant fate 71 % of the time (23 % by chance). Day-2 blocks and mouse blocks are aligned
(sister coherence 26-44 at day 2; clones cohort-specific), so the suspicion is the cell order WITHIN in vivo libraries.
Candidate orders of the clone-matrix columns vs metadata rows are scored by within-mouse same-type enrichment (and
cohort structure must survive). Metadata + clone matrix only, no expression.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import larry_vivo_prep as VP  # noqa: E402
from larry_vivo_typecheck import same_type_enrichment  # noqa: E402


def block_order(lib):
    """Library blocks in order of first appearance in the metadata."""
    _, first = np.unique(lib, return_index=True)
    rank = {lib[i]: r for r, i in enumerate(sorted(first))}
    return np.array([rank[x] for x in lib])


def orders(meta):
    lib = meta["Library"].astype(str).to_numpy()
    bc = meta["Cell barcode"].astype(str).to_numpy()
    n = len(meta)
    blk = block_order(lib)
    pos = np.arange(n)
    return {
        "as_is": pos,
        "barcode_within_library": np.lexsort((bc, blk)),
        "reverse_within_library": np.lexsort((-pos, blk)),
        "libraries_sorted": np.lexsort((pos, lib)),
        "libraries_sorted_barcode": np.lexsort((bc, lib)),
    }


def main():
    meta = pd.read_csv(os.path.join(VP.D, VP.P + "metadata.txt.gz"), sep="\t")
    C = scipy.io.mmread(os.path.join(VP.D, VP.P + "clone_matrix.mtx.gz"))
    C = (C if C.shape[0] == len(meta) else C.T).tocsr()
    nclo = np.diff(C.indptr)
    col_clone = np.full(C.shape[0], -1)
    one = np.where(nclo == 1)[0]
    col_clone[one] = C.indices[C.indptr[one]]
    pop = meta["Starting population"].astype(str).to_numpy()
    mouse = meta["Recipient mouse"].astype(str).to_numpy()
    ct = meta["Cell type annotation"].astype(str).to_numpy()
    lib = meta["Library"].astype(str).to_numpy()
    day = meta["Time point"].to_numpy()
    mice = sorted(set(mouse[pop == VP.POP]) - {"nan", "NA"})
    for name, perm in orders(meta).items():
        clone = np.full(len(meta), -1)
        clone[perm] = col_clone             # matrix column j belongs to metadata row perm[j]
        ok = (pop == VP.POP) & (clone >= 0)
        e = [same_type_enrichment(clone[ok & (mouse == m)], ct[ok & (mouse == m)]) for m in mice]
        d2a = set(clone[ok & (day == 2) & np.char.startswith(lib.astype(str), "d2A_")])
        d2b = set(clone[ok & (day == 2) & np.char.startswith(lib.astype(str), "d2B_")])
        same = sum(len((d2a if m[0] == "A" else d2b) & set(clone[ok & (mouse == m)])) for m in mice)
        other = sum(len((d2b if m[0] == "A" else d2a) & set(clone[ok & (mouse == m)])) for m in mice)
        print(f"{name}: within-mouse same-type enrichment {np.round(e, 2).tolist()} mean {np.mean(e):.2f}; "
              f"day-2 clones shared with own-cohort mice {same}, other-cohort {other}", flush=True)


if __name__ == "__main__":
    main()
