"""Design summary of LARRY in vivo (Weinreb et al. 2020, GEO GSM4185643, ~/rafm_data/weinreb2020_invivo), read BEFORE
the pre-registration is frozen: only metadata and the clone matrix (no expression). Prints cells per time point x
starting population x recipient mouse / library, clonally labelled cells and clones shared between time points (the
units a coupling test can use). Usage: python -I larry_vivo_design.py
"""
import gzip
import os

import numpy as np
import pandas as pd
from scipy.io import mmread

D = os.path.expanduser("~/rafm_data/weinreb2020_invivo")
P = "GSM4185643_stateFate_inVivo_"


def main():
    meta = pd.read_csv(os.path.join(D, P + "metadata.txt.gz"), sep="\t")
    print("cells", len(meta), "columns", list(meta.columns))
    print(pd.crosstab(meta["Time point"], meta["Starting population"], margins=True))
    print(pd.crosstab(meta["Time point"], meta["Recipient mouse"].fillna("NA"), margins=True))
    print(meta.groupby(["Time point", "Library"]).size().to_string())
    print(pd.crosstab(meta["Cell type annotation"], meta["Time point"]))
    with gzip.open(os.path.join(D, P + "clone_matrix.mtx.gz"), "rb") as fh:
        C = mmread(fh).tocsr()
    if C.shape[0] != len(meta):
        C = C.T.tocsr()
    print("clone matrix (cells x clones)", C.shape, "nnz", C.nnz)
    nb = np.asarray((C > 0).sum(1)).ravel()
    meta["n_barcodes"] = nb
    print("cells by number of barcodes", pd.Series(nb).value_counts().sort_index().head(6).to_dict())
    one = nb == 1
    clone = np.full(len(meta), -1)
    clone[one] = C[one].argmax(1).A.ravel()
    meta["clone"] = clone
    lab = meta[one]
    print("single-clone cells per time point x population")
    print(pd.crosstab(lab["Time point"], lab["Starting population"]))
    tps = sorted(meta["Time point"].astype(str).unique())
    for pop in sorted(meta["Starting population"].astype(str).unique()):
        sets = {t: set(lab.loc[(lab["Time point"].astype(str) == t) & (lab["Starting population"].astype(str) == pop),
                               "clone"]) for t in tps}
        for i, a in enumerate(tps):
            for b in tps[i + 1:]:
                s = sets[a] & sets[b]
                if s:
                    na = int(lab[(lab["Time point"].astype(str) == a) & lab["clone"].isin(s)
                                 & (lab["Starting population"].astype(str) == pop)].shape[0])
                    nb_ = int(lab[(lab["Time point"].astype(str) == b) & lab["clone"].isin(s)
                                  & (lab["Starting population"].astype(str) == pop)].shape[0])
                    print(f"pop {pop}: {a} -> {b} shared clones {len(s)} cells in shared clones {na} / {nb_}")
    # per group: day-2 cohort (library d2A_* / d2B_*) or recipient mouse; clones shared between groups
    lab = lab.copy()
    lib = lab["Library"].astype(str)
    lab["group"] = np.where(lab["Time point"] == 2, "d2" + lib.str[2],
                            lab["Recipient mouse"].astype(str) + "_d" + lab["Time point"].astype(str))
    lab = lab[lab["Starting population"].astype(str) == "Lin-Kit+Sca1hi"]
    groups = sorted(lab["group"].unique())
    sets = {g: set(lab.loc[lab["group"] == g, "clone"]) for g in groups}
    n = {g: int((lab["group"] == g).sum()) for g in groups}
    print("Sca1hi single-clone cells per group", n)
    print("shared clones between groups (Sca1hi)")
    print(pd.DataFrame([[len(sets[a] & sets[b]) for b in groups] for a in groups], index=groups, columns=groups))
    for a in groups:
        if a.startswith("d2"):
            for b in groups:
                if not b.startswith("d2"):
                    s = sets[a] & sets[b]
                    print(f"{a} -> {b}: shared {len(s)}, source cells in shared clones "
                          f"{int(lab[(lab['group'] == a) & lab['clone'].isin(s)].shape[0])} of {n[a]}, target "
                          f"{int(lab[(lab['group'] == b) & lab['clone'].isin(s)].shape[0])} of {n[b]}")


if __name__ == "__main__":
    main()
