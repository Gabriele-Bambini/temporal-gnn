"""EXPLORATORY alignment check, metadata + clone matrix only (no expression), after larry_vivo_sanity found sisters NOT
near each other inside a mouse (kNN enrichment ~1) although they are inside the day-2 sample (26-44): is the clone
matrix aligned with the metadata rows for the in vivo cells? Same-type coupling within each mouse (each cell uniform
over the OTHER cells of its annotated type): enrichment > 1 = sisters share fate more than chance (clone labels carry
cell-level information); ~1 = no fate bias OR misaligned labels. Also the same between mice of a cohort (sisters split
at transplant), and a type-label permutation null inside each mouse.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import larry_vivo_prep as VP  # noqa: E402


def same_type_enrichment(clone, ct):
    """Coupling: each cell uniform over the other cells of its type; enrichment as larry_confirm.enrichment."""
    n = len(clone)
    df = pd.DataFrame({"c": clone, "t": ct})
    m = df.groupby(["c", "t"]).size()
    nt = df.groupby("t").size()
    mass = sum(v * (v - 1) / max(nt[t] - 1, 1) for (c, t), v in m.items()) / n
    base = float(((df.groupby("c").size() / n) ** 2).sum())
    return mass / base


def cross_type_enrichment(ca, ta, cb, tb):
    """Coupling from mouse a to mouse b: each a-cell uniform over b-cells of its type."""
    a = pd.DataFrame({"c": ca, "t": ta})
    b = pd.DataFrame({"c": cb, "t": tb})
    nb_t = b.groupby("t").size()
    mb = b.groupby(["c", "t"]).size()
    mass = sum(mb.get((c, t), 0) / nb_t[t] for c, t in zip(ca, ta) if t in nb_t.index) / len(ca)
    pa, pb = a.groupby("c").size() / len(a), b.groupby("c").size() / len(b)
    base = float(sum(pa[c] * pb.get(c, 0.0) for c in pa.index))
    return mass / base


def main():
    meta = pd.read_csv(os.path.join(VP.D, VP.P + "metadata.txt.gz"), sep="\t")
    C = scipy.io.mmread(os.path.join(VP.D, VP.P + "clone_matrix.mtx.gz"))
    C = (C if C.shape[0] == len(meta) else C.T).tocsr()
    nclo = np.diff(C.indptr)
    clone = np.full(C.shape[0], -1)
    one = np.where(nclo == 1)[0]
    clone[one] = C.indices[C.indptr[one]]
    ok = (meta["Starting population"].astype(str).to_numpy() == VP.POP) & (clone >= 0)
    mouse = meta["Recipient mouse"].astype(str).to_numpy()
    ct = meta["Cell type annotation"].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    mice = sorted(set(mouse[ok]) - {"nan", "NA"})
    for m in mice:
        i = np.where(ok & (mouse == m))[0]
        e = same_type_enrichment(clone[i], ct[i])
        null = [same_type_enrichment(clone[i], rng.permutation(ct[i])) for _ in range(5)]
        print(f"within {m}: n {len(i)} clones {len(set(clone[i]))} same-type enrichment {e:.2f} "
              f"(type-permutation null {np.mean(null):.2f} +- {np.std(null):.2f})", flush=True)
    for a in mice:
        for b in mice:
            if a < b and a[0] == b[0]:
                ia, ib = np.where(ok & (mouse == a))[0], np.where(ok & (mouse == b))[0]
                if len(set(clone[ia]) & set(clone[ib])) >= 20:
                    e = cross_type_enrichment(clone[ia], ct[ia], clone[ib], ct[ib])
                    print(f"across {a} -> {b}: shared {len(set(clone[ia]) & set(clone[ib]))} same-type enrichment "
                          f"{e:.2f}", flush=True)


if __name__ == "__main__":
    main()
