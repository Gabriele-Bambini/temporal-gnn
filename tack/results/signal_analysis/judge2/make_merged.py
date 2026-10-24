"""Judge 2: re-binned robustness builds of gastrulation-human for the frozen NASS (same cells, different resolution).
Consecutive snapshot bins are merged in pairs (offset 0: (0,1),(2,3),...; offset 1: (1,2),(3,4),..., bin 0 dropped):
mean and nonzero fraction are cell-weighted means, E[c^2] is pooled exactly (population sd recomputed), sum adds,
the median column (unused by NASS) is the cell-weighted mean of medians. Writes judge2/_node_<newname>.npz in the
dsdata cache format (B, G, 5)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsdata as D

d = D.Data("gastrulation-human")
X = d.X
for off in (0, 1):
    groups = [(b, b + 1) for b in range(off, d.B - 1, 2)]
    out = []
    for g in groups:
        n = np.array([D.DB.n_cells(X[b]) for b in g])
        w = n / n.sum()
        m = sum(wi * X[b][:, 0] for wi, b in zip(w, g))
        e2 = sum(wi * (X[b][:, 2] ** 2 + X[b][:, 0] ** 2) for wi, b in zip(w, g))
        f = sum(wi * X[b][:, 3] for wi, b in zip(w, g))
        med = sum(wi * X[b][:, 1] for wi, b in zip(w, g))
        s = np.sqrt(np.maximum(e2 - m ** 2, 0))
        sm = sum(X[b][:, 4] for b in g)
        out.append(np.stack([m, med, s, f, sm], 1))
    Xm = np.stack(out)
    name = f"gastr-merged2-off{off}"
    np.savez(os.path.join(HERE, f"_node_{name}.npz"), X=Xm)
    print(name, Xm.shape, "cells", [round(D.DB.n_cells(x)) for x in Xm])
