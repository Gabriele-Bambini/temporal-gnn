"""Shared loaders for verify_1 (numpy / pandas only, no torch). Reads the Dileo '-human' node files directly.

Masks: the CONTEXT convention, np.sort(np.random.default_rng(1000*s + i).permutation(G)[:n_test]), s in {0,1,2},
n_test = round(0.25 G); transition i maps snapshot i -> i+1 (0-based snapshot index; gastrulation snapshots are the
raw bins 2..31, pancreas 0..14).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
HPC = os.path.join(ROOT, "tack", "hpc")
DBS = os.path.join(ROOT, "tack", "results", "signal_analysis", "design_best-signal")
OUT = os.path.dirname(os.path.abspath(__file__))
EDGE_CACHE = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\edge_cache"
NAMES = ("gastrulation-human", "pancreas-human")
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
ALLBINS = {"gastrulation-human": 32, "pancreas-human": 15}
STATS = ("s00_mean", "s01_median", "s02_std", "s03_frac_nonzero", "s04_sum")
sys.path.insert(0, HPC)
import dileo_protocol  # noqa: E402
import dileo_rescore  # noqa: E402


def n_genes(name):
    return len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))


def load_all_bins(name):
    """(G, B_all, 5) over every raw bin in the file, first N rows."""
    N = n_genes(name)
    B = ALLBINS[name]
    cols = [f"t{b:03d}_{s}" for b in range(B) for s in STATS]
    node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=cols, nrows=N)
    X = np.stack([np.stack([node[f"t{b:03d}_{s}"].to_numpy(np.float64) for s in STATS], 1) for b in range(B)], 1)
    return X


def n_cells(xb):
    ok = xb[:, 0] > 0
    return float(np.median(xb[ok, 4] / xb[ok, 0]))


class Data:
    """Snapshot-bin data. m, s, f: (G, B); n: cells per bin; CP = CP10k; L = log1p(CP)."""

    def __init__(self, name):
        Xa = load_all_bins(name)
        self.name = name
        self.bins = SNAPS[name]
        X = Xa[:, self.bins, :]
        self.X = X
        self.G, self.B = X.shape[0], X.shape[1]
        self.T = self.B - 1
        self.m = np.maximum(X[:, :, 0], 0)
        self.s = X[:, :, 2]
        self.f = X[:, :, 3]
        self.n = np.array([n_cells(X[:, b]) for b in range(self.B)])
        self.lib = self.m.sum(0)
        self.kap = 1e4 / self.lib
        self.CP = self.m * self.kap[None, :]
        self.L = np.log1p(self.CP)
        self.n_test = int(round(0.25 * self.G))

    def Y(self, tg="l1"):
        return np.stack([np.asarray(dileo_protocol.TARGET_NP[tg](self.m[:, i], self.m[:, i + 1]), np.float64)
                         for i in range(self.T)])

    def masks(self):
        return np.array([[np.sort(np.random.default_rng(1000 * s + i).permutation(self.G)[:self.n_test])
                          for i in range(self.T)] for s in range(3)])

    def R_L(self, floor=False):
        """Delta-method noise variance of L = log1p(CP10k) of a bin mean: kap^2 Var(mean) / (1 + CP)^2,
        Var(mean) = s^2 / (n - 1) (population sd). floor=True uses NASS's max(s^2, running mean / kap) + 1/n."""
        nn = np.maximum(self.n - 1, 1)[None, :]
        if floor:
            w = np.cumsum(self.n)
            mbar = np.cumsum(self.m * self.n[None, :], 1) / w[None, :]
            v = (np.maximum(self.s ** 2, mbar) + 1.0 / self.n[None, :]) / nn
        else:
            v = self.s ** 2 / nn
        return self.kap[None, :] ** 2 * v / (1 + self.CP) ** 2


def pcc(a, b):
    a = np.asarray(a, np.float64) - np.mean(a)
    b = np.asarray(b, np.float64) - np.mean(b)
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")
