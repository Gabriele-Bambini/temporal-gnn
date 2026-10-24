"""Shared data, targets, strict per-bin features and metrics for the design_best-signal prototype (numpy only).

Data: Dileo & Sottoriva '-human' node files, first N genes (N = rows of <name>_y_outdeg.csv), snapshot bins only
(gastrulation-human 2..31, pancreas-human 0..14). Transition i maps snapshot i -> i+1 (29 / 14 transitions).
Per bin we keep X[b] = (G, 5) = (mean, median, sd, frac_nonzero, sum) as dist_bridge expects; n_b = median sum/mean.
Targets: tack/hpc/dileo_protocol.TARGET_NP (orig, l1, linear), imported.
STRICT: every feature of transition i is a function of snapshots 0..i only (filters are causal, the SVD basis and the
empirical-Bayes signal variance use bins <= i, the GRN is the bin-i graph).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
HPC = os.path.join(ROOT, "tack", "hpc")
OUT = os.path.dirname(os.path.abspath(__file__))
EDGE_CACHE = os.environ.get("GRAPH_CACHE", r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\edge_cache")
sys.path.insert(0, HPC)
import dileo_protocol  # noqa: E402  (numpy only)
import dist_bridge as DB  # noqa: E402  (numpy + scipy + own sinkhorn, no torch)

SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
TARGETS = ("orig", "l1", "linear")
EPS = 1e-8
K = 200


class Data:
    def __init__(self, name):
        cache = os.path.join(OUT, f"_node_{name}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            Xs = z["X"]
        else:
            n_genes = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
            stats = ("s00_mean", "s01_median", "s02_std", "s03_frac_nonzero", "s04_sum")
            cols = [f"t{b:03d}_{s}" for b in SNAPS[name] for s in stats]
            node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=cols, nrows=n_genes)
            Xs = np.stack([np.stack([node[f"t{b:03d}_{s}"].to_numpy(np.float64) for s in stats], 1)
                           for b in SNAPS[name]])
            np.savez(cache, X=Xs)
        self.name = name
        self.bins = SNAPS[name]
        self.X = Xs                                   # (B, G, 5)
        self.B, self.G = Xs.shape[0], Xs.shape[1]
        self.T = self.B - 1
        self.m = np.maximum(Xs[:, :, 0].T, 0)         # (G, B)
        self.s = Xs[:, :, 2].T
        self.f = Xs[:, :, 3].T
        self.n = np.array([DB.n_cells(Xs[b]) for b in range(self.B)])
        self.lib = self.m.sum(0)
        self.CP = 1e4 * self.m / self.lib[None, :]
        self.L = np.log1p(self.CP)
        self.n_test = int(round(0.25 * self.G))

    def y(self, tg, i):
        return np.asarray(dileo_protocol.TARGET_NP[tg](self.m[:, i], self.m[:, i + 1]), np.float64)

    def Y(self, tg):
        return np.stack([self.y(tg, i) for i in range(self.T)])


def tfun(tg, now, nxt):
    """Target formula between two base-scale level vectors (orig/linear: raw counts per cell; l1: CP10k-like)."""
    if tg == "orig":
        return np.clip(np.log(np.maximum(nxt, 0) + EPS) - np.log(np.maximum(now, 0) + EPS), -5, 5)
    if tg == "linear":
        return nxt - now
    a = 1e4 * np.maximum(nxt, 0) / max(np.maximum(nxt, 0).sum(), 1e-12)
    b = 1e4 * np.maximum(now, 0) / max(np.maximum(now, 0).sum(), 1e-12)
    return np.log1p(a) - np.log1p(b)


# ------------------------------------------------------------------ metrics (their convention + tie-aware)
def _incl(v, k):
    thr = np.sort(v)[::-1][k - 1]
    above, eq = v > thr, v == thr
    p = above.astype(float)
    p[eq] = (k - above.sum()) / eq.sum()
    return p


def metrics(y, p, k=K):
    """(pearson [nan if a side is constant], P@200 up, P@200 down [np.argsort, as their _delta_metrics],
    tie-aware expected P@200 up, down)."""
    y = np.asarray(y, np.float32)
    p = np.asarray(p, np.float32)
    k = min(k, len(y))
    if np.all(y == y[0]) or np.all(p == p[0]):
        r = np.nan
    else:
        yc, pc = y.astype(np.float64) - y.mean(), p.astype(np.float64) - p.mean()
        r = float((yc * pc).sum() / np.sqrt((yc ** 2).sum() * (pc ** 2).sum()))
    up = len(np.intersect1d(np.argsort(-y)[:k], np.argsort(-p)[:k])) / k
    dn = len(np.intersect1d(np.argsort(y)[:k], np.argsort(p)[:k])) / k
    tu = float((_incl(y, k) * _incl(p, k)).sum() / k)
    td = float((_incl(-y, k) * _incl(-p, k)).sum() / k)
    return r, up, dn, tu, td


def criterion(y, p, n_test):
    """Selection criterion J (all genes): Pearson + tie-aware P@K up + down, K = 200 * N / n_test (dist_bridge)."""
    return DB.criterion(y, p, 200.0 / n_test)


def masks(G, T, draws=20, seed=20261022):
    rng = np.random.default_rng(seed)
    n_test = int(round(0.25 * G))
    return np.array([[np.sort(rng.permutation(G)[:n_test]) for _ in range(T)] for _ in range(draws)])


def score(Y, P, mk):
    """Y, P: (T, G) (P rows may be None). Returns (draws, T, 5) metrics on the fixed masks."""
    D, T = mk.shape[0], len(Y)
    out = np.full((D, T, 5), np.nan)
    for i in range(T):
        if P[i] is None:
            continue
        for k in range(D):
            out[k, i] = metrics(Y[i][mk[k, i]], P[i][mk[k, i]])
    return out


def summarize(S, idx):
    """S: (draws, T, 5); idx: transitions to average. Mean over draws of the transition nanmean, and mask sd."""
    sub = S[:, idx]
    per_draw = np.nanmean(sub, axis=1)                 # (draws, 5)
    keys = ("pcc", "p200_up", "p200_down", "pta_up", "pta_down")
    return {k: round(float(np.mean(per_draw[:, j])), 4) for j, k in enumerate(keys)} | \
        {k + "_sd": round(float(np.std(per_draw[:, j])), 4) for j, k in enumerate(keys[:3])}
