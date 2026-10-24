"""Shared loading, targets and scoring for the temporal-structure analysis (Dileo Task 2, '-human' builds).

Targets: same numpy formulas as tack/hpc/dileo_protocol.TARGET_NP (re-implemented here, identical maths).
Scoring: Pearson over genes, P@200 up/down on a random 25 % of genes per transition, mean over transitions
i >= 1 (i = 0 has no past); strict predictions do not depend on the mask, so every model is scored on the same
n_draws masks (paired) and the mean over draws is reported.
"""
import os
import numpy as np
import pandas as pd

GRN = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tgl-grn-1CCD\GRN"
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
TRAIN_SPLIT = {"gastrulation-human": 29, "pancreas-human": 14}
TARGETS = ("orig", "l1", "linear")
EPS = 1e-8
K = 200


def load(name):
    """Return dict with arrays (genes x snapshot bins): m (mean), s (std), f (frac nonzero), n (cells per bin)."""
    cache = os.path.join(os.path.dirname(__file__), f"_cache_{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    n_genes = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    bins = SNAPS[name]
    cols = []
    for b in bins:
        cols += [f"t{b:03d}_s00_mean", f"t{b:03d}_s02_std", f"t{b:03d}_s03_frac_nonzero", f"t{b:03d}_s04_sum"]
    node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=cols, nrows=n_genes)
    m = np.stack([node[f"t{b:03d}_s00_mean"].to_numpy(np.float64) for b in bins], 1)
    s = np.stack([node[f"t{b:03d}_s02_std"].to_numpy(np.float64) for b in bins], 1)
    f = np.stack([node[f"t{b:03d}_s03_frac_nonzero"].to_numpy(np.float64) for b in bins], 1)
    S = np.stack([node[f"t{b:03d}_s04_sum"].to_numpy(np.float64) for b in bins], 1)
    n = np.array([np.median(S[m[:, j] > 0, j] / m[m[:, j] > 0, j]) for j in range(len(bins))])
    out = dict(m=m, s=s, f=f, n=np.round(n), bins=np.array(bins))
    np.savez(cache, **out)
    return out


def tgt(target, m0, m1):
    m0 = np.maximum(m0, 0)
    m1 = np.maximum(m1, 0)
    if target == "orig":
        d = np.log(m1.astype(np.float32) + np.float32(EPS)) - np.log(m0.astype(np.float32) + np.float32(EPS))
        return np.clip(d, -5.0, 5.0).astype(np.float64)
    if target == "l1":
        c0 = 1e4 * m0 / max(m0.sum(), 1e-12)
        c1 = 1e4 * m1 / max(m1.sum(), 1e-12)
        return np.log1p(c1) - np.log1p(c0)
    if target == "linear":
        return m1 - m0
    raise ValueError(target)


def targets(d, target, name):
    m = d["m"]
    T = TRAIN_SPLIT[name]
    return np.stack([tgt(target, m[:, i], m[:, i + 1]) for i in range(T)])  # (T, G)


def level_space(target, m):
    """Per-bin level X such that the target is approximately X_{t+1} - X_t (orig: no exact level, use log(m+eps))."""
    if target == "l1":
        return np.log1p(1e4 * m / np.maximum(m.sum(0, keepdims=True), 1e-12))
    if target == "linear":
        return m.copy()
    return np.log(m + EPS)


def _metrics(y, p, k=K):
    yc, pc = y - y.mean(), p - p.mean()
    den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
    r = float((yc * pc).sum() / den) if den > 0 else 0.0
    up = len(np.intersect1d(np.argsort(-y, kind="quicksort")[:k], np.argsort(-p, kind="quicksort")[:k])) / k
    dn = len(np.intersect1d(np.argsort(y, kind="quicksort")[:k], np.argsort(p, kind="quicksort")[:k])) / k
    return r, up, dn


class Scorer:
    """Fixed random 25 % masks per (draw, transition); identical for every model -> paired comparisons."""

    def __init__(self, n_genes, n_trans, n_draws=10, seed=0, frac=0.25):
        rng = np.random.default_rng(seed)
        n_test = int(round(frac * n_genes))
        self.masks = [[np.sort(rng.permutation(n_genes)[:n_test]) for _ in range(n_trans)] for _ in range(n_draws)]

    def per_transition(self, Y, P, start=1):
        """Returns array (T, 3) of mask-averaged (PCC, P@200 up, P@200 down) for transitions >= start (NaN before)."""
        T = len(Y)
        out = np.full((T, 3), np.nan)
        for i in range(start, T):
            vals = [_metrics(Y[i][mk[i]], P[i][mk[i]]) for mk in self.masks]
            out[i] = np.mean(vals, 0)
        return out

    def summary(self, Y, P, start=1):
        pt = self.per_transition(Y, P, start)
        return {"pcc": float(np.nanmean(pt[:, 0])), "p200_up": float(np.nanmean(pt[:, 1])),
                "p200_down": float(np.nanmean(pt[:, 2])), "per_transition": pt[start:].round(4).tolist()}


def allgene_metrics(y, p, k=K):
    """Fast criterion on all genes (used only for rolling-origin selection on past transitions)."""
    return _metrics(y, p, k)
