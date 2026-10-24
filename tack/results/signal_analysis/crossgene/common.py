"""Shared loading, targets, masks, metrics and strict rolling-origin selection for the cross-gene signal analysis.

Data: Dileo & Sottoriva '-human' node files, first N genes (N = rows of <name>_y_outdeg.csv), snapshot bins only
(gastrulation-human 2..31, pancreas-human 0..14). Snapshot index s = 0..S-1; transition i maps snapshot i -> i+1
(29 / 14 transitions). Targets come from dileo_protocol.TARGET_NP (l1, linear, orig) on the raw bin means.
Masks: per draw and transition a fresh random 25 % of genes (round(0.25 N)), fixed seed, shared by every method.
Metrics (as dileo_task2.metrics): Pearson over genes (NaN if a side is constant), P@200 up / down with np.argsort.
STRICT: a forecast for transition i reads only snapshots 0..i. Rolling-origin selection at transition i picks the
config with the best mean J = pearson + P@200 up + P@200 down (all genes, targets of past transitions j < i, j >= 1,
window of the last 8) -- the same rule DistBridge uses; with no past transition j >= 1 the family default is used.
"""
import os
import sys

import numpy as np
import pandas as pd

HPC = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\hpc"
GRN = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tgl-grn-1CCD\GRN"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HPC)
import dileo_protocol  # noqa: E402  (numpy only, no torch)

SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
K_TOP = 200
WINDOW = 8
N_DRAWS = 10
SEED = 20261022


class Data:
    def __init__(self, name):
        node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"))
        n = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
        b = SNAPS[name]
        col = lambda s: node[[f"t{x:03d}_{s}" for x in b]].to_numpy(np.float64)[:n]
        self.name = name
        self.genes = node["gene"].astype(str).to_numpy()[:n]
        self.bins = b
        self.M = np.maximum(col("s00_mean"), 0)        # raw UMI bin means, N x S
        self.SD = col("s02_std")                       # population sd of counts in the bin
        self.F = col("s03_frac_nonzero")
        sums = col("s04_sum")
        ok = self.M > 0
        self.nc = np.array([np.median(sums[ok[:, s], s] / self.M[ok[:, s], s]) for s in range(len(b))])
        self.lib = self.M.sum(0)                       # UMI per cell over the N genes
        self.CP = 1e4 * self.M / self.lib              # CP10k exactly as dileo_protocol._cp10k
        self.L = np.log1p(self.CP)
        self.N, self.S = self.M.shape
        self.T = self.S - 1                            # number of transitions


def target(d, tname, i):
    return np.asarray(dileo_protocol.TARGET_NP[tname](d.M[:, i].astype(np.float32), d.M[:, i + 1].astype(np.float32)),
                      np.float64)


def all_targets(d, tname):
    return np.stack([target(d, tname, i) for i in range(d.T)])


def masks(d):
    rng = np.random.default_rng(SEED)
    n_test = int(round(0.25 * d.N))
    out = np.zeros((N_DRAWS, d.T, n_test), np.int64)
    for k in range(N_DRAWS):
        for i in range(d.T):
            out[k, i] = np.sort(rng.permutation(d.N)[d.N - n_test:])
    return out


def metrics3(y, p):
    """(pearson, P@200 up, P@200 down); pearson NaN if a side is constant."""
    y = np.asarray(y, np.float32)
    p = np.asarray(p, np.float32)
    if np.all(y == y[0]) or np.all(p == p[0]) or not np.all(np.isfinite(p)):
        r = np.nan
    else:
        yc, pc = y - y.mean(), p - p.mean()
        den = np.sqrt(float((yc.astype(np.float64) ** 2).sum()) * float((pc.astype(np.float64) ** 2).sum()))
        r = float((yc.astype(np.float64) * pc).sum() / den) if den > 0 else np.nan
    k = min(K_TOP, len(y))
    up = len(np.intersect1d(np.argsort(-y)[:k], np.argsort(-p)[:k], assume_unique=True)) / k
    dn = len(np.intersect1d(np.argsort(y)[:k], np.argsort(p)[:k], assume_unique=True)) / k
    return r, up, dn


def J_of(y, p):
    r, up, dn = metrics3(y, p)
    return (0.0 if np.isnan(r) else r) + up + dn


def test_scores(Y, preds, mk):
    """Y, preds: T x N. Returns T x 3 (pearson, up, down) averaged over draws (pearson nanmean over draws)."""
    T = Y.shape[0]
    out = np.zeros((T, 3))
    for i in range(T):
        rows = np.array([metrics3(Y[i, mk[k, i]], preds[i, mk[k, i]]) for k in range(mk.shape[0])], float)
        with np.errstate(all="ignore"):
            out[i] = [np.nanmean(rows[:, 0]) if np.isfinite(rows[:, 0]).any() else np.nan,
                      rows[:, 1].mean(), rows[:, 2].mean()]
    return out


def select(Jmat, default=0, start=1, window=WINDOW):
    """Jmat: C x T (J of config c at transition j, all genes). Returns the selected config per transition."""
    C, T = Jmat.shape
    sel = np.full(T, default, int)
    for i in range(T):
        lo = max(start, i - window)
        if i - lo >= 1:
            sel[i] = int(np.argmax(np.nanmean(Jmat[:, lo:i], axis=1)))
    return sel


def summarize(per_t, first=1, drop_last=False):
    """per_t: T x 3 test scores. Mean over transitions first..T-1 (pearson NaN counted as 0)."""
    x = per_t[first:-1] if drop_last else per_t[first:]
    x = np.where(np.isnan(x), 0.0, x)
    return {"pearson": float(x[:, 0].mean()), "p_at_200_up": float(x[:, 1].mean()),
            "p_at_200_down": float(x[:, 2].mean())}


def report(per_t, d):
    return {"i>=1": summarize(per_t, 1), "i>=1_no_last": summarize(per_t, 1, True),
            "per_transition": [{"i": i, "bins": f"{d.bins[i]}->{d.bins[i + 1]}", "pearson": _f(per_t[i, 0]),
                                "up": _f(per_t[i, 1]), "down": _f(per_t[i, 2])} for i in range(d.T)]}


def _f(x):
    return None if not np.isfinite(x) else round(float(x), 4)


def paired(a, b, first=1):
    """Paired per-transition comparison of two T x 3 score arrays (a - b), transitions >= first."""
    from scipy import stats
    out = {}
    for j, k in enumerate(("pearson", "p_at_200_up", "p_at_200_down")):
        da = np.nan_to_num(a[first:, j]) - np.nan_to_num(b[first:, j])
        nz = da[np.abs(da) > 1e-12]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else None
        out[k] = {"mean_delta": round(float(da.mean()), 4), "n_better": int((da > 1e-12).sum()),
                  "n_worse": int((da < -1e-12).sum()), "wilcoxon_two_sided_p": p}
    Ja = np.nan_to_num(a[first:]).sum(1)
    Jb = np.nan_to_num(b[first:]).sum(1)
    dJ = Ja - Jb
    nz = dJ[np.abs(dJ) > 1e-12]
    out["J"] = {"mean_delta": round(float(dJ.mean()), 4), "n_better": int((dJ > 1e-12).sum()),
                "n_worse": int((dJ < -1e-12).sum()),
                "wilcoxon_two_sided_p": float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else None}
    return out


def cw_mean(X, w):
    """Weighted mean over the last axis."""
    w = np.asarray(w, float)
    return (X * w).sum(-1) / w.sum()
