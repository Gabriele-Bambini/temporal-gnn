"""Shared data, targets, masks, metrics and strict rolling-origin selection for the latent-SB prototype.

Data: Dileo & Sottoriva '-human' node files, first N genes (N = rows of <name>_y_outdeg.csv), snapshot bins only
(gastrulation-human 2..31, pancreas-human 0..14). Snapshot s = 0..S-1, transition i maps snapshot i -> i+1.
Targets: dileo_protocol.TARGET_NP (orig, l1, linear) on the raw bin means (float32 inputs, as dileo_rescore).
Test masks: N_DRAWS fixed draws of a random 25 % of genes per transition (seed SEED), shared by every method, so
all comparisons are paired. Test metrics: dileo_rescore.metrics (pearson_nan0, P@200 up/down with np.argsort,
plus tie-aware p_ta). Selection criterion (strict): J = Pearson + tie-aware P@K up + down on ALL genes with
K = round(200 / n_test * N) (the same quantile as P@200 on a 25 % mask), averaged over past transitions
j in [max(1, i - WINDOW), i - 1]; a config's prediction at origin j uses only snapshots <= j.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
HPC = os.path.join(ROOT, "tack", "hpc")
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HPC)
import dileo_protocol  # noqa: E402  (numpy only; patch() never called, torch never imported)
import dileo_rescore  # noqa: E402

SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
TARGETS = ("orig", "l1", "linear")
N_DRAWS = 10
SEED = 20261022
WINDOW = 8
K_TEST = 200


class Data:
    def __init__(self, name):
        cache = os.path.join(OUT, f"_cache_{name}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            self.M, self.SD, self.F, self.SUM = z["M"], z["SD"], z["F"], z["SUM"]
        else:
            n = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
            b = SNAPS[name]
            cols = [f"t{x:03d}_{s}" for x in b for s in ("s00_mean", "s02_std", "s03_frac_nonzero", "s04_sum")]
            node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=cols, nrows=n)
            col = lambda s: node[[f"t{x:03d}_{s}" for x in b]].to_numpy(np.float64)
            self.M, self.SD, self.F, self.SUM = col("s00_mean"), col("s02_std"), col("s03_frac_nonzero"), col("s04_sum")
            np.savez(cache, M=self.M, SD=self.SD, F=self.F, SUM=self.SUM)
        self.name = name
        self.bins = SNAPS[name]
        self.M = np.maximum(self.M, 0)
        ok = self.M > 0
        self.nc = np.array([np.round(np.median(self.SUM[ok[:, s], s] / self.M[ok[:, s], s]))
                            for s in range(self.M.shape[1])])
        self.N, self.S = self.M.shape
        self.T = self.S - 1
        self.lib = self.M.sum(0)
        self.CP = 1e4 * self.M / self.lib
        self.X = np.log1p(self.CP)
        # law triplets (m, f, E[c^2]) and raw node rows as dist_bridge expects them (cols m, median, sd, f, sum)
        self.trip = [np.stack([self.M[:, s], self.F[:, s], self.SD[:, s] ** 2 + self.M[:, s] ** 2], 1)
                     for s in range(self.S)]


def target(d, tname, i):
    return np.asarray(dileo_protocol.TARGET_NP[tname](d.M[:, i].astype(np.float32),
                                                      d.M[:, i + 1].astype(np.float32)), np.float64)


def all_targets(d, tname):
    return np.stack([target(d, tname, i) for i in range(d.T)])


def masks(d):
    rng = np.random.default_rng(SEED)
    n_test = int(round(0.25 * d.N))
    return np.array([[np.sort(rng.permutation(d.N)[:n_test]) for _ in range(d.T)] for _ in range(N_DRAWS)])


def _incl(v, k):
    thr = np.partition(v, len(v) - k)[len(v) - k]
    above, eq = v > thr, v == thr
    p = above.astype(float)
    p[eq] = (k - above.sum()) / eq.sum()
    return p


def J_all(y, p, k):
    """Selection criterion on all genes: Pearson (0 if constant) + tie-aware P@k up + down."""
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    if not np.all(np.isfinite(p)):
        return -np.inf
    yc, pc = y - y.mean(), p - p.mean()
    den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
    r = float((yc * pc).sum() / den) if den > 0 else 0.0
    up = float((_incl(y, k) * _incl(p, k)).sum() / k)
    dn = float((_incl(-y, k) * _incl(-p, k)).sum() / k)
    return r + up + dn


def k_sel(d):
    return int(round(K_TEST / round(0.25 * d.N) * d.N))


def select(Jmat, default, start=1, window=WINDOW):
    """Jmat: C x T (J of config c at past transition j). Returns the selected config index per transition."""
    C, T = Jmat.shape
    sel = np.full(T, default, int)
    for i in range(T):
        lo = max(start, i - window)
        if i - lo >= 1 and np.isfinite(Jmat[:, lo:i]).any():
            with np.errstate(all="ignore"):
                sc = np.nanmean(np.where(np.isfinite(Jmat[:, lo:i]), Jmat[:, lo:i], np.nan), axis=1)
            sc = np.where(np.isfinite(sc), sc, -np.inf)
            sel[i] = int(np.argmax(sc))
    return sel


def test_scores(Y, P, mk):
    """Y, P: T x N. Returns (draws, T, 5): pearson_nan0, P@200 up, P@200 down, p_ta_up, p_ta_down."""
    D, T = mk.shape[0], Y.shape[0]
    out = np.zeros((D, T, 5))
    for i in range(T):
        for k in range(D):
            r = dileo_rescore.metrics(Y[i][mk[k, i]], P[i][mk[k, i]])
            out[k, i] = [r["pearson_nan0"], r["p_at_200_up"], r["p_at_200_down"], r["p_ta_up"], r["p_ta_down"]]
    return out


def conventions(d):
    n = d.nc
    T = d.T
    big = np.array([min(n[i], n[i + 1]) >= 100 for i in range(T)])
    return {"all": np.arange(T) >= 0, "i>=1": np.arange(T) >= 1, "i>=3": np.arange(T) >= 3,
            "i>=1_nmin>=100": (np.arange(T) >= 1) & big}


def summarize(sc, d):
    """sc: (draws, T, 5). Mean over transitions of each convention; mask sd = sd over draws of that mean."""
    out = {}
    for cn, sel in conventions(d).items():
        per_draw = sc[:, sel, :].mean(1)  # (draws, 5)
        mu = per_draw.mean(0)
        out[cn] = {"pcc": round(float(mu[0]), 4), "up": round(float(mu[1]), 4), "down": round(float(mu[2]), 4),
                   "ta_up": round(float(mu[3]), 4), "ta_down": round(float(mu[4]), 4),
                   "pcc_mask_sd": round(float(per_draw[:, 0].std()), 4), "n_trans": int(sel.sum())}
    out["per_transition_pcc"] = [round(float(x), 4) for x in sc[:, :, 0].mean(0)]
    return out


def paired(a, b, d, conv="i>=1"):
    """Paired per-transition comparison (a - b) of mask-averaged scores on a convention."""
    sel = conventions(d)[conv]
    A, B = a.mean(0)[sel], b.mean(0)[sel]
    out = {}
    for j, key in enumerate(("pcc", "up", "down")):
        dd = A[:, j] - B[:, j]
        nz = dd[np.abs(dd) > 1e-12]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else None
        out[key] = {"mean_delta": round(float(dd.mean()), 4), "better": int((dd > 1e-12).sum()),
                    "worse": int((dd < -1e-12).sum()), "wilcoxon_p": None if p is None else round(p, 4)}
    return out
