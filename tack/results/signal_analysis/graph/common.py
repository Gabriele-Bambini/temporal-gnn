"""Shared loaders for the graph / artefact / heterogeneity signal analysis (numpy only, no torch).

Data: Dileo & Sottoriva '-human' builds in tgl-grn-1CCD/GRN. Node file: per gene per bin t{bbb}_s00_mean,
s01_median, s02_std, s03_frac_nonzero, s04_sum (raw UMI counts); only the first N rows are used
(N = rows of <name>_y_outdeg.csv). Edge file <name>.csv: u (regulator), i (target), ts (bin), label (GRNBoost2
importance). Edges are cached once as compact arrays (u, i int32, ts int16, w float32) in CACHE (a temp dir,
rebuildable in ~30 s) so the 2-3.5 GB csv files are parsed only once.
Targets: the numpy formulas of tack/hpc/dileo_protocol.TARGET_NP (orig, l1, linear), imported, not copied.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
HPC = os.path.join(ROOT, "tack", "hpc")
OUT = os.path.join(ROOT, "tack", "results", "signal_analysis", "graph")
CACHE = os.environ.get("GRAPH_CACHE", r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\edge_cache")
sys.path.insert(0, HPC)
import dileo_protocol  # noqa: E402  (numpy only; torch is imported only inside patch())

SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
TARGETS = ("orig", "l1", "linear")
K = 200
TEST_FRAC = 0.25


def load_node(name):
    """Return dict with arrays (N, B) over ALL bins 0..B-1: m (mean), med, s (sd), f (frac nonzero), S (sum),
    plus n (cells per bin, B) and genes."""
    node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"))
    n = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    B = len([c for c in node.columns if c.endswith("_s00_mean")])
    def arr(stat):
        return node[[f"t{b:03d}_{stat}" for b in range(B)]].to_numpy(np.float64)[:n]
    d = {"m": arr("s00_mean"), "med": arr("s01_median"), "s": arr("s02_std"), "f": arr("s03_frac_nonzero"),
         "S": arr("s04_sum"), "genes": node["gene"].astype(str).to_numpy()[:n]}
    ncell = []
    for b in range(B):
        ok = d["m"][:, b] > 0
        ncell.append(float(np.median(d["S"][ok, b] / d["m"][ok, b])) if ok.any() else 0.0)
    d["n"] = np.array(ncell)
    return d


def target(tname, m_now, m_next):
    return dileo_protocol.TARGET_NP[tname](m_now, m_next).astype(np.float64)


def transitions(name):
    s = SNAPS[name]
    return [(s[i], s[i + 1]) for i in range(len(s) - 1)]


def load_edges(name):
    """u, i, ts, w arrays for the whole edge file (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{name}_edges.npz")
    if not os.path.exists(f):
        parts = {k: [] for k in ("u", "i", "ts", "w")}
        for c in pd.read_csv(os.path.join(GRN, f"{name}.csv"), usecols=["u", "i", "ts", "label"],
                             dtype={"u": "int32", "i": "int32", "ts": "int16", "label": "float32"},
                             chunksize=4_000_000):
            parts["u"].append(c["u"].to_numpy()); parts["i"].append(c["i"].to_numpy())
            parts["ts"].append(c["ts"].to_numpy()); parts["w"].append(c["label"].to_numpy())
        np.savez(f, **{k: np.concatenate(v) for k, v in parts.items()})
    z = np.load(f)
    return z["u"], z["i"], z["ts"], z["w"]


def masks(n_genes, n_trans, draws, seed=0):
    """Random 25 % test masks per (draw, transition), as in their protocol (fresh per transition)."""
    rng = np.random.default_rng(seed)
    n_test = int(round(TEST_FRAC * n_genes))
    return [[np.sort(rng.permutation(n_genes)[:n_test]) for _ in range(n_trans)] for _ in range(draws)]


def pearson(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    yc, pc = y - y.mean(), p - p.mean()
    den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
    return float((yc * pc).sum() / den) if den > 0 else 0.0


def p_at_k(y, p, k=K):
    """Their P@k (np.argsort default kind) and the tie-aware expected version."""
    k = min(k, len(y))
    up = len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k
    dn = len(set(np.argsort(y)[:k]) & set(np.argsort(p)[:k])) / k
    return up, dn


def _incl(v, k):
    thr = np.sort(v)[::-1][k - 1]
    above, eq = v > thr, v == thr
    q = above.astype(float)
    q[eq] = (k - above.sum()) / eq.sum()
    return q


def p_tie_aware(y, p, k=K):
    k = min(k, len(y))
    out = {}
    for side, s in (("up", 1), ("down", -1)):
        ty, tp = _incl(s * y, k), _incl(s * p, k)
        out[side] = float((ty * tp).sum() / k)
        out[f"max_{side}"] = float((ty ** 2).sum() / k)
    return out


def score_masked(Y, P, mk):
    """Y, P: lists (per transition) of full-gene vectors; mk: masks(...) output. Mean over draws of the
    transition-averaged Pearson, P@200 up, P@200 down; also per-transition means over draws."""
    T = len(Y)
    per = np.zeros((T, 3))
    for draw in mk:
        for t in range(T):
            idx = draw[t]
            y, p = Y[t][idx], P[t][idx]
            up, dn = p_at_k(y, p)
            per[t] += np.array([pearson(y, p), up, dn]) / len(mk)
    return {"pcc": float(per[:, 0].mean()), "p200_up": float(per[:, 1].mean()), "p200_down": float(per[:, 2].mean()),
            "per_transition": per.round(4).tolist()}
