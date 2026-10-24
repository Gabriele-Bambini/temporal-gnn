"""verify_3 shared helpers: data (dsdata cache, read-only), l1-scale known noise R, masks, scoring.

Masks: default_rng(1000 s + i).permutation(G)[:round(0.25 G)], s = 0, 1, 2 (as uot_judge/indep_check.py).
Metrics: tack/hpc/dileo_rescore.metrics (pearson, tie-aware p_ta_up / p_ta_down, spearman).
Known noise of today's log1p(CP10k) value, per gene and bin (declared choice, C2 is not visible here):
  raw scale   R_raw = s^2 / (n - 1)   (s = population sd of the bin, so Var(mean) = s^2/(n-1)),
              floored at its m = 0 value R0 = 1 / n^2 (= the value one count in one cell gives: s^2 = (n-1)/n^2);
  l1 scale    R = R_raw * (kap / (1 + CP))^2, kap = 1e4 / lib_b (delta method of log1p(1e4 m / lib)).
"""
import os
import sys

import numpy as np

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
SA = os.path.join(ROOT, "tack", "results", "signal_analysis", "design_best-signal")
HPC = os.path.join(ROOT, "tack", "hpc")
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HPC)
sys.path.insert(0, SA)
import dsdata as D  # noqa: E402  (reads design_best-signal/_node_*.npz, never writes when the cache exists)
import dileo_rescore as RS  # noqa: E402
import dileo_protocol as PR  # noqa: E402

NAMES = ("gastrulation-human", "pancreas-human")
SHORT = {"gastrulation-human": "gastr", "pancreas-human": "pancr"}
KEYS = ("pearson", "p_ta_up", "p_ta_down", "spearman")


def load(name):
    d = D.Data(name)
    return d


def R_raw(d, floor=True):
    n = d.n[None, :]
    R = d.s ** 2 / np.maximum(n - 1, 1)
    if floor:
        R = np.maximum(R, 1.0 / n ** 2)
    return R


def R_l1(d, floor=True):
    kap = 1e4 / d.lib
    return R_raw(d, floor) * (kap[None, :] / (1 + d.CP)) ** 2


def masks(G, T):
    nt = int(round(0.25 * G))
    return [[np.sort(np.random.default_rng(1000 * s + i).permutation(G)[:nt]) for i in range(T)] for s in range(3)]


def score(Y, P, mk):
    """rows (3 seeds, T, 4) of pearson / p_ta_up / p_ta_down / spearman; NaN rows where P[i] is None."""
    T = len(Y)
    out = np.full((3, T, 4), np.nan)
    for s in range(3):
        for i in range(T):
            if P[i] is None:
                continue
            t = mk[s][i]
            m = RS.metrics(Y[i][t], np.asarray(P[i])[t])
            out[s, i] = [m[k] for k in KEYS]
    return out


def summ(rows, idx):
    """mean over seeds of the nanmean over transitions idx -> dict."""
    idx = list(idx)
    if len(idx) == 0:
        return None
    v = np.nanmean(rows[:, idx, :], axis=1).mean(0)
    return {k: round(float(x), 4) for k, x in zip(("pcc", "up", "down", "spearman"), v)}


def fmt(s):
    return "n/a" if s is None else f"{s['pcc']:+.3f}/{s['up']:.3f}/{s['down']:.3f}"
