"""Shared helpers for verify_2 (step 3 of the chain of verification). numpy only, no torch.

Data: dsdata.Data (Dileo '-human' node files, first N genes, snapshot bins only: gastrulation-human bins 2..31,
pancreas-human bins 0..14; transition i = snapshot i -> i+1). Target: l1 = log1p(CP10k_{i+1}) - log1p(CP10k_i).
Noise (log scale, delta method): R_{g,b} = kappa_b^2 s^2 / (n_b - 1) / (1 + CP_{g,b})^2, kappa_b = 1e4 / sum_g m_{g,b},
s = s02_std (population sd). CV2_lib_b = 5th percentile over genes with m > 1 of (s^2 - m) / m^2.
Masks (CONTEXT): np.sort(default_rng(1000 s + i).permutation(G)[:round(G/4)]), s in {0, 1, 2}.
Metrics: tack/hpc/dileo_rescore.metrics (pearson, P@200 up/down, tie-aware, spearman).
"""
import os
import sys

import numpy as np

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
NASS_DIR = os.path.join(ROOT, "tack", "results", "signal_analysis", "design_best-signal")
HPC = os.path.join(ROOT, "tack", "hpc")
OUT = os.path.dirname(os.path.abspath(__file__))
for p in (NASS_DIR, HPC):
    if p not in sys.path:
        sys.path.insert(0, p)

import dsdata as D  # noqa: E402
import nass as N  # noqa: E402
import dileo_rescore as RS  # noqa: E402

NAMES = ("gastrulation-human", "pancreas-human")
SEEDS = (0, 1, 2)
CP_EDGES = (0.0, 0.1, 1.0, 10.0, 100.0)
CP_LABELS = ("0", "(0,0.1]", "(0.1,1]", "(1,10]", "(10,100]", ">100")


def cp_class(cp):
    """0 for CP == 0, 1..4 for (0,0.1], (0.1,1], (1,10], (10,100], 5 for > 100."""
    c = np.digitize(cp, CP_EDGES, right=True)          # cp == 0 -> 0; (0,0.1] -> 1; ...; > 100 -> 5
    return c


class Ext:
    """Data + log-scale noise quantities."""

    def __init__(self, name):
        d = D.Data(name)
        self.d = d
        self.name = name
        self.G, self.B, self.T = d.G, d.B, d.T
        self.m, self.s, self.f, self.n, self.lib = d.m, d.s, d.f, d.n, d.lib
        self.med = d.X[:, :, 1].T
        self.CP, self.L = d.CP, d.L
        self.kap = 1e4 / d.lib
        self.Y = d.Y("l1")                               # (T, G), float64 of the float32 target
        self.Yd = np.diff(self.L, axis=1).T              # same quantity in float64 (L_{i+1} - L_i)
        s2 = self.s ** 2
        m = self.m
        cv2 = np.zeros(self.B)
        for b in range(self.B):
            ok = m[:, b] > 1
            cv2[b] = np.percentile((s2[ok, b] - m[ok, b]) / m[ok, b] ** 2, 5)
        self.cv2lib = cv2
        self.s2 = s2
        self.s2c = np.maximum(s2 - m ** 2 * cv2[None, :], 0.0)
        self.R = self.r_of(s2)
        self.Rc = self.r_of(self.s2c)

    def r_of(self, s2):
        k = self.kap[None, :]
        return k ** 2 * s2 / np.maximum(self.n - 1, 1)[None, :] / (1 + self.CP) ** 2

    def r_floor(self):
        """Delta-method R with s^2 floored at 1/n (a single count in one of n cells): finite noise for zero genes."""
        return self.r_of(np.maximum(self.s2, 1.0 / self.n[None, :]))


def masks3(G, T):
    nt = int(round(0.25 * G))
    return {s: [np.sort(np.random.default_rng(1000 * s + i).permutation(G)[:nt]) for i in range(T)] for s in SEEDS}


def pcc_table(Y, P, mk, keys=("pearson", "p_ta_up", "p_ta_down")):
    """Per (seed, transition) metrics; returns array (S, T, len(keys)) (nan where P[i] is None)."""
    T = len(Y)
    out = np.full((len(SEEDS), T, len(keys)), np.nan)
    for a, s in enumerate(SEEDS):
        for i in range(T):
            if P[i] is None:
                continue
            t = mk[s][i]
            y, p = np.asarray(Y[i])[t], np.asarray(P[i])[t]
            y32, p32 = y.astype(np.float32), p.astype(np.float32)
            if np.all(y32 == y32[0]) or np.all(p32 == p32[0]):
                r = np.nan
            else:
                r = float(np.corrcoef(y32, p32)[0, 1])
            vals = {"pearson": r}
            if len(keys) > 1:
                k = min(200, len(y))
                for side, sg in (("up", 1), ("down", -1)):
                    ty, tp = RS._incl(sg * y32, k), RS._incl(sg * p32, k)
                    vals[f"p_ta_{side}"] = float((ty * tp).sum() / k)
            out[a, i] = [vals[k_] for k_ in keys]
    return out


def mean_from(tab, start):
    """Mean over transitions >= start (nanmean), then over seeds."""
    return np.nanmean(np.nanmean(tab[:, start:], axis=1), axis=0)


def nass_full_l1(e, cache=True):
    """Re-run NASS 'full' for l1 from nass.py pieces (no graph, no OT; 'full' uses neither).
    Returns (predictions list, dict with 'cls' and 'F_<feature>' arrays). Only the predictions are cached; the
    features are rebuilt each call (a few seconds)."""
    f = os.path.join(OUT, f"_nass_full_l1_{e.name}.npz")
    d = e.d
    Y = d.Y("l1")
    F, cls = N.build_features(d, "l1", Y, None, None)
    zd = {"cls": cls, **{f"F_{k}": F[k] for k in F}}
    zd["files"] = list(zd.keys())
    if cache and os.path.exists(f):
        return list(np.load(f)["P"]), _Z(zd)
    fb = N.fallback(d, "l1", Y, F)
    G_ = N.GROUPS
    full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"]
    st = N.Stacker(F, cls, Y, full)
    P, chosen = N.run_learned(st, [full], Y, d, fb)
    np.savez_compressed(f, P=np.stack(P))
    return P, _Z(zd)


class _Z(dict):
    """dict with an npz-like .files attribute."""

    @property
    def files(self):
        return [k for k in self.keys() if k != "files"]
