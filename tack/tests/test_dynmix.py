"""Synthetic checks for tack/hpc/dynmix.py: with a curved (logistic) change of composition the dynamic mixture beats the
50/50 union at the held timepoint; with no change it is not worse; the forecast follows a steady trend."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc"))
import dynmix as DM  # noqa: E402

DM.N_STATES = 6


def world(shares_fn, seed=0, n_tp=7, n_emb=6, n_cell=150, dim=5):
    """Three fixed Gaussian states; the share of each state at timepoint t is shares_fn(t)."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3, dim) * 6.0
    Z, t, unit = [], [], []
    for tp in range(n_tp):
        p = shares_fn(tp)
        for e in range(n_emb):
            lab = rng.choice(3, n_cell, p=p)
            Z.append(centres[lab] + rng.normal(size=(n_cell, dim)))
            t += [tp] * n_cell
            unit += [f"{tp}_{e}"] * n_cell
    return np.vstack(Z), np.array(t), np.array(unit)


def logistic(tp):
    a = 1 / (1 + np.exp(-2.5 * (tp - 3.6)))
    return np.array([0.85 * (1 - a) + 0.05, 0.85 * a + 0.05, 0.05])


def flat(tp):
    return np.array([0.5, 0.3, 0.2])


def split(Z, t, u, k):
    f = (np.abs(t - k) <= 2) & (t != k)
    h = t == k
    return Z[f], t[f], u[f], Z[h], u[h]


def test_beats_union_on_curved_composition():
    wins = 0
    for seed in range(3):
        Z, t, u = world(logistic, seed)
        r = DM.interp_unit(*split(Z, t, u, 3), 3)
        wins += r["dynmix"]["ED_U"] < r["union"]["ED_U"]
    assert wins >= 2


def test_not_worse_without_change():
    Z, t, u = world(flat, 4)
    r = DM.interp_unit(*split(Z, t, u, 3), 3)
    assert r["dynmix"]["ED_U"] < r["union"]["ED_U"] + 0.02


def test_forecast_follows_trend():
    lin = lambda tp: np.array([0.7 - 0.1 * tp, 0.2 + 0.1 * tp, 0.1])  # noqa: E731
    Z, t, u = world(lin, 5, n_tp=6)
    past, fut = t <= 3, t == 4
    r = DM.forecast_unit(Z[past], t[past], u[past], Z[fut], u[fut], 3)
    assert r["dynmix"]["ED_U"] < r["persist"]["ED_U"]
