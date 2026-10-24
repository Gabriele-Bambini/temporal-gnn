"""Couplings between source and target minibatches.

A coupling is always computed WITHIN a compatible group (same condition, replicate and
biological time pair). Any permutation applied to the target is applied to its metadata
through the returned index arrays, so cells and labels can never drift apart.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class IncompatibleCouplingError(ValueError):
    """Raised when a source group has no compatible target group."""


def independent_coupling(n0: int, n1: int, rng: np.random.Generator, size: int | None = None):
    """Random pairs of indices (no optimal transport)."""
    n = min(n0, n1) if size is None else min(n0, n1, size)
    return rng.choice(n0, n, replace=False), rng.choice(n1, n, replace=False)


def ot_coupling(x0: np.ndarray, x1: np.ndarray, rng: np.random.Generator, max_size: int = 512):
    """Exact minibatch optimal transport (squared Euclidean cost) via the Hungarian assignment.

    Both sides are subsampled to n = min(len(x0), len(x1), max_size) so the problem is balanced.
    Returns index arrays (i0, i1) such that x0[i0[k]] is paired with x1[i1[k]].
    """
    n = min(len(x0), len(x1), max_size)
    if n < 1:
        raise IncompatibleCouplingError("cannot couple empty sets")
    i0 = rng.choice(len(x0), n, replace=False)
    i1 = rng.choice(len(x1), n, replace=False)
    a = np.asarray(x0[i0], dtype=np.float64)
    b = np.asarray(x1[i1], dtype=np.float64)
    cost = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    row, col = linear_sum_assignment(cost)
    return i0[row], i1[col]


def chain_ot(xs: list, rng: np.random.Generator, max_cells: int = 1000, mode: str = "ot") -> np.ndarray:
    """Chain assignments across an ordered list of snapshots x_0, ..., x_{m-1} so that row k of the result
    follows one pseudo-trajectory: every snapshot is subsampled to n = min(len(x_k), max_cells) cells and
    consecutive snapshots are matched by the exact OT (Hungarian, squared Euclidean cost) assignment, or,
    with mode="random", by a random permutation (the ablation that removes optimal transport).
    Returns an array of shape (m, n, genes)."""
    if mode not in ("ot", "random"):
        raise ValueError(f"unknown chain mode {mode!r} (expected 'ot' or 'random')")
    if len(xs) < 2:
        raise IncompatibleCouplingError("chaining needs at least two snapshots")
    n = min(min(len(x) for x in xs), max_cells)
    if n < 1:
        raise IncompatibleCouplingError("cannot chain empty snapshots")
    cur = np.asarray(xs[0], dtype=np.float64)[rng.choice(len(xs[0]), n, replace=False)]
    out = [cur]
    for x in xs[1:]:
        nxt = np.asarray(x, dtype=np.float64)[rng.choice(len(x), n, replace=False)]
        if mode == "ot":
            cost = (cur * cur).sum(1)[:, None] + (nxt * nxt).sum(1)[None, :] - 2.0 * cur @ nxt.T
            row, col = linear_sum_assignment(cost)
            order = np.empty(n, dtype=int)
            order[row] = col
        else:
            order = rng.permutation(n)
        nxt = nxt[order]
        out.append(nxt)
        cur = nxt
    return np.stack(out)


def group_keys(meta: dict, keys: tuple) -> np.ndarray:
    cols = [np.asarray(meta[k]).astype(str) for k in keys]
    return np.array(["|".join(t) for t in zip(*cols)])


def couple_within_groups(x0, meta0, x1, meta1, keys=("condition", "replicate"), method="ot",
                         rng: np.random.Generator | None = None, max_size: int = 512):
    """Couple source and target cells group by group. Returns (i0, i1) with equal group keys."""
    rng = np.random.default_rng() if rng is None else rng
    g0 = group_keys(meta0, keys)
    g1 = group_keys(meta1, keys)
    out0, out1 = [], []
    for g in np.unique(g0):
        idx0 = np.flatnonzero(g0 == g)
        idx1 = np.flatnonzero(g1 == g)
        if len(idx1) == 0:
            raise IncompatibleCouplingError(f"group {g!r} has source cells but no target cells")
        if method == "ot":
            a, b = ot_coupling(x0[idx0], x1[idx1], rng, max_size)
        elif method == "independent":
            a, b = independent_coupling(len(idx0), len(idx1), rng, size=max_size)
        else:
            raise ValueError(f"unknown coupling method {method!r}")
        out0.append(idx0[a])
        out1.append(idx1[b])
    i0 = np.concatenate(out0)
    i1 = np.concatenate(out1)
    if not (g0[i0] == g1[i1]).all():
        raise IncompatibleCouplingError("internal error: coupled cells across groups")
    return i0, i1
