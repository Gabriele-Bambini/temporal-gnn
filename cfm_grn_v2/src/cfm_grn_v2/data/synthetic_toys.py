"""Two-dimensional toy transitions with known answers (biological_time 0 -> 1).

translation      : N(0, I) -> N((3, -2), I)
variance         : N(0, I) -> N(0, diag(4, 0.25))
bimodal          : N(0, I) -> 0.5 N((3, 3), 0.25 I) + 0.5 N((-3, -3), 0.25 I)
condition_shift  : ctrl -> +2 on dim0, stim -> -2 on dim0 (independent cohorts per time)
Every replicate is an independent cohort, so replicate-level splits are honest.
"""
from __future__ import annotations

import numpy as np

from .schema import SnapshotSet

TOY_NAMES = ("translation", "variance", "bimodal", "condition_shift")


def _sample_t1(name: str, cond: str, rng: np.random.Generator, n: int) -> np.ndarray:
    if name == "translation":
        return rng.normal(0.0, 1.0, size=(n, 2)) + np.array([3.0, -2.0])
    if name == "variance":
        return rng.normal(0.0, 1.0, size=(n, 2)) * np.array([2.0, 0.5])
    if name == "bimodal":
        sign = rng.choice([-1.0, 1.0], size=(n, 1))
        return rng.normal(0.0, 0.5, size=(n, 2)) + sign * 3.0
    if name == "condition_shift":
        shift = np.array([2.0, 0.0]) if cond == "ctrl" else np.array([-2.0, 0.0])
        return rng.normal(0.0, 1.0, size=(n, 2)) + shift
    raise ValueError(f"unknown toy {name!r}; choose one of {TOY_NAMES}")


def make_toy(name: str, n_per_group: int = 1000, replicates=("r1", "r2", "r3"), seed: int = 0) -> SnapshotSet:
    if name not in TOY_NAMES:
        raise ValueError(f"unknown toy {name!r}; choose one of {TOY_NAMES}")
    conditions = ("ctrl", "stim") if name == "condition_shift" else ("ctrl",)
    rng = np.random.default_rng(seed)
    xs, ts, cs, rs = [], [], [], []
    for rep in replicates:
        for cond in conditions:
            x0 = rng.normal(0.0, 1.0, size=(n_per_group, 2))
            x1 = _sample_t1(name, cond, rng, n_per_group)
            xs += [x0, x1]
            ts += [np.zeros(n_per_group), np.ones(n_per_group)]
            cs += [np.array([cond] * n_per_group)] * 2
            rs += [np.array([rep] * n_per_group)] * 2
    return SnapshotSet(np.concatenate(xs), np.concatenate(ts), np.concatenate(cs), np.concatenate(rs),
                       ["dim0", "dim1"], name=f"toy_{name}")
