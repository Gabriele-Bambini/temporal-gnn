"""Splits by whole biological units: timepoints, replicates, perturbation targets. Never by random cells."""
from __future__ import annotations

import numpy as np


class SplitError(ValueError):
    """Raised when a split cannot be built honestly."""


def _sorted_unique_times(times) -> np.ndarray:
    return np.unique(np.asarray(list(times), dtype=np.float64))


def forecast_split(times, n_test: int = 1) -> dict:
    """Hold out the LAST n_test timepoints (extrapolation in biological_time)."""
    t = _sorted_unique_times(times)
    if len(t) < n_test + 2:
        raise SplitError(f"need at least {n_test + 2} timepoints for a forecast split, got {len(t)}")
    return {"train_times": t[:-n_test].tolist(), "test_times": t[-n_test:].tolist()}


def inner_validation_split(train_times) -> dict:
    """Inside the training timepoints, the last one is the validation target."""
    t = _sorted_unique_times(train_times)
    if len(t) < 2:
        raise SplitError("need at least two training timepoints to hold one out for validation")
    return {"fit_times": t[:-1].tolist(), "val_times": [float(t[-1])]}


def interpolation_folds(times) -> list:
    """Leave-one-inner-timepoint-out folds (never the first or the last timepoint)."""
    t = _sorted_unique_times(times)
    if len(t) < 3:
        raise SplitError("interpolation needs at least three timepoints")
    return [{"train_times": np.delete(t, k).tolist(), "test_times": [float(t[k])]} for k in range(1, len(t) - 1)]


def replicate_split(replicates, n_val: int = 1, n_test: int = 1, seed: int = 0) -> dict:
    reps = sorted({str(r) for r in replicates})
    if len(reps) < n_val + n_test + 1:
        raise SplitError(f"need at least {n_val + n_test + 1} replicates, got {len(reps)}")
    order = [str(r) for r in np.random.default_rng(seed).permutation(reps)]
    return {"test": order[:n_test], "val": order[n_test:n_test + n_val], "fit": order[n_test + n_val:]}


def grouped_folds(groups, n_folds: int, seed: int = 0) -> list:
    """K folds over whole groups (e.g. perturbation targets); every group is tested exactly once."""
    g = sorted({str(x) for x in groups})
    if n_folds < 2 or n_folds > len(g):
        raise SplitError(f"n_folds must be between 2 and {len(g)}")
    order = np.random.default_rng(seed).permutation(g)
    return [{"train_groups": sorted(set(g) - set(f.tolist())), "test_groups": sorted(f.tolist())}
            for f in np.array_split(order, n_folds)]


def assert_disjoint(a, b, what: str = "sets") -> None:
    inter = set(a) & set(b)
    if inter:
        raise SplitError(f"{what} overlap: {sorted(inter)[:5]}")
