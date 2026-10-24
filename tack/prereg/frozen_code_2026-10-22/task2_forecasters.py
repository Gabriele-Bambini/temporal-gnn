"""Forecasters for Dileo & Sottoriva Task 2, run inside their original loop by dileo_harness.py.

Interface: predict(hist, ts, ctx) -> np.ndarray of shape (N,) with the predicted target
    clip(log(m_next + 1e-8) - log(m_now + 1e-8), -5, 5)
hist = raw node features of the snapshots <= t (list of (N, 5) arrays: mean, median, std, frac_nonzero, sum),
ts = their timestamps, ctx = {"train_idx", "y_train", "edge_index", "edge_attr"} of the current transition.
Strict forecasters (strict = True) never read ctx; the same-information tier may read train_idx / y_train,
exactly the targets their models are fitted on. No forecaster may use torch's global RNG: the gene masks of
their loop are drawn from it.
"""
import numpy as np

EPS = np.float32(1e-8)
CLIP = 5.0


def log_delta(m_now, m_next):
    d = np.log(np.maximum(m_next, 0).astype(np.float32) + EPS) - np.log(np.maximum(m_now, 0).astype(np.float32) + EPS)
    return np.clip(d, -CLIP, CLIP)


def n_cells(x):
    """Cells in a bin = sum / mean (identical for every expressed gene)."""
    ok = x[:, 0] > 0
    return float(np.median(x[ok, 4] / x[ok, 0])) if ok.any() else 0.0


class Level:
    """Next mean = cell-weighted average of the last k snapshot means."""
    strict = True

    def __init__(self, k):
        self.k = k
        self.name = f"level{k}"

    def predict(self, hist, ts, ctx):
        if len(hist) < 2:
            return np.zeros(hist[-1].shape[0], np.float32)
        h = hist[-self.k:]
        w = np.array([n_cells(x) for x in h])
        m = sum(wi * x[:, 0] for wi, x in zip(w, h)) / w.sum()
        return log_delta(hist[-1][:, 0], m)


class ObsLevel:
    """Hidden population = cell-weighted mixture of the last k bins (zero-inflated per gene); the next bin is a
    finite sample of n cells (n = cells of the current bin); prediction = expected clipped target (Monte Carlo)."""
    strict = True

    def __init__(self, k, draws=256, seed=0):
        self.k, self.draws, self.seed = k, draws, seed
        self.name = f"obslevel{k}"

    def predict(self, hist, ts, ctx):
        if len(hist) < 2:
            return np.zeros(hist[-1].shape[0], np.float32)
        h = hist[-self.k:]
        w = np.array([n_cells(x) for x in h])
        w = w / w.sum()
        f = sum(wi * x[:, 3] for wi, x in zip(w, h))
        m = sum(wi * x[:, 0] for wi, x in zip(w, h))
        ex2 = sum(wi * (x[:, 2] ** 2 + x[:, 0] ** 2) for wi, x in zip(w, h))
        return expected_target(hist[-1][:, 0], f, m, ex2, int(round(n_cells(hist[-1]))), self.draws,
                               np.random.default_rng(self.seed + len(hist)))


def expected_target(m_now, f, m, ex2, n, draws, rng):
    """E[clip(log(mean of n cells + eps) - log(m_now + eps))] for a zero-inflated population with nonzero
    fraction f, mean m and second moment ex2 (positive part by moment matching, sum of positives by CLT)."""
    f = np.clip(f, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(f > 0, m / f, 0.0)
        var = np.where(f > 0, np.maximum(ex2 / np.where(f > 0, f, 1) - mu ** 2, 0), 0.0)
    K = rng.binomial(n, f[:, None], size=(len(f), draws))
    tot = np.maximum(K * mu[:, None] + np.sqrt(K * var[:, None]) * rng.standard_normal((len(f), draws)), 0)
    return log_delta(m_now[:, None], tot / max(n, 1)).mean(1).astype(np.float32)


REGISTRY = {
    "level4": lambda: Level(4), "level6": lambda: Level(6),
    "obslevel4": lambda: ObsLevel(4), "obslevel6": lambda: ObsLevel(6),
}

try:  # the OT / Schroedinger-bridge models live in their own module
    import dist_bridge
    REGISTRY.update(dist_bridge.REGISTRY)
except ImportError:
    pass
