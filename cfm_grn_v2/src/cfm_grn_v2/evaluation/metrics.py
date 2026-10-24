"""Distributional metrics computed under a protocol fixed before the test is looked at."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _mean_offdiag(d: np.ndarray) -> float:
    n = d.shape[0]
    if n < 2:
        raise ValueError("need at least two samples")
    return float((d.sum() - np.trace(d)) / (n * (n - 1)))


def energy_distance(a, b) -> float:
    """2 E|X - Y| - E|X - X'| - E|Y - Y'| with unbiased within-sample terms (U-statistic):
    expectation 0 for identical distributions at any sample size (can be slightly negative)."""
    a, b = _f64(a), _f64(b)
    return float(2.0 * cdist(a, b).mean() - _mean_offdiag(cdist(a, a)) - _mean_offdiag(cdist(b, b)))


def median_bandwidth(x, rng: np.random.Generator | None = None, n_max: int = 500) -> float:
    """Median pairwise distance heuristic (computed on TRAINING cells, then frozen)."""
    x = _f64(x)
    rng = np.random.default_rng(0) if rng is None else rng
    if len(x) > n_max:
        x = x[rng.choice(len(x), n_max, replace=False)]
    d = cdist(x, x)[np.triu_indices(len(x), 1)]
    return float(max(np.median(d), 1e-6))


def mmd_rbf(a, b, bandwidth: float) -> float:
    """RBF MMD^2 with unbiased within-sample terms."""
    a, b = _f64(a), _f64(b)
    g = 1.0 / (2.0 * bandwidth ** 2)
    kaa = np.exp(-g * cdist(a, a, "sqeuclidean"))
    kbb = np.exp(-g * cdist(b, b, "sqeuclidean"))
    kab = np.exp(-g * cdist(a, b, "sqeuclidean"))
    return float(_mean_offdiag(kaa) + _mean_offdiag(kbb) - 2.0 * kab.mean())


def w2_assignment(a, b, rng: np.random.Generator, n_max: int = 500) -> float:
    """Exact 2-Wasserstein between balanced subsamples (Hungarian assignment)."""
    a, b = _f64(a), _f64(b)
    n = min(len(a), len(b), n_max)
    ia = rng.choice(len(a), n, replace=False)
    ib = rng.choice(len(b), n, replace=False)
    cost = cdist(a[ia], b[ib], "sqeuclidean")
    r, c = linear_sum_assignment(cost)
    return float(np.sqrt(cost[r, c].mean()))


def mean_error(a, b) -> float:
    return float(np.linalg.norm(_f64(a).mean(0) - _f64(b).mean(0)))


def var_ratio(a, b) -> float:
    """Total variance of a over that of b (trace of the covariance): > 1 means a is spread wider than b."""
    return float(_f64(a).var(0, ddof=1).sum() / max(_f64(b).var(0, ddof=1).sum(), 1e-12))


def ed_mean_trace_split(a, b) -> dict:
    """High-dimensional approximation of the energy distance (E|X - Y| ~ sqrt(tr Sa + tr Sb + |ma - mb|^2),
    E|X - X'| ~ sqrt(2 tr Sa)) split into the part a matching spread would leave (trace part: means set equal)
    and the part added by the distance between the means. A diagnostic of what the score is made of; compare
    ed_approx with the measured energy distance before reading the split."""
    a, b = _f64(a), _f64(b)
    ta, tb = a.var(0, ddof=1).sum(), b.var(0, ddof=1).sum()
    m2 = float(((a.mean(0) - b.mean(0)) ** 2).sum())
    trace = 2.0 * np.sqrt(ta + tb) - np.sqrt(2.0 * ta) - np.sqrt(2.0 * tb)
    approx = 2.0 * np.sqrt(ta + tb + m2) - np.sqrt(2.0 * ta) - np.sqrt(2.0 * tb)
    return {"ed_approx": float(approx), "ed_mean_part": float(approx - trace), "ed_trace_part": float(trace)}


def per_gene_w1(a, b, rng: np.random.Generator, n_max: int = 2000) -> float:
    """Mean over genes of the 1-D Wasserstein distance (sorted-sample form)."""
    a, b = _f64(a), _f64(b)
    n = min(len(a), len(b), n_max)
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]
    return float(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean())


def permutation_null(pred, target, rng: np.random.Generator, n_perm: int = 50, q: float = 0.95) -> float:
    """q-quantile of the energy distance between two random splits of the pooled samples: the null
    distribution of ED under 'pred and target share a distribution', at the SAME sample sizes as
    the evaluation. A prediction with ED below this value is indistinguishable from the target."""
    pred, target = _f64(pred), _f64(target)
    pooled = np.concatenate([pred, target])
    n = len(pred)
    vals = [energy_distance(pooled[perm[:n]], pooled[perm[n:]])
            for perm in (rng.permutation(len(pooled)) for _ in range(n_perm))]
    return float(np.quantile(vals, q))


@dataclass
class EvalProtocol:
    n_samples: int = 1000
    seed: int = 0
    bandwidth: float = 1.0
    w2_max: int = 500
    n_perm: int = 50

    def subsample(self, x, rng: np.random.Generator) -> np.ndarray:
        x = np.asarray(x)
        if len(x) <= self.n_samples:
            return x
        return x[rng.choice(len(x), self.n_samples, replace=False)]

    def evaluate(self, pred, target, seed_offset: int = 0) -> dict:
        rng = np.random.default_rng(self.seed + seed_offset)
        p = self.subsample(pred, rng)
        t = self.subsample(target, rng)
        return {
            "energy_distance": energy_distance(p, t),
            "mmd": mmd_rbf(p, t, self.bandwidth),
            "w2": w2_assignment(p, t, rng, self.w2_max),
            "mean_error": mean_error(p, t),
            "per_gene_w1": per_gene_w1(p, t, rng),
            "null_q95": permutation_null(p, t, rng, self.n_perm),
            "var_ratio": var_ratio(p, t),                   # deterministic: added last, no random draw
            **ed_mean_trace_split(p, t),
        }


def noise_floor(target, protocol: EvalProtocol, n_rep: int = 5) -> dict:
    """Metrics between two disjoint halves of the target: below this, models are indistinguishable."""
    target = np.asarray(target)
    acc: dict = {}
    for r in range(n_rep):
        rng = np.random.default_rng(protocol.seed + 1000 + r)
        perm = rng.permutation(len(target))
        h = len(target) // 2
        m = protocol.evaluate(target[perm[:h]], target[perm[h:2 * h]], seed_offset=r)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}
