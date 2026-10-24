"""Controls every flow-matching result must be reported against.

persistence     : predict x1 = x0 (no change)
train_mean      : degenerate distribution at the mean of the last observed snapshot
mean_shift      : x0 + (mean of last observed target - mean of its source)
LinearMap       : ridge regression of the OT displacement on the source state
MLPMap          : deterministic MLP regression of the OT displacement (no generative noise)
freefm_replay   : replay of observed transitions (nearest training source's displacement), the
                  "trajectory replay" critique of flow matching turned into a baseline
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from torch import nn

from ..flows.coupling import ot_coupling


def persistence(x0) -> np.ndarray:
    return np.asarray(x0).copy()


def train_mean(x_train_t1, n: int) -> np.ndarray:
    return np.repeat(np.asarray(x_train_t1).mean(0, keepdims=True), n, axis=0)


def mean_shift(x0, x_train_t0, x_train_t1) -> np.ndarray:
    return np.asarray(x0) + (np.asarray(x_train_t1).mean(0) - np.asarray(x_train_t0).mean(0))


def ot_pairs(x0, x1, rng: np.random.Generator, max_size: int = 1000):
    """Balanced OT pairs used to fit deterministic controls (training data only)."""
    x0, x1 = np.asarray(x0), np.asarray(x1)
    i0, i1 = ot_coupling(x0, x1, rng, max_size)
    return x0[i0], x1[i1]


class LinearMap:
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, x0p, x1p) -> "LinearMap":
        self.model.fit(np.asarray(x0p), np.asarray(x1p) - np.asarray(x0p))
        return self

    def predict(self, x0) -> np.ndarray:
        x0 = np.asarray(x0)
        return x0 + self.model.predict(x0)


class MLPMap:
    def __init__(self, hidden: int = 128, n_iters: int = 2000, lr: float = 1e-3, seed: int = 0):
        self.hidden, self.n_iters, self.lr, self.seed = int(hidden), int(n_iters), float(lr), int(seed)
        self.net = None

    def fit(self, x0p, x1p) -> "MLPMap":
        torch.manual_seed(self.seed)
        x = torch.as_tensor(np.asarray(x0p), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(x1p) - np.asarray(x0p), dtype=torch.float32)
        d = x.shape[1]
        self.net = nn.Sequential(nn.Linear(d, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                                 nn.Linear(self.hidden, d))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(self.seed)
        b = min(256, len(x))
        for _ in range(self.n_iters):
            idx = torch.randint(0, len(x), (b,), generator=gen)
            loss = ((self.net(x[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def predict(self, x0) -> np.ndarray:
        x0 = np.asarray(x0)
        with torch.no_grad():
            return x0 + self.net(torch.as_tensor(x0, dtype=torch.float32)).numpy()


def freefm_replay(x0, x0p, x1p, k: int = 1) -> np.ndarray:
    """Move every cell by the mean displacement of its k nearest TRAINING sources."""
    x0p, x1p = np.asarray(x0p), np.asarray(x1p)
    _, idx = NearestNeighbors(n_neighbors=k).fit(x0p).kneighbors(np.asarray(x0))
    return np.asarray(x0) + (x1p - x0p)[idx].mean(1)
