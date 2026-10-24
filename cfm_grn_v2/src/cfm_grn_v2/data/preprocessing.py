"""Preprocessing fitted on training cells only and serialised with the model."""
from __future__ import annotations

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when transform is called before fit."""


class Standardizer:
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean_ = None
        self.std_ = None

    def fit(self, x) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("Standardizer.fit needs a 2-D array with at least two rows")
        self.mean_ = x.mean(0)
        self.std_ = np.maximum(x.std(0), self.eps)
        return self

    def _check(self) -> None:
        if self.mean_ is None:
            raise NotFittedError("Standardizer.fit must run on training cells before transform")

    def transform(self, x) -> np.ndarray:
        self._check()
        return ((np.asarray(x, dtype=np.float64) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, z) -> np.ndarray:
        self._check()
        return (np.asarray(z, dtype=np.float64) * self.std_ + self.mean_).astype(np.float32)

    def to_dict(self) -> dict:
        self._check()
        return {"eps": self.eps, "mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        st = cls(d["eps"])
        st.mean_ = np.asarray(d["mean"], dtype=np.float64)
        st.std_ = np.asarray(d["std"], dtype=np.float64)
        return st


def fit_on_train(x, train_mask) -> Standardizer:
    """Fit on the training cells only; refuses a mask that hides nothing."""
    train_mask = np.asarray(train_mask, dtype=bool)
    x = np.asarray(x)
    if train_mask.shape != (len(x),):
        raise ValueError("train_mask must have one boolean entry per cell")
    if train_mask.all():
        raise ValueError("train_mask covers every cell: nothing is held out")
    if train_mask.sum() < 2:
        raise ValueError("train_mask selects fewer than two cells")
    return Standardizer().fit(x[train_mask])
