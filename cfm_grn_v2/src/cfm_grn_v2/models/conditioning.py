"""Conditioning context built from biological_time and condition labels.

The context never contains ``flow_time``: the generative clock enters the vector field
separately, so the two clocks cannot be confused inside the model.
"""
from __future__ import annotations

import torch
from torch import nn


class UnknownConditionError(KeyError):
    """Raised when a condition label was not present at training time."""


class ConditionRegistry:
    """Maps condition names to embedding indices; refuses unknown names loudly."""

    def __init__(self, names):
        self.names = sorted({str(n) for n in names})
        self._index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, names) -> torch.Tensor:
        idx = []
        for n in names:
            if str(n) not in self._index:
                raise UnknownConditionError(f"condition {n!r} not in registry {self.names}; "
                                            "a condition unseen in training cannot be represented")
            idx.append(self._index[str(n)])
        return torch.tensor(idx, dtype=torch.long)

    def to_dict(self) -> dict:
        return {"names": list(self.names)}

    @classmethod
    def from_dict(cls, d: dict) -> "ConditionRegistry":
        return cls(d["names"])


class ContextEncoder(nn.Module):
    """context = [embedding(condition), time features].

    time_conditioning:
      "delta"    -> [(t1 - t0)/scale]                      time-homogeneous dynamics; extrapolates to unseen times
      "absolute" -> [t0/scale, t1/scale, (t1 - t0)/scale]  non-autonomous; cannot extrapolate past training times
      "none"     -> []
    """

    MODES = ("delta", "absolute", "none")

    def __init__(self, registry: ConditionRegistry, cond_dim: int = 8, time_scale: float = 1.0,
                 time_conditioning: str = "delta"):
        super().__init__()
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if time_conditioning not in self.MODES:
            raise ValueError(f"time_conditioning must be one of {self.MODES}, got {time_conditioning!r}")
        self.registry = registry
        self.cond_dim = int(cond_dim)
        self.time_scale = float(time_scale)
        self.time_conditioning = time_conditioning
        self.cond_embed = nn.Embedding(len(registry), self.cond_dim)

    @property
    def n_time_features(self) -> int:
        return {"delta": 1, "absolute": 3, "none": 0}[self.time_conditioning]

    @property
    def out_dim(self) -> int:
        return self.cond_dim + self.n_time_features

    def forward(self, cond_idx: torch.Tensor, bio_t0: torch.Tensor, bio_t1: torch.Tensor) -> torch.Tensor:
        t0 = (bio_t0.to(torch.float32) / self.time_scale)[:, None]
        t1 = (bio_t1.to(torch.float32) / self.time_scale)[:, None]
        parts = [self.cond_embed(cond_idx)]
        if self.time_conditioning == "delta":
            parts.append(t1 - t0)
        elif self.time_conditioning == "absolute":
            parts += [t0, t1, t1 - t0]
        return torch.cat(parts, dim=-1)
