"""Versioned training configuration and hyper-parameter grids."""
from __future__ import annotations

import dataclasses
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 256
    depth: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    n_iters: int = 2000
    sigma_min: float = 1e-3
    coupling: str = "ot"              # "ot" | "independent"
    n_ode_steps: int = 20
    ode_method: str = "rk4"
    time_embed_dim: int = 32
    cond_dim: int = 8
    time_conditioning: str = "delta"      # "delta" | "absolute" | "none" (see ContextEncoder)
    eval_every: int = 200
    patience: int = 5
    seed: int = 0
    shuffle_biological_time: bool = False   # control: destroys temporal order in the training labels
    ema_decay: float = 0.999          # exponential moving average of the weights, used for validation and prediction; 0 = off
    grad_clip: float = 1.0            # max gradient norm per step; 0 = off
    pair_mode: str = "consecutive"    # "consecutive": (t_k, t_k+1) transitions | "all": every ordered pair (t_i, t_j), i < j
    clock: str = "flow"               # "flow": v(x, s, ctx) over s in [0, 1] | "duration": autonomous v(x, cond) over the biological horizon
    path: str = "linear"              # "linear": straight pair interpolants | "spline": cubic spline through OT-chained trajectories (needs clock="duration")
    chain: str = "ot"                 # spline path only: "ot" (chained optimal transport) | "random" (random pairing, ablation)
    spline_kind: str = "natural"      # spline boundary condition: "natural" | "not-a-knot" | "clamped"
    field_time: str = "none"          # duration clock: "none" = autonomous v(x, cond) | "bio" = v(x, t/time_scale, cond), t biological (v0.1.11)
    path_center: str = "spline"       # spline path centre: "spline" (cubic through the knot means) | "linear" (straight segments between them, v0.1.10)
    marginal_iters: int = 0           # v0.1.9 distributional fine-tune: marginal-matching steps after the path stage (0 = off)
    marginal_weight: float = 1.0      # weight of the two-sample divergence between the pushforward and the next snapshot
    marginal_prior: float = 1.0       # weight of the retained path loss during the fine-tune (0 = pure marginal matching)
    marginal_batch: int = 256         # cells per side of the divergence
    marginal_steps: int = 8           # ODE steps per unit of biological time inside the differentiable rollout
    marginal_metric: str = "energy"   # "energy" (U-statistic energy distance) | "sliced_w2"
    marginal_window: float = 0.0      # during the fine-tune, supervise the path only within this many time units of an observed time (0 = everywhere)
    marginal_lr: float = 3e-4
    marginal_eval_every: int = 25
    marginal_patience: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise TypeError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replace(self, **kw) -> "TrainConfig":
        return dataclasses.replace(self, **kw)


def expand_grid(grid: dict) -> list:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*[grid[k] for k in keys])]


def random_subset(configs: list, n: int, seed: int = 0) -> list:
    if n >= len(configs):
        return list(configs)
    idx = np.random.default_rng(seed).choice(len(configs), n, replace=False)
    return [configs[i] for i in sorted(idx)]
