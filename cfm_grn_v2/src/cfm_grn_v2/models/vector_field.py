"""MLP vector field v(x, flow_time, context) -> dx/ds."""
from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_embedding(flow_time: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of flow_time in [0, 1] (scaled by 1000 like diffusion timesteps)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=flow_time.device, dtype=torch.float32) / max(half, 1))
    args = flow_time.to(torch.float32)[:, None] * 1000.0 * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLPVectorField(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden: int = 256, depth: int = 3, time_embed_dim: int = 32,
                 use_time: bool = True, time_embed: str = "sinusoidal"):
        super().__init__()
        if time_embed not in ("sinusoidal", "scalar"):
            raise ValueError(f"unknown time_embed {time_embed!r} (expected 'sinusoidal' or 'scalar')")
        self.dim, self.context_dim, self.time_embed_dim = int(dim), int(context_dim), int(time_embed_dim)
        self.use_time = bool(use_time)            # False: autonomous field v(x, context), flow_time ignored
        self.time_embed = time_embed              # "scalar": the time as one raw feature (biological time, v0.1.11)
        t_width = (self.time_embed_dim if time_embed == "sinusoidal" else 1) if self.use_time else 0
        layers, width = [], self.dim + t_width + self.context_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, self.dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, flow_time: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x]
        if self.use_time:
            parts.append(flow_time.to(torch.float32)[:, None] if self.time_embed == "scalar"
                         else sinusoidal_embedding(flow_time, self.time_embed_dim))
        if self.context_dim > 0:
            if context is None:
                raise ValueError("this vector field expects a context tensor (biological_time + condition)")
            parts.append(context)
        return self.net(torch.cat(parts, dim=-1))
