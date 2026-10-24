"""Distributional (marginal-matching) objective (v0.1.9).

Conditional flow matching regresses the field on the velocity of a CONSTRUCTED path — a straight segment
between coupled cells, or a cubic spline through chained pseudo-trajectories. The learned field can therefore
never be better than that construction: if the path family's own marginal at an unobserved time is wrong, the
field faithfully reproduces the error (measured: spline marginal 0.636, trained model 0.641 on the synthetic
regulatory benchmark).

The objective here references no path. It asks the field to transport an observed snapshot onto the NEXT
observed snapshot and scores the result with a differentiable two-sample divergence, so the interior of the
trajectory is left to the field's own smoothness and to consistency with the other observed transitions.
Related in spirit to TrajectoryNet (Tong et al. 2020) and MIOFlow (Huguet et al. 2022), which match observed
marginals through a neural ODE; the implementation here is ours and is used as a fine-tuning stage on top of
the flow-matching pre-training, with the path loss retained as a regulariser.
"""
from __future__ import annotations

import torch

from .solver import integrate


def energy_distance_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable U-statistic energy distance 2 E|X - Y| - E|X - X'| - E|Y - Y'|.

    Expectation zero for identical distributions at any sample size (the same estimator as the evaluation
    metric, so the training signal and the reported number are the same quantity on different data)."""
    if a.dim() != 2 or b.dim() != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"expected two 2-D tensors with the same width, got {tuple(a.shape)} and {tuple(b.shape)}")
    n, m = a.shape[0], b.shape[0]
    if n < 2 or m < 2:
        raise ValueError("the U-statistic energy distance needs at least two samples on each side")
    ab = torch.cdist(a, b).mean()
    daa, dbb = torch.cdist(a, a), torch.cdist(b, b)
    aa = (daa.sum() - daa.diagonal().sum()) / (n * (n - 1))
    bb = (dbb.sum() - dbb.diagonal().sum()) / (m * (m - 1))
    return 2.0 * ab - aa - bb


def sliced_w2(a: torch.Tensor, b: torch.Tensor, n_projections: int = 64,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Sliced squared 2-Wasserstein distance: mean over random one-dimensional projections of the squared
    difference between the sorted projected samples (needs equal sample sizes)."""
    if a.shape != b.shape:
        raise ValueError(f"sliced_w2 needs equal shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    dirs = torch.randn(a.shape[1], int(n_projections), generator=generator, dtype=a.dtype, device=a.device)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)
    pa = torch.sort(a @ dirs, dim=0).values
    pb = torch.sort(b @ dirs, dim=0).values
    return ((pa - pb) ** 2).mean()


def divergence(pred: torch.Tensor, target: torch.Tensor, metric: str = "energy",
               generator: torch.Generator | None = None) -> torch.Tensor:
    if metric == "energy":
        return energy_distance_torch(pred, target)
    if metric == "sliced_w2":
        return sliced_w2(pred, target, generator=generator)
    raise ValueError(f"unknown marginal metric {metric!r} (expected 'energy' or 'sliced_w2')")


def marginal_loss(vector_field, x0: torch.Tensor, x1: torch.Tensor, context, horizon: float, n_steps: int = 10,
                  method: str = "rk4", metric: str = "energy",
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Divergence between the field's pushforward of x0 over `horizon` and the independently sampled x1.

    x0 and x1 are unpaired samples of the two snapshots: no coupling is computed and none is implied."""
    pred = integrate(vector_field, x0, context, int(n_steps), method, t_end=float(horizon))
    return divergence(pred, x1, metric, generator)
