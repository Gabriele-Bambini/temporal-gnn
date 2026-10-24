"""Linear probability path and the conditional flow matching objective; cubic-spline paths (v0.1.6).

``flow_time`` (s in [0, 1]) parametrises the generative path between a source sample x0
and a target sample x1. It is a mathematical construction, not biological time: the
latter only reaches the model through the conditioning ``context``.
"""
from __future__ import annotations

import numpy as np
import torch


def sample_flow_time(batch_size: int, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    """Uniform flow_time in [0, 1)."""
    return torch.rand(batch_size, generator=generator, device=device)


def interpolate(x0: torch.Tensor, x1: torch.Tensor, flow_time: torch.Tensor, sigma_min: float = 1e-3,
                noise: torch.Tensor | None = None) -> torch.Tensor:
    """x_s = (1 - s) x0 + s x1 + sigma_min * eps."""
    if x0.shape != x1.shape:
        raise ValueError(f"x0 {tuple(x0.shape)} and x1 {tuple(x1.shape)} must have the same shape")
    s = flow_time.view(-1, *([1] * (x0.dim() - 1))).to(x0.dtype)
    if noise is None:
        noise = torch.randn_like(x0)
    return (1.0 - s) * x0 + s * x1 + sigma_min * noise


def velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Conditional velocity of the linear path (constant along the segment)."""
    return x1 - x0


def cfm_loss(vector_field, x0: torch.Tensor, x1: torch.Tensor, context, flow_time: torch.Tensor,
             sigma_min: float = 1e-3, target_scale: float = 1.0) -> torch.Tensor:
    """Mean squared error between v(x_s, s, context) and the conditional velocity target_scale * (x1 - x0)
    (target_scale = 1/dt turns the displacement into a velocity per unit of biological time)."""
    x_s = interpolate(x0, x1, flow_time, sigma_min)
    return ((vector_field(x_s, flow_time, context) - float(target_scale) * velocity_target(x0, x1)) ** 2).mean()


class SplinePath:
    """Cubic spline (natural boundary) through chained pseudo-trajectories: times (m,), trajectories (m, n, D).

    Sampling a biological time u ~ U[t_0, t_{m-1}] gives a state on the curve and the curve's biological
    velocity, the regression target of an autonomous field v(x, condition). Curved paths let the field
    learn the curvature between observed snapshots, which straight (McCann) interpolants cannot express.
    """

    def __init__(self, times, trajectories, bc_type: str = "natural", center: str = "spline"):
        from scipy.interpolate import CubicSpline
        if bc_type not in ("natural", "not-a-knot", "clamped"):
            raise ValueError(f"unknown spline boundary condition {bc_type!r}")
        if center not in ("spline", "linear"):
            raise ValueError(f"unknown path centre {center!r} (expected 'spline' or 'linear')")
        t = np.asarray(times, dtype=np.float64)
        y = np.asarray(trajectories, dtype=np.float64)
        if t.ndim != 1 or y.ndim != 3 or len(t) != y.shape[0]:
            raise ValueError("times must have shape (m,) and trajectories (m, n, D) with the same m")
        if len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly increasing with at least two entries")
        self.times, self.n, self.dim = t, int(y.shape[1]), int(y.shape[2])
        self.coef = CubicSpline(t, y, axis=0, bc_type=bc_type).c        # (4, m-1, n, D): c0 dt^3 + ... + c3
        self.center = center
        if center == "linear":
            # The mean over trajectories of a spline through chained cells is the spline through the knot means
            # (the spline is linear in its knots). "linear" swaps that curve for straight segments between the
            # knot means and keeps each trajectory's deviation from it: the spline supplies the shape, the data
            # the centre. The two centres agree at the knots, so the path still passes through observed cells.
            means = y.mean(1)                                            # (m, D)
            self.mean_coef = CubicSpline(t, means, axis=0, bc_type=bc_type).c
            self.knot_means = means

    def evaluate(self, u, idx):
        """State and velocity of trajectories idx at times u (both arrays of shape (b,))."""
        u, idx = np.asarray(u, dtype=np.float64), np.asarray(idx, dtype=int)
        j = np.clip(np.searchsorted(self.times, u, side="right") - 1, 0, len(self.times) - 2)
        dt = (u - self.times[j])[:, None]
        c = self.coef[:, j, idx, :]                                    # (4, b, D)
        x = ((c[0] * dt + c[1]) * dt + c[2]) * dt + c[3]
        v = (3.0 * c[0] * dt + 2.0 * c[1]) * dt + c[2]
        if self.center == "linear":
            cm = self.mean_coef[:, j, :]                               # (4, b, D)
            xm = ((cm[0] * dt + cm[1]) * dt + cm[2]) * dt + cm[3]
            vm = (3.0 * cm[0] * dt + 2.0 * cm[1]) * dt + cm[2]
            h = (self.times[j + 1] - self.times[j])[:, None]
            a, b = self.knot_means[j], self.knot_means[j + 1]
            x = x - xm + a + (dt / h) * (b - a)
            v = v - vm + (b - a) / h
        return x, v

    def sample(self, rng: np.random.Generator, batch_size: int, window: float = 0.0):
        """(x, v, u) for batch_size random (trajectory, biological time) pairs.

        ``window`` > 0 restricts u to a neighbourhood of the observed times: the interpolant is anchored by
        data at its knots and is a guess in the middle of a long gap, so a windowed sample supervises the field
        only where the path is supported and leaves the interior of long gaps to the field's own smoothness.
        It is meant to be combined with the marginal-matching stage, which constrains the endpoints of those
        gaps from the data (see flows/marginal.py)."""
        idx = rng.integers(0, self.n, size=batch_size)
        if window and window > 0:
            k = rng.integers(0, len(self.times), size=batch_size)
            lo = np.maximum(self.times[k] - float(window), self.times[0])
            hi = np.minimum(self.times[k] + float(window), self.times[-1])
            u = rng.uniform(lo, hi)
        else:
            u = rng.uniform(self.times[0], self.times[-1], size=batch_size)
        x, v = self.evaluate(u, idx)
        return x, v, u


def spline_loss(vector_field, x: torch.Tensor, v: torch.Tensor, context, sigma_min: float = 1e-3,
                time: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between the field at a noisy state on the curve and the curve's biological velocity. ``time`` is the
    (scaled) biological time of each state for a non-autonomous field; None for the autonomous one."""
    x_n = x + sigma_min * torch.randn_like(x)
    t = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device) if time is None else time
    return ((vector_field(x_n, t, context) - v) ** 2).mean()
