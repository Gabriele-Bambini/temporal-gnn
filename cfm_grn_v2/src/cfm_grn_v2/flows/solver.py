"""Fixed-step ODE integration of dx/ds = v(x, s, context) over s in [0, t_end] (t_end = 1 for the flow clock,
the biological horizon for the duration clock)."""
from __future__ import annotations

import torch


def integrate(vector_field, x0: torch.Tensor, context, n_steps: int = 20, method: str = "rk4",
              return_trajectory: bool = False, t_end: float = 1.0):
    if method not in ("euler", "midpoint", "rk4"):
        raise ValueError(f"unknown ODE method {method!r}")
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    x = x0
    grid = torch.linspace(0.0, float(t_end), n_steps + 1, device=x0.device, dtype=x0.dtype)
    traj = [x0]
    for k in range(n_steps):
        s0, h = grid[k], grid[k + 1] - grid[k]
        t0 = s0.expand(x.shape[0])
        if method == "euler":
            x = x + h * vector_field(x, t0, context)
        elif method == "midpoint":
            k1 = vector_field(x, t0, context)
            x = x + h * vector_field(x + 0.5 * h * k1, t0 + 0.5 * h, context)
        else:
            k1 = vector_field(x, t0, context)
            k2 = vector_field(x + 0.5 * h * k1, t0 + 0.5 * h, context)
            k3 = vector_field(x + 0.5 * h * k2, t0 + 0.5 * h, context)
            k4 = vector_field(x + h * k3, t0 + h, context)
            x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if return_trajectory:
            traj.append(x)
    if return_trajectory:
        return x, torch.stack(traj)
    return x
