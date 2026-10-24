import math

import pytest
import torch

from cfm_grn_v2.flows.solver import integrate


def test_constant_field_translates():
    x0 = torch.zeros(3, 2)
    c = torch.tensor([1.0, -1.0])
    x1 = integrate(lambda x, s, ctx: c.expand_as(x), x0, None, n_steps=10, method="euler")
    assert torch.allclose(x1, c.expand_as(x0), atol=1e-6)


@pytest.mark.parametrize("method,tol", [("euler", 0.2), ("midpoint", 1e-2), ("rk4", 1e-4)])
def test_linear_field_accuracy(method, tol):
    x0 = torch.ones(4, 3)
    x1 = integrate(lambda x, s, ctx: x, x0, None, n_steps=20, method=method)
    assert float((x1 - math.e * x0).abs().max()) < tol


def test_trajectory_shape_and_endpoints():
    x0 = torch.randn(5, 2)
    x1, traj = integrate(lambda x, s, ctx: torch.zeros_like(x), x0, None, n_steps=7, return_trajectory=True)
    assert traj.shape == (8, 5, 2)
    assert torch.allclose(traj[0], x0) and torch.allclose(traj[-1], x1)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        integrate(lambda x, s, ctx: x, torch.zeros(1, 1), None, method="leapfrog")


def test_t_end_scales_the_horizon():
    x0 = torch.zeros(3, 2)
    c = torch.tensor([1.0, -1.0])
    x1 = integrate(lambda x, s, ctx: c.expand_as(x), x0, None, n_steps=10, method="rk4", t_end=2.5)
    assert torch.allclose(x1, 2.5 * c.expand_as(x0), atol=1e-6)
    with pytest.raises(ValueError):
        integrate(lambda x, s, ctx: x, torch.zeros(1, 1), None, t_end=0.0)
