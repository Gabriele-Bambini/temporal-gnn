import pytest
import torch

import numpy as np

from cfm_grn_v2.flows.path import SplinePath, cfm_loss, interpolate, sample_flow_time, spline_loss, velocity_target


def test_path_endpoints_without_noise():
    x0 = torch.zeros(4, 3)
    x1 = torch.ones(4, 3)
    assert torch.allclose(interpolate(x0, x1, torch.zeros(4), sigma_min=0.0), x0)
    assert torch.allclose(interpolate(x0, x1, torch.ones(4), sigma_min=0.0), x1)


def test_velocity_target_is_displacement():
    x0 = torch.randn(5, 2)
    x1 = torch.randn(5, 2)
    assert torch.allclose(velocity_target(x0, x1), x1 - x0)


def test_flow_time_in_unit_interval():
    s = sample_flow_time(1000, generator=torch.Generator().manual_seed(0))
    assert s.shape == (1000,) and float(s.min()) >= 0.0 and float(s.max()) < 1.0


def test_cfm_loss_zero_for_exact_field():
    x0 = torch.randn(8, 2)
    shift = torch.tensor([1.0, -2.0])
    x1 = x0 + shift
    field = lambda x, s, c: shift.expand_as(x)
    loss = cfm_loss(field, x0, x1, context=None, flow_time=torch.rand(8), sigma_min=0.0)
    assert float(loss) < 1e-12


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        interpolate(torch.zeros(2, 3), torch.zeros(3, 3), torch.zeros(2))


def test_spline_path_hits_knots_and_recovers_constant_velocity():
    times = np.array([0.0, 1.0, 2.0, 3.5])
    v = np.array([[1.0, -2.0], [0.5, 0.5], [-1.0, 3.0]])                  # three trajectories, constant velocity
    traj = np.stack([k * v + np.arange(3)[:, None] for k in times])       # (4, 3, 2)
    path = SplinePath(times, traj)
    x, vel = path.evaluate(times, [0, 1, 2, 0])
    assert np.allclose(x, traj[np.arange(4), [0, 1, 2, 0]])
    assert np.allclose(vel, v[[0, 1, 2, 0]], atol=1e-8)
    xs, vs, us = path.sample(np.random.default_rng(0), 64)
    assert xs.shape == (64, 2) and vs.shape == (64, 2) and us.min() >= 0.0 and us.max() <= 3.5
    two = SplinePath([0.0, 2.0], traj[[0, 2]])                             # two knots: linear segment
    assert np.allclose(two.evaluate([1.0], [1])[1], v[1])
    with pytest.raises(ValueError):
        SplinePath([0.0, 0.0], traj[:2])
    field = lambda x, s, c: torch.as_tensor(v[1], dtype=x.dtype).expand_as(x)
    xt, vt = torch.as_tensor(xs[:8], dtype=torch.float32), torch.as_tensor(np.tile(v[1], (8, 1)), dtype=torch.float32)
    assert float(spline_loss(field, xt, vt, None, sigma_min=0.0)) < 1e-12


def test_cfm_loss_target_scale_turns_displacement_into_velocity():
    x0 = torch.randn(8, 2)
    shift = torch.tensor([2.0, -4.0])
    x1 = x0 + shift
    field = lambda x, s, c: (shift / 2.0).expand_as(x)          # velocity per unit time for a horizon of 2
    assert float(cfm_loss(field, x0, x1, None, torch.rand(8), sigma_min=0.0, target_scale=0.5)) < 1e-12
    assert float(cfm_loss(field, x0, x1, None, torch.rand(8), sigma_min=0.0)) > 1.0


def test_spline_path_linear_centre():
    """center="linear": the path mean follows straight segments between the knot means, the deviation of each
    trajectory from the mean is the spline's, the path still passes through every observed cell at the knots,
    and the velocity is the derivative of the state."""
    import numpy as np
    from cfm_grn_v2.flows.path import SplinePath
    rng = np.random.default_rng(0)
    t = np.array([0.0, 1.0, 3.0, 4.0])
    traj = rng.normal(size=(4, 50, 3)) + np.array([0.0, 1.0, 4.0, 4.5])[:, None, None]
    sp_ = SplinePath(t, traj, center="spline")
    li = SplinePath(t, traj, center="linear")
    idx = np.arange(50)
    for k, tk in enumerate(t):
        x, _ = li.evaluate(np.full(50, tk), idx)
        assert np.allclose(x, traj[k], atol=1e-9)                                  # through the observed cells
    u = np.full(50, 2.0)
    xl, _ = li.evaluate(u, idx)
    xs, _ = sp_.evaluate(u, idx)
    m = traj.mean(1)
    assert np.allclose(xl.mean(0), m[1] + 0.5 * (m[2] - m[1]), atol=1e-9)        # straight centre
    assert np.allclose(xl - xl.mean(0), xs - xs.mean(0), atol=1e-9)              # spline shape
    eps = 1e-6
    for uu in (0.4, 2.2, 3.7):
        x0, v0 = li.evaluate(np.full(50, uu), idx)
        x1, _ = li.evaluate(np.full(50, uu + eps), idx)
        assert np.allclose((x1 - x0) / eps, v0, atol=1e-4)
    with np.testing.assert_raises(ValueError):
        SplinePath(t, traj, center="quadratic")

