import numpy as np
import pytest
import torch

from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.flows.marginal import divergence, energy_distance_torch, marginal_loss, sliced_w2
from cfm_grn_v2.models.vector_field import MLPVectorField
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import replicate_task, train_flow


def test_energy_distance_matches_the_numpy_estimator():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(60, 3)), rng.normal(size=(50, 3)) + 1.0
    got = float(energy_distance_torch(torch.as_tensor(a), torch.as_tensor(b)))
    assert got == pytest.approx(energy_distance(a, b), rel=1e-6, abs=1e-8)


def test_energy_distance_is_zero_for_two_halves_of_one_sample():
    x = torch.as_tensor(np.random.default_rng(1).normal(size=(400, 4)))
    assert abs(float(energy_distance_torch(x[:200], x[200:]))) < 0.05
    assert float(energy_distance_torch(x[:200], x[200:] + 2.0)) > 1.0


def test_energy_distance_rejects_degenerate_input():
    with pytest.raises(ValueError):
        energy_distance_torch(torch.zeros(1, 2), torch.zeros(5, 2))
    with pytest.raises(ValueError):
        energy_distance_torch(torch.zeros(5, 2), torch.zeros(5, 3))


def test_sliced_w2_zero_for_identical_and_positive_for_shifted():
    x = torch.as_tensor(np.random.default_rng(2).normal(size=(128, 5)))
    assert float(sliced_w2(x, x.clone())) == pytest.approx(0.0, abs=1e-9)
    assert float(sliced_w2(x, x + 1.0)) > 0.5
    with pytest.raises(ValueError):
        sliced_w2(x, x[:10])


def test_divergence_rejects_an_unknown_metric():
    x = torch.zeros(4, 2)
    with pytest.raises(ValueError):
        divergence(x, x, metric="wasserstein")


def test_marginal_loss_is_differentiable_and_learns_a_translation():
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    x0 = torch.as_tensor(rng.normal(size=(96, 2)), dtype=torch.float32)
    x1 = torch.as_tensor(rng.normal(size=(96, 2)) + np.array([2.0, -1.0]), dtype=torch.float32)
    field = MLPVectorField(2, 0, hidden=32, depth=2, use_time=False)
    opt = torch.optim.Adam(field.parameters(), lr=1e-2)
    first, loss = None, None
    for _ in range(80):
        loss = marginal_loss(field, x0, x1, None, horizon=1.0, n_steps=4)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in field.parameters())
        opt.step()
    assert torch.isfinite(loss) and loss.item() < 0.3 * first


def test_finetune_stage_runs_and_never_worsens_the_validation_score():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    proto = EvalProtocol(n_samples=150, seed=0)
    base = TrainConfig(hidden=32, depth=2, n_iters=200, eval_every=100, patience=3, clock="duration",
                       batch_size=64, seed=0)
    pre, _ = train_flow(s, fit, [0.0, 1.0], base, val, proto)
    tuned, hist = train_flow(s, fit, [0.0, 1.0], base.replace(marginal_iters=40, marginal_eval_every=10,
                                                             marginal_batch=64, marginal_steps=4), val, proto)
    assert any(r.get("stage") == "marginal" for r in hist)
    assert tuned.marginal_iter is not None
    assert tuned.best_val <= pre.best_val + 1e-9          # the stage keeps the pre-trained weights unless it wins
    assert np.isfinite(tuned.predict_standardized(np.zeros((5, 2), dtype=np.float32), "ctrl", 0.0, 1.0)).all()
