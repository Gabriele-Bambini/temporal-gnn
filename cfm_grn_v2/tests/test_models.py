import inspect

import pytest
import torch

from cfm_grn_v2.models.conditioning import ConditionRegistry, ContextEncoder, UnknownConditionError
from cfm_grn_v2.models.vector_field import MLPVectorField, sinusoidal_embedding


def test_registry_roundtrip_and_unknown():
    reg = ConditionRegistry(["stim", "ctrl", "ctrl"])
    assert reg.names == ["ctrl", "stim"] and len(reg) == 2
    assert reg.encode(["stim", "ctrl"]).tolist() == [1, 0]
    with pytest.raises(UnknownConditionError):
        reg.encode(["never_seen"])
    assert ConditionRegistry.from_dict(reg.to_dict()).names == reg.names


def test_context_encoder_never_receives_flow_time():
    params = inspect.signature(ContextEncoder.forward).parameters
    assert "flow_time" not in params and "bio_t0" in params and "bio_t1" in params


def test_context_encoder_output():
    enc = ContextEncoder(ConditionRegistry(["a", "b"]), cond_dim=4, time_scale=10.0, time_conditioning="absolute")
    out = enc(torch.tensor([0, 1, 1]), torch.tensor([0.0, 5.0, 5.0]), torch.tensor([5.0, 10.0, 10.0]))
    assert out.shape == (3, enc.out_dim) == (3, 7)
    assert torch.allclose(out[1], out[2])
    assert torch.allclose(out[0, -3:], torch.tensor([0.0, 0.5, 0.5]))


def test_context_encoder_delta_mode_ignores_absolute_time():
    enc = ContextEncoder(ConditionRegistry(["a"]), cond_dim=2, time_scale=1.0, time_conditioning="delta")
    a = enc(torch.tensor([0]), torch.tensor([0.0]), torch.tensor([1.0]))
    b = enc(torch.tensor([0]), torch.tensor([5.0]), torch.tensor([6.0]))
    assert enc.out_dim == 3 and torch.allclose(a, b)
    with pytest.raises(ValueError):
        ContextEncoder(ConditionRegistry(["a"]), time_conditioning="sinusoidal")


def test_sinusoidal_embedding_shape():
    emb = sinusoidal_embedding(torch.tensor([0.0, 0.5, 1.0]), 16)
    assert emb.shape == (3, 16) and not torch.allclose(emb[0], emb[1])


def test_autonomous_vector_field_ignores_flow_time():
    torch.manual_seed(0)
    vf = MLPVectorField(dim=4, context_dim=2, hidden=8, depth=1, time_embed_dim=8, use_time=False)
    x, c = torch.randn(5, 4), torch.randn(5, 2)
    assert torch.allclose(vf(x, torch.zeros(5), c), vf(x, torch.ones(5), c))
    assert vf.net[0].in_features == 6


def test_vector_field_shapes_and_context_required():
    vf = MLPVectorField(dim=5, context_dim=3, hidden=16, depth=2, time_embed_dim=8)
    x = torch.randn(6, 5)
    out = vf(x, torch.rand(6), torch.randn(6, 3))
    assert out.shape == (6, 5)
    out.sum().backward()
    assert vf.net[0].weight.grad is not None
    with pytest.raises(ValueError):
        vf(x, torch.rand(6), None)


def test_vector_field_scalar_time_input():
    """time_embed="scalar": the field sees the time as one raw feature (smooth in biological time) instead of
    the high-frequency sinusoidal flow-time embedding."""
    torch.manual_seed(0)
    f = MLPVectorField(3, 0, hidden=16, depth=2, use_time=True, time_embed="scalar")
    assert f.net[0].in_features == 3 + 1
    x = torch.randn(5, 3)
    a, b = f(x, torch.zeros(5)), f(x, torch.ones(5))
    assert a.shape == (5, 3) and not torch.allclose(a, b)
    with pytest.raises(ValueError):
        MLPVectorField(3, 0, time_embed="fourier")
