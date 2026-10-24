import numpy as np
import pytest

from cfm_grn_v2.flows.coupling import (IncompatibleCouplingError, chain_ot, couple_within_groups, independent_coupling,
                                       ot_coupling)


def test_chain_ot_recovers_identifiable_trajectories():
    rng = np.random.default_rng(0)
    base = 10.0 * np.arange(6)[:, None] * np.ones((6, 2))
    v = np.array([0.3, -0.2])
    xs = [rng.permutation(base + k * v) for k in range(4)]
    out = chain_ot(xs, rng, max_cells=100)
    assert out.shape == (4, 6, 2)
    for k in range(1, 4):
        assert np.allclose(out[k] - out[0], k * v)
    with pytest.raises(IncompatibleCouplingError):
        chain_ot([xs[0]], rng)
    rnd = chain_ot(xs, np.random.default_rng(0), max_cells=100, mode="random")
    assert rnd.shape == (4, 6, 2) and not all(np.allclose(rnd[k] - rnd[0], k * v) for k in range(1, 4))
    with pytest.raises(ValueError):
        chain_ot(xs, rng, mode="nearest")


def test_ot_coupling_recovers_translation_pairs():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(50, 2))
    x1 = x0 + 5.0
    i0, i1 = ot_coupling(x0, x1, rng, max_size=50)
    assert len(i0) == 50 and len(np.unique(i0)) == 50 and len(np.unique(i1)) == 50
    assert np.allclose(x0[i0] + 5.0, x1[i1])


def test_ot_cost_not_worse_than_independent():
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(200, 4))
    x1 = rng.normal(size=(200, 4)) + 1.0
    i0, i1 = ot_coupling(x0, x1, np.random.default_rng(2), max_size=200)
    j0, j1 = independent_coupling(200, 200, np.random.default_rng(3))
    assert ((x0[i0] - x1[i1]) ** 2).sum(1).mean() <= ((x0[j0] - x1[j1]) ** 2).sum(1).mean()


def test_independent_coupling_size():
    i0, i1 = independent_coupling(30, 50, np.random.default_rng(0), size=20)
    assert len(i0) == len(i1) == 20 and i0.max() < 30 and i1.max() < 50


def test_groups_never_mix():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(40, 3))
    x1 = rng.normal(size=(40, 3))
    meta0 = {"condition": ["a"] * 20 + ["b"] * 20, "replicate": ["r1"] * 40}
    meta1 = {"condition": ["b"] * 20 + ["a"] * 20, "replicate": ["r1"] * 40}
    i0, i1 = couple_within_groups(x0, meta0, x1, meta1, rng=rng)
    assert (np.asarray(meta0["condition"])[i0] == np.asarray(meta1["condition"])[i1]).all()
    assert len(i0) == 40


def test_missing_group_raises():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(10, 2))
    x1 = rng.normal(size=(10, 2))
    meta0 = {"condition": ["a"] * 10, "replicate": ["r1"] * 10}
    meta1 = {"condition": ["b"] * 10, "replicate": ["r1"] * 10}
    with pytest.raises(IncompatibleCouplingError):
        couple_within_groups(x0, meta0, x1, meta1, rng=rng)
