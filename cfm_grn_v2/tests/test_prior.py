import numpy as np
import pytest

from cfm_grn_v2.data.prior import GRNPrior


def _prior():
    rng = np.random.default_rng(0)
    W = np.zeros((12, 12))
    for j in range(4):
        for i in rng.choice(np.arange(4, 12), 4, replace=False):
            W[j, i] = rng.choice([-1.5, 2.0])
    return GRNPrior.from_weights(W, [f"g{i}" for i in range(12)]), W


def test_prior_from_weights_and_edges():
    p, W = _prior()
    assert p.n_genes == 12 and p.n_edges == 16 and p.regulators.tolist() == [0, 1, 2, 3]
    e = p.edges
    assert np.all(W[e[:, 0], e[:, 1]] != 0) and np.all(p.signs[e[:, 0], e[:, 1]] == np.sign(W[e[:, 0], e[:, 1]]))
    assert GRNPrior.from_dict(p.to_dict()).n_edges == 16
    with pytest.raises(ValueError):
        GRNPrior(np.eye(3, dtype=bool))
    with pytest.raises(ValueError):
        GRNPrior(np.zeros((3, 3), dtype=bool))


def test_randomized_prior_preserves_degrees_and_signs_but_not_edges():
    p, _ = _prior()
    q = p.randomized(np.random.default_rng(1))
    assert q.n_edges == p.n_edges
    assert np.array_equal(q.support.sum(1), p.support.sum(1)) and np.array_equal(q.support.sum(0), p.support.sum(0))
    assert sorted(q.signs[q.support].tolist()) == sorted(p.signs[p.support].tolist())
    assert (q.support != p.support).sum() > 0 and not q.support.diagonal().any()
