import numpy as np

from cfm_grn_v2.evaluation.controls import (LinearMap, MLPMap, freefm_replay, mean_shift, ot_pairs, persistence,
                                            train_mean)


def _shift_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, 3))
    x1 = rng.normal(size=(n, 3)) + np.array([2.0, 0.0, -1.0])
    return x0, x1


def test_persistence_and_train_mean():
    x0, x1 = _shift_data()
    assert np.array_equal(persistence(x0), x0)
    m = train_mean(x1, 5)
    assert m.shape == (5, 3) and np.allclose(m[0], x1.mean(0))


def test_mean_shift_moves_by_training_displacement():
    x0, x1 = _shift_data()
    assert np.allclose(mean_shift(x0, x0, x1).mean(0), x1.mean(0), atol=1e-6)


def test_linear_and_mlp_maps_learn_a_shift():
    x0, x1 = _shift_data()
    p0, p1 = ot_pairs(x0, x1, np.random.default_rng(0), max_size=400)
    lin = LinearMap(alpha=1.0).fit(p0, p1)
    mlp = MLPMap(hidden=32, n_iters=1500, lr=1e-2, seed=0).fit(p0, p1)
    xt, _ = _shift_data(seed=5)
    for pred in (lin.predict(xt), mlp.predict(xt)):
        assert np.allclose(pred.mean(0) - xt.mean(0), [2.0, 0.0, -1.0], atol=0.4)


def test_freefm_replay_reuses_observed_displacements():
    x0, x1 = _shift_data()
    p0, p1 = ot_pairs(x0, x1, np.random.default_rng(0), max_size=400)
    assert np.allclose(freefm_replay(p0, p0, p1, k=1), p1)
