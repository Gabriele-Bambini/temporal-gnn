"""Synthetic checks for tack/hpc/wfr_confirm.py: the HK plan of two Diracs, the semi-coupling identity on a random
instance, the two limits (delta = 0 -> mix / persistence, large delta -> McCann / OT extrapolation) and the
mean-rank choice of delta."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc"))
import wfr_confirm as W  # noqa: E402


def test_two_diracs_plan_and_value():
    ZA, ZB = np.zeros((1, 2)), np.array([[1.0, 0.0]])
    dl = 1.0                                             # delta = 1 * sqrt(median d^2) = 1, theta = 1/2
    P, theta, _ = W.wfr_plan(ZA, ZB, dl, eps_rel=0.02)    # eps << 1 slows the KL-unbalanced Sinkhorn (k = 1/(1+eps))
    assert abs(P[0, 0] - np.cos(0.5)) < 1e-3              # gamma = sqrt(m0 m1) cos(theta), m0 = m1 = 1


def test_semicoupling_identity():
    rng = np.random.default_rng(0)
    ZA, ZB = rng.normal(size=(30, 3)), rng.normal(size=(25, 3)) + 0.5
    P, theta, _ = W.wfr_plan(ZA, ZB, 0.3, eps_rel=1e-3)
    mu0, mu1 = 1 / 30, 1 / 25
    r, c = P.sum(1), P.sum(0)
    a, b = P * (mu0 / r)[:, None], P * (mu1 / c)[None, :]
    lifted = (a + b - 2 * np.sqrt(a * b) * np.cos(np.minimum(theta, np.pi / 2))).sum()
    static = 2.0 - 2 * P.sum()                            # m0 + m1 - 2 sum(gamma) at the optimum
    assert abs(lifted - static) < 0.05 * abs(static)


def test_delta_zero_is_mix_and_persistence():
    rng = np.random.default_rng(1)
    ZA, ZB = rng.normal(size=(40, 2)), rng.normal(size=(60, 2)) + 3
    Z, X, w, tf = W.wfr_predict(ZA, ZA, ZB, ZB, 0.0, 0.5, 50, rng)
    assert tf == 0 and len(Z) == 100
    assert abs(w[:40].sum() - 0.5) < 1e-12 and abs(w[40:].sum() - 0.5) < 1e-12
    Z, X, w, tf = W.wfr_predict(ZA, ZA, ZB, ZB, 0.0, 2.0, 50, rng)
    assert np.allclose(Z, ZB) and np.allclose(w, 1 / 60)


def test_large_delta_is_ot_midpoint_and_extrapolation():
    rng = np.random.default_rng(2)
    ZA = rng.normal(size=(50, 2))
    ZB = ZA + np.array([2.0, 0.0])                        # pure translation: OT pairs i with i
    Z, X, w, tf = W.wfr_predict(ZA, ZA, ZB, ZB, 50.0, 0.5, 50, rng)
    assert tf > 0.99
    assert np.abs((w[:, None] * Z).sum(0) - (ZA.mean(0) + np.array([1.0, 0.0]))).max() < 0.05
    Z, X, w, tf = W.wfr_predict(ZA, ZA, ZB, ZB, 50.0, 2.0, 50, rng)
    assert np.abs((w[:, None] * Z).sum(0) - (ZA.mean(0) + np.array([4.0, 0.0]))).max() < 0.1


def test_far_mass_fades():
    rng = np.random.default_rng(3)
    ZA = np.vstack([rng.normal(size=(20, 2)), rng.normal(size=(20, 2)) + 40])   # a far source cluster
    ZB = rng.normal(size=(40, 2)) + 0.3
    Z, X, w, tf = W.wfr_predict(ZA, ZA, ZB, ZB, 0.1, 0.5, 40, rng)
    far = np.linalg.norm(Z - 40, axis=1) < 10
    assert far.any() and np.allclose(Z[far], ZA[20:][np.isin(ZA[20:, 0], Z[far, 0])])  # faded in place, not moved
    assert 0 < tf < 1


def test_mean_rank_choice_ties_to_smaller():
    files = [{d: 1.0 for d in W.DELTAS}, {d: 1.0 for d in W.DELTAS}]
    assert W.mean_rank_choice(files) == W.DELTAS[0]
    files = [{d: (0.5 if d == 0.2 else 1.0) for d in W.DELTAS}]
    assert W.mean_rank_choice(files) == 0.2
