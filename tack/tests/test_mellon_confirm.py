"""Synthetic checks for tack/hpc/mellon_confirm.py: weights / weighted mix, the diffusion map, and that Mellon's
per-time densities are on one scale when time points have unequal cell counts (needed for the density ratio)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc"))
import mellon_confirm as MC  # noqa: E402


def test_constant_density_gives_the_mix():
    side_a = np.r_[np.ones(30, bool), np.zeros(50, bool)]
    lr = np.full(80, -2.0)
    w = MC.weights(lr, lr, lr, side_a)
    assert np.allclose(w, 1.0)
    rng = np.random.default_rng(0)
    ZA, ZB = rng.normal(size=(30, 2)), rng.normal(size=(50, 2))
    Z, X, v = MC.weighted_pred(ZA, ZA, ZB, ZB, w[side_a], w[~side_a])
    assert abs(v[:30].sum() - 0.5) < 1e-12 and abs(v[30:].sum() - 0.5) < 1e-12


def test_weights_follow_the_ratio_and_clip():
    side_a = np.r_[np.ones(30, bool), np.zeros(30, bool)]
    mid = np.zeros(60)
    mid[0], mid[30] = np.log(1e6), np.log(1e-6)
    w = MC.weights(np.zeros(60), mid, np.zeros(60), side_a)
    assert w[0] == MC.CLIP[1] and w[30] == MC.CLIP[0]
    lr0 = np.zeros(60)
    lr0[5] = np.log(2.0)
    w2 = MC.weights(np.zeros(60), lr0, np.zeros(60), side_a)
    assert w2[5] > w2[4]


def test_diffusion_map_shape():
    rng = np.random.default_rng(1)
    Z = np.vstack([rng.normal(size=(300, 5)), rng.normal(size=(300, 5)) + 4])
    E = MC.diffusion_map(Z, n_comp=5, knn=15, ka=5)
    assert E.shape == (600, 5) and np.isfinite(E).all()


def test_mellon_slices_on_one_scale():
    rng = np.random.default_rng(2)
    E = np.vstack([rng.normal(size=(400, 3)), rng.normal(size=(1200, 3)), rng.normal(size=(400, 3))])
    t = np.r_[np.zeros(400), np.ones(1200), 2 * np.ones(400)]      # same distribution, unequal counts
    lr, _ = MC.mellon_logdens(E, t, np.arange(0, len(E), 10), (0.0, 1.0, 2.0))
    assert np.median(np.abs(lr[0.0] - lr[1.0])) < 0.5 and np.median(np.abs(lr[2.0] - lr[1.0])) < 0.5
