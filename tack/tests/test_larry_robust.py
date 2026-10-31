"""Synthetic checks for tack/hpc/larry_robust.py: the spread of a kNN coupling is K; a very sharp soft kernel is the
1-nearest-neighbour coupling; the matched-spread interpolation is exact on grid points; with clones that sit together
in state space the OT plan is clonally enriched and its spread grows with eps."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc"))
import larry_robust as LR  # noqa: E402


def clones(seed=0, n_clone=40, per=8, dim=5):
    """Each clone is a small Gaussian blob; the target cells of a clone sit near its source cells, shifted."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4.0, size=(n_clone, dim))
    ca = np.repeat(np.arange(n_clone), per)
    cb = np.repeat(np.arange(n_clone), per)
    Za = centres[ca] + rng.normal(scale=0.5, size=(len(ca), dim))
    Zb = centres[cb] + 0.3 + rng.normal(scale=0.5, size=(len(cb), dim))
    return Za, Zb, ca, cb


def test_knn_spread_is_k():
    Za, Zb, _, _ = clones()
    for k in (1, 5, 20):
        assert abs(LR.spread(LR.knn_plan(Za, Zb, k)) - k) < 1e-9


def test_sharp_soft_is_nearest_neighbour():
    Za, Zb, ca, cb = clones(1)
    P1, Ps = LR.knn_plan(Za, Zb, 1), LR.soft_plan(Za, Zb, 1e-6)
    assert LR.spread(Ps) < 1.01
    assert abs(LR.enrichment(Ps, ca, cb)[0] - LR.enrichment(P1, ca, cb)[0]) < 1e-3


def test_at_spread_exact_on_grid():
    cur = [{"spread": s, "enrichment": e} for s, e in ((1.0, 5.0), (4.0, 3.0), (16.0, 2.0))]
    e, inside = LR.at_spread(cur, 4.0)
    assert abs(e - 3.0) < 1e-12 and inside
    assert not LR.at_spread(cur, 0.5)[1]


def test_ot_enriched_and_spread_grows():
    Za, Zb, ca, cb = clones(2)
    u = LR.run_unit(Za, Zb, ca, cb, ot_grid=(0.02, 0.05, 0.2))
    sp = [c["spread"] for c in u["curves"]["ot"]]
    assert sp[0] < sp[1] < sp[2]
    assert u["ot_frozen"]["enrichment"] > 5.0
