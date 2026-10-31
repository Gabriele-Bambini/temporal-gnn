"""Synthetic checks for tack/hpc/larry_vivo_confirm.py: the decision rule (wins + geometric mean, NaN = loss) and one
unit end to end on clones that sit together in state space (OT converged at the frozen eps, clonally enriched)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc"))
import larry_vivo_confirm as VC  # noqa: E402
from test_larry_robust import clones  # noqa: E402


def test_rule_passes_and_fails():
    r = VC.rule([2.0, 2.0, 2.0, 0.5])
    assert r["wins"] == 3 and r["need"] == 3 and abs(r["gmean"] - 4 ** 0.25) < 1e-12 and r["pass"]
    assert not VC.rule([2.0, 0.4, 0.4, 2.0])["pass"]


def test_rule_nan_is_a_loss():
    r = VC.rule([2.0, 2.0, float("nan")])
    assert r["wins"] == 2 and r["need"] == 2 and r["pass"]
    assert not VC.rule([2.0, float("nan"), float("nan")])["pass"]


def test_unit_end_to_end():
    Za, Zb, ca, cb = clones(3)
    u = VC.unit(Za, Zb, ca, cb, eps_ot=(0.02, 0.05))
    assert u["ot0"]["marg_err"] <= VC.TOL
    assert u["ot0"]["enrichment"] > 5.0
    assert u["best_ot"] is not None and np.isfinite(u["ratio_best"])
    assert set(u["matched"]) == {"knn", "soft"} and u["matched"]["knn"]["ratio"] > 0
