"""Synthetic checks for tack/hpc/larry_cyto_prep.py (unit rule: consecutive within-condition transitions, day-2 source
shared) and tack/hpc/larry_cyto_confirm.py (verdicts over a unit dict, split by transition type)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc"))
import larry_cyto_confirm as CC  # noqa: E402
import larry_cyto_prep as CP  # noqa: E402
import larry_vivo_confirm as VC  # noqa: E402
from test_larry_robust import clones  # noqa: E402


def test_transitions_follow_the_design():
    day = np.array([2, 2, 4, 6, 4, 6, 4, 4, 6])
    cond = np.array(["full cocktail", "full cocktail", "Epo", "Epo", "SCF", "Epo", "full cocktail", "M-csf", "M-csf"])
    ok = np.ones(len(day), bool)
    ok[7] = False                                    # unlabelled day-4 M-csf cell: no M-csf day-4 group
    tr = {n: (list(a), list(b)) for n, a, b in CP.transitions(day, cond, ok)}
    assert set(tr) == {"Epo_24", "Epo_46", "SCF_24", "full_24"}
    assert tr["Epo_24"] == ([0, 1], [2]) and tr["Epo_46"] == ([2], [3, 5]) and tr["full_24"] == ([0, 1], [6])


def test_verdicts_on_units():
    Za, Zb, ca, cb = clones(3)
    u = VC.unit(Za, Zb, ca, cb, eps_ot=(0.02, 0.05))
    v = CC.verdicts({"a_24": u, "b_46": u})
    assert v["H1_ot_vs_knn_matched"]["of"] == 2 and v["desc_ot0_max_marg_err"] <= VC.TOL
    assert set(v) == {"H1_ot_vs_knn_matched", "H2_ot_vs_soft_matched", "H3_best_ot_vs_best_knn", "desc_ot_vs_knn10",
                      "desc_all_inside_grid", "desc_ot0_max_marg_err"}
