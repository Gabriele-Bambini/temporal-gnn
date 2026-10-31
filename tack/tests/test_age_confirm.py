"""Synthetic checks for tack/hpc/age_confirm.py: the age model recovers embryo age, the age mixture wins when pools are
asynchronous, and it does not win (no free lunch) when the stage label is the true age."""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpc"))
import age_confirm as AC  # noqa: E402

AC.SUB_PER_POOL, AC.NPERM, AC.REPS, AC.TRAIN_PER_POOL = 80, 4, 2, 200


def world(spread, seed=0, n_stage=7, n_pool=3, n_cell=200, dim=10):
    """Cells of pool p ~ N(age_p * u, I); age_p = stage + N(0, spread)."""
    rng = np.random.default_rng(seed)
    u = np.zeros(dim)
    u[0] = 1.5
    Z, pools, stages, true_age = [], [], [], {}
    for s in range(n_stage):
        for j in range(n_pool):
            p = f"{s}_{j}"
            a = s + rng.normal(0, spread)
            true_age[p] = a
            Z.append(a * u + rng.normal(size=(n_cell, dim)))
            pools += [p] * n_cell
            stages += [s] * n_cell
    return np.vstack(Z), np.array(pools), np.array(stages), true_age


def split(Z, pools, stages, k):
    r, h = stages != k, stages == k
    nominal = {p: int(stages[pools == p][0]) for p in sorted(set(pools[r]))}
    return Z[r], pools[r], nominal, Z[h], pools[h]


def test_age_model_recovers_true_age():
    Z, pools, stages, true_age = world(0.5)
    ages = AC.loo_ages(Z, pools, stages)
    ps = sorted(ages)
    assert spearmanr([ages[p] for p in ps], [true_age[p] for p in ps])[0] > 0.9


def test_age_mixture_wins_when_asynchronous():
    wins = 0
    for seed in range(3):
        Z, pools, stages, _ = world(0.6, seed=seed)
        Zr, pr, nominal, Zt, pt = split(Z, pools, stages, 3)
        base = AC.stage_mass(pr, nominal, {2: 0.5, 4: 0.5})
        r = AC.run_unit(Zr, pr, nominal, 3, base, Zt, pt, seed=seed)
        wins += r["age"]["ED_U"] < min(r["base"]["ED_U"], r["stagek"]["ED_U"], r["shufw"]["ED_U"])
    assert wins >= 2


def test_no_gain_when_label_is_true_age():
    Z, pools, stages, _ = world(0.0, seed=5)
    Zr, pr, nominal, Zt, pt = split(Z, pools, stages, 3)
    base = AC.stage_mass(pr, nominal, {2: 0.5, 4: 0.5})
    r = AC.run_unit(Zr, pr, nominal, 3, base, Zt, pt, seed=5)
    # with synchronous pools the age kernel cannot beat the nominal-stage kernel by much
    assert r["age"]["ED_U"] > 0.8 * r["stagek"]["ED_U"] - 0.01
