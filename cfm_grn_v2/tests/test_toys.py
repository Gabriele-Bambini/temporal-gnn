import numpy as np
import pytest

from cfm_grn_v2.data.synthetic_toys import TOY_NAMES, make_toy


@pytest.mark.parametrize("name", TOY_NAMES)
def test_toys_have_two_times_and_replicates(name):
    s = make_toy(name, n_per_group=200, seed=1)
    assert s.timepoints().tolist() == [0.0, 1.0]
    assert s.replicates() == ["r1", "r2", "r3"] and s.n_genes == 2 and s.name == f"toy_{name}"


def test_translation_moves_the_mean():
    s = make_toy("translation", n_per_group=2000)
    m0 = s.x[s.mask(times=[0.0])].mean(0)
    m1 = s.x[s.mask(times=[1.0])].mean(0)
    assert np.allclose(m1 - m0, [3.0, -2.0], atol=0.15)


def test_condition_shift_has_opposite_shifts():
    s = make_toy("condition_shift", n_per_group=2000)
    assert s.conditions() == ["ctrl", "stim"]
    d = {c: s.x[s.mask(times=[1.0], conditions=[c])].mean(0)[0] - s.x[s.mask(times=[0.0], conditions=[c])].mean(0)[0]
         for c in s.conditions()}
    assert d["ctrl"] > 1.5 and d["stim"] < -1.5


def test_unknown_toy_raises():
    with pytest.raises(ValueError):
        make_toy("spiral")
