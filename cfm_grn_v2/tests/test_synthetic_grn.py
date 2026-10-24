import numpy as np

from cfm_grn_v2.data.synthetic_grn import TimeVaryingGRNSimulator
from cfm_grn_v2.evaluation.metrics import energy_distance


def test_activity_bounds_and_schedules():
    sim = TimeVaryingGRNSimulator(n_genes=12, n_regulators=4, seed=0)
    a = np.stack([sim.activity(t) for t in np.linspace(0, sim.t_max, 11)])
    assert a.shape == (11, len(sim.edges)) and a.min() >= 0 and a.max() <= 1
    const = [e for e, s in enumerate(sim.schedules) if s.kind == "constant"]
    dyn = [e for e, s in enumerate(sim.schedules) if s.kind != "constant"]
    assert np.allclose(a[:, const], 1.0)
    assert len(dyn) > 0 and (np.abs(a[0, dyn] - a[-1, dyn]) > 0.8).all()


def test_snapshot_set_is_reproducible_finite_nonnegative():
    sim = TimeVaryingGRNSimulator(n_genes=12, n_regulators=4, seed=0)
    s1, truth = sim.snapshot_set([0.0, 2.0, 4.0], n_cells_per_cohort=50, replicates=("r1",), seed=3)
    s2, _ = sim.snapshot_set([0.0, 2.0, 4.0], n_cells_per_cohort=50, replicates=("r1",), seed=3)
    assert np.array_equal(s1.x, s2.x) and np.isfinite(s1.x).all() and (s1.x >= 0).all()
    assert truth["activity"].shape == (3, len(sim.edges)) and truth["W"].shape == (12, 12)
    assert s1.summary()["n_cells"].tolist() == [50, 50, 50]


def test_time_changes_distribution_more_than_noise():
    sim = TimeVaryingGRNSimulator(n_genes=20, n_regulators=6, seed=1)
    s, _ = sim.snapshot_set([0.0, 6.0], n_cells_per_cohort=300, replicates=("r1",), seed=0)
    x0 = s.x[s.mask(times=[0.0])]
    x1 = s.x[s.mask(times=[6.0])]
    assert energy_distance(x0, x1) > 5 * energy_distance(x1[:150], x1[150:])


def test_knockout_gene_stays_silent():
    sim = TimeVaryingGRNSimulator(n_genes=12, n_regulators=4, seed=0)
    s, _ = sim.snapshot_set([0.0, 3.0], n_cells_per_cohort=40, replicates=("r1",), knockouts=(0,), seed=0)
    assert s.conditions() == ["ctrl", "ko_g0"]
    assert np.all(s.x[s.mask(conditions=["ko_g0"])][:, 0] == 0)
