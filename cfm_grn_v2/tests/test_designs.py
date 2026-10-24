import numpy as np
import pytest

from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.evaluation.metrics import EvalProtocol, median_bandwidth
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.designs import CONTROLS, SHORT, grid_gap, interpolation_controls, multi_removal_design
from cfm_grn_v2.training.search import interpolation_design, run_interpolation_benchmark
from cfm_grn_v2.training.trainer import task_groups, time_task

VERES = [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0]                 # day 4 (the test stage) removed
PIJUAN = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]           # E8.0 = quarter day 6 removed


def _drift_set(times=(0.0, 1.0, 2.0, 3.0), n=120, seed=0, reps=("r1",)):
    rng = np.random.default_rng(seed)
    v = np.array([1.0, -0.5])
    xs, ts, rs = [], [], []
    for r in reps:
        for t in times:
            xs.append(rng.normal(size=(n, 2)) * 0.1 + t * v)
            ts.append(np.full(n, t))
            rs.append(np.full(n, r))
    x = np.concatenate(xs).astype(np.float32)
    return SnapshotSet(x, np.concatenate(ts), np.array(["ctrl"] * len(x)), np.concatenate(rs), ["a", "b"])


@pytest.mark.parametrize("grid", [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], VERES, PIJUAN,
                                  [0.0, 1.0, 2.0, 4.0, 5.0, 7.0]])
def test_single_removal_agrees_with_interpolation_design(grid):
    for t in grid[1:-1]:
        ref = interpolation_design(grid, t)
        d = multi_removal_design(grid, [t])
        (p,) = d["predictions"]
        assert d["obs_times"] == ref["obs_times"] and d["fit_times"] == ref["fit_times"]
        assert d["val_time"] == ref["val_time"] and d["v_prev"] == ref["v_prev"]
        assert d["source"] == ref["t_prev"] and d["t_next"] == ref["t_next"]
        assert p["target"] == ref["test_time"] and p["source"] == ref["t_prev"] and p["kind"] == "near"
        assert p["horizon"] == ref["horizon"] and p["frac"] == pytest.approx(ref["frac"])


def test_multi_removal_designs_of_the_preregistered_folds():
    v12 = multi_removal_design(VERES, [1.0, 2.0], full_times=range(8))
    assert v12["source"] == 0.0 and v12["t_next"] == 3.0 and v12["boundary_source"]
    assert v12["obs_times"] == [0.0, 3.0, 5.0, 6.0, 7.0] and v12["val_time"] == 6.0 and v12["v_prev"] == 5.0
    assert v12["fit_times"] == [0.0, 3.0, 5.0, 7.0] and v12["g"] == 1.0
    near, far = v12["predictions"]
    assert (near["target"], near["kind"], near["h_rel"]) == (1.0, "near", 1.0) and near["frac"] == pytest.approx(1 / 3)
    assert (far["target"], far["kind"], far["h_rel"]) == (2.0, "far", 2.0) and far["frac"] == pytest.approx(2 / 3)
    v23 = multi_removal_design(VERES, [2.0, 3.0], full_times=range(8))
    assert v23["source"] == 1.0 and v23["t_next"] == 5.0 and not v23["boundary_source"]
    assert v23["val_time"] == 6.0 and v23["fit_times"] == [0.0, 1.0, 5.0, 7.0]
    assert v23["predictions"][0]["frac"] == pytest.approx(0.25) and v23["predictions"][1]["h_rel"] == 2.0
    v56 = multi_removal_design(VERES, [5.0, 6.0], full_times=range(8))
    assert v56["source"] == 3.0 and v56["t_next"] == 7.0 and v56["val_time"] == 1.0 and v56["v_prev"] == 0.0
    assert [p["h_rel"] for p in v56["predictions"]] == [2.0, 3.0]
    assert v56["span"] == 7.0 and [p["h_span"] for p in v56["predictions"]] == pytest.approx([2 / 7, 3 / 7])
    v5 = multi_removal_design(VERES, [5.0], full_times=range(8))
    assert v5["predictions"][0]["h_rel"] == 2.0 and not v5["predictions"][0]["h_rel"] < SHORT
    p234 = multi_removal_design(PIJUAN, [2.0, 3.0, 4.0], full_times=range(9), targets=[2.0, 4.0])
    assert p234["source"] == 1.0 and p234["t_next"] == 5.0 and p234["centre"] == 3.0
    assert p234["val_time"] == 7.0 and p234["v_prev"] == 5.0 and p234["fit_times"] == [0.0, 1.0, 5.0, 8.0]
    assert [(p["target"], p["kind"], p["h_rel"]) for p in p234["predictions"]] == [(2.0, "near", 1.0), (4.0, "far", 3.0)]
    assert p234["predictions"][0]["frac"] == pytest.approx(0.25)
    assert p234["span"] == 8.0 and [p["h_span"] for p in p234["predictions"]] == pytest.approx([1 / 8, 3 / 8])
    p34 = multi_removal_design(PIJUAN, [3.0, 4.0], full_times=range(9))
    assert p34["val_time"] == 7.0 and p34["predictions"][0]["frac"] == pytest.approx(1 / 3)
    p5 = multi_removal_design(PIJUAN, [5.0], full_times=range(9))
    assert p5["source"] == 4.0 and p5["t_next"] == 7.0 and p5["predictions"][0]["frac"] == pytest.approx(1 / 3)
    p7 = multi_removal_design(PIJUAN, [7.0], full_times=range(9))
    assert p7["source"] == 5.0 and p7["predictions"][0]["h_rel"] == 2.0 and p7["val_time"] == 1.0


def test_validation_ties_go_to_the_later_time_and_h_rel_uses_the_full_grid():
    d = multi_removal_design([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3.0])
    assert d["val_time"] == 5.0                              # 1 and 5 are both 2 away from 3: the later one
    d = multi_removal_design([0.0, 2.0, 4.0, 6.0, 8.0], [4.0], full_times=[0.0, 1.0, 2.0, 4.0, 6.0, 8.0])
    assert d["g"] == 2.0 and d["predictions"][0]["h_rel"] == 1.0
    d = multi_removal_design([0.0, 2.0, 4.0, 6.0, 8.0], [4.0], full_times=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    assert d["g"] == 1.0 and d["predictions"][0]["h_rel"] == 2.0
    assert d["span"] == 8.0 and d["predictions"][0]["h_span"] == 0.25
    d = multi_removal_design([0.0, 2.0, 4.0, 6.0, 8.0], [4.0])
    assert d["span"] == 8.0 and d["predictions"][0]["h_span"] == 0.25
    assert grid_gap([3.0, 0.0, 1.0, 1.0, 2.0, 7.0]) == 1.0


def test_invalid_designs_raise():
    with pytest.raises(ValueError):
        multi_removal_design(VERES, [0.0])                   # the first time cannot be removed
    with pytest.raises(ValueError):
        multi_removal_design(VERES, [7.0])                   # nor the last
    with pytest.raises(ValueError):
        multi_removal_design(VERES, [4.0])                   # not on the grid
    with pytest.raises(ValueError):
        multi_removal_design(VERES, [1.0, 3.0])              # an observed time inside the removed block
    with pytest.raises(ValueError):
        multi_removal_design(VERES, [2.0, 3.0], targets=[5.0])
    with pytest.raises(ValueError):
        multi_removal_design([0.0, 1.0, 2.0], [1.0])         # no inner observed time for validation
    with pytest.raises(ValueError):
        multi_removal_design(VERES, [])
    with pytest.raises(ValueError):
        grid_gap([1.0])


def test_controls_equal_run_interpolation_benchmark_for_a_single_removal():
    s = _drift_set()
    proto = EvalProtocol(n_samples=100, n_perm=5)
    mlp = {"hidden": 16, "n_iters": 50}
    cfg = TrainConfig(hidden=16, depth=1, n_iters=40, batch_size=64, eval_every=20, patience=2, clock="duration")
    ref = run_interpolation_benchmark(s, [0.0, 1.0, 2.0, 3.0], 2.0, cfg, seeds=[0, 1], protocol=proto,
                                      dataset_name="drift", shuffled_control=False, mlp_kwargs=mlp)
    d = multi_removal_design([0.0, 1.0, 2.0, 3.0], [2.0])
    all_mask = s.mask(times=d["obs_times"])
    x_eval = Standardizer().fit(s.x[all_mask]).transform(s.x)
    ev = EvalProtocol(n_samples=100, n_perm=5, bandwidth=median_bandwidth(x_eval[all_mask]))
    (p,) = d["predictions"]
    groups = task_groups(s, time_task(s, p["source"], p["target"]))
    for seed in (0, 1):
        out = interpolation_controls(s, x_eval, all_mask, d["obs_times"], p["source"], p["target"], p["t_next"], groups,
                                     seed, mlp_kwargs=mlp)
        assert [name for name, *_ in out] == list(CONTROLS)
        for name, cond, rep, pred in out:
            m1 = groups[0][3]
            ed = ev.evaluate(pred, x_eval[m1], seed_offset=seed)["energy_distance"]
            r = ref[(ref.model == name) & (ref.seed == seed)]
            assert len(r) == 1 and ed == pytest.approx(float(r.energy_distance.iloc[0]), abs=1e-6)


def test_controls_of_a_multi_removal_prediction_use_the_bracket():
    """Constant drift, times 0..5, {2, 3} removed: the far target 3 is predicted from 1 at frac 2/3 of the bracket
    1 -> 4, so mean and McCann interpolation land on it; the source is interior, so the velocity replay uses 0 -> 1."""
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    s = _drift_set(times, reps=("r1", "r2"))
    d = multi_removal_design(times, [2.0, 3.0])
    all_mask = s.mask(times=d["obs_times"])
    x_eval = Standardizer().fit(s.x[all_mask]).transform(s.x)
    far = d["predictions"][1]
    assert far["target"] == 3.0 and far["frac"] == pytest.approx(2 / 3)
    groups = task_groups(s, time_task(s, far["source"], far["target"]))
    assert len(groups) == 2
    out = interpolation_controls(s, x_eval, all_mask, d["obs_times"], far["source"], far["target"], far["t_next"], groups,
                                 0, mlp_kwargs={"hidden": 16, "n_iters": 50})
    assert len(out) == 2 * len(CONTROLS)
    for name, cond, rep, pred in out:
        target = x_eval[[m1 for c, r, _, m1 in groups if r == rep][0]]
        err = np.linalg.norm(pred.mean(0) - target.mean(0))
        if name in ("mean_interp", "ot_interp", "linear", "freefm_replay"):
            assert err < 0.2, name
        if name == "persistence":
            assert err > 1.0
