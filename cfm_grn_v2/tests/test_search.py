import numpy as np
import pandas as pd
import pytest

from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.evaluation.metrics import EvalProtocol
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import (interpolation_design, interpolation_val_folds, nested_grid_search,
                                        run_forecast_benchmark, run_interpolation_benchmark, select_best)
from cfm_grn_v2.training.trainer import replicate_task


def _drift_set(n=120, seed=0):
    """Four timepoints of a constant-velocity drift: the McCann interpolant recovers the inner one exactly."""
    rng = np.random.default_rng(seed)
    v = np.array([1.0, -0.5])
    xs, ts = [], []
    for t in (0.0, 1.0, 2.0, 3.0):
        xs.append(rng.normal(size=(n, 2)) * 0.1 + t * v)
        ts.append(np.full(n, t))
    x = np.concatenate(xs).astype(np.float32)
    return SnapshotSet(x, np.concatenate(ts), np.array(["ctrl"] * (4 * n)), np.array(["r1"] * (4 * n)), ["a", "b"])


def test_val_folds_and_select_best_over_folds():
    assert interpolation_val_folds([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3.0) == [1.0, 5.0]
    assert interpolation_val_folds([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2.0) == [1.0, 5.0]
    assert interpolation_val_folds([0.0, 1.0, 2.0, 3.0], 2.0, n_folds=2) == [1.0]
    table = pd.DataFrame([{"config_id": 0, "hidden": 16, "seed": 0, "val_energy_distance": 0.1, "val_time": 1.0},
                          {"config_id": 0, "hidden": 16, "seed": 0, "val_energy_distance": 0.9, "val_time": 5.0},
                          {"config_id": 1, "hidden": 24, "seed": 0, "val_energy_distance": 0.4, "val_time": 1.0},
                          {"config_id": 1, "hidden": 24, "seed": 0, "val_energy_distance": 0.4, "val_time": 5.0}])
    best = select_best(table, TrainConfig())
    assert best.hidden == 24 and isinstance(best.hidden, int)


def test_interpolation_design_and_benchmark_rows():
    d = interpolation_design([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3.0)
    assert d["val_time"] == 5.0 and d["fit_times"] == [0.0, 1.0, 2.0, 4.0, 6.0] and d["v_prev"] == 4.0
    assert d["t_prev"] == 2.0 and d["t_next"] == 4.0 and d["horizon"] == 1.0 and d["frac"] == 0.5
    with pytest.raises(ValueError):
        interpolation_design([0.0, 1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        interpolation_design([0.0, 1.0, 2.0], 1.0)            # no second inner timepoint for validation
    s = _drift_set()
    cfg = TrainConfig(hidden=16, depth=1, n_iters=40, batch_size=64, eval_every=20, patience=2, clock="duration")
    df = run_interpolation_benchmark(s, [0.0, 1.0, 2.0, 3.0], 2.0, cfg, seeds=[0],
                                     protocol=EvalProtocol(n_samples=100, n_perm=5), dataset_name="drift",
                                     mlp_kwargs={"hidden": 16, "n_iters": 50})
    models = set(df["model"])
    for m in ("flow_matching", "flow_matching_shuffled_time", "persistence", "mean_interp", "ot_interp", "linear",
              "mlp", "freefm_replay", "noise_floor"):
        assert m in models
    assert "train_mean" not in models and df["energy_distance"].notna().all()
    ed = df.groupby("model")["energy_distance"].mean()
    assert ed["ot_interp"] < ed["persistence"] and ed["mean_interp"] < ed["persistence"]
    assert ed["freefm_replay"] < ed["persistence"]


def test_nested_search_and_benchmark_rows():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    test = replicate_task(s, "r3", 0.0, 1.0)
    base = TrainConfig(hidden=16, depth=1, n_iters=40, batch_size=64, eval_every=20, patience=2)
    proto = EvalProtocol(n_samples=150, seed=0)
    best, table = nested_grid_search(s, fit, [0.0, 1.0], val, base, {"hidden": [16, 24]}, seeds=[0], protocol=proto)
    assert best.hidden in (16, 24) and len(table) == 2 and "val_energy_distance" in table
    _, explicit = nested_grid_search(s, fit, [0.0, 1.0], val, base, [{"hidden": 16}], seeds=[0], protocol=proto)
    assert len(explicit) == 1
    df = run_forecast_benchmark(s, fit, [0.0, 1.0], val, test, best, seeds=[0, 1], protocol=proto,
                                dataset_name="toy", mlp_kwargs={"hidden": 16, "n_iters": 50})
    models = set(df["model"])
    for m in ("flow_matching", "flow_matching_shuffled_time", "persistence", "train_mean", "mean_shift", "linear",
              "mlp", "freefm_replay", "noise_floor"):
        assert m in models
    assert set(df["seed"]) == {0, 1} and df["energy_distance"].notna().all()
    assert (df[df["model"] == "flow_matching"]["best_iter"] >= 20).all()
