import pytest

from cfm_grn_v2.data.splits import (SplitError, assert_disjoint, forecast_split, grouped_folds,
                                    inner_validation_split, interpolation_folds, replicate_split)


def test_forecast_split_holds_out_terminal_time():
    s = forecast_split([3.0, 1.0, 2.0, 0.0])
    assert s["train_times"] == [0.0, 1.0, 2.0] and s["test_times"] == [3.0]
    inner = inner_validation_split(s["train_times"])
    assert inner["fit_times"] == [0.0, 1.0] and inner["val_times"] == [2.0]


def test_forecast_split_needs_three_times():
    with pytest.raises(SplitError):
        forecast_split([0.0, 1.0])


def test_interpolation_folds_only_inner_times():
    folds = interpolation_folds([0, 1, 2, 3])
    assert [f["test_times"] for f in folds] == [[1.0], [2.0]]
    assert folds[0]["train_times"] == [0.0, 2.0, 3.0]


def test_replicate_split_is_disjoint_and_complete():
    s = replicate_split(["r1", "r2", "r3", "r3"], seed=0)
    assert sorted(s["fit"] + s["val"] + s["test"]) == ["r1", "r2", "r3"]
    assert len(s["test"]) == 1 and len(s["val"]) == 1


def test_grouped_folds_cover_every_group_once():
    folds = grouped_folds(["a", "b", "c", "d", "e"], n_folds=2, seed=1)
    tested = sum((f["test_groups"] for f in folds), [])
    assert sorted(tested) == ["a", "b", "c", "d", "e"]
    for f in folds:
        assert not set(f["train_groups"]) & set(f["test_groups"])


def test_assert_disjoint():
    assert_disjoint([0.0, 1.0], [2.0])
    with pytest.raises(SplitError):
        assert_disjoint([0.0, 1.0], [1.0])
