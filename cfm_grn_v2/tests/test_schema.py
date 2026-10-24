import numpy as np
import pytest

from cfm_grn_v2.data.schema import SchemaError, SnapshotSet


def _snaps():
    x = np.random.default_rng(0).normal(size=(6, 2)).astype(np.float32)
    return SnapshotSet(x, [0, 0, 0, 1, 1, 1], ["a", "a", "b", "a", "b", "b"], ["r1"] * 6, ["g1", "g2"])


def test_valid_snapshot_set():
    s = _snaps()
    assert s.n_cells == 6 and s.n_genes == 2
    assert s.timepoints().tolist() == [0.0, 1.0] and s.conditions() == ["a", "b"] and s.replicates() == ["r1"]


def test_mask_and_subset():
    s = _snaps()
    m = s.mask(times=[1.0], conditions=["b"])
    assert m.sum() == 2
    sub = s.subset(m)
    assert sub.n_cells == 2 and (sub.condition == "b").all()


def test_summary_counts():
    df = _snaps().summary()
    assert df["n_cells"].sum() == 6 and set(df.columns) == {"biological_time", "condition", "replicate", "n_cells"}


def test_length_mismatch_raises():
    with pytest.raises(SchemaError):
        SnapshotSet(np.zeros((3, 2)), [0, 1], ["a"] * 3, ["r"] * 3, ["g1", "g2"])


def test_nan_raises():
    x = np.zeros((4, 2))
    x[0, 0] = np.nan
    with pytest.raises(SchemaError):
        SnapshotSet(x, [0, 0, 1, 1], ["a"] * 4, ["r"] * 4, ["g1", "g2"])


def test_single_timepoint_raises():
    with pytest.raises(SchemaError):
        SnapshotSet(np.zeros((4, 2)), [0, 0, 0, 0], ["a"] * 4, ["r"] * 4, ["g1", "g2"])


def test_duplicate_gene_names_raise():
    with pytest.raises(SchemaError):
        SnapshotSet(np.zeros((4, 2)), [0, 0, 1, 1], ["a"] * 4, ["r"] * 4, ["g", "g"])
