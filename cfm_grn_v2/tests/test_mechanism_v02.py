import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_dev"))
import mechanism_v02 as mv  # noqa: E402

ARMS = {"base": 1.0, "noft": 1.6, "gated": 1.0, "transfer": 1.0, "time": 1.0, "time_noft": 1.5, "spline_noft": 1.1}


def _write_dataset(root: Path, ds: str, long_folds_field) -> Path:
    """Fake Job-A output of one data set: long fold 3.0, short fold 1.0, seeds 0-2, plus the v0.1.11 reference."""
    out = root / f"mech_{ds}"
    out.mkdir()
    rows, ref = [], []
    for fold in (1.0, 3.0):
        for seed in range(3):
            for arm, ed in ARMS.items():
                rows.append({"dataset": ds, "fold": fold, "seed": seed, "arm": arm, "variant": "main",
                             "source": "t_prev", "steps": 20, "energy_distance": ed + 0.01 * seed})
                if arm in ("base", "noft", "time") and seed < 2:
                    ref.append({"fold": fold, "seed": seed, "arm": arm, "energy_distance": ed + 0.01 * seed})
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    ref_path = root / f"{ds}_ref.csv"
    pd.DataFrame(ref).to_csv(ref_path, index=False)
    (out / "summary.json").write_text(json.dumps({"dataset": ds, "long_folds": long_folds_field,
                                                  "reference": str(ref_path)}), encoding="utf-8")
    return out


@pytest.mark.parametrize("fmt", ["plain", "decided"])
def test_combine_reads_long_folds_in_both_summary_formats(tmp_path, fmt):
    # a non-smoke per-data-set run overwrote summary['long_folds'] with decide()'s [[dataset, fold]] form (job 670008)
    dirs = [str(_write_dataset(tmp_path, ds, [3.0] if fmt == "plain" else [[ds, 3.0]])) for ds in ("eb", "argelaguet")]
    out = tmp_path / "decision.json"
    mv.combine(dirs, str(out))
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["long_folds"] == [["eb", 3.0], ["argelaguet", 3.0]]
    assert res["reproducibility_gate"]["passes"]
    rb = res["rule_b"]
    assert rb["time_noft"]["n_pairs"] == 6 and rb["spline_noft"]["n_pairs"] == 6
    # g = (noft - M) / (noft - base): time_noft (1.6 - 1.5) / 0.6 = 0.17 -> does not repair; spline 0.83 -> repairs
    assert rb["time_noft"]["verdict"] == "does not repair"
    assert rb["spline_noft"]["verdict"] == "repairs"
    assert res["admitted_to_step6"] == ["base", "noft", "gated", "transfer", "spline_noft"]


def test_main_summary_keeps_the_per_dataset_long_folds(tmp_path):
    """decide()'s keys must not replace the per-data-set fields the combine step reads."""
    df = pd.read_csv(_write_dataset(tmp_path, "eb", [3.0]) / "results.csv")
    summary = {"dataset": "eb", "long_folds": [3.0]}
    mv.merge_decision(summary, mv.decide(df, [("eb", 3.0)], {"eb": str(tmp_path / "eb_ref.csv")}))
    assert summary["long_folds"] == [3.0]
    assert summary["reproducibility_gate"]["passes"]
