import numpy as np
import pandas as pd

from cfm_grn_v2.evaluation.report import METRICS, paired_difference, summarize, verdict, write_report


def _fake_rows():
    rows = []
    rng = np.random.default_rng(0)
    levels = (("flow_matching", 0.1), ("flow_matching_shuffled_time", 0.9), ("persistence", 1.0), ("mean_shift", 0.6),
              ("linear", 0.5), ("mlp", 0.5), ("train_mean", 1.5), ("freefm_replay", 0.7), ("noise_floor", 0.05))
    for seed in range(3):
        for model, base in levels:
            rows.append({"dataset": "toy", "model": model, "seed": seed, "group": "ctrl|r3", "null_q95": 0.12,
                         **{m: base + 0.01 * rng.normal() for m in METRICS}})
    return pd.DataFrame(rows)


def test_summary_paired_and_verdict(tmp_path):
    df = _fake_rows()
    summ = summarize(df)
    assert {"dataset", "model", "metric", "mean", "std", "ci95", "n"} <= set(summ.columns)
    d = paired_difference(df, "toy", "flow_matching", "persistence")
    assert d["ci_high"] < 0 and d["n"] == 3
    v = verdict(df, "toy")
    assert v["pass"] and v["uses_time"] and v["best_control"] in ("linear", "mlp") and v["mode"] == "benchmark"
    assert verdict(df, "toy", mode="sanity")["pass"]
    worse = df.copy()
    worse.loc[worse["model"] == "flow_matching", "energy_distance"] = 0.3
    assert not verdict(worse, "toy", mode="sanity")["pass"]
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    out = write_report(df, md, js)
    assert md.exists() and js.exists() and "flow_matching" in md.read_text(encoding="utf-8") and out["toy"]["pass"]
