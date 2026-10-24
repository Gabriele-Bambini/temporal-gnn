"""Merge the Phase 1 result tables (toys, GRN forecast diagnostic, GRN interpolation) into one report.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/merge_phase1_report.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.evaluation.report import write_report

RUNS = {"outputs/phase1_v1_toys": "toys (v0.1.4 protocol, refit)",
        "outputs/phase1_v1_grn": "GRN forecast, diagnostic (v0.1.4 protocol, refit)",
        "outputs/phase1_v2_grn_interp": "GRN interpolation t=3, gate G1 (v0.1.7 protocol, no refit)"}
out = Path("outputs/phase1_report")
out.mkdir(parents=True, exist_ok=True)
tables, modes, notes = [], {}, []
for run, label in RUNS.items():
    p = Path(run)
    if not (p / "results.csv").exists():
        notes.append(f"- {run}: results.csv missing (run not finished)")
        continue
    df = pd.read_csv(p / "results.csv")
    meta = json.loads((p / "run_meta.json").read_text(encoding="utf-8"))
    for ds in df["dataset"].unique():
        modes[ds] = "sanity" if ds.startswith("toy") else "benchmark"
    tables.append(df)
    notes.append(f"- {run}: {label}; code version {meta.get('version')}; finished {meta.get('finished')}; "
                 f"best configs: {json.dumps(meta.get('best_configs'))}")
df = pd.concat(tables, ignore_index=True)
df.to_csv(out / "results.csv", index=False)
verdicts = write_report(df, out / "report.md", out / "verdict.json", modes=modes)
with open(out / "report.md", "a", encoding="utf-8") as fh:
    fh.write("\n## Runs merged\n\n" + "\n".join(notes) + "\n")
for ds, v in verdicts.items():
    print(f"{ds} ({v['mode']}): {'PASS' if v['pass'] else 'FAIL'} fm={v['fm_mean']:.3f} best_control={v['best_control']} "
          f"uses_time={v['uses_time']} null_q95={v['null_q95']:.3f} matches={v['matches_target']}")
