"""JUDGE (not exploratory): leaf-by-leaf comparison of my local re-run of the frozen atlas_confirm.py (confirm_rerun.json,
produced by `python tack/hpc/atlas_confirm.py --data <scratchpad/atlas> --out judge/confirm_rerun.json`, unmodified)
against the HPC result tack/results/atlas/confirm.json. Both-NaN counts as a match. Config paths are skipped.
Usage: python j_diff.py -> j_diff.json
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "..", "confirm.json")))
B = json.load(open(os.path.join(HERE, "confirm_rerun.json")))
SKIP = {"config", "seconds"}
stats = {"n_leaves": 0, "n_numeric": 0, "max_abs": 0.0, "argmax": None, "mismatch_nonnumeric": [], "missing": []}


def walk(a, b, path):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            stats["missing"].append(path)
            return
        for key in set(a) | set(b):
            if path == "" and key in SKIP:
                continue
            if key not in a or key not in b:
                stats["missing"].append(f"{path}/{key}")
                continue
            walk(a[key], b[key], f"{path}/{key}")
        return
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            stats["missing"].append(path)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
        return
    stats["n_leaves"] += 1
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        if a != b:
            stats["mismatch_nonnumeric"].append((path, a, b))
        return
    stats["n_numeric"] += 1
    if isinstance(a, float) and math.isnan(a):
        if not (isinstance(b, float) and math.isnan(b)):
            stats["mismatch_nonnumeric"].append((path, a, b))
        return
    d = abs(a - b)
    if d > stats["max_abs"]:
        stats["max_abs"], stats["argmax"] = d, path


walk(A, B, "")
stats["verdicts_equal"] = A["verdicts"] == B["verdicts"]
json.dump(stats, open(os.path.join(HERE, "j_diff.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
