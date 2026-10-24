"""Judge: diff of the judge rerun (judge/ copies of the build scripts) against the build outputs.
Walks results_l1.json, results_linear.json, orig/results_orig.json (every numeric leaf, every string / bool leaf such
as chosen configurations, pools and detection settings) and the pertrans_*.npz arrays.
Ignored keys: 'seconds' (wall time). Output: judge_compare.json."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(os.path.dirname(HERE), "build")
FILES = ["results_l1.json", "results_linear.json", os.path.join("orig", "results_orig.json")]
NPZ = ["pertrans_l1_gastrulation-human.npz", "pertrans_l1_pancreas-human.npz", "pertrans_linear_gastrulation-human.npz",
       "pertrans_linear_pancreas-human.npz", os.path.join("orig", "pertrans_orig_gastrulation-human.npz"),
       os.path.join("orig", "pertrans_orig_pancreas-human.npz")]


def walk(a, b, path, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k == "seconds":
                continue
            if k not in a or k not in b:
                acc["missing"].append(f"{path}/{k}")
                continue
            walk(a[k], b[k], f"{path}/{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["len_mismatch"].append(path)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        acc["n_other"] += 1
        if a != b:
            acc["other_mismatch"].append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        acc["n_num"] += 1
        if not (np.isnan(a) and np.isnan(b)):
            d = abs(float(a) - float(b))
            if d > acc["max_num_diff"]:
                acc["max_num_diff"], acc["max_num_path"] = d, path
            if d > 0:
                acc["n_num_diff"] += 1
    else:
        acc["other_mismatch"].append(f"{path}: type {type(a)} vs {type(b)}")


def main():
    out = {}
    for f in FILES:
        a = json.load(open(os.path.join(HERE, f)))
        b = json.load(open(os.path.join(BUILD, f)))
        acc = {"n_num": 0, "n_num_diff": 0, "max_num_diff": 0.0, "max_num_path": None, "n_other": 0,
               "other_mismatch": [], "missing": [], "len_mismatch": []}
        walk(a, b, "", acc)
        acc["other_mismatch"] = acc["other_mismatch"][:50]
        out[f] = acc
        print(f, {k: (v if not isinstance(v, list) else len(v)) for k, v in acc.items()})
    for f in NPZ:
        a, b = np.load(os.path.join(HERE, f)), np.load(os.path.join(BUILD, f))
        ka, kb = set(a.files), set(b.files)
        md = max((float(np.nanmax(np.abs(a[k] - b[k]))) if a[k].size else 0.0) for k in ka & kb)
        out[f] = {"keys_equal": ka == kb, "max_abs_diff": md,
                  "all_equal_nan_aware": bool(all(np.array_equal(a[k], b[k], equal_nan=True) for k in ka & kb))}
        print(f, out[f])
    json.dump(out, open(os.path.join(HERE, "judge_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
