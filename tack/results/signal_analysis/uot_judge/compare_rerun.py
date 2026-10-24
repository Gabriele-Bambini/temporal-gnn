"""Judge: compare the judge reruns (uot_judge/p1, uot_judge/p2) with the prototypes' saved outputs.
Recursive numeric diff of the result JSONs (max abs diff, paths with |diff| > 1e-3, string / selection mismatches)
and elementwise diff of the saved arrays (p1 *_scores.npz, p2 sel_preds_*.npz).
Usage: python compare_rerun.py  ->  compare_rerun.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.dirname(HERE)


def walk(a, b, path, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in ("seconds", "protocol"):
                continue
            if k not in a or k not in b:
                acc["missing"].append(f"{path}/{k}")
                continue
            walk(a[k], b[k], f"{path}/{k}", acc)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["missing"].append(f"{path} len {len(a)} vs {len(b)}")
            return
        for j, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{j}]", acc)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if np.isnan(a) and np.isnan(b):
            return
        dd = abs(float(a) - float(b))
        acc["n_num"] += 1
        if dd > acc["max"]:
            acc["max"], acc["argmax"] = dd, path
        if dd > 1e-3:
            acc["big"].append([path, a, b])
    else:
        if a != b:
            acc["str_mismatch"].append([path, str(a)[:80], str(b)[:80]])


def jdiff(fa, fb):
    a, b = json.load(open(fa)), json.load(open(fb))
    acc = {"max": 0.0, "argmax": None, "n_num": 0, "big": [], "missing": [], "str_mismatch": []}
    walk(a, b, "", acc)
    acc["n_big"] = len(acc["big"])
    acc["big"] = acc["big"][:40]
    acc["n_str_mismatch"] = len(acc["str_mismatch"])
    acc["str_mismatch"] = acc["str_mismatch"][:40]
    return acc


def ndiff(fa, fb):
    a, b = np.load(fa), np.load(fb)
    out = {}
    for k in a.files:
        if k not in b.files:
            out[k] = "missing in prototype"
            continue
        x, y = a[k], b[k]
        if x.dtype.kind in "US":
            out[k] = "equal" if np.array_equal(x, y) else "labels differ"
            continue
        out[k] = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.shape == y.shape else \
            f"shape {x.shape} vs {y.shape}"
    num = [v for v in out.values() if isinstance(v, float)]
    return {"max_abs_diff_over_arrays": max(num) if num else None, "n_arrays": len(out),
            "n_arrays_nonzero_diff": int(sum(v > 0 for v in num)),
            "worst": sorted([(k, v) for k, v in out.items() if isinstance(v, float)], key=lambda x: -x[1])[:8]}


def main():
    out = {}
    for name in ("pancreas-human", "gastrulation-human"):
        o = {}
        fa, fb = os.path.join(HERE, "p1", f"uot_sink_{name}.json"), os.path.join(SA, "uot_pop_sink", f"uot_sink_{name}.json")
        if os.path.exists(fa):
            o["p1_json"] = jdiff(fa, fb)
            o["p1_scores_npz"] = ndiff(os.path.join(HERE, "p1", f"uot_sink_{name}_scores.npz"),
                                       os.path.join(SA, "uot_pop_sink", f"uot_sink_{name}_scores.npz"))
        fa, fb = os.path.join(HERE, "p2", f"results_{name}.json"), os.path.join(SA, "uot_gene_sink", f"results_{name}.json")
        if os.path.exists(fa):
            o["p2_json"] = jdiff(fa, fb)
            o["p2_sel_preds_npz"] = ndiff(os.path.join(HERE, "p2", f"sel_preds_{name}.npz"),
                                          os.path.join(SA, "uot_gene_sink", f"sel_preds_{name}.npz"))
        out[name] = o
        print(name, json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("max", "argmax", "n_num", "n_big", "n_str_mismatch",
                                                                           "max_abs_diff_over_arrays", "n_arrays_nonzero_diff",
                                                                           "missing")}
                                for k, v in o.items()}, indent=1))
    with open(os.path.join(HERE, "compare_rerun.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
