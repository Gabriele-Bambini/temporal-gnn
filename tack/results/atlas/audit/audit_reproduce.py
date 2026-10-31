"""AUDIT 0: the frozen atlas_confirm.py, run unmodified on the LOCAL products (confirm_local.json, produced by
`python tack/hpc/atlas_confirm.py --data <scratchpad/atlas> --out audit/confirm_local.json`), reproduces the HPC
confirm.json. Diffs every numeric leaf and every hp / verdict. Output: audit/reproduce.json.
"""
import json
import os

from _common import AUDIT, confirm, dump


def leaves(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{pre}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, o


def main():
    hpc = confirm()
    with open(os.path.join(AUDIT, "confirm_local.json")) as fh:
        loc = json.load(fh)
    a = dict(leaves({k: hpc[k] for k in ("interp", "forecast", "verdicts")}))
    b = dict(leaves({k: loc[k] for k in ("interp", "forecast", "verdicts")}))
    missing = sorted(set(a) ^ set(b))
    num_diff, other_diff = [], []
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            if x != y:
                other_diff.append((k, x, y))
        elif x != x or y != y:  # NaN: equal only if both NaN
            if not (x != x and y != y):
                other_diff.append((k, x, y))
        else:
            num_diff.append((abs(x - y), k, x, y))
    num_diff.sort(reverse=True)
    out = {"n_leaves": len(a), "keys_only_in_one": missing, "non_numeric_mismatches": other_diff,
           "max_abs_numeric_diff": num_diff[0][0] if num_diff else 0.0, "top5_numeric_diffs": num_diff[:5],
           "verdicts_identical": hpc["verdicts"] == loc["verdicts"]}
    print(json.dumps(out, indent=1, default=str))
    dump("reproduce.json", out)


if __name__ == "__main__":
    main()
