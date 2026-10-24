"""Compare judge2 re-runs of run_lsb.py (and popot.py) with the designs' own outputs, cell by cell.
Usage: python compare_lsb.py lsb <dataset> [judge_json] | popot <dataset>"""
import json
import sys

S = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\signal_analysis"


def walk(a, b, path=""):
    """Yield (path, |a-b|) for every numeric leaf present in both."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                yield from walk(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        yield path, abs(a - b)


kind, name = sys.argv[1], sys.argv[2]
if kind == "lsb":
    A = json.load(open(fr"{S}\design_latent-sb\results_{name}.json"))
    B = json.load(open(sys.argv[3] if len(sys.argv) > 3 else fr"{S}\judge2\results_{name}.json"))
    keys = ("methods", "paired")
else:
    A = json.load(open(fr"{S}\design_population-ot\popot_{name}.json"))
    B = json.load(open(fr"{S}\judge2\popot_{name}.json"))
    keys = ("targets", "composition_forecast_tv_error_vs_oracle_projection_i>=2", "reproduction_crossgene_nmf_linear_props_l1")
for key in keys:
    diffs = list(walk(A[key], B[key]))
    diffs = [(p, d) for p, d in diffs if "seconds" not in p]
    worst = max(diffs, key=lambda x: x[1]) if diffs else None
    big = [p for p, d in diffs if d > 0.002]
    print(f"{kind} {name} [{key}] leaves {len(diffs)}, max |diff| {worst}, n>0.002: {len(big)} e.g. {big[:5]}")
