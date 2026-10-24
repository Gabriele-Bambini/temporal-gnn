"""Compare the judge2 re-run of nass.py (judge2/results.json) with the design's own results.json, cell by cell."""
import json
import sys

import numpy as np

A = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\signal_analysis\design_best-signal\results.json"
B = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\signal_analysis\judge2\results.json"
a, b = json.load(open(A)), json.load(open(B))
for name in b:
    if name not in a:
        continue
    for tg, tb in b[name]["targets"].items():
        ta = a[name]["targets"].get(tg)
        if ta is None:
            print(name, tg, "missing in design results")
            continue
        worst, where, n = 0.0, None, 0
        missing = []
        for meth, rr in tb["table"].items():
            if meth not in ta["table"]:
                missing.append(meth)
                continue
            for rng, cell in rr.items():
                for k, v in cell.items():
                    w = ta["table"][meth][rng][k]
                    d = abs(v - w)
                    n += 1
                    if d > worst:
                        worst, where = d, (meth, rng, k, v, w)
        pw, pwhere = 0.0, None
        for key, v in tb["paired"].items():
            if key in ta["paired"]:
                for m in ("pcc", "J"):
                    d = abs(v[m]["mean_delta"] - ta["paired"][key][m]["mean_delta"])
                    if d > pw:
                        pw, pwhere = d, (key, m, v[m]["mean_delta"], ta["paired"][key][m]["mean_delta"])
        print(f"{name} {tg}: {n} table cells, max |diff| {worst:.4g} at {where}; paired max |diff| {pw:.4g} at {pwhere};"
              f" methods missing in design: {missing}")
