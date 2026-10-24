"""Judge: matched-config sink ablation for the population prototype, free of selection noise.
For every balanced config (K, L, R, eps) and its sink twins (sink_norm / sink_abs with the same K, L, R, eps, each
lambda), the per-transition all-gene criterion J (Pearson + tie-aware P@K up + down)/3 saved by the judge rerun
(J_<target> in uot_sink_<ds>_scores.npz) is paired over transitions i >= 2. Reported per family and lambda: mean dJ
over the 24 matched (K, L, R, eps) cells, how many cells have mean dJ > 0, and the pooled sign test over (cell,
transition) (cells share transitions, so that p-value is optimistic). This is a hindsight profile (every config is
scored on every transition), not a strict forecaster.
Usage: python profile_p1.py <dataset>  ->  profile_p1_<dataset>.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(lbl):
    parts = lbl.split("|")
    hp = {}
    for p in parts[3:]:
        for key in ("R", "eps", "lam", "tau"):
            if p.startswith(key):
                hp[key] = p[len(key):]
                break
    return parts[0], parts[1], parts[2], hp


def main():
    name = sys.argv[1]
    z = np.load(os.path.join(HERE, "p1", f"uot_sink_{name}_scores.npz"))
    labels = [str(x) for x in z["labels"]]
    info = [parse(x) for x in labels]
    out = {}
    for t in ("l1", "linear", "orig"):
        J = z[f"J_{t}"]
        base = {}
        for r, (fam, K, L, hp) in enumerate(info):
            if fam == "balanced":
                base[(K, L, hp["R"], hp["eps"])] = r
        res = {}
        for fam in ("sink_norm", "sink_abs"):
            for lam in ("0.1", "0.25", "0.5", "1"):
                cell_means, pooled = [], []
                for r, (f, K, L, hp) in enumerate(info):
                    if f != fam or hp.get("lam") != lam:
                        continue
                    b = base[(K, L, hp["R"], hp["eps"])]
                    d = J[r, 2:] - J[b, 2:]
                    d = d[np.isfinite(d)]
                    cell_means.append(float(d.mean()))
                    pooled.extend(d.tolist())
                pooled = np.array(pooled)
                nz = pooled[np.abs(pooled) > 1e-12]
                res[f"{fam}|lam{lam}"] = {"mean_dJ_over_cells": round(float(np.mean(cell_means)), 4),
                                          "cells_with_mean_dJ>0": f"{int(np.sum(np.array(cell_means) > 0))}/{len(cell_means)}",
                                          "pooled_wins": int((nz > 0).sum()), "pooled_losses": int((nz < 0).sum()),
                                          "pooled_sign_p": round(float(stats.binomtest(int((nz > 0).sum()), len(nz)).pvalue), 4)
                                          if len(nz) else None}
        out[t] = res
        print("==", name, t)
        for k, v in res.items():
            print("  ", k, json.dumps(v))
    with open(os.path.join(HERE, f"profile_p1_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
