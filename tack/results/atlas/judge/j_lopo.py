"""JUDGE, EXPLORATORY (descriptive, uses the held stage): model-free pool-level reference for the "noise floor" claim.

For every held pool p (>= 20 cells) the leave-one-pool-out replicate is the rest of the held stage (<= 1500 cells,
IB.sub seed 5, the frozen interp_bench own_loso construction) scored by ED_U against pool p. A real same-stage
replicate from other embryos; a prediction built from neighbour stages that matches it is "at the pool floor".
Compared with the stored 3-rep pool means of union and ot_pairs (confirm.json / parity_erythroid.json).
Usage: python j_lopo.py -> j_lopo.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
SETS = {"atlas": (os.path.join(SCR, "atlas"), range(1, 8), os.path.join(HERE, "..", "confirm.json")),
        "ery": (os.path.join(SCR, "ery"), range(2, 6), os.path.join(HERE, "..", "parity_erythroid.json"))}


def main():
    res = {}
    for name, (data, ks, sp) in SETS.items():
        stored = json.load(open(sp))["interp"]
        rows = []
        for k in ks:
            d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
            Zf, Zh = d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
            smp = d["sample_held"].astype(str)
            S = IB.med_scale(Zf)
            pools = [p for p in sorted(set(smp)) if (smp == p).sum() >= 20]
            if len(pools) < 2:
                continue
            for p in pools:
                rest = np.where(smp != p)[0]
                rest = rest[IB.sub(len(rest), seed=5)]
                pt = IB.Target(Zh[smp == p], None, S, with_sd=False)
                lo = IB.ed_uv(Zh[rest], IB.unif(len(rest)), pt)[0]
                u = stored[str(k)]["pool_mean"]["union"][p]
                o = stored[str(k)]["pool_mean"]["ot_pairs"][p]
                rows.append({"k": k, "pool": p, "n": int((smp == p).sum()), "lopo": lo, "union": u, "ot_pairs": o,
                             "union_over_lopo": u / lo, "ot_over_lopo": o / lo})
        res[name] = {"rows": rows, "summary": {
            "n_pools": len(rows),
            "union_lt_lopo": sum(r["union"] < r["lopo"] for r in rows),
            "ot_pairs_lt_lopo": sum(r["ot_pairs"] < r["lopo"] for r in rows),
            "ot_pairs_lt_union": sum(r["ot_pairs"] < r["union"] for r in rows),
            "median_union_over_lopo": float(np.median([r["union_over_lopo"] for r in rows])),
            "median_ot_over_lopo": float(np.median([r["ot_over_lopo"] for r in rows])),
            "median_lopo": float(np.median([r["lopo"] for r in rows])),
            "lopo_range": [float(min(r["lopo"] for r in rows)), float(max(r["lopo"] for r in rows))]}}
        print(name, json.dumps(res[name]["summary"]))
        for r in rows:
            print("  ", name, r["k"], r["pool"], r["n"], "lopo", round(r["lopo"], 3), "union", round(r["union"], 3),
                  "ot", round(r["ot_pairs"], 3))
    json.dump(res, open(os.path.join(HERE, "j_lopo.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
