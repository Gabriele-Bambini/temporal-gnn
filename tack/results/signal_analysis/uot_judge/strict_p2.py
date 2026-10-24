"""Judge: future-scramble strictness test for the gene-sink prototype (judge copy p2/run_uot.py + uot_core.py).
For transition i, every bin > i (law triplet, cell count, mean) is replaced by random values in the in-process data
cache; R.job(...) is recomputed for several couplings (uot cheap / mid sink, L = 1 / 3, balanced, persist) and must be
bit-identical to the judge rerun's cached job output (all deltas, all targets).
Usage: python strict_p2.py <dataset>  ->  strict_p2_<dataset>.json
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "p2"))
import run_uot as R  # noqa: E402

SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\judge_p2"


def main():
    name = sys.argv[1]
    X, trip, n, m = R.U.load(name)
    with open(os.path.join(SCR, f"uot_res_{name}.pkl"), "rb") as fh:
        res = pickle.load(fh)
    T = len(trip) - 1
    tests = [5, 10] if T < 20 else [6, 20]
    out = {}
    for i in tests:
        rng = np.random.default_rng(7 + i)
        trip2 = [t.copy() for t in trip]
        m2, n2 = m.copy(), n.copy()
        for b in range(i + 1, len(trip)):
            G = trip[b].shape[0]
            mm = rng.gamma(0.2, 2.0, G)
            ff = np.minimum(rng.random(G), mm)
            trip2[b] = np.stack([mm, ff, mm ** 2 + rng.gamma(0.2, 2.0, G)], 1)
            m2[b] = mm
            n2[b] = rng.uniform(5, 1500)
        R._DATA[name] = (X, trip2, n2, m2)
        # positive control: transition i + 1 reads bin i + 1, which is scrambled, so its output must change
        pa = (name, "persist", i + 1, 0, 0.0)
        g_, r_ = R.job(pa)[1][0.0], res[pa][0][0.0]
        out[f"positive_control_i{i + 1}_persist"] = max(float(np.max(np.abs(np.asarray(g_[tg], np.float64) -
                                                                            np.asarray(r_[tg], np.float64))))
                                                        for tg in ("l1", "orig"))
        print(name, pa, "positive control diff", out[f"positive_control_i{i + 1}_persist"], flush=True)
        if len(sys.argv) > 2 and sys.argv[2] == "pos":
            continue
        for args in ((name, "uot", i, 1, 0.0625), (name, "uot", i, min(3, i), 1.0), (name, "bal", i, min(3, i), 0.0),
                     (name, "persist", i, 0, 0.0)):
            got = R.job(args)[1]
            ref = res[args][0]
            worst = 0.0
            for dl in ref:
                for tg in R.TARGETS:
                    worst = max(worst, float(np.max(np.abs(np.asarray(got[dl][tg], np.float64) -
                                                           np.asarray(ref[dl][tg], np.float64)))))
            out[f"i{i}_{args[1]}_Le{args[3]}_lam{args[4]:g}"] = worst
            print(name, args, "max abs diff vs unscrambled", worst, flush=True)
    with open(os.path.join(HERE, f"strict_p2_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
