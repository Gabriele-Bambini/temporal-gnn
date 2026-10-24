"""Reference: hpc/dist_bridge.py DistBridge on orig, run unchanged (no SB: 'distbridge_strict_nosb'; and the full
'distbridge_strict' with the SB step when --full), scored on the 1000 s + i masks, i >= 1. Saves predictions.
Usage: python db_check.py [--full]"""
import json
import os
import sys
import time

import numpy as np

import common as C

DB = C.D.DB


def main(full):
    key = "distbridge_strict" if full else "distbridge_strict_nosb"
    out = {}
    for name in C.NAMES:
        t0 = time.time()
        d = C.load(name)
        Y = d.Y("orig")
        model = DB.REGISTRY[key]()
        hist = [d.X[b] for b in range(d.B)]
        P = [np.asarray(model.predict(hist[:i + 1], None, None), np.float64) for i in range(d.T)]
        np.savez_compressed(os.path.join(C.HERE, f"db_{key}_orig_{name}.npz"), pred=np.stack(P))
        rows = C.score(Y, P, C.masks(d.G, d.T))
        sel = [list(v) for k, v in sorted(model.selected.items(), key=lambda kv: kv[0][1])]
        out[name] = {"i>=1": C.summ(rows, range(1, d.T)), "i>=2": C.summ(rows, range(2, d.T)),
                     "selected": [str(s) for s in sel], "seconds": round(time.time() - t0)}
        print(name, key, json.dumps(out[name]), flush=True)
    json.dump(out, open(os.path.join(C.HERE, f"db_check_{key}.json"), "w"), indent=1)


if __name__ == "__main__":
    main("--full" in sys.argv)
