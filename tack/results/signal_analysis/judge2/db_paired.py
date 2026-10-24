"""Judge 2: run our DistBridge (tack/hpc/dist_bridge.py, unmodified, strict rolling origin) on the SAME 20 masks that
nass.py uses (dsdata.masks, seed 20261022), so NASS / trend vs DistBridge can be paired by transition.
Writes db_<dataset>.json: per-transition mask-averaged [PCC, P@200 up, P@200 down] per model and the summaries.
Usage: python db_paired.py <dataset> [orig|l1|linear ...]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsdata as D  # judge2 copy (reads judge2/_node_*.npz)
from dsdata import DB

MODELS = {"orig": ["distbridge_strict", "distbridge_strict_nosb"],
          "l1": ["distbridge_strict_l1", "distbridge_strict_nosb_l1"],
          "linear": ["distbridge_strict_linear", "distbridge_strict_nosb_linear"]}


def main(name, targets):
    d = D.Data(name)
    mk = D.masks(d.G, d.T)
    tag = os.environ.get("DB_TAG", "")
    out_f = os.path.join(HERE, f"db_{name}{('_' + tag) if tag else ''}.json")
    res = json.load(open(out_f)) if os.path.exists(out_f) else {}
    hist_all = [d.X[b] for b in range(d.B)]
    for tg in targets:
        Y = d.Y(tg)
        only = os.environ.get("DB_MODELS")
        for key in (only.split(",") if only else MODELS[tg]):
            t0 = time.time()
            m = DB.REGISTRY[key]()
            preds = []
            for i in range(d.T):
                preds.append(np.asarray(m.predict(hist_all[:i + 1], None, None), np.float64))
            S = D.score(Y, preds, mk)
            per_t = np.nanmean(S, 0)[:, :3]
            ranges = {"all": list(range(d.T)), "i>=1": list(range(1, d.T)), "i>=2": list(range(2, d.T)),
                      "i>=3": list(range(3, d.T))}
            sel = [str(m.selected.get(k)) for k in sorted(m.selected)]
            res[f"{tg}|{key}"] = {"per_transition": np.round(per_t, 5).tolist(),
                                  "table": {r: D.summarize(S, idx) for r, idx in ranges.items()},
                                  "selected": sel, "seconds": round(time.time() - t0, 1)}
            print(name, tg, key, {r: [res[f'{tg}|{key}']['table'][r][k] for k in ('pcc', 'p200_up', 'p200_down')]
                                  for r in ranges}, f"{time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(out_f, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["orig", "l1", "linear"])
