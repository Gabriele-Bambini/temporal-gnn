"""Judge 2: paired (same 20 masks, per transition) comparison of NASS rows (judge2/results.json, the re-run) with our
DistBridge (judge2/db_<dataset>.json). Writes pair_db.json and prints the table.
Usage: python pair_db.py <dataset> [...]"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NASS_ROWS = ("full", "full+grn", "regime", "trend", "level(box4+obs)", "level(box4)", "full[h=0.5 fixed]")
DB_KEY = {"orig": ("distbridge_strict", "distbridge_strict_nosb"), "l1": ("distbridge_strict_l1", "distbridge_strict_nosb_l1"),
          "linear": ("distbridge_strict_linear", "distbridge_strict_nosb_linear")}


def wil(d):
    nz = d[np.abs(d) > 1e-12]
    return round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None


out_f = os.path.join(HERE, "pair_db.json")
out = json.load(open(out_f)) if os.path.exists(out_f) else {}
nass = json.load(open(os.path.join(HERE, "results.json")))
for name in sys.argv[1:]:
    import glob
    db = {}
    for fn in sorted(glob.glob(os.path.join(HERE, f"db_{name}*.json"))):
        db.update(json.load(open(fn)))
    for tg, keys in DB_KEY.items():
        pt = nass[name]["targets"][tg]["per_transition_pcc_up_down"]
        for dk in keys:
            if f"{tg}|{dk}" not in db:
                continue
            B = np.array(db[f"{tg}|{dk}"]["per_transition"], float)
            for row in NASS_ROWS:
                if row not in pt:
                    continue
                A = np.array(pt[row], float)
                for rng, lo in (("all", 0), ("i>=2", 2)):
                    a, b = np.nan_to_num(A[lo:]), np.nan_to_num(B[lo:])
                    dP = a[:, 0] - b[:, 0]
                    dJ = a.sum(1) - b.sum(1)
                    rec = {"nass_row": [round(float(x), 4) for x in np.nanmean(A[lo:], 0)],
                           "distbridge": [round(float(x), 4) for x in np.nanmean(B[lo:], 0)],
                           "dPCC": round(float(dP.mean()), 4), "better/worse": [int((dP > 1e-12).sum()), int((dP < -1e-12).sum())],
                           "p_PCC": wil(dP), "dJ": round(float(dJ.mean()), 4), "p_J": wil(dJ)}
                    out[f"{name}|{tg}|{row} vs {dk}|{rng}"] = rec
                    print(f"{name[:5]} {tg:6s} {rng:4s} {row:18s} vs {dk:30s} {rec['nass_row']} vs {rec['distbridge']}"
                          f" dPCC {rec['dPCC']:+.3f} {rec['better/worse']} p={rec['p_PCC']} dJ {rec['dJ']:+.3f} p={rec['p_J']}")
json.dump(out, open(out_f, "w"), indent=1)
