"""DESCRIPTIVE: what the reweighting forecasts do. For origins 2-5 and the PRIMARY ('roll') hp under sd_fixed and
energy, the atlas cell-type composition implied by the weights (uot_rw, mk_km, tilt_rw, comp_km) next to the current
stage s and the TRUE stage s+1 (the latter only for reading, never used), plus the effective sample size.
Output: explain_weights.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attempt_forecast as A  # noqa: E402

F = A.F
TYPES = ["Blood progenitors 1", "Blood progenitors 2", "Erythroid1", "Erythroid2", "Erythroid3"]


def comp(lab, w=None):
    w = np.full(len(lab), 1 / len(lab)) if w is None else w / w.sum()
    return {t[:6] + t[-1]: round(float(w[lab == t].sum()), 3) for t in TYPES}


def main():
    obs, codes = A.load_obs(A.DATA)
    out = {}
    for s in (2, 3, 4, 5):
        cur, prev = A.load_scores(s), A.load_scores(s - 1)
        F._DRIFT.clear()
        d, smp_past, smp_fut, km, S, P = A.build_final(A.DATA, s, obs, codes)
        r = {"now": comp(P.ct_n), "true_next": comp(d["celltype_fut"].astype(str))}
        for m in ("sd_fixed", "energy"):
            for fam in ("uot_rw", "mk_km", "tilt_rw", "comp_km"):
                hp = A.selected_hps(fam, s, m, cur, prev)["roll"]
                _, _, w = A.FAMILIES[fam][1](P, hp)
                key = f"{fam}|{m}|{A.hkey(hp)}"
                r[key] = {**comp(P.ct_n, w), "ess": round(float(1 / ((w / w.sum()) ** 2).sum()), 1) if w is not None
                          else len(P.Zn)}
        out[str(s)] = r
        print(s, json.dumps(r, indent=0)[:3000], flush=True)
    with open(os.path.join(HERE, "explain_weights.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
