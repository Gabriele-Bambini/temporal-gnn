"""Judge 1: standalone value of the per-gene OT push on the corrected l1 target and on orig (strict, i >= 2).
Arms (same masks as indep.py, seed 4242): level = box-4 reversion move (l1) or box-4 + finite-cell head (orig, from
the saved unpushed expectation = level); push = level + OT increment (dist_bridge coupling/sb_step, saved by
design_best-signal/ot_feature.py); inc alone; trend. Pearson / P@200 up / down, mean over transitions and masks."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rerun_nass_p"))
import dsdata as D
from dsdata import DB
SRC = os.path.join(os.path.dirname(HERE), "design_best-signal")

def met(y, p):
    r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 and np.std(y) > 0 else np.nan
    up = len(np.intersect1d(np.argsort(-y)[:200], np.argsort(-p)[:200])) / 200
    dn = len(np.intersect1d(np.argsort(y)[:200], np.argsort(p)[:200])) / 200
    return r, up, dn

out = {}
for name in ("pancreas-human", "gastrulation-human"):
    d = D.Data(name)
    ot = np.load(os.path.join(SRC, f"_ot_{name}.npz"))
    trip = [DB.triplet(d.X[b]) for b in range(d.B)]
    box = DB.box_pool(trip, list(d.n), 4)
    rng = np.random.default_rng(4242)
    nt = int(round(0.25 * d.G))
    mk = np.array([[np.sort(rng.permutation(d.G)[:nt]) for _ in range(d.T)] for _ in range(20)])
    res = {}
    for tg in ("l1", "orig"):
        Y = d.Y(tg)
        Z = d.CP if tg == "l1" else d.m
        arms = {k: [] for k in ("trend", "level", "level+OT", "level+2OT", "OT_inc_alone")}
        for i in range(2, d.T):
            if tg == "l1":
                lev = D.tfun("l1", Z[:, i], (Z[:, max(0, i - 3):i + 1] * d.n[max(0, i - 3):i + 1]).sum(1) / d.n[max(0, i - 3):i + 1].sum())
            else:
                lev = DB.expected_target(box[i], d.m[:, i], d.n[i], np.random.default_rng([1 + (name == "pancreas-human"), i, 0])).astype(np.float64)
            inc = ot[f"inc_{tg}"][i].astype(np.float64)
            P = {"trend": Y[i - 1], "level": lev, "level+OT": lev + inc, "level+2OT": lev + 2 * inc, "OT_inc_alone": inc}
            for k, p in P.items():
                arms[k].append(np.mean([met(Y[i][mk[q, i]], p[mk[q, i]]) for q in range(20)], 0))
        res[tg] = {k: [round(float(x), 4) for x in np.nanmean(np.array(v), 0)] for k, v in arms.items()}
        dl = np.array(arms["level+OT"])[:, 0] - np.array(arms["level"])[:, 0]
        res[tg]["paired_level+OT_minus_level_pcc"] = [round(float(dl.mean()), 4), int((dl > 0).sum()), int((dl < 0).sum())]
        print(name, tg, json.dumps(res[tg]), flush=True)
    out[name] = res
json.dump(out, open(os.path.join(HERE, "ot_standalone.json"), "w"), indent=1)
