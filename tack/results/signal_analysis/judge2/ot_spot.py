"""Judge 2 spot-check of the NASS OT feature (_ot_<dataset>.npz): recompute the OT/SB increment of one transition
with the unmodified ot_feature recipe (dist_bridge.coupling / sb_step / expected_target / expected_value, same
generator keys) on (a) the real data and (b) data whose bins > i are garbage; compare both with the stored row.
Usage: python ot_spot.py <dataset> <i>"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsdata as D
from dsdata import DB
import ot_feature as OF


def inc_at(d, i, key):
    trip = [DB.triplet(d.X[b]) for b in range(d.B)]
    box = DB.box_pool(trip, list(d.n), 4)
    L = min(3, i)
    H = box[i]
    cp = DB.coupling(box[i - L], H, L)
    Hot = DB.sb_step(H, cp, L, 1.0)
    m_now = d.m[:, i]
    out = {"orig": np.zeros(d.G, np.float32), "l1": np.zeros(d.G, np.float32)}
    for arm, HH in ((0, H), (1, Hot)):
        sgn = 1 if arm else -1
        eo = DB.expected_target(HH, m_now, d.n[i], np.random.default_rng([key, i, 0, arm]), draws=OF.DRAWS)
        el = DB.expected_value(HH, m_now, d.n[i], np.random.default_rng([key, i, 1, arm]),
                               OF.l1_fn(m_now, max(float(HH[:, 0].sum()), 1e-12)), draws=OF.DRAWS)
        out["orig"] += sgn * eo
        out["l1"] += sgn * el
    return out, len(cp["idx"])


name, i = sys.argv[1], int(sys.argv[2])
key = 1 + (name == "pancreas-human")
stored = np.load(os.path.join(HERE, "..", "design_best-signal", f"_ot_{name}.npz"))
d0 = D.Data(name)
d1 = D.Data(name)
X = d1.X.copy()
rng = np.random.default_rng(7)
for b in range(i + 1, d1.B):
    Xb = X[b][rng.permutation(d1.G)].copy()
    Xb[:, [0, 1, 2]] *= rng.uniform(0.2, 5, d1.G)[:, None]
    Xb[:, 4] = Xb[:, 0] * 3 * DB.n_cells(X[b])
    X[b] = Xb
d1.X = X
d1.m = np.maximum(X[:, :, 0].T, 0)
d1.n = np.array([DB.n_cells(X[b]) for b in range(d1.B)])
r0, nc0 = inc_at(d0, i, key)
r1, nc1 = inc_at(d1, i, key)
res = {"dataset": name, "i": i, "coupled_genes": nc0}
for tg in ("orig", "l1"):
    s = stored[f"inc_{tg}"][i]
    res[tg] = {"stored_vs_recomputed_maxdiff": float(np.abs(s - r0[tg]).max()),
               "real_vs_garbage_future_maxdiff": float(np.abs(r0[tg] - r1[tg]).max()),
               "mean_abs_inc": float(np.abs(s).mean()), "frac_nonzero": float((np.abs(s) > 1e-6).mean()),
               "corr_with_last_change": None}
Y = d0.Y("l1")
res["l1"]["corr_with_last_change"] = float(np.corrcoef(stored["inc_l1"][i], Y[i - 1])[0, 1])
res["l1"]["corr_with_truth"] = float(np.corrcoef(stored["inc_l1"][i], Y[i])[0, 1])
print(json.dumps(res, indent=1))
with open(os.path.join(HERE, "ot_spot.jsonl"), "a") as fh:
    fh.write(json.dumps(res) + "\n")
