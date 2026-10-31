"""JUDGE, EXPLORATORY: how much of the erythroid ot_pairs gain over union is already given by random-pair midpoints
(indep)? From parity_erythroid.json (frozen code on the erythroid products, 3-rep means, ED_U in PCA).
share = (union - indep) / (union - ot_pairs) at the stages where ot_pairs beats union.
Usage: python j_ery_indep.py -> j_ery_indep.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "..", "parity_erythroid.json")))["interp"]
out = {}
for k, r in P.items():
    m = {x: r["mean"][x]["ED_U"] for x in ("union", "indep", "ot_pairs")}
    g_ot, g_ind = m["union"] - m["ot_pairs"], m["union"] - m["indep"]
    out[k] = {**m, "gain_ot_pairs": g_ot, "gain_indep": g_ind,
              "indep_share_of_ot_gain": g_ind / g_ot if g_ot > 0 else None}
    print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in out[k].items()})
json.dump(out, open(os.path.join(HERE, "j_ery_indep.json"), "w"), indent=1)
