"""Post-hoc paired checks from the cached run (no new fits): does a cheap sink (lam0 = 0.0625 / 0.25) change the
DOWN side of the ranking against the forced-only sink (lam0 = 128) and the balanced ablation? Strict families as
selected in results_<dataset>.json; per-transition mask-averaged metrics, Wilcoxon over transitions i >= 1.
Also: how often the strict uot selection picked each lam0 / L / delta."""
import collections
import json
import os
import pickle
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_uot as R  # noqa: E402

out = {}
for name in ("pancreas-human", "gastrulation-human"):
    X, trip, n, m = R.U.load(name)
    T, G = len(trip) - 1, trip[0].shape[0]
    with open(os.path.join(R.SCRATCH, f"uot_res_{name}.pkl"), "rb") as fh:
        res = pickle.load(fh)
    J = json.load(open(os.path.join(HERE, f"results_{name}.json")))
    mk = R.masks(G, T)
    out[name] = {}
    for tg in ("l1", "linear"):
        Y = np.stack([R.dileo_protocol.TARGET_NP[tg](m[i], m[i + 1]).astype(np.float64) for i in range(T)])

        def pred(fam, i):
            c = J["selected"][fam][tg][str(i)]
            if c[0] == "uot":
                return res[(name, "uot", i, min(c[2], i), c[1])][0][c[3]][tg]
            return res[(name, "bal", i, min(c[1], i), 0.0)][0][c[2]][tg]

        def table(fam):
            rows = []
            for i in range(1, T):
                p = pred(fam, i)
                v = [R.dileo_rescore.metrics(Y[i][mk[s][i]], p[mk[s][i]]) for s in R.SEEDS]
                rows.append([np.nanmean([x[k] for x in v]) for k in ("pearson", "p_ta_up", "p_ta_down")])
            return np.array(rows)
        tabs = {f: table(f) for f in ("uot_lam0.0625", "uot_lam0.25", "uot_lam128", "bal")}
        res_t = {}
        for a, b in (("uot_lam0.0625", "uot_lam128"), ("uot_lam0.25", "uot_lam128"), ("uot_lam0.0625", "bal"),
                     ("uot_lam0.25", "bal")):
            d = tabs[a] - tabs[b]
            res_t[f"{a} vs {b}"] = {k: {"mean_diff": round(float(np.nanmean(d[:, j])), 4),
                                        "wins": int((d[:, j] > 0).sum()), "losses": int((d[:, j] < 0).sum()),
                                        "wilcoxon_p": round(float(stats.wilcoxon(d[np.isfinite(d[:, j]), j]).pvalue), 4)}
                                    for j, k in enumerate(("pcc", "p_up", "p_down"))}
        cnt = collections.Counter(tuple(J["selected"]["uot"][tg][str(i)][1:]) for i in range(2, T))
        lam_cnt = collections.Counter(J["selected"]["uot"][tg][str(i)][1] for i in range(2, T))
        res_t["uot_selected_lam0_counts_i>=2"] = {str(k): v for k, v in sorted(lam_cnt.items())}
        res_t["uot_selected_top_configs"] = [[list(k), v] for k, v in cnt.most_common(4)]
        out[name][tg] = res_t
        print(name, tg)
        for k, v in res_t.items():
            print("  ", k, json.dumps(v))
with open(os.path.join(HERE, "posthoc_down_side.json"), "w") as fh:
    json.dump(out, fh, indent=1)
