"""Judge: free (non-forced) turnover of the population sink plans at the last strict fit (judge rerun JSON).
sink_abs: forced turnover = |N_src - N_dst|; free turnover = min(deaths, births) / N_src per pair.
sink_norm: equal totals, every unit of turnover is free: deaths / 1 per pair.
Usage: python turnover_p1.py <dataset>  ->  turnover_p1_<dataset>.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    name = sys.argv[1]
    r = json.load(open(os.path.join(HERE, "p1", f"uot_sink_{name}.json")))
    out = {}
    for K, ent in r["sink_interpretation_last_fit"].items():
        for k, v in ent.items():
            if "|lam" not in k:
                continue
            pp = v["per_pair"]
            D = np.array([x["death_total"] for x in pp])
            B = np.array([x["birth_total"] for x in pp])
            Tr = np.array([x["transported"] for x in pp])
            src = D + Tr
            free = np.minimum(D, B) / src
            out[f"{K}|{k}"] = {"free_turnover_share_median": round(float(np.median(free)), 4),
                               "free_turnover_share_max": round(float(free.max()), 4),
                               "free_turnover_share_mean": round(float(free.mean()), 4),
                               "pairs_free_gt_1pct": int((free > 0.01).sum()), "n_pairs": len(pp),
                               "total_turnover_share": v["turnover_share_of_source_mass"],
                               "spearman_turnover_vs_abs_dlogn": v["corr_gross_turnover_vs_abs_log_cellcount_ratio"]}
            print(name, K, k, json.dumps(out[f"{K}|{k}"]))
    with open(os.path.join(HERE, f"turnover_p1_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
