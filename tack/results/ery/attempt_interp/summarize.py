"""Compact summary of interp_bench.py outputs (reads table.csv, pools.csv, results.json, batch_matched.csv,
sensitivity_oracle.csv in this folder; writes headline.csv and summary.txt). No new computation on cells."""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = ["prev", "next", "union", "indep", "ot_pairs", "ot_bary", "ot_bary_sym", "sb", "sb_half", "uot",
        "vel_ss", "vel_dyn", "vel_ss_cED", "vel_ss_back", "vel_dyn_back"]
out_lines = []


def p(*a):
    s = " ".join(str(x) for x in a)
    out_lines.append(s)
    print(s)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    t = pd.read_csv(os.path.join(HERE, "table.csv"))
    pools = pd.read_csv(os.path.join(HERE, "pools.csv"))
    res = json.load(open(os.path.join(HERE, "results.json")))
    ks = sorted(t["k"].unique())
    rows = []
    for met in ("sd_fixed", "ED_U", "W2", "W1", "ED_V", "ED_U_gene", "comp_tv", "spread_ratio"):
        pv = t.pivot(index="method", columns="k", values=met).reindex(MAIN)
        pv["mean"] = pv[ks].mean(1)
        sdv = t.pivot(index="method", columns="k", values=met + "_sd").reindex(MAIN)
        p(f"\n== {met} (mean over reps; rep sd max {np.nanmax(sdv.values):.3f}) ==")
        p(pv.round(3).to_string())
        for m in MAIN:
            for k in ks:
                rows.append({"metric": met, "method": m, "k": k, "held": t[t.k == k]["held"].iloc[0],
                             "value": pv.loc[m, k], "rep_sd": sdv.loc[m, k]})
    p("\n== floors (DESCRIPTIVE; use the held stage) ==")
    fl = []
    for k in ks:
        f = res["k"][str(k)]["floors"]
        best_ed = t[t.k == k].set_index("method").loc[MAIN, "ED_U"]
        fl.append({"k": k, "held": t[t.k == k]["held"].iloc[0], "loso_ED_U_mean": f["pca"]["loso_ED_U_mean"],
                   "loso_ED_U_min": f["pca"]["loso_ED_U_min"], "loso_ED_U_max": f["pca"]["loso_ED_U_max"],
                   "oracle_ED_U": f["pca"]["oracle_ED_U"], "halves_ED_U": f["pca"]["halves_ED_U"],
                   "own_loso_sd_mean": f["own_loso_mean"]["sd_fixed"], "own_halves_sd": f["own_halves"]["sd_fixed"],
                   "own_halves_W2": f["own_halves"]["W2"], "halves_n_each": f["own_halves"]["n_each"],
                   "best_ED_U": float(best_ed.min()), "best_ED_U_method": str(best_ed.idxmin()),
                   "best_over_loso": float(best_ed.min() / f["pca"]["loso_ED_U_mean"]),
                   "best_over_oracle": float(best_ed.min() / f["pca"]["oracle_ED_U"]),
                   "union_over_loso": float(best_ed["union"] / f["pca"]["loso_ED_U_mean"])})
    fl = pd.DataFrame(fl)
    p(fl.round(3).to_string())
    p("\n== verdict summary ==")
    for kk, v in res["verdicts"]["summary"].items():
        p(f"{kk}: {v}")
    p("\n== pool consistency (ED_U per held pool; does the method beat prev, next AND union in every pool?) ==")
    for k, v in res["verdicts"]["pool_consistency"].items():
        p(f"k={k}: best per pool {v['best_method_per_pool']}")
        for m, c in v["vs_prev_next_union"].items():
            p(f"   {m}: {c['pools_won']} all={c['all_pools']}")
    pv = pools[pools.method.isin(["prev", "next", "union", "indep", "ot_pairs", "ot_bary_sym", "sb_half", "uot",
                                  "vel_ss"])]
    p("\n== per pool ED_U (and the pool's leave-one-sample-out reference) ==")
    p(pv.pivot_table(index=["k", "pool", "batch", "n", "loso_ED_U"], columns="method", values="ED_U").round(2)
      .to_string())
    bm = os.path.join(HERE, "batch_matched.csv")
    if os.path.exists(bm):
        b = pd.read_csv(bm)
        p("\n== batch-matched block (target = held batch-2 cells; 1 rep) ==")
        for met in ("sd_fixed", "ED_U", "W2"):
            p(f"-- {met}")
            p(b.pivot_table(index=["k", "method"], columns="fit", values=met).round(3).unstack(0).to_string())
    se = os.path.join(HERE, "sensitivity_oracle.csv")
    if os.path.exists(se):
        s = pd.read_csv(se)
        p("\n== sensitivity grid (ORACLE-LABELLED: chosen by looking at the held stage; 1 rep) ==")
        p(s.pivot_table(index=["grid", "setting", "method"], columns="k", values="ED_U").round(3).to_string())
        p(s.pivot_table(index=["grid", "setting", "method"], columns="k", values="sd_fixed").round(3).to_string())
    p("\n== gene level (mean change from k-1; balanced methods share one direction) ==")
    p(t.pivot(index="method", columns="k", values="pcc").reindex(MAIN).round(3).to_string())
    p(t.pivot(index="method", columns="k", values="dev_pcc").reindex(MAIN).round(3).to_string())
    p("\n== velocity c (strict fit on k-1 <-> k+1) and share of |V| inside the PCA ==")
    for k in ks:
        r = res["k"][str(k)]
        p(k, r["velocity_fit"], {kk: round(v, 3) for kk, v in r["velocity_pca_share"].items()})
    p("\n== checks ==")
    p("gene list origin2 corr", res.get("gene_list_check_origin2_corr"))
    for k in ks:
        r = res["k"][str(k)]
        p(k, "cc corr origin5", round(r["cc_corr_origin5"], 3), "knn acc", round(r["knn_accuracy_on_held"], 3),
          "endpoint", {kk: round(v, 4) for kk, v in r["reps"][0]["endpoint_check_ED_U"].items()},
          "uot", r["reps"][0]["uot"], "plan", r["reps"][0]["plan"])
    p("solver", res.get("stats"), "seconds", res.get("seconds_total"))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "headline.csv"), index=False)
    fl.to_csv(os.path.join(HERE, "floors_vs_best.csv"), index=False)
    with open(os.path.join(HERE, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
