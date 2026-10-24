"""Collect the headline numbers of grn_signal / grn_check / grn_combo / artefacts / heterogeneity / followups into
results.json (every number is read from those files, nothing is recomputed here)."""
import json
import os

import common

J = {k: json.load(open(os.path.join(common.OUT, f"{k}.json"))) for k in
     ("grn_signal", "grn_check", "grn_combo", "artefacts", "heterogeneity", "followups")}


def r3(x):
    return round(x, 3) if isinstance(x, float) else x


def pick(sc):
    return {k: r3(sc[k]) for k in ("pcc", "p200_up", "p200_down")}


out = {"note": "STRICT, rolling origin; metrics = Pearson / P@200 up / down on 25 % random test genes, mean over "
               "transitions (eval windows in the source files). Detailed per-transition values in the source jsons.",
       "datasets": {}}
for name in common.SNAPS:
    g, c, cb, a, h, f = (J[k][name] for k in ("grn_signal", "grn_check", "grn_combo", "artefacts", "heterogeneity", "followups"))
    ds = {}
    ds["grn_coverage_frac_genes_with_regulator_mean"] = r3(sum(x["frac_with_regulator"] for x in g["coverage"]) / len(g["coverage"]))
    for T in common.TARGETS:
        t = {}
        A = g["A_partial_corr"][T]["mean"]
        t["partial_corr_lag1"] = {k: r3(A[f"partial_{k}"]) for k in ("grn", "grn_top10", "shuf", "prof", "knn")}
        t["partial_corr_lag0_NONSTRICT"] = {k: r3(A[f"lag0_partial_{k}"]) for k in ("g", "shuf", "prof", "knn")}
        t["partial_corr_check"] = {k: (r3(v["mean"]), f"{v['n_positive']}/{v['n']}") for k, v in c[T]["partial_summary"].items()}
        t["strict_ridge"] = {k: pick(v) for k, v in g["B_strict_ridge"][T].items()}
        t["strict_ridge_lean"] = {k: pick(v) for k, v in c[T]["strict_ridge"].items()}
        t["strict_ridge_with_noise_weighting"] = {k: pick(v) for k, v in cb[T].items()}
        t["noise_ceiling"] = {k: r3(v) for k, v in a["ceiling_vs_reference"][T].items()}
        t["lag1_corr_last_change_by_quintile"] = {k: r3(v) for k, v in f[T]["F1_lag1_corr_by_quintile"].items()}
        t["followup_ridge"] = {k: pick(v) for k, v in f[T]["F2_F3_strict_ridge"].items()}
        het = {}
        for part in ("crossing", "expr", "top1", "outdeg", "frac", "indeg"):
            het[part] = {grp: {k: r3(v) for k, v in vals.items()
                               if k in ("share_genes", "share_var", "share_true_top_up", "share_true_top_down")
                               or k.endswith("pcc_contrib") or k.endswith("pcc_without_group")}
                         for grp, vals in h[T][part].items()}
        t["heterogeneity"] = het
        t["reference_all_genes_k800"] = {k: {kk: r3(vv) for kk, vv in v.items()} for k, v in h[T]["_all"].items()}
        ds[T] = t
    ds["turn_on"] = {"strict_logistic_auc": {k: r3(v["mean_auc"]) for k, v in g["C_turn_on"]["strict_logistic"].items()},
                     "single_feature_auc": {k: r3(v) for k, v in g["C_turn_on"]["mean_single_feature"].items()}}
    ds["hub_expression"] = {k: r3(v) for k, v in h["hub_expression"].items()}
    ds["binning"] = {"n_prediction_abs_log_error": {k: r3(v) for k, v in a["n_prediction_abs_log_error"].items()},
                     "reference_strict": {k: pick(v) for k, v in a["reference_strict"].items()},
                     "library": {k: r3(v) for k, v in a["library"].items()},
                     "cross_transition_spearman": a["cross_transition_spearman"],
                     "tiny_bins": a["tiny_bins"],
                     "mean_over_transitions": {k: r3(v) for k, v in a["mean_over_transitions"].items()},
                     "per_transition_brief": [{k: r[k] for k in ("transition", "n_a", "n_b", "log_lib_ratio", "frac_on", "frac_off",
                                                                 "orig_test_ties_at_max", "orig_p200_ceiling_ties_up",
                                                                 "orig_pcc_ceiling", "l1_pcc_ceiling", "linear_pcc_ceiling",
                                                                 "linear_corr_y_with_m_a")} for r in a["per_transition"]]}
    out["datasets"][name] = ds
with open(os.path.join(common.OUT, "results.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print("written", os.path.join(common.OUT, "results.json"))
