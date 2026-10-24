"""Aggregate results_raw.json (variance_decomp.py) into results_summary.json and print the tables.

Averages are over transitions: 'all' = every transition, 'i>=1' = transitions with a past change (the strict
trend predictors exist only there). Per-transition series of the key quantities are kept for inspection.
"""
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
TARGETS = ("orig", "l1", "linear")


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def main():
    raw = json.load(open(os.path.join(OUT, "results_raw.json")))
    out = {"settings": raw["settings"]}
    for name in ("gastrulation-human", "pancreas-human"):
        R = raw[name]
        trs = R["transitions"]
        ds = {"n_genes": R["n_genes"], "cells_per_bin": R["cells_per_bin"], "library_per_bin": R["library_per_bin"],
              "n_transitions": len(trs)}
        ds["zero_crossing_counts"] = {
            "mean_frac_genes": {k: mean([t["group_frac"][k] for t in trs]) for k in ("BZ", "UP", "DN", "EX")},
            "obs_DN_total": int(sum(t["obs_DN"] for t in trs)),
            "obs_UP_total": int(sum(t["obs_UP"] for t in trs)),
        }
        for lab in ("interp_noB", "plugin_B", "smooth_B"):
            ds["zero_crossing_counts"][f"expected_DN_from_sampling_{lab}"] = float(sum(t[f"exp_DN_{lab}"] for t in trs))
            ds["zero_crossing_counts"][f"expected_UP_from_sampling_{lab}"] = float(sum(t[f"exp_UP_{lab}"] for t in trs))
        lag = [t["lag1_linear"] for t in trs if "lag1_linear" in t]
        ds["lag1_linear_check"] = {
            "sum_cov_obs": float(sum(x["cov_obs"] for x in lag)),
            "sum_cov_pred_by_noise": float(sum(x["cov_noise_pred"] for x in lag)),
            "mean_corr_obs": mean([x["cov_obs"] / np.sqrt(x["var_y_i"] * x["var_y_next"]) for x in lag]),
            "mean_corr_pred_by_noise": mean([x["cov_noise_pred"] / np.sqrt(x["var_y_i"] * x["var_y_next"]) for x in lag]),
        }
        ds["lag1_corr_mean"] = {tg: mean([t["lag1_corr"][tg] for t in trs if "lag1_corr" in t]) for tg in TARGETS}
        for tg in TARGETS:
            S = [t[tg] for t in trs]
            agg = {"ss_share": {k: mean([s["ss_share"][k] for s in S]) for k in ("BZ", "UP", "DN", "EX")},
                   "R2_zc_groups": mean([s["R2_zc_groups"] for s in S])}
            for k in ("noise_share_next_plugin", "noise_share_next_smooth", "noise_share_now_smooth",
                      "noise_share_both_smooth", "noise_share_analytic_next", "noise_share_analytic_now",
                      "pearson_ceiling_next_plugin", "pearson_ceiling_next_smooth"):
                if k in S[0]:
                    agg[k] = mean([s[k] for s in S])
                    agg[k + "_median"] = float(np.median([s[k] for s in S]))
            agg["modelworld_ceiling"] = {k: mean([s["modelworld_ceiling"][k] for s in S]) for k in S[0]["modelworld_ceiling"]}
            if tg == "orig":
                agg["ties"] = {"mean_n_at_+5": mean([s["n_at_+5"] for s in S]), "mean_n_at_-5": mean([s["n_at_-5"] for s in S]),
                               "mean_n_at_0": mean([s["n_at_0"] for s in S]),
                               "p_max_up_mean": mean([s["pred"]["strict_level_point"]["p_max_up"] for s in S]),
                               "p_max_down_mean": mean([s["pred"]["strict_level_point"]["p_max_down"] for s in S])}
            preds = sorted({k for s in S for k in s["pred"]})
            pa = {}
            for p in preds:
                rows_all = [s["pred"][p] for s in S if p in s["pred"]]
                rows_1 = [s["pred"][p] for j, s in enumerate(S) if p in s["pred"] and j >= 1]
                pa[p] = {"all": {k: mean([r[k] for r in rows_all]) for k in ("pearson_nan0", "pearson_all_genes", "p_at_200_up", "p_at_200_down", "p_ta_up", "p_ta_down")},
                         "i>=1": {k: mean([r[k] for r in rows_1]) for k in ("pearson_nan0", "pearson_all_genes", "p_at_200_up", "p_at_200_down", "p_ta_up", "p_ta_down")},
                         "n_transitions": len(rows_all)}
            agg["predictors"] = pa
            agg["per_transition"] = [{"bins": t["bins"], "n_a": t["n_a"], "n_b": t["n_b"], "log_L_ratio": round(t["log_L_ratio"], 4),
                                      "V_tot": t[tg]["V_tot"],
                                      "ss_share_ZC": t[tg]["ss_share"]["UP"] + t[tg]["ss_share"]["DN"],
                                      "noise_next_smooth": t[tg]["noise_share_next_smooth"],
                                      "noise_now_smooth": t[tg]["noise_share_now_smooth"],
                                      "ceil_next_smooth": t[tg]["pearson_ceiling_next_smooth"],
                                      "mw_pearson": t[tg]["modelworld_ceiling"]["pearson_nan0"],
                                      "mw_p_up": t[tg]["modelworld_ceiling"]["p_at_200_up"],
                                      "mw_p_down": t[tg]["modelworld_ceiling"]["p_at_200_down"],
                                      "oracle_interp2": t[tg]["pred"]["oracle_next_interp2_nb"]["pearson_nan0"],
                                      "strict_level_obs": t[tg]["pred"]["strict_level_obs_na"]["pearson_nan0"],
                                      "trend1": t[tg]["pred"].get("strict_trend1", {}).get("pearson_nan0"),
                                      **({"n_at_+5": t[tg]["n_at_+5"], "n_at_-5": t[tg]["n_at_-5"]} if tg == "orig" else {})}
                                     for t in trs]
            # across-transition association of noise with cell counts
            nb = np.array([t["n_b"] for t in trs], float)
            na = np.array([t["n_a"] for t in trs], float)
            ns = np.array([t[tg]["noise_share_next_smooth"] for t in trs])
            zc = np.array([t[tg]["ss_share"]["UP"] + t[tg]["ss_share"]["DN"] for t in trs])
            agg["spearman_noise_next_vs_n_b"] = float(_spear(ns, nb))
            agg["spearman_ZCshare_vs_log_nb_over_na"] = float(_spear(zc, np.log(nb / na)))
            ds[tg] = agg
        ds["snr"] = R["snr"]
        out[name] = ds
    with open(os.path.join(OUT, "results_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    show(out)


def _spear(a, b):
    from scipy import stats
    return stats.spearmanr(a, b).correlation


def show(out):
    for name in ("gastrulation-human", "pancreas-human"):
        ds = out[name]
        print(f"\n===== {name}  genes={ds['n_genes']}  transitions={ds['n_transitions']}")
        print(" groups (mean frac genes):", {k: round(v, 3) for k, v in ds["zero_crossing_counts"]["mean_frac_genes"].items()})
        z = ds["zero_crossing_counts"]
        print(f" DN obs {z['obs_DN_total']}  exp(interp,noB) {z['expected_DN_from_sampling_interp_noB']:.0f}  exp(plugin B) {z['expected_DN_from_sampling_plugin_B']:.0f}  exp(smooth B) {z['expected_DN_from_sampling_smooth_B']:.0f}")
        print(f" UP obs {z['obs_UP_total']}  exp(interp,noB) {z['expected_UP_from_sampling_interp_noB']:.0f}  exp(plugin B) {z['expected_UP_from_sampling_plugin_B']:.0f}  exp(smooth B) {z['expected_UP_from_sampling_smooth_B']:.0f}")
        print(" lag1 linear:", {k: round(v, 4) for k, v in ds["lag1_linear_check"].items()}, " lag1 corr:", {k: round(v, 3) for k, v in ds["lag1_corr_mean"].items()})
        for tg in TARGETS:
            a = ds[tg]
            print(f" --- target {tg}")
            print("  ss share:", {k: round(v, 3) for k, v in a["ss_share"].items()}, " R2 zc groups", round(a["R2_zc_groups"], 3))
            print("  noise share next(plug) %.3f next(smooth) %.3f [median %.3f] now(smooth) %.3f both %.3f" % (
                a["noise_share_next_plugin"], a["noise_share_next_smooth"], a["noise_share_next_smooth_median"],
                a["noise_share_now_smooth"], a["noise_share_both_smooth"]))
            if "noise_share_analytic_next" in a:
                print("  analytic linear noise next %.3f now %.3f" % (a["noise_share_analytic_next"], a["noise_share_analytic_now"]))
            print("  pearson ceiling next(plug) %.3f next(smooth) %.3f" % (a["pearson_ceiling_next_plugin"], a["pearson_ceiling_next_smooth"]))
            print("  model-world ceiling:", {k: round(v, 3) for k, v in a["modelworld_ceiling"].items()})
            if tg == "orig":
                print("  ties:", {k: round(v, 3) for k, v in a["ties"].items()})
            print("  spearman(noise_next, n_b) %.2f  spearman(ZC share, log nb/na) %.2f" % (a["spearman_noise_next_vs_n_b"], a["spearman_ZCshare_vs_log_nb_over_na"]))
            for p, v in a["predictors"].items():
                r = v["i>=1"]
                print(f"   {p:38s} i>=1: r={r['pearson_nan0']:.3f} P@200 up={r['p_at_200_up']:.3f} dn={r['p_at_200_down']:.3f} | all: r={v['all']['pearson_nan0']:.3f} up={v['all']['p_at_200_up']:.3f} dn={v['all']['p_at_200_down']:.3f}")
        for tg in TARGETS:
            print(f"  SNR {tg} by level:", [(r["level_range"], round(r["share_of_target_ss"], 3), None if r["signal_fraction"] is None else round(r["signal_fraction"], 3)) for r in ds["snr"][tg]["by_level"]])
            print(f"  SNR {tg} by n:", [(r["n_min_range"], r["n_transitions"], round(r["signal_fraction"], 3)) for r in ds["snr"][tg]["by_n"]])
        print("  linear analytic by level:", [(r["level_range"], None if r["signal_fraction"] is None else round(r["signal_fraction"], 3)) for r in ds["snr"]["linear_analytic_by_level"]])


if __name__ == "__main__":
    main()
