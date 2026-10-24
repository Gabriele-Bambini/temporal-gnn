"""Merge noise_acf.json, predictors.json, kalman_ml.json into results.json (+ a printed table).

Conventions: 'i>=1' = mean over transitions 1..T-1 (our default); 'all' = mean over all transitions incl. i = 0
(nanmean, the convention of the DistBridge / their-model numbers quoted in CONTEXT).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT = {
    "gastrulation-human": {"orig": {"DistBridge": (0.585, 0.406, 0.541), "theirs_same_transition": (0.386, 0.084, 0.367),
                                    "theirs_corrected": (0.313, 0.073, 0.329)}},
    "pancreas-human": {"orig": {"DistBridge": (0.488, 0.301, 0.442), "theirs_same_transition": (0.411, 0.155, 0.334),
                                "theirs_corrected": (0.249, 0.124, 0.255)},
                       "l1": {"DistBridge": (0.10, 0.17, 0.21), "DistBridge_noOT": (-0.05, 0.10, 0.15), "trend": (0.15, 0.21, 0.21)},
                       "linear": {"DistBridge": (0.30, 0.25, 0.30), "DistBridge_noOT": (-0.14, 0.06, 0.06), "trend": (0.40, 0.27, 0.35)}},
}
WINDOW_FAMILIES = {"level_box": 7, "lpoly_raw": 7, "ar_pooled": 5}   # first oracle index with no tied configs


def trio(x, pre=""):
    return [round(x[pre + "pcc"], 3), round(x[pre + "p200_up"], 3), round(x[pre + "p200_down"], 3)]


def main():
    na = json.load(open(os.path.join(HERE, "noise_acf.json")))
    pr = json.load(open(os.path.join(HERE, "predictors.json")))
    km = json.load(open(os.path.join(HERE, "kalman_ml.json")))
    lr = json.load(open(os.path.join(HERE, "lowrank_probe.json")))
    out = {"conventions": __doc__, "context_reference": CONTEXT, "datasets": {}}
    for name in pr:
        D = {"cells_per_snapshot": pr[name]["cells"], "n_genes": pr[name]["n_genes"]}
        # ---------------- signal structure
        sig = {}
        for res in ("native", "merged_pairs"):
            sig[res] = {}
            for t in ("orig", "l1", "linear"):
                x = na[name][res][t]
                e = {"xs_rho_lags": x["xs_rho"], "within_gene_rho_lags": x["within_gene_rho_all"],
                     "xs_rho1_by_expr_decile": x["xs_rho1_by_decile"],
                     "within_gene_rho1_by_expr_decile": {q: v["1"] for q, v in x["within_gene_rho_by_decile"].items()},
                     "xs_rho1_by_shared_bin_size_tercile": x["xs_rho1_by_shared_n_tercile"],
                     "xs_rho1_spearman_vs_shared_bin_size": x["xs_rho1_vs_shared_n_spearman"]}
                if "noise_budget" in x:
                    nb = x["noise_budget"]
                    S, Rv = nb["signal_share_mean"], nb["reversion_share_mean"]
                    rho = x["noise_corrected_xs_rho"]["1"]
                    e.update({"signal_share": S, "reversion_share": Rv, "next_bin_noise_share": nb["next_noise_share_mean"],
                              "ceiling_pcc_true_curve_known": nb["ceiling_pcc_mean_i>=1"],
                              "pcc_if_only_bin_t_noise_known": nb["pcc_denoise_only_mean"],
                              "pcc_if_only_true_change_known": nb["pcc_true_change_only_mean"],
                              "noise_corrected_rho_lags": x["noise_corrected_xs_rho"],
                              "pure_noise_expected_rho1": x["pure_noise_expected_rho1"],
                              "bound_pcc_if_current_true_level_known_and_AR1_true_change": float(np.sqrt(max(0.0, Rv + S * max(rho, 0) ** 2))),
                              "R2_secdiff_over_noise_median": nb["R2_median"],
                              "effective_n_genes_in_pearson": x["effective_n_genes_mean"],
                              "top1pct_genes_share_of_variance": x["top1pct_share_of_V"],
                              "by_decile": nb["by_decile"]})
                if t == "orig":
                    e.update({"xs_rho1_genes_without_zero_crossing": x["xs_rho1_no_zero_crossing"],
                              "clip_mass_expressed_genes": x["clip_mass_mean"], "frac_pairs_with_clip_atom": x["frac_pairs_with_atom"]})
                sig[res][t] = e
        D["signal"] = sig
        # ---------------- forecasters
        fc = {}
        for t in ("orig", "l1", "linear"):
            rows = {}
            for k, v in pr[name]["targets"][t]["families"].items():
                x = v["strict_selected"]
                hb = v["hindsight_best_fixed_NOT_STRICT"]
                rows[k] = {"strict_i>=1": trio(x), "strict_i>=1_sd": [round(x["pcc_sd"], 3), round(x["p200_up_sd"], 3), round(x["p200_down_sd"], 3)],
                           "strict_all": trio(x, "all_trans_"),
                           "hindsight_fixed_NOT_STRICT": {"config": hb["config"], "i>=1": trio(hb)},
                           "selection_agrees_with_oracle": round(v["selection_agrees_with_oracle"], 2),
                           "per_transition_from_i0": x["per_transition_from_i0"]}
            x = pr[name]["targets"][t]["meta"]
            rows["meta"] = {"strict_i>=1": trio(x), "strict_all": trio(x, "all_trans_"), "selected_family": x["selected_family"]}
            for space in ("raw", "l1"):
                for k in ("kd_RO", "ml_strat", "ml_gene", "obs_ml_strat", "obs_kd_bestJ_raw_RO"):
                    if k in km[name][space][t]:
                        x = km[name][space][t][k]
                        rows[f"kalman_{space}_{k}"] = {"strict_i>=1": trio(x), "strict_all": trio(x, "all_trans_"),
                                                       "per_transition_from_i0": x["per_transition_from_i0"]}
                hb = km[name][space][t]["kd_hindsight_best_NOT_STRICT"]
                rows[f"kalman_{space}_damped_hindsight_NOT_STRICT"] = {"config": hb["config"], "i>=1": trio(hb)}
            x = lr[name][t]["strict_RO"]
            rows["lowrank_crossgene_probe"] = {"strict_i>=1": trio(x), "strict_all": trio(x, "all_trans_"),
                                               "RO_selected": lr[name][t]["RO_selected"],
                                               "per_transition_from_i0": x["per_transition_from_i0"]}
            if t == "orig":
                x = lr[name][t]["obs_strict_RO"]
                rows["lowrank_crossgene_probe_obs"] = {"strict_i>=1": trio(x), "strict_all": trio(x, "all_trans_")}
            fc[t] = rows
        D["forecasters"] = fc
        D["kalman_damped_choice"] = {space: {k: km[name][space][k] for k in km[name][space] if k.startswith("kd_RO_") or k.startswith("ml_strat_")}
                                     for space in ("raw", "l1")}
        # ---------------- drift (ties removed for window families)
        dr = {}
        for k, dk in pr[name]["drift"].items():
            dr[k] = {}
            for t, x in dk.items():
                v = np.array(x["oracle_param_per_transition"], float)
                s0 = WINDOW_FAMILIES.get(k, 0)
                vv = v[s0:]
                dr[k][t] = {"oracle_param_sequence": v.tolist(), "from_index": s0,
                            "frac_consecutive_equal": float(np.mean(vv[1:] == vv[:-1])) if len(vv) > 1 else None,
                            "median_run_length": float(np.median(np.diff(np.flatnonzero(np.r_[True, vv[1:] != vv[:-1], True])))) if len(vv) > 1 else None,
                            "n_distinct": int(len(np.unique(vv))),
                            "selection_agrees_with_oracle": pr[name]["targets"][t]["families"][k]["selection_agrees_with_oracle"]}
        D["drift"] = dr
        D["kalman_innovations"] = pr[name]["kalman_innovations"]
        out["datasets"][name] = D
        # ---------------- print
        print(f"\n=== {name}  cells {D['cells_per_snapshot']}")
        for t in ("orig", "l1", "linear"):
            s = sig["native"][t]
            line = f"[{t}] rho(L1..L3) xs {s['xs_rho_lags']['1']:+.2f} {s['xs_rho_lags']['2']:+.2f} {s['xs_rho_lags']['3']:+.2f}"
            if "signal_share" in s:
                line += (f" | true-curve rho {s['noise_corrected_rho_lags']['1']:+.2f} {s['noise_corrected_rho_lags']['2']:+.2f}"
                         f" | shares signal {s['signal_share']:.2f} rev {s['reversion_share']:.2f} next {s['next_bin_noise_share']:.2f}"
                         f" | ceiling {s['ceiling_pcc_true_curve_known']:.2f} bound(mu_t known) {s['bound_pcc_if_current_true_level_known_and_AR1_true_change']:.2f}")
            print(line)
            ranked = sorted(((k, v) for k, v in fc[t].items() if "strict_i>=1" in v), key=lambda kv: -kv[1]["strict_i>=1"][0])
            for k, v in ranked[:6]:
                print(f"     {k:28s} i>=1 {v['strict_i>=1']}  all {v['strict_all']}")
            for k, v in CONTEXT[name].get(t, {}).items():
                print(f"     CONTEXT {k:20s} {list(v)}")
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
