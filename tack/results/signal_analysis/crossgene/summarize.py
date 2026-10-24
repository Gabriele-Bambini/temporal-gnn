"""Collect the headline numbers of structure.json, composition.json, forecasts_<ds>.json and extra_svdkf_<ds>.json
into results.json (means over transitions i >= 1, 10 mask draws, strict unless the key says ORACLE / HINDSIGHT)."""
import json
import os

import common

OUT = common.OUT
HEAD = ("trend", "pergene_all", "pergene_kalman_scaled", "pergene_level_only", "svd_all", "svd_denoise_only",
        "svd_factor_change_only", "svd_lowrank_only", "svd_fullrank_pergene", "nmf_all", "nmf_persist_denoise",
        "nmf_linear_props", "nmf_ot_props", "nmf_ot_balanced", "nmf_uot", "nmf_change_only_ot",
        "nmf_change_only_linear", "nmf_K2", "nmf_K3", "nmf_K4", "nmf_K6", "nmf_K8", "svd_k1", "svd_k3", "svd_k4",
        "svd_k8", "svd_kfull", "blend_pergene+svd_all", "blend_pergene+nmf_all", "library_term_strict",
        "library_term_ORACLE", "ORACLE_svd_k3_zero", "ORACLE_svd_k3_last", "ORACLE_svd_k8_zero", "ORACLE_svd_k8_last",
        "ORACLE_nmf_K3_zero", "ORACLE_nmf_K3_last", "ORACLE_nmf_K8_zero", "ORACLE_nmf_K8_last",
        "pergene_all__HINDSIGHT_not_strict", "svd_all__HINDSIGHT_not_strict", "nmf_all__HINDSIGHT_not_strict")
DISTBRIDGE_REF = {"pancreas-human": {"l1": {"distbridge": [0.10, 0.17, 0.21], "distbridge_noOT": [-0.05, 0.10, 0.15],
                                            "trend_main_agent": [0.15, 0.21, 0.21]},
                                     "linear": {"distbridge": [0.30, 0.25, 0.30], "distbridge_noOT": [-0.14, 0.06, 0.06],
                                                "trend_main_agent": [0.40, 0.27, 0.35]}}}


def r3(x):
    return [round(x["pearson"], 3), round(x["p_at_200_up"], 3), round(x["p_at_200_down"], 3)]


def main():
    st = json.load(open(os.path.join(OUT, "structure.json")))
    co = json.load(open(os.path.join(OUT, "composition.json")))
    res = {"folder": OUT,
           "files": sorted(f for f in os.listdir(OUT) if f.endswith((".py", ".json", ".log"))),
           "protocol": "STRICT rolling origin: forecast of transition t->t+1 uses snapshots <= t only; config "
                       "selection per transition by mean J = PCC + P@200 up + down over the last 8 past transitions "
                       "(all genes); fallback per-gene trend at i <= 1. Scores: random 25 % test genes per "
                       "transition, 10 mask draws (seed 20261022), mean over transitions i >= 1; '| no_last' drops "
                       "the last transition. Triples are [PCC, P@200 up, P@200 down].",
           "datasets": {}}
    for ds in common.SNAPS:
        f = json.load(open(os.path.join(OUT, f"forecasts_{ds}.json")))
        x = json.load(open(os.path.join(OUT, f"extra_svdkf_{ds}.json")))
        S = st[ds]
        D = {"cells_per_snapshot": f["cells"],
             "structure": {
                 "noise_budget_l1_mean_i>=1": S["noise_budget_l1"]["mean_i>=1"],
                 "noise_budget_linear_mean_i>=1": S["noise_budget_linear"]["mean_i>=1"],
                 "noise_model_calibration_l1": S["noise_budget_l1"]["calibration_var_delta_on_predicted_noise"],
                 "noise_model_calibration_linear": S["noise_budget_linear"]["calibration_var_delta_on_predicted_noise"],
                 "raw_lag1_corr_of_changes": {t: S[f"noise_budget_{t}"]["mean_raw_lag1_corr"] for t in ("l1", "linear")},
                 "signal_lag1_corr_pooled_noise_corrected": {t: S[f"noise_budget_{t}"]["pooled_signal_lag1_corr"]
                                                             for t in ("l1", "linear")},
                 "svd_full_matrix": S["spectrum_full"],
                 "delta_l1_signal_share_in_topk_of_past_bins_le_t-1": {
                     k: S["delta_in_topk"]["mean_over_i>=1"][f"strict_level_bins_le_t-1_k{k}"]["pooled_signal_share"]
                     for k in (1, 2, 3, 5, 8)},
                 "delta_l1_signal_share_in_topk_of_past_changes_le_t-1": {
                     k: S["delta_in_topk"]["mean_over_i>=1"][f"strict_change_bins_le_t-1_k{k}"]["pooled_signal_share"]
                     for k in (1, 2, 3, 5, 8)},
                 "delta_l1_raw_energy_share_in_topk_ORACLE_all_bins": {
                     k: S["delta_in_topk"]["mean_over_i>=1"][f"oracle_all_bins_k{k}"]["pooled_raw"] for k in (1, 3, 8)},
                 "delta_l1_signal_share_of_energy_pooled": S["delta_in_topk"]["pooled_signal_share_of_delta_energy"],
                 "linear_target_library_term": S["library_term_linear"]["mean_i>=1"],
                 "linear_lib_sign_persistence": S["library_term_linear"]["lib_sign_persistence_i>=1"],
                 "nmf_cv_heldout_deviance": S["nmf_cv"]["heldout_poisson_deviance_per_entry"],
                 "nmf_cv_K": S["nmf_cv"]["K_within_1pct"],
                 "nmf_markers_at_cv_K": S["nmf_all_bins"]["markers"]},
             "composition_dynamics": {k: {kk: vv for kk, vv in v.items() if kk != "props"} for k, v in co[ds].items()},
             "forecasts": {}, "paired": {}}
        for t in ("l1", "linear"):
            M = f["methods"][t]
            tab = {k: {"i>=1": r3(M[k]["i>=1"]), "no_last": r3(M[k]["i>=1_no_last"])} for k in HEAD if k in M}
            for k in ("svdkf_all", "svdkf_avg3", "svdkf_all__HINDSIGHT_not_strict"):
                tab[k] = {"i>=1": r3(x["methods"][t][k]["i>=1"]), "no_last": r3(x["methods"][t][k]["i>=1_no_last"])}
            if ds in DISTBRIDGE_REF:
                tab["REFERENCE_main_agent_numbers"] = DISTBRIDGE_REF[ds][t]
            D["forecasts"][t] = tab
            P = f["paired"][t]
            pick = {k: P[k] for k in ("pergene_all", "svd_all", "nmf_all", "nmf_ot_props", "nmf_linear_props",
                                      "nmf_persist_denoise", "svd_denoise_only", "svd_factor_change_only")}
            pick.update({k: P[k] for k in ("nmf_ot_props_vs_nmf_linear_props", "nmf_ot_props_vs_nmf_persist_denoise",
                                           "nmf_change_only_ot_vs_linear", "svd_lowrank_vs_fullrank")})
            pick.update(x["paired"][t])
            D["paired"][t] = pick
            D.setdefault("partial_corr_given_pergene_selected", {})[t] = f["partial_corr_given_pergene"][t]
        res["datasets"][ds] = D
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    for ds, D in res["datasets"].items():
        for t, tab in D["forecasts"].items():
            print("==", ds, t)
            for k, v in tab.items():
                print(f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
