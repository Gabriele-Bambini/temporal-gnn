"""Merge every output of this folder into results.json with a headline table per dataset x target.

Inputs: results_summary.json (summarize.py), results_posthoc.json,
results_trendfit.json, results_kalman.json, results_regime.json. Strict = only bins <= t; oracle / descriptive
rows are non-causal and labelled. 'i>=1' / 'i>=2' / 'i>=3' = transitions averaged over.
"""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))


def load(f):
    return json.load(open(os.path.join(OUT, f)))


def pick(d, keys=("pearson_nan0", "p_at_200_up", "p_at_200_down")):
    return {k: round(d[k], 3) for k in keys}


def main():
    S, PH, TF, KF, RG = (load(f) for f in ("results_summary.json", "results_posthoc.json", "results_trendfit.json",
                                            "results_kalman.json", "results_regime.json"))
    head = {}
    for ds in ("gastrulation-human", "pancreas-human"):
        head[ds] = {}
        for tg in ("orig", "l1", "linear"):
            a, ph = S[ds][tg], PH[ds][tg]
            pr = a["predictors"]
            h = {
                "(a) zero-crossings": {
                    "ss_share_UP_plus_DN": round(a["ss_share"]["UP"] + a["ss_share"]["DN"], 3),
                    "R2_of_4group_oracle (BZ/UP/DN/EX known, descriptive)": round(a["R2_zc_groups"], 3),
                },
                "(b) global factor": {
                    "oracle_knows_L_next_pearson_i>=1": round(pr["oracle_scalar_L"]["i>=1"]["pearson_nan0"], 3),
                    "oracle_k1_k3_k10_scalars_on_past_loadings_pearson_i>=1": [round(pr[f"oracle_k{k}_scalars_pastloadings"]["i>=1"]["pearson_nan0"], 3) for k in (1, 3, 10)],
                    "descriptive_function_of_level_pearson": round(pr["descr_level_fn"]["all"]["pearson_nan0"], 3),
                    "strict_function_of_level_pearson_i>=1": round(pr["strict_level_fn"]["i>=1"]["pearson_nan0"], 3),
                },
                "(c) finite-sample noise (share of per-transition target variance, mean over transitions; median in brackets)": {
                    "next_bin": f"{a['noise_share_next_smooth']:.3f} [{a['noise_share_next_smooth_median']:.3f}]",
                    "current_bin": f"{a['noise_share_now_smooth']:.3f} [{a['noise_share_now_smooth_median']:.3f}]",
                    "both_bins": f"{a['noise_share_both_smooth']:.3f} [{a['noise_share_both_smooth_median']:.3f}]",
                    "plugin_law_next_bin": round(a["noise_share_next_plugin"], 3),
                    "second_diff_obs_over_noise_pred_median": round(ph["second_diff_obs_over_noise_median"], 2),
                    "true_change_persistence_median": round(ph["true_change_persistence_median"], 2),
                    "N_eff_genes_median": round(ph["N_eff_genes_median"]),
                    "genes_for_50pct_of_SS_median": ph["genes_for_50pct_ss_median"],
                },
                "ceilings (non-causal)": {
                    "pearson_ceiling_sqrt(1-noise_next)": round(a["pearson_ceiling_next_smooth"], 3),
                    "model_world (truth = smoothed law, E[T] vs fresh draws)": pick(a["modelworld_ceiling"]),
                    "oracle_next_bin_interpolated_without_bin_t+1 (best of 2), i>=1": max(
                        (pick(pr[k]["i>=1"]) for k in ("oracle_next_interp1_nb", "oracle_next_interp2_nb")), key=lambda d: d["pearson_nan0"]),
                    "oracle_knows_n_next (level law, finite-cell obs), i>=1": pick(pr["oracle_n_level_obs_nb"]["i>=1"]),
                    "oracle_coef_per_gene_history_by_level (in-sample coefficients), i>=1": pick(ph["per_gene_history"]["oracle_coef_by_level"]),
                },
                "strict": {
                    "level4_law_finite_cell_obs_n_t, i>=1": pick(pr["strict_level_obs_na"]["i>=1"]),
                    "repeat_last_change, i>=1": pick(pr["strict_trend1"]["i>=1"]),
                    "lowrank3_projected_last_change, i>=1": pick(pr["strict_lowrank3_trend"]["i>=1"]),
                    "local_polynomial_rolling_selected, i>=2": pick(TF[ds][tg]["strict_rolling_selected_i>=2"]),
                    "per_gene_history_pooled_coef_by_level, i>=2": pick(ph["per_gene_history"]["strict_coef_by_level"]),
                },
            }
            if tg in ("l1", "linear"):
                h["strict"]["kalman_known_noise_rolling_selected, i>=2"] = pick(KF[ds][tg]["strict_rolling_selected_i>=2"])
                h["strict"]["per_gene_history_recency_halflife0.5_by_level, i>=2"] = pick(RG[ds][tg]["halflife_0.5_i>=2"])
                h["strict"]["per_gene_history_recency_rolling_selected, i>=3"] = pick(RG[ds][tg]["rolling_selected_halflife_i>=3"])
            if tg == "orig":
                h["(a) zero-crossings"]["mean_genes_at_+5 / -5 (all genes)"] = [round(a["ties"]["mean_n_at_+5"]), round(a["ties"]["mean_n_at_-5"])]
                h["(a) zero-crossings"]["tie_limited_P@200_max_up/down"] = [round(a["ties"]["p_max_up_mean"], 3), round(a["ties"]["p_max_down_mean"], 3)]
            if tg == "linear":
                h["(b) global factor"]["oracle_best_scalar_on_m_a_abs_corr"] = round(ph["oracle_abs_corr_y_m_a_mean"], 3)
            head[ds][tg] = h
        z = S[ds]["zero_crossing_counts"]
        head[ds]["zero_crossing_counts_vs_pure_sampling"] = {
            "DN (>0 -> 0) observed": z["obs_DN_total"],
            "DN expected from (1-f)^n_next, law of t+1 interpolated from t and t+2": round(z["expected_DN_from_sampling_interp_noB"]),
            "UP (0 -> >0) observed": z["obs_UP_total"],
            "UP expected, same law": round(z["expected_UP_from_sampling_interp_noB"]),
        }
        head[ds]["lag1_corr_of_consecutive_changes"] = S[ds]["lag1_corr_mean"]
        pf = {}
        for tg in ("orig", "l1", "linear"):
            h, a = head[ds][tg], S[ds][tg]
            mw = h["ceilings (non-causal)"]["model_world (truth = smoothed law, E[T] vs fresh draws)"]["pearson_nan0"]
            io = h["ceilings (non-causal)"]["oracle_next_bin_interpolated_without_bin_t+1 (best of 2), i>=1"]["pearson_nan0"]
            st = {k: v["pearson_nan0"] for k, v in h["strict"].items()}
            bk = max(st, key=st.get)
            pf[tg] = {"R2_upper_1_minus_noise_next_mean": round(1 - a["noise_share_next_smooth"], 3),
                      "R2_upper_1_minus_noise_next_median": round(1 - a["noise_share_next_smooth_median"], 3),
                      "R2_model_world_ceiling": round(mw ** 2, 3),
                      "R2_noncausal_interp_oracle": round(io ** 2, 3),
                      "R2_best_strict": round(st[bk] ** 2, 3), "best_strict": bk,
                      "R2_current_bin_noise_share (recoverable only by denoising m_t)": round(a["noise_share_now_smooth"], 3),
                      "R2_true_change_share (1 - noise both bins)": round(1 - a["noise_share_both_smooth"], 3)}
        head[ds]["predictable_fraction (R2 = squared Pearson; masks differ between scripts, +/-0.01)"] = pf
    out = {"headline": head, "summary": S, "posthoc": PH, "trendfit": TF, "kalman": KF, "regime": RG,
           "files": {"variance_decomp.py": "main decomposition (results_raw.json)", "summarize.py": "aggregation",
                     "posthoc.py": "per-transition-normalised tables, persistence, per-gene history ceilings",
                     "trend_fit.py": "strict local polynomial forecasts", "kalman_fit.py": "strict heteroscedastic Kalman",
                     "regime_fit.py": "recency-weighted (regime-adaptive) per-gene history", "assemble.py": "this merge"}}
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
