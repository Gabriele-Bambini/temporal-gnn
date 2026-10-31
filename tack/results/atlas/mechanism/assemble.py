"""EXPLORATORY: merge the mechanism parts into mechanism.json with cross-unit summaries (no new computation on cells).
Inputs (this folder): check_repro.json (check_repro.py), interp_mechanism.json (interp_mechanism.py),
within_lineage.json (within_lineage.py), forecast_oracle.json (forecast_oracle.py); confirm.json read-only.
Usage: python assemble.py   -> mechanism.json
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

import common as C


def load(name):
    p = os.path.join(C.HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    ref = C.confirm()
    rep, im, wl, fo = (load(x) for x in ("check_repro.json", "interp_mechanism.json", "within_lineage.json",
                                          "forecast_oracle.json"))
    out = {"title": "Atlas mechanism analysis (EXPLORATORY)",
           "status": "EXPLORATORY, not pre-registered. The pre-registered verdicts in confirm.json (H1 FAIL, H1b FAIL, "
                     "H1c PASS, H2 FAIL, H2b FAIL) are final and unchanged by anything here. ORACLE-labelled rows use the "
                     "held / future stage to choose a setting and are never valid as methods.",
           "scripts": {"check_repro.py": "local reproduction of confirm.json", "interp_mechanism.py": "Q1 composition + "
                       "interpolation geometry", "within_lineage.py": "Q2 within-lineage rerun of the frozen methods",
                       "forecast_oracle.py": "Q3 oracle forecasting grid", "assemble.py": "this file",
                       "spread_eps.py": "Q1 dispersion of the OT midpoint clouds + ORACLE eps_rel scan",
                       "common.py": "helpers (mirror of atlas_confirm.run_interp with masks)"},
           "data": C.DATA, "reproduction": rep}
    ks = sorted(im["k"], key=int) if im else []
    # ---------------- Q1 composition + mechanism tables
    if im:
        q1, geo, pools, vote = {}, {}, [], {}
        for k in ks:
            r = im["k"][k]
            c = r["composition"]
            q1[k] = {x: round(c[x], 4) for x in ("TV_held_prev", "TV_held_next", "TV_held_mix5050", "best_alpha_next",
                                                 "TV_held_best_alpha_mix", "TV_prev_next",
                                                 "share_of_TV_prev_next_from_one_sided_types",
                                                 "next_mass_types_absent_at_prev(appearing)",
                                                 "prev_mass_types_absent_at_next(vanishing)",
                                                 "held_mass_types_absent_at_prev", "held_mass_types_absent_at_next",
                                                 "held_transient_peak_excess", "held_mass_in_types_between_neighbours")}
            q1[k]["held_closer_to_mix_than_both_neighbours"] = bool(c["TV_held_mix5050"] < min(c["TV_held_prev"], c["TV_held_next"]))
            e = r["ed_identity"]
            ts, al, spr = r["t_scan_ORACLE"], r["alpha_scan_ORACLE"], r["t_spread_scan_ORACLE"]
            om = r["off_manifold"]["methods"]
            geo[k] = {"union": e["recorded_union_ED_U"], "ot_pairs": e["recorded_ot_pairs_ED_U"], "uot": e["recorded_uot_ED_U"],
                      "prev": e["ED_prev_held"], "next": e["ED_next_held"], "ED_prev_next": e["ED_prev_next"],
                      "quarter_ED_prev_next(bonus)": e["convexity_bonus_quarter_ED_prev_next"],
                      "identity_pred_union": e["identity_pred_union"],
                      "ORACLE_best_t": ts["best_t"], "ORACLE_best_t_ED_U": ts["best_ED_U"], "t0.5_own_draw_ED_U": ts["at_t0.5"],
                      "ORACLE_best_alpha": al["best_alpha"], "ORACLE_best_alpha_ED_U": al["best_ED_U"],
                      "ORACLE_best_spread": spr["best"], "ORACLE_best_spread_ED_U": spr["best_ED_U"],
                      "spread_uniform_h0.5": spr["ED_U"]["uniform_h0.5"],
                      "off_manifold_frac": {m: round(om[m]["frac_off_manifold"], 3) for m in ("prev", "next", "union", "ot_pairs", "uot", "indep")},
                      "target_uncovered_frac": {m: round(om[m]["frac_target_uncovered"], 3) for m in ("prev", "next", "union", "ot_pairs", "uot", "indep")},
                      "plan_mass_same_type": r["pair_types"]["plan_mass_same_type"],
                      "off_manifold_same_vs_cross_type_midpoints": [r["pair_types"]["off_manifold_same_type_midpoints"],
                                                                    r["pair_types"]["off_manifold_cross_type_midpoints"]]}
            sv = r["stage_vote"]
            vote[k] = {"classifier_acc_prev_next": [round(sv["acc_on_prev_rep0"], 3), round(sv["acc_on_next_rep0"], 3)],
                       "intermediate_frac": {m: round(sv[m]["frac_intermediate[0.2,0.8]"], 3) for m in ("held_all", "union", "ot_pairs", "uot")},
                       "held_mean_P_next": round(sv["held_all"]["mean_P_next"], 3),
                       "held_pool_mean_P_next": {p: round(v, 3) for p, v in sv["held_per_pool_mean_P_next"].items()}}
            for p, po in r["pool_oracle_scans"].items():
                pc = r["pools_confirm"].get(p)
                pools.append({"k": int(k), "pool": p, "n": po["n"], "P_next": sv["held_per_pool_mean_P_next"].get(p),
                              "union": pc["union"] if pc else None, "ot_pairs": pc["ot_pairs"] if pc else None,
                              "uot": pc["uot"] if pc else None,
                              "union_beats_ot_pairs": bool(pc["union"] < pc["ot_pairs"]) if pc else None,
                              "ORACLE_pool_best_t": po["best_t"], "ORACLE_pool_best_alpha": po["best_alpha"],
                              "ORACLE_pool_best_t_OT_beats_best_mix": po["best_t_OT_beats_best_mix"],
                              "ORACLE_pool_best_t_ED_U": po["best_t_ED_U"], "ORACLE_pool_best_alpha_ED_U": po["best_alpha_ED_U"]})
        multi = [p for p in pools if sum(q["k"] == p["k"] for q in pools) > 1]
        pv = [p for p in pools if p["P_next"] is not None]
        rho = spearmanr([p["P_next"] for p in pv], [p["ORACLE_pool_best_t"] for p in pv])
        out["Q1_composition"] = {"per_k": q1, "n_k_held_closer_to_mix_than_both": int(sum(v["held_closer_to_mix_than_both_neighbours"] for v in q1.values())),
                                 "n_k": len(q1)}
        out["Q1_geometry"] = {"per_k": geo,
                              "n_k_ORACLE_best_t_OT_beats_best_alpha_mix": int(sum(g["ORACLE_best_t_ED_U"] < g["ORACLE_best_alpha_ED_U"] for g in geo.values())),
                              "n_k_ORACLE_best_spread_beats_best_alpha_mix": int(sum(g["ORACLE_best_spread_ED_U"] < g["ORACLE_best_alpha_ED_U"] for g in geo.values())),
                              "n_k_ot_pairs_less_off_manifold_than_union": int(sum(g["off_manifold_frac"]["ot_pairs"] < g["off_manifold_frac"]["union"] for g in geo.values())),
                              "n_k_ot_pairs_covers_target_better_than_union": int(sum(g["target_uncovered_frac"]["ot_pairs"] < g["target_uncovered_frac"]["union"] for g in geo.values()))}
        out["Q1_stage_vote"] = vote
        out["Q1_pools"] = {"rows": pools, "n_pools": len(pools),
                           "n_union_beats_ot_pairs": int(sum(bool(p["union_beats_ot_pairs"]) for p in pools)),
                           "n_ORACLE_pool_best_t_OT_beats_best_mix": int(sum(p["ORACLE_pool_best_t_OT_beats_best_mix"] for p in pools)),
                           "n_ORACLE_pool_best_t_OT_beats_best_mix_multi_pool_stages": int(sum(p["ORACLE_pool_best_t_OT_beats_best_mix"] for p in multi)),
                           "n_pools_multi_pool_stages": len(multi),
                           "spearman_pool_P_next_vs_pool_best_t": float(rho.correlation),
                           "best_t_range_per_k": {k: [min(p["ORACLE_pool_best_t"] for p in pools if p["k"] == int(k)),
                                                      max(p["ORACLE_pool_best_t"] for p in pools if p["k"] == int(k))] for k in ks}}
    if wl:
        out["Q2_within_lineage"] = {"summary": wl["summary"], "reps": wl["reps"], "min_n": wl["min_n"],
                                    "units": [{x: u[x] for x in ("family", "k", "name", "n_prev", "n_held", "n_next")} |
                                              {m: round(u["mean"][m]["ED_U"], 4) for m in C.METHODS} |
                                              {"sd_union": round(u["sd_ED_U"]["union"], 4), "sd_ot_pairs": round(u["sd_ED_U"]["ot_pairs"], 4),
                                               "quarter_ED_prev_next": round(u["quarter_ED_prev_next"], 4)}
                                              for u in wl["units"]]}
    se = load("spread_eps.json")
    if se:
        out["Q1_spread_eps"] = {"summary": se["summary"],
                                "full_per_k": {k: {"spread": {m: round(v, 3) for m, v in u["spread"].items()},
                                                   "eps_ORACLE_ED_U": {e: round(x["ED_U"], 4) for e, x in u["eps"].items()},
                                                   "eps_median_eff_partners": {e: round(x["median_eff_partners"], 1) for e, x in u["eps"].items()},
                                                   "union_ED_U": round(u["ED_U"]["union"], 4)} for k, u in se["full"].items()}}
    if fo:
        out["Q3_forecast"] = {"summary": fo["summary"],
                              "per_origin": {s: {"forecast_stage": v["forecast_stage"], "persist": v["persist_ED_U"],
                                                 **{f"{fam}_{x}": v["families"][fam][x] for fam in ("uot_rw", "tilt_rw")
                                                    for x in ("rolling_choice", "rolling_ED_U", "best_lam_pos_hp", "best_lam_pos_ED_U",
                                                              "any_lam_pos_beats_persist", "oracle_rel_gain_vs_persist",
                                                              "n_grid_lam_pos_beating_persist", "spearman_backtest_vs_final")},
                                                 "uot_rw_extended_ORACLE_best": v["families"]["uot_rw"]["extended_ORACLE"]["best_ED_U"],
                                                 "truth_bound_best": v["truth_bounds_ORACLE"]["best"],
                                                 "truth_bound_best_ED_U": v["truth_bounds_ORACLE"]["best_ED_U"],
                                                 "truth_bound_rel_gain": v["truth_bounds_ORACLE"]["best_rel_gain_vs_persist"],
                                                 "celltype_weights_ED_U": v["truth_bounds_ORACLE"]["celltype_composition_weights"]["ED_U"],
                                                 "truth_bound_best_ess": v["truth_bounds_ORACLE"]["best_ess"],
                                                 "truth_bound_insample_best_ED_U": v["truth_bounds_ORACLE_insample"]["best_ED_U"],
                                                 "uot_rw_min_ess_lam_pos": v["families"]["uot_rw"]["min_ess_lam_pos"],
                                                 "uot_rw_rolling_weight_by_type": v.get("uot_rw_rolling_weight_by_type"),
                                                 "floor_ED_U_future_rest": v["floor_ED_U_future_rest"], **v["support"]}
                                             for s, v in fo["s"].items()}}
    if im:
        rows = {}
        for k in ks:
            g, v = out["Q1_geometry"]["per_k"][k], out["Q1_stage_vote"][k]["intermediate_frac"]
            rows[k] = {"ot_pairs/union": g["ot_pairs"] / g["union"], "ot_minus_union": g["ot_pairs"] - g["union"],
                       "intermediate_excess_ot_minus_held": v["ot_pairs"] - v["held_all"],
                       "pair_draw_noise_abs(own_t0.5_draw_vs_recorded)": abs(g["t0.5_own_draw_ED_U"] - g["ot_pairs"]),
                       "ORACLE_best_t": g["ORACLE_best_t"],
                       "composition_best_alpha": out["Q1_composition"]["per_k"][k]["best_alpha_next"]}
        hl = {"per_k": rows,
              "spearman_k_intermediate_excess_vs_ot_over_union": float(spearmanr(
                  [rows[k]["intermediate_excess_ot_minus_held"] for k in ks], [rows[k]["ot_pairs/union"] for k in ks]).correlation),
              "spearman_k_best_t_vs_composition_alpha": float(spearmanr(
                  [rows[k]["ORACLE_best_t"] for k in ks], [rows[k]["composition_best_alpha"] for k in ks]).correlation)}
        if se:
            hl["full_spread_mean"] = se["summary"]["full_mean_spread"]
        out["headline_numbers"] = hl
    out["confirm_verdicts"] = ref["verdicts"]
    C.dump(os.path.join(C.HERE, "mechanism.json"), out)
    print(json.dumps({x: out[x] for x in out if x.startswith("Q") and x != "Q2_within_lineage"}, default=str)[:6000])


if __name__ == "__main__":
    main()
