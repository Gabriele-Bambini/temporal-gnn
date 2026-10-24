"""Judge: extract from the judge rerun JSON of the population prototype every number its report quotes
(group scores, oracle, paired tests, selected lambda / K / R shares, sink interpretation at the last fit).
Usage: python claims_p1.py <dataset>  ->  claims_p1_<dataset>.json
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def tri(e):
    return [e["pearson_nan0"], e["p_ta_up"], e["p_ta_down"]]


def main():
    name = sys.argv[1]
    r = json.load(open(os.path.join(HERE, "p1", f"uot_sink_{name}.json")))
    out = {}
    for t in ("l1", "linear", "orig"):
        R = r["targets"][t]
        o = {"trend": tri(R["trend"]["i>=1"])}
        for g in ("persist|Kany", "persist|L3|Kany", "lin1|Kany", "balanced|Kany", "sink_abs|Kany", "sink_norm|Kany",
                  "kluot_abs|Kany", "full|Kany", "sink_any|Kany"):
            o[g] = tri(R["groups"][g]["i>=1"])
        for K in (4, 8, 16):
            for f in ("balanced", "sink_abs", "sink_norm"):
                o[f"{f}|K{K}"] = tri(R["groups"][f"{f}|K{K}"]["i>=1"])
            o[f"oracle|K{K}|Lsel"] = tri(R["oracle"][f"oracle|K{K}|Lsel"]["i>=1"])
        o["oracle|Ksel|Lsel"] = tri(R["oracle"]["oracle|Ksel|Lsel"]["i>=1"])
        o["paired"] = {k: v for k, v in R["paired"].items()
                       if k.startswith(("sink_abs|Kany - balanced", "sink_norm|Kany - balanced", "kluot_abs|Kany - bal",
                                        "sink_norm|Kany - lin1", "sink_any|Kany - persist", "balanced|Kany - lin1"))}
        for g in ("sink_norm|Kany", "sink_abs|Kany", "balanced|Kany", "full|Kany"):
            sel = [s for s in R["groups"][g]["selected"][2:] if s and not s.startswith("fallback")]
            o[f"selection_shares::{g}"] = {
                "lam": dict(collections.Counter([p for s in sel for p in s.split("|") if p.startswith("lam")])),
                "K": dict(collections.Counter([s.split("|")[1] for s in sel])),
                "L": dict(collections.Counter([s.split("|")[2] for s in sel])),
                "R": dict(collections.Counter([p for s in sel for p in s.split("|") if p.startswith("R")])),
                "fam": dict(collections.Counter([s.split("|")[0] for s in sel]))}
        out[t] = o
    out["composition_tv_vs_oracle_i>=2"] = r["composition_tv_vs_oracle_i>=2"]
    si = r["sink_interpretation_last_fit"]
    interp = {}
    for K, ent in si.items():
        e = {"markers": ent["markers"]}
        for k, v in ent.items():
            if "|lam" in k:
                pp = v["per_pair"]
                e[k] = {"turnover_share_of_source_mass": v["turnover_share_of_source_mass"],
                        "mean_death_frac_by_state": v["mean_death_frac_by_state"],
                        "birth_share_by_state": v["birth_share_by_state"],
                        "corr_gross_turnover_vs_abs_log_cellcount_ratio": v["corr_gross_turnover_vs_abs_log_cellcount_ratio"],
                        "corr_net_change_vs_cell_count_change": v["corr_net_change_vs_cell_count_change"],
                        "pairs_with_both_death_and_birth_gt_1pct_of_src": sum(
                            1 for x in pp if min(x["death_total"], x["birth_total"]) > 0.01 * (x["death_total"] + x["transported"])),
                        "n_pairs": len(pp), "first_pairs": pp[:3]}
        interp[K] = e
    out["sink_interpretation_last_fit"] = interp
    with open(os.path.join(HERE, f"claims_p1_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for t in ("l1", "linear", "orig"):
        print("==", name, t)
        for k, v in out[t].items():
            print("  ", k, json.dumps(v))
    print(json.dumps(out["composition_tv_vs_oracle_i>=2"]))
    for K, e in interp.items():
        print(K, json.dumps(e)[:3000])


if __name__ == "__main__":
    main()
