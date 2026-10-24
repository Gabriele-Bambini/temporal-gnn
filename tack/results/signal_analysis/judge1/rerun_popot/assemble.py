"""Merge popot_<dataset>.json into results.json and print a headline table (headline.txt).

Triples are [Pearson, P@200 up, P@200 down] on crossgene common.masks (10 draws x random 25 % per transition);
'i>=1' = mean over transitions 1..T-1, 'all' = nanmean over all transitions (CONTEXT DistBridge convention).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DS = ("gastrulation-human", "pancreas-human")
SHOW = ("full", "noOT", "OT_only", "persist_only(level_obs_RO)", "composition_only(box_states,beta0)",
        "two_pathway(box1+beta)", "mixavg3_denoised", "fc=persist", "fc=lin1", "fc=loglin1", "fc=ot0.05",
        "fc=uot0.05t1", "fc=sb3", "fc=sbW")
PAIRS = ("full - noOT", "OT_only - noOT", "fc=ot0.05 - fc=lin1", "fc=ot0.05 - fc=persist", "fc=uot0.05t1 - fc=loglin1",
         "fc=sb3 - fc=persist", "fc=sbW - fc=persist", "fc=sb3 - fc=lin1", "full - persist_only(level_obs_RO)",
         "full - trend", "noOT - trend", "two_pathway(box1+beta) - trend", "mixavg3_denoised - trend",
         "composition_only(box_states,beta0) - trend")


def main():
    res = {"folder": HERE, "files": sorted(os.listdir(HERE)), "datasets": {}}
    lines = []
    for name in DS:
        p = os.path.join(HERE, f"popot_{name}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        ds = {"cells": r["cells"], "n_test": r["n_test"], "context_numbers_quoted_not_rerun": r["context_numbers_quoted_not_rerun"],
              "reproduction_crossgene_nmf_linear_props_l1": r["reproduction_crossgene_nmf_linear_props_l1"],
              "composition_forecast_tv_error": {k: {kk: vv for kk, vv in v.items() if kk != "proportions_final_fit"}
                                                for k, v in r["composition_forecast_tv_error_vs_oracle_projection_i>=2"].items()},
              "turn_on_auc_history_fraction_of_past_bins_expressed": r["turn_on_auc_history_fraction_of_past_bins_expressed"],
              "targets": {}}
        lines.append(f"===== {name}  (cells {r['cells']})")
        lines.append(f"reproduction crossgene nmf_linear_props l1: {r['reproduction_crossgene_nmf_linear_props_l1']}")
        for t, v in r["targets"].items():
            tt = {"comparators": v["comparators"], "groups": {}, "paired": {}, "oracle": v["oracle_next_composition_NOT_strict"],
                  "hindsight_fixed_NOT_strict": v["fixed_configs_hindsight_NOT_strict"]}
            lines.append(f"--- target {t}   [PCC, P@200 up, down]   i>=1 | all")
            for k, x in v["comparators"].items():
                lines.append(f"  {k:44s} {x['i>=1']} | {x['all']}")
            for g in SHOW:
                if g in v["groups"]:
                    x = v["groups"][g]
                    tt["groups"][g] = {k: x[k] for k in x if k not in ("per_transition",)}
                    extra = f" OTsel {x['share_transitions_OT_selected']}"
                    if "turn_on_auc_mean_i>=1" in x:
                        extra += f" AUCon {x['turn_on_auc_mean_i>=1']} AUCoff {x['turn_off_auc_mean_i>=1']}"
                    lines.append(f"  {g:44s} {x['i>=1']} | {x['all']}{extra}")
            best = max(v["oracle_next_composition_NOT_strict"].items(), key=lambda kv: kv[1]["i>=1"][0])
            lines.append(f"  ORACLE next composition (best, NOT strict) {best[0]} {best[1]['i>=1']}")
            for k in PAIRS:
                if k in v["paired"]:
                    pp = v["paired"][k]
                    tt["paired"][k] = pp
                    lines.append(f"  paired {k:44s} dPCC {pp['pearson']['mean_delta']:+.4f} "
                                 f"({pp['pearson']['n_better']}/{pp['pearson']['n_worse']}, p={_p(pp['pearson'])}) "
                                 f"dJ {pp['J']['mean_delta']:+.4f} ({pp['J']['n_better']}/{pp['J']['n_worse']}, p={_p(pp['J'])})")
            ds["targets"][t] = tt
        lines.append("composition forecast TV error vs oracle projection (i>=2): " +
                     json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("persist", "lin1", "loglin1", "ot0.05",
                                                                             "uot0.05t1", "sb3", "sbW")}
                                 for k, v in ds["composition_forecast_tv_error"].items()}))
        lines.append(f"turn-on AUC, history fraction of past bins expressed: {ds['turn_on_auc_history_fraction_of_past_bins_expressed']}")
        lines.append(f"context (quoted): {r['context_numbers_quoted_not_rerun']}")
        res["datasets"][name] = ds
    # ---- NOT strict probes: known cell-state dictionary (popot_atlas.py) and the strict orig-head side probe
    res["atlas_probes_NOT_strict"] = {}
    for mode, tag in (("full", "atlas"), ("loo", "atlas_loo"), ("terminal", "atlas_terminal")):
        for name in DS:
            p = os.path.join(HERE, f"popot_{tag}_{name}.json")
            if not os.path.exists(p):
                continue
            a = json.load(open(p))
            lines.append(f"===== ATLAS probe (NOT strict) mode={mode} {name}  terminal bin {a.get('terminal_bin')}")
            blk = {}
            for t, v in a["targets"].items():
                row = {k: x["i>=1"] for k, x in v.items() if k.startswith(("fc=", "trend"))}
                blk[t] = {"i>=1": row, "paired": {k: {"dPCC": pp["pearson"]["mean_delta"],
                                                       "better/worse": [pp["pearson"]["n_better"], pp["pearson"]["n_worse"]],
                                                       "p": pp["pearson"]["wilcoxon_two_sided_p"],
                                                       "dJ": pp["J"]["mean_delta"], "p_J": pp["J"]["wilcoxon_two_sided_p"]}
                                                   for k, pp in v["paired"].items()}}
                lines.append(f"  {t}: " + "  ".join(f"{k.replace('fc=', '')} {x}" for k, x in row.items()))
                for k, pp in blk[t]["paired"].items():
                    lines.append(f"     paired {k:22s} dPCC {pp['dPCC']:+.4f} {pp['better/worse']} p={_p({'wilcoxon_two_sided_p': pp['p']})}"
                                 f"  dJ {pp['dJ']:+.4f} p={_p({'wilcoxon_two_sided_p': pp['p_J']})}")
            res["atlas_probes_NOT_strict"].setdefault(mode, {})[name] = blk
    p = os.path.join(HERE, "turnon_probe.json")
    if os.path.exists(p):
        tp = json.load(open(p))
        res["orig_head_turnon_probe_strict"] = {k: v for k, v in tp.items() if k != "protocol"}
        lines.append("===== orig head side probe (strict): long-memory law for genes at zero in bin i")
        for name in DS:
            for vn, x in tp.get(name, {}).items():
                lines.append(f"  {name} {vn:20s} i>=1 {x['i>=1']} all {x['all']} turn-on AUC {x['turn_on_auc_i>=1']}")
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    open(os.path.join(HERE, "headline.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _p(x):
    v = x["wilcoxon_two_sided_p"]
    return "na" if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
