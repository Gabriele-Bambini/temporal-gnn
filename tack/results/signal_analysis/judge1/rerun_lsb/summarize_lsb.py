"""Merge results_<dataset>.json into results.json and print the headline table (strict, paired masks)."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
DS = ("gastrulation-human", "pancreas-human")
ROWS = {"orig": ["trend", "levelobs", "lsb_obs"],
        "l1": ["trend", "regime", "pergene_nomom", "pergene", "pergene_fixed", "k4_level_fixed", "lsb_fixed_k4ar",
               "k4_var_fixed", "k4_lagbasis_ar_fixed", "lsb_nodrift", "lsb_full", "lsb_forced", "lsb_stack",
               "regime+lsb", "HINDSIGHT_fixed_cfg"],
        "linear": ["trend", "regime", "pergene_nomom", "pergene", "pergene_fixed", "k4_level_fixed", "lsb_fixed_k4ar",
                   "k4_var_fixed", "k4_lagbasis_ar_fixed", "lsb_nodrift", "lsb_full", "lsb_full_beta0",
                   "lsb_forced", "lsb_stack", "regime+lsb", "HINDSIGHT_fixed_cfg"]}
KEYS = [("lsb_full", "trend"), ("lsb_stack", "trend"), ("lsb_stack", "regime"), ("regime+lsb", "regime"),
        ("lsb_fixed_k4ar", "pergene_fixed"), ("lsb_fixed_k4ar", "k4_level_fixed"), ("k4_var_fixed", "k4_level_fixed"),
        ("pergene", "trend"), ("regime", "trend"), ("lsb_obs", "levelobs")]


def main():
    merged = {}
    lines = []
    for ds in DS:
        p = os.path.join(OUT, f"results_{ds}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        merged[ds] = {"methods": {tg: {s: {k: v for k, v in rep.items() if k != "per_transition_pcc"}
                                       for s, rep in r["methods"][tg].items()} for tg in r["methods"]},
                      "paired": r["paired"], "hindsight_cfg": r["hindsight_cfg"],
                      "context_distbridge": r["context_distbridge"], "n_cells": r["n_cells"]}
        lines.append(f"===== {ds}  strict rolling origin, 10 paired random 25 % masks; PCC / P@200 up / down")
        for tg in ("orig", "l1", "linear"):
            lines.append(f"--- {tg}            {'all':>20s} {'i>=1':>20s} {'i>=3':>20s} {'i>=1,n>=100':>20s}  sd")
            for s in ROWS[tg]:
                if s not in r["methods"][tg]:
                    continue
                rep = r["methods"][tg][s]
                cells = " ".join(f"{rep[c]['pcc']:6.3f}/{rep[c]['up']:.3f}/{rep[c]['down']:.3f}"
                                 for c in ("all", "i>=1", "i>=3", "i>=1_nmin>=100"))
                lines.append(f"  {s:22s} {cells}  {rep['i>=1']['pcc_mask_sd']:.3f}")
            ctx = r["context_distbridge"]
            for k, v in ctx.items():
                if k == tg or k.startswith(tg + "_"):
                    lines.append(f"  CONTEXT {k:14s} {v} (their masks, convention all)")
            for a, b in KEYS:
                for conv in ("i>=1", "i>=3"):
                    key = f"{a}_vs_{b}_{conv}"
                    if key in r["paired"].get(tg, {}):
                        v = r["paired"][tg][key]
                        lines.append(f"    {key:42s} dPCC {v['pcc']['mean_delta']:+.3f} ({v['pcc']['better']}/"
                                     f"{v['pcc']['worse']}, p={v['pcc']['wilcoxon_p']})  dUp {v['up']['mean_delta']:+.3f}"
                                     f"  dDn {v['down']['mean_delta']:+.3f}")
    json.dump(merged, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    txt = "\n".join(lines)
    open(os.path.join(OUT, "headline.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
