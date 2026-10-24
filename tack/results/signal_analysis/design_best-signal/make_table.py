"""Compact headline table from results.json (nass.py), with the CONTEXT DistBridge numbers (strict, 'all' convention)
quoted for reference (they were not re-run here). Writes headline.txt."""
import json
import os

OUT = os.path.dirname(os.path.abspath(__file__))
DISTBRIDGE = {  # CONTEXT, strict, convention 'all' (nanmean over transitions incl. i = 0)
    ("gastrulation-human", "orig"): (0.585, 0.406, 0.541),
    ("pancreas-human", "orig"): (0.488, 0.301, 0.442),
    ("pancreas-human", "l1"): (0.10, 0.17, 0.21),
    ("pancreas-human", "linear"): (0.30, 0.25, 0.30),
}
SHOW = ("trend", "level(box4+obs)", "level(box4)", "regime", "full", "full+grn", "full+grn_null", "full+ot",
        "full+grn+ot", "auto(full|full-cg)", "auto(full|full-cg|trend)", "full-cg", "full-kal", "full-snr", "full-obs",
        "regime+obs", "full[RO h in 0.5,1,2]", "regime[h=0.5 fixed]", "full[h=0.5 fixed]", "full+grn[h=0.5 fixed]")


def main():
    res = json.load(open(os.path.join(OUT, "results.json")))
    lines = []
    for name, r in res.items():
        for tg, t in r["targets"].items():
            lines.append(f"== {name} {tg}   [PCC / P@200 up / P@200 down; 20 fixed random 25 % masks per transition]")
            db = DISTBRIDGE.get((name, tg))
            if db:
                lines.append(f"  {'DistBridge (CONTEXT)':26s} all {db[0]:.3f}/{db[1]:.3f}/{db[2]:.3f}")
            for k in SHOW:
                if k not in t["table"]:
                    continue
                a, b, c = t["table"][k]["all"], t["table"][k]["i>=2"], t["table"][k]["i>=2_bins>=100"]
                lines.append(f"  {k:26s} all {a['pcc']:.3f}/{a['p200_up']:.3f}/{a['p200_down']:.3f}"
                             f" | i>=2 {b['pcc']:.3f}/{b['p200_up']:.3f}/{b['p200_down']:.3f} (pcc sd {b['pcc_sd']:.3f})"
                             f" | i>=2,bins>=100 {c['pcc']:.3f}/{c['p200_up']:.3f}/{c['p200_down']:.3f}")
            for k, v in t["paired"].items():
                lines.append(f"  paired {k:36s} dPCC {v['pcc']['mean_delta']:+.3f} ({v['pcc']['better']}/"
                             f"{v['pcc']['worse']}, p={v['pcc']['wilcoxon_p']})  dJ {v['J']['mean_delta']:+.3f} "
                             f"({v['J']['better']}/{v['J']['worse']}, p={v['J']['wilcoxon_p']})")
            for k, v in t["partial_corr_given_full"].items():
                lines.append(f"  partial corr(y, {k} | full prediction) {v['mean']:+.4f} ({v['n_pos']}/{v['n']} > 0)")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "headline.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
