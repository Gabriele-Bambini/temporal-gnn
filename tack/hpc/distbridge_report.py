"""Apply the pre-registered gates (tack/prereg/2026-10-22-dileo-task2-distbridge-prereg.md) to harness outputs.

Reads <root>/<dataset>/<model>/seed<s>.json (their per-transition metric dicts) and summary.json, prints and
writes <root>/report.json:
  G1  mean - 2.132 sd / sqrt(5) over seeds > paper best, per dataset x metric (PCC, P@200 up, P@200 down);
  G3  no NaN-Pearson transition;
  G6  distbridge_strict vs distbridge_strict_nosb on identical masks: 5-seed mean Delta >= +0.01 on one metric and
      >= -0.005 on the others, one-sided Wilcoxon on the per-transition seed-averaged Delta of
      J = (pearson + p_at_200_up + p_at_200_down) / 3, p < 0.05, and delta > 0 selected in >= 25 % of transitions;
plus the averages without the last transition (28 / 13) and the test genes tied at +/-5.
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

PAPER = {
    "gastrulation-human": {"pearson": 0.400, "p_at_200_up": 0.110, "p_at_200_down": 0.387},
    "pancreas-human": {"pearson": 0.434, "p_at_200_up": 0.178, "p_at_200_down": 0.354},
}
METRICS = ("pearson", "p_at_200_up", "p_at_200_down")
SEEDS = (1234, 42, 7, 0, 1)


def load(root, ds, model):
    return [json.load(open(os.path.join(root, ds, model, f"seed{s}.json"))) for s in SEEDS]


def seed_means(runs, drop_last=False):
    out = []
    for r in runs:
        rr = r[:-1] if drop_last else r
        out.append({k: float(np.nanmean([x[k] for x in rr])) for k in METRICS + ("mae",)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--headline", default="distbridge_strict")
    ap.add_argument("--ablation", default="distbridge_strict_nosb")
    args = ap.parse_args()
    rep = {}
    for ds in PAPER:
        if not os.path.isdir(os.path.join(args.root, ds)):
            continue
        summ = json.load(open(os.path.join(args.root, ds, "summary.json")))
        R = {"models": {}}
        for model in summ["models"]:
            runs = load(args.root, ds, model)
            sm = seed_means(runs)
            sm_drop = seed_means(runs, drop_last=True)
            row = {}
            for k in METRICS + ("mae",):
                v = np.array([s[k] for s in sm])
                row[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                          "lower95": float(v.mean() - 2.132 * v.std(ddof=1) / np.sqrt(len(v))),
                          "mean_without_last": float(np.mean([s[k] for s in sm_drop]))}
                if k in METRICS:
                    row[k]["G1"] = bool(row[k]["lower95"] > PAPER[ds][k])
            row["G1_all"] = all(row[k]["G1"] for k in METRICS)
            row["G3_nan_pearson"] = int(sum(np.isnan(x["pearson"]) for r in runs for x in r))
            R["models"][model] = row
        if args.headline in summ["models"] and args.ablation in summ["models"]:
            a, b = load(args.root, ds, args.headline), load(args.root, ds, args.ablation)
            d = {k: float(np.mean([sa[k] - sb[k] for sa, sb in zip(seed_means(a), seed_means(b))])) for k in METRICS}
            J = lambda x: (x["pearson"] + x["p_at_200_up"] + x["p_at_200_down"]) / 3
            dJ = np.mean([[J(xa) - J(xb) for xa, xb in zip(ra, rb)] for ra, rb in zip(a, b)], axis=0)
            nz = dJ[np.abs(dJ) > 0]
            p = float(stats.wilcoxon(nz, alternative="greater").pvalue) if len(nz) >= 5 else 1.0
            sel = summ["models"][args.headline].get("selected", {})
            share = float(np.mean([c[2] > 0 for c in sel.values()])) if sel else 0.0
            g6 = (max(d.values()) >= 0.01 and min(d.values()) >= -0.005 and p < 0.05 and share >= 0.25)
            R["G6"] = {"mean_delta": d, "wilcoxon_p": p, "n_transitions_nonzero": int(len(nz)),
                       "delta_selected_share": share, "pass": bool(g6),
                       "selected": sel}
        r0 = load(args.root, ds, next(iter(summ["models"])))[0]
        R["ties"] = [{"i": x["snapshot_idx"], "plus5": x.get("n_test_at_plus5"), "minus5": x.get("n_test_at_minus5")} for x in r0]
        R["paper_best"] = PAPER[ds]
        rep[ds] = R
        print(f"== {ds}  paper best " + " / ".join(f"{PAPER[ds][k]:.3f}" for k in METRICS))
        for model, row in R["models"].items():
            print(f"  {model:24s} " + "  ".join(f"{k}={row[k]['mean']:.3f}±{row[k]['sd']:.3f}" for k in METRICS)
                  + f"  mae={row['mae']['mean']:.3f}  G1={'PASS' if row['G1_all'] else 'fail'}  nanPCC={row['G3_nan_pearson']}")
        if "G6" in R:
            g = R["G6"]
            print(f"  G6 (OT step contributes): delta={g['mean_delta']}  wilcoxon p={g['wilcoxon_p']:.3g}  "
                  f"OT selected in {g['delta_selected_share']:.0%} -> {'PASS' if g['pass'] else 'fail'}")
    with open(os.path.join(args.root, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
