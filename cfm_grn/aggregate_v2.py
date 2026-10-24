"""
Aggregate results_v2/*.json into summary tables and pre-registered paired comparisons.

Pre-registered predictions (written before any v2 result was seen):
  P1  per-condition OT > global OT                     (bug fix, oracle scenario)
  P2  graph_cfm (per-edge pert) > graph_cfm_dense      (mask fix, all scenarios)
  P3  static_graph > mlp                               (graph prior helps unseen perturbations)
  P4  static_learned > static_graph                    only where the prior is noisy
  P5  graph_cfm > static_learned                       only in noisy_context
  P6  every learned model > persistence                on effect metrics

Pairs are matched on (scenario, fold, seed). Folds of one seed share a dataset, so the
15 pairs are not fully independent: treat the CI as optimistic and look at the
per-seed signs as well.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

RES = sys.argv[1] if len(sys.argv) > 1 else "results_v2"

PRIMARY = "pearson_delta_downstream"
METRICS = [
    ("pearson_delta_downstream", +1), ("direction_acc_downstream", +1),
    ("pearson_delta", +1), ("knockdown_fraction", +1),
    ("energy_distance_de", -1), ("w1_de", -1),
    # original (absolute-level) metrics, kept for comparison only
    ("pearson_mean_corr", +1), ("energy_distance", -1), ("mean_w1_distance", -1),
]

rows = []
for f in glob.glob(os.path.join(RES, "*.json")):
    r = json.load(open(f))
    row = {k: r[k] for k in ("scenario", "model", "ot", "fold", "seed")}
    row.update(r["mean"])
    row["train_seconds"] = r["train_seconds"]
    rows.append(row)
df = pd.DataFrame(rows)
print(f"loaded {len(df)} runs from {RES}")
if df.empty:
    sys.exit(0)
df.to_csv(os.path.join(RES, "_all_runs.csv"), index=False)


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

print("\n" + "=" * 100)
print("SUMMARY  mean +/- 95% CI over fold x seed")
print("=" * 100)
summ = []
for (sc, m, ot), g in df.groupby(["scenario", "model", "ot"]):
    rec = {"scenario": sc, "model": m, "ot": ot, "n": len(g)}
    for k, _ in METRICS:
        if k in g:
            rec[k] = f"{np.nanmean(g[k]):.3f} +/- {ci95(g[k]):.3f}"
    summ.append(rec)
summ = pd.DataFrame(summ).sort_values(["scenario", "ot", "model"])
summ.to_csv(os.path.join(RES, "_summary.csv"), index=False)
for sc in ["oracle", "noisy", "noisy_context"]:
    s = summ[summ.scenario == sc]
    if len(s):
        print(f"\n--- {sc} ---")
        print(s.drop(columns=["scenario"]).to_string(index=False))


def paired(sc, a, b, ot_a="per_condition", ot_b="per_condition", metric=PRIMARY, sign=+1):
    A = df[(df.scenario == sc) & (df.model == a) & (df.ot == ot_a)].set_index(["fold", "seed"])[metric]
    B = df[(df.scenario == sc) & (df.model == b) & (df.ot == ot_b)].set_index(["fold", "seed"])[metric]
    j = pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 3:
        return None
    d = sign * (j["a"] - j["b"]).values
    try:
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    per_seed = j.assign(d=d).groupby(level="seed")["d"].mean()
    return dict(n=len(d), mean=d.mean(), ci=ci95(d), p=p,
                wins=int((d > 0).sum()), seed_signs="".join("+" if v > 0 else "-" for v in per_seed))


print("\n" + "=" * 100)
print("PRE-REGISTERED COMPARISONS  (positive = first model better; primary metric: "
      f"{PRIMARY}, secondary: energy_distance_de)")
print("=" * 100)
tests = []
for m in ["mlp", "static_graph", "graph_cfm"]:
    tests.append(("P1", "oracle", m, m, "per_condition", "global"))
for sc in ["oracle", "noisy", "noisy_context"]:
    tests.append(("P2", sc, "graph_cfm", "graph_cfm_dense", "per_condition", "per_condition"))
    tests.append(("P3", sc, "static_graph", "mlp", "per_condition", "per_condition"))
    tests.append(("P4", sc, "static_learned", "static_graph", "per_condition", "per_condition"))
    tests.append(("P5", sc, "graph_cfm", "static_learned", "per_condition", "per_condition"))
    for m in ["mlp", "static_graph", "static_learned", "graph_cfm"]:
        tests.append(("P6", sc, m, "persistence", "per_condition", "per_condition"))

out = []
for tag, sc, a, b, oa, ob in tests:
    for metric, sign in [(PRIMARY, +1), ("energy_distance_de", -1), ("knockdown_fraction", +1)]:
        r = paired(sc, a, b, oa, ob, metric, sign)
        if r is None:
            continue
        la = a if oa == "per_condition" else f"{a}[global OT]"
        lb = b if ob == "per_condition" else f"{b}[global OT]"
        if tag == "P1":
            la, lb = f"{a}[per-cond OT]", f"{a}[global OT]"
        sig = "SIG" if (np.isfinite(r["ci"]) and abs(r["mean"]) > r["ci"]) else "n.s."
        out.append(dict(pred=tag, scenario=sc, comparison=f"{la} vs {lb}", metric=metric,
                        n=r["n"], diff=round(r["mean"], 4), ci95=round(r["ci"], 4), verdict=sig,
                        wins=f"{r['wins']}/{r['n']}", seeds=r["seed_signs"],
                        wilcoxon_p=round(r["p"], 4) if np.isfinite(r["p"]) else np.nan))
res = pd.DataFrame(out)
if res.empty:
    print("\nno paired comparisons yet (need >= 3 matched (fold, seed) pairs per comparison)")
    sys.exit(0)
res.to_csv(os.path.join(RES, "_comparisons.csv"), index=False)
for tag in ["P1", "P2", "P3", "P4", "P5", "P6"]:
    s = res[res.pred == tag]
    if len(s):
        print(f"\n--- {tag} ---")
        print(s.drop(columns=["pred"]).to_string(index=False))
