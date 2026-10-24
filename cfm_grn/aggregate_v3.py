"""
Aggregate results_v3 (and results_v2 for pairing) into tables + pre-registered tests.

Pre-registered predictions (written before any v3 result was seen):
  Q1  a SIGNED prior lowers downstream error of the snapshot (cfm) graph models in the
      noisy scenarios -- missing signs were part of why their cascade was wrong
  Q2  learned edge weights do NOT separate true from spurious prior edges (AUC ~ 0.5)
  Q3  mechanistic graph model (clean velocity) predicts the downstream cascade better
      than the best snapshot model -- clearly positive downstream correlation
  Q4  mechanistic graph > mechanistic MLP on unseen knockouts (structure helps
      extrapolate to the clamped state)
  Q5  velocity noise (SNR 1) degrades the mechanistic model but does not remove Q3
  Q6  signed > unsigned for the mechanistic graph model in the noisy scenarios

Usage: python aggregate_v3.py <results_v3_dir> [<results_v2_dir>]
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R3 = sys.argv[1] if len(sys.argv) > 1 else "results_v3"
R2 = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "results_v2")

KEYS = ["pearson_delta_downstream", "direction_acc_downstream", "mse_delta_downstream",
        "energy_distance_de", "knockdown_fraction"]


def load(path, v2=False):
    rows = []
    for f in glob.glob(os.path.join(path, "*.json")):
        r = json.load(open(f))
        if v2 and r.get("ot") != "per_condition":
            continue
        row = dict(family="cfm" if not v2 else "cfm_v2", scenario=r["scenario"], model=r["model"],
                   signed=bool(r.get("signed", False)), vnoise=float(r.get("vnoise", 0.0)),
                   fold=r["fold"], seed=r["seed"])
        row.update({k: r["mean"].get(k, np.nan) for k in KEYS})
        if not v2:
            row["family"] = r["family"]
            row.update({k: v for k, v in (r.get("edge_diag") or {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


d3 = load(R3)
d2 = load(R2, v2=True) if os.path.isdir(R2) else pd.DataFrame()
df = pd.concat([d3, d2], ignore_index=True)
print(f"loaded {len(d3)} v3 runs, {len(d2)} v2 runs (per-condition OT)")
df.to_csv(os.path.join(R3, "_all_runs_v3.csv"), index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def sel(**kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m]


def paired(a_kw, b_kw, metric, sign=+1):
    """sign=+1: higher is better; -1: lower is better. Positive diff = A better."""
    A = sel(**a_kw).set_index(["scenario", "fold", "seed"])[metric]
    B = sel(**b_kw).set_index(["scenario", "fold", "seed"])[metric]
    j = pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 3:
        return None
    d = sign * (j["a"] - j["b"]).values
    try:
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    per_seed = pd.Series(d, index=j.index.get_level_values("seed")).groupby(level=0).mean()
    ci = ci95(d)
    return dict(n=len(d), diff=round(float(d.mean()), 4), ci95=round(ci, 4),
                verdict="SIG" if np.isfinite(ci) and abs(d.mean()) > ci else "n.s.",
                wins=f"{int((d > 0).sum())}/{len(d)}",
                seeds="".join("+" if v > 0 else "-" for v in per_seed),
                wilcoxon_p=round(float(p), 4) if np.isfinite(p) else np.nan)


# --------------------------------------------------------------------------- tables
print("\n" + "=" * 110)
print("SUMMARY (mean +/- 95% CI over fold x seed)")
print("=" * 110)
rows = []
for (fam, sc, m, sg, vn), g in df.groupby(["family", "scenario", "model", "signed", "vnoise"]):
    rec = dict(family=fam, scenario=sc, model=m, signed=sg, vnoise=vn, n=len(g))
    for k in KEYS:
        rec[k] = f"{np.nanmean(g[k]):.3f} +/- {ci95(g[k]):.3f}"
    if "edge_auc_true_vs_spurious" in g and g["edge_auc_true_vs_spurious"].notna().any():
        rec["edge_auc"] = f"{np.nanmean(g['edge_auc_true_vs_spurious']):.3f} +/- {ci95(g['edge_auc_true_vs_spurious']):.3f}"
    rows.append(rec)
summ = pd.DataFrame(rows).sort_values(["scenario", "family", "model", "signed", "vnoise"])
summ.to_csv(os.path.join(R3, "_summary_v3.csv"), index=False)
for sc in ["oracle", "noisy", "noisy_context"]:
    s = summ[summ.scenario == sc]
    if len(s):
        print(f"\n--- {sc} ---")
        print(s.drop(columns=["scenario"]).to_string(index=False))

# --------------------------------------------------------------------------- tests
out = []


def add(tag, label, a_kw, b_kw, scenarios, metrics=(("pearson_delta_downstream", 1),
                                                     ("mse_delta_downstream", -1),
                                                     ("direction_acc_downstream", 1))):
    for sc in scenarios:
        for metric, sign in metrics:
            r = paired(dict(a_kw, scenario=sc), dict(b_kw, scenario=sc), metric, sign)
            if r:
                out.append(dict(pred=tag, scenario=sc, comparison=label, metric=metric, **r))


NOISY = ["noisy", "noisy_context"]
ALL = ["oracle", "noisy", "noisy_context"]
for m in ["static_graph", "static_learned", "graph_cfm"]:
    add("Q1", f"{m}: signed vs unsigned",
        dict(family="cfm", model=m, signed=True), dict(family="cfm", model=m, signed=False), NOISY)

best_cfm = dict(family="cfm_v2", model="static_graph", signed=False)
add("Q3", "mech_graph(v0) vs static_graph CFM",
    dict(family="mech", model="mech_graph", signed=False, vnoise=0.0), best_cfm, ALL)
add("Q3", "mech_graph_learned(v0) vs static_graph CFM",
    dict(family="mech", model="mech_graph_learned", signed=False, vnoise=0.0), best_cfm, ALL)
for vn in (0.0, 1.0):
    add("Q4", f"mech_graph vs mech_mlp (v{vn:g})",
        dict(family="mech", model="mech_graph", signed=False, vnoise=vn),
        dict(family="mech", model="mech_mlp", signed=False, vnoise=vn), ALL)
add("Q5", "mech_graph: clean vs noisy velocity",
    dict(family="mech", model="mech_graph", signed=False, vnoise=0.0),
    dict(family="mech", model="mech_graph", signed=False, vnoise=1.0), ALL)
for m in ["mech_graph", "mech_graph_learned"]:
    for vn in (0.0, 1.0):
        add("Q6", f"{m} (v{vn:g}): signed vs unsigned",
            dict(family="mech", model=m, signed=True, vnoise=vn),
            dict(family="mech", model=m, signed=False, vnoise=vn), NOISY)
# consistency: v3 unsigned CFM should reproduce v2
for m in ["static_graph", "static_learned", "graph_cfm"]:
    add("CHK", f"{m}: v3 unsigned vs v2 (should be ~0)",
        dict(family="cfm", model=m, signed=False), dict(family="cfm_v2", model=m), NOISY,
        metrics=(("mse_delta_downstream", -1),))

res = pd.DataFrame(out)
if res.empty:
    print("\nno paired comparisons available yet")
    sys.exit(0)
res.to_csv(os.path.join(R3, "_comparisons_v3.csv"), index=False)
print("\n" + "=" * 110)
print("PRE-REGISTERED COMPARISONS  (positive diff = first better)")
print("=" * 110)
for tag in ["Q1", "Q3", "Q4", "Q5", "Q6", "CHK"]:
    s = res[res.pred == tag]
    if len(s):
        print(f"\n--- {tag} ---")
        print(s.drop(columns=["pred"]).to_string(index=False))

# Q2: edge separation
print("\n--- Q2: do learned edge weights separate true vs spurious prior edges? (AUC 0.5 = no) ---")
e = df[df.get("edge_auc_true_vs_spurious", pd.Series(dtype=float)).notna()] if "edge_auc_true_vs_spurious" in df else pd.DataFrame()
if len(e):
    q2 = e.groupby(["family", "scenario", "model", "signed", "vnoise"]).agg(
        auc=("edge_auc_true_vs_spurious", "mean"),
        auc_ci=("edge_auc_true_vs_spurious", ci95),
        w_true=("edge_w_true_mean", "mean"), w_spur=("edge_w_spurious_mean", "mean"),
        n=("edge_auc_true_vs_spurious", "size")).round(3)
    print(q2.to_string())
