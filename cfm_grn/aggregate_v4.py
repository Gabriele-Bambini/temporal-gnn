"""
Degradation curves from results_v3 + results_v4 (identical data/folds/seeds).

  (a) velocity noise: mechanistic signed models vs velocity noise (SNR = 1/vnoise)
  (b) sign errors: signed models vs fraction of flipped true-edge signs, with the
      unsigned counterpart as reference -- where does a signed prior stop helping?

Usage: python aggregate_v4.py <results_v4_dir> <results_v3_dir>
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R4 = sys.argv[1] if len(sys.argv) > 1 else "results_v4"
R3 = sys.argv[2] if len(sys.argv) > 2 else os.path.join("output", "results_v3")
M = ["pearson_delta_downstream", "mse_delta_downstream", "direction_acc_downstream"]

rows = []
for path in (R3, R4):
    for f in glob.glob(os.path.join(path, "*.json")):
        r = json.load(open(f))
        row = {k: r.get(k) for k in ("family", "scenario", "model", "signed", "vnoise", "fold", "seed")}
        row["sign_error"] = float(r.get("sign_error", 0.0))
        row.update({k: r["mean"].get(k, np.nan) for k in M})
        rows.append(row)
df = pd.DataFrame(rows)
df = df[df.scenario.isin(["noisy", "noisy_context"])]
print(f"loaded {len(df)} runs (noisy scenarios)")


def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def curve(sub, xcol):
    g = sub.groupby(["scenario", xcol])
    out = g.agg(**{f"{k}_mean": (k, "mean") for k in M}, n=("seed", "size")).reset_index()
    for k in M:
        out[f"{k}_ci"] = g[k].apply(ci95).values
    return out


pd.set_option("display.width", 200)

# (a) velocity noise
print("\n=== (a) VELOCITY NOISE  (mechanistic, signed prior, no sign errors) ===")
va = []
for m in ["mech_graph", "mech_graph_learned"]:
    sub = df[(df.family == "mech") & (df.model == m) & (df.signed == True) & (df.sign_error == 0)]
    c = curve(sub, "vnoise"); c.insert(0, "model", m); va.append(c)
va = pd.concat(va)
va["SNR"] = np.where(va.vnoise > 0, 1 / va.vnoise.replace(0, np.nan), np.inf)
print(va[["model", "scenario", "vnoise", "SNR", "n", "pearson_delta_downstream_mean",
          "pearson_delta_downstream_ci", "mse_delta_downstream_mean"]].round(3).to_string(index=False))

# reference: best snapshot model (signed static_graph CFM, no sign errors)
ref = df[(df.family == "cfm") & (df.model == "static_graph") & (df.signed == True) & (df.sign_error == 0)]
print("\nreference, snapshot CFM static_graph signed:",
      ref.groupby("scenario")["pearson_delta_downstream"].mean().round(3).to_dict())

# (b) sign errors
print("\n=== (b) SIGN ERRORS  (signed models vs flipped-sign fraction; unsigned = reference) ===")
arms = [("cfm", "static_graph", 0.0, "CFM static_graph"),
        ("mech", "mech_graph", 0.0, "mech_graph clean vel"),
        ("mech", "mech_graph", 1.0, "mech_graph SNR1 vel")]
vb, tests = [], []
for fam, m, vn, label in arms:
    sub = df[(df.family == fam) & (df.model == m) & (df.signed == True) & (df.vnoise == vn)]
    c = curve(sub, "sign_error"); c.insert(0, "arm", label); vb.append(c)
    uns = df[(df.family == fam) & (df.model == m) & (df.signed == False) & (df.vnoise == vn)]
    U = uns.set_index(["scenario", "fold", "seed"])["pearson_delta_downstream"]
    for se, s in sub.groupby("sign_error"):
        S = s.set_index(["scenario", "fold", "seed"])["pearson_delta_downstream"]
        for sc in ["noisy", "noisy_context"]:
            j = pd.concat([S, U], axis=1, keys=["s", "u"]).dropna()
            j = j[j.index.get_level_values("scenario") == sc]
            if len(j) < 3:
                continue
            d = (j["s"] - j["u"]).values
            ci = ci95(d)
            tests.append(dict(arm=label, scenario=sc, sign_error=se, n=len(d),
                              signed_minus_unsigned=round(d.mean(), 3), ci95=round(ci, 3),
                              verdict=("signed better" if d.mean() > ci else
                                       "signed WORSE" if d.mean() < -ci else "n.s.")))
vb = pd.concat(vb)
print(vb[["arm", "scenario", "sign_error", "n", "pearson_delta_downstream_mean",
          "pearson_delta_downstream_ci", "mse_delta_downstream_mean"]].round(3).to_string(index=False))
print("\n--- signed vs unsigned, paired, by sign-error level ---")
tt = pd.DataFrame(tests)
print(tt.to_string(index=False))

va.to_csv(os.path.join(R4, "_curve_velocity_noise.csv"), index=False)
vb.to_csv(os.path.join(R4, "_curve_sign_error.csv"), index=False)
tt.to_csv(os.path.join(R4, "_sign_error_vs_unsigned.csv"), index=False)

# figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
    for i, sc in enumerate(["noisy", "noisy_context"]):
        a = ax[i, 0]
        for m, sub in va[va.scenario == sc].groupby("model"):
            sub = sub.sort_values("vnoise")
            a.errorbar(sub.vnoise, sub.pearson_delta_downstream_mean, yerr=sub.pearson_delta_downstream_ci,
                       marker="o", capsize=3, label=m)
        r = ref[ref.scenario == sc]["pearson_delta_downstream"].mean()
        a.axhline(r, ls="--", c="gray", label="snapshot CFM (signed)")
        a.set_title(f"{sc}: velocity noise"); a.set_xlabel("velocity noise (x per-gene std; SNR = 1/x)")
        a.set_ylabel("downstream delta Pearson"); a.legend(fontsize=8)
        b = ax[i, 1]
        for arm, sub in vb[vb.scenario == sc].groupby("arm"):
            sub = sub.sort_values("sign_error")
            b.errorbar(sub.sign_error, sub.pearson_delta_downstream_mean, yerr=sub.pearson_delta_downstream_ci,
                       marker="o", capsize=3, label=arm)
        b.axhline(0, c="k", lw=0.5)
        b.set_title(f"{sc}: sign errors in prior"); b.set_xlabel("fraction of true-edge signs flipped")
        b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(R4, "_degradation_curves.png"), dpi=150)
    print(f"\nfigure: {os.path.join(R4, '_degradation_curves.png')}")
except Exception as e:
    print("figure skipped:", e)
