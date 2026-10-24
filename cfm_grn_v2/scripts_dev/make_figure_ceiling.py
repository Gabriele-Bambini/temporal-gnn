"""Figure 10: why the residual error is what it is (target-path ceiling, mean-vs-shape, quadratic law).

Reads outputs/diag_ceiling/{results.csv,mean_vs_shape.csv,mean_curve.csv} and the benchmark results.
Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/make_figure_ceiling.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
from cfm_grn_v2.evaluation.report import per_seed  # noqa: E402

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, LIGHT, ORANGE, GREEN, RED, GREY = "#1f4e79", "#2e75b6", "#9dc3e6", "#d9822b", "#3a7d44", "#b23a48", "#6e6e6e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "figure.facecolor": "white", "legend.frameon": False})

ceil = pd.read_csv("outputs/diag_ceiling/results.csv").groupby("measurement").energy_distance.agg(["mean", "std"])
shape = pd.read_csv("outputs/diag_ceiling/mean_vs_shape.csv")
bench = per_seed(pd.read_csv("outputs/phase1_v2_grn_interp/results.csv"))
model_ed = bench[bench.model == "flow_matching"].energy_distance.mean()
final = Path("outputs/final_test/results.csv")
model_v019 = (per_seed(pd.read_csv(final)).query("model == 'flow_v019'").energy_distance.mean()
              if final.exists() else None)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), gridspec_kw={"width_ratios": [1.6, 1.0, 1.0]})

# (a) the training target is the ceiling
ax = axes[0]
names = ["spline target\n(no network)", "flow model\nv0.1.8", "spline target\nall knots",
         "flow model\nv0.1.9", "McCann / OT\ninterpolant", "oracle chain\nthrough t = 3"]
vals = [ceil.loc["ceiling_fit", "mean"], model_ed, ceil.loc["ceiling_obs", "mean"],
        model_v019 if model_v019 is not None else float("nan"),
        ceil.loc["mccann_2_4", "mean"], ceil.loc["oracle_chain", "mean"]]
cols = [ORANGE, BLUE, LIGHT, NAVY, "#b0b7c0", GREEN]
n = len(vals)
ax.bar(range(n), vals, color=cols, width=0.66, edgecolor="white")
for i, v in enumerate(vals):
    if v == v:
        ax.text(i, max(v, 0) + 0.03, f"{v:.3f}", ha="center", fontsize=7.4)
ax.axhline(ceil.loc["noise_floor", "mean"], color=GREEN, ls="--", lw=1)
ax.text(n - 0.55, 0.115, "noise floor", color=GREEN, fontsize=7, ha="right")
ax.set_xticks(range(n))
ax.set_xticklabels(names, fontsize=6.7)
ax.set_ylabel("energy distance at the held-out t = 3")
ax.set_title("(a) the constructed path, not the network, is the limit")
ax.set_ylim(-0.05, 1.32)
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.annotate("", xy=(1, vals[1] + 0.09), xytext=(0, vals[0] + 0.09),
            arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1))
ax.text(0.5, vals[0] + 0.17, "difference 0.005:\nthe field reproduces\nits own target", ha="center", fontsize=6.8, color=NAVY)
if model_v019 is not None:
    ax.annotate("", xy=(3, vals[3] + 0.10), xytext=(2, vals[2] + 0.10),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1))
    ax.text(2.5, vals[2] + 0.155, "version 0.1.9 reaches\nthe predicted ceiling", ha="center", fontsize=6.8, color=NAVY)

# (b) all of the error is in the mean
ax = axes[1]
g = shape[shape.variant != "mean_curve"].groupby(["variant", "correction"]).energy_distance.mean().unstack()
order = ["spline", "mccann", "persistence"]
lab = {"spline": "spline paths\n(our model)", "mccann": "McCann / OT\ninterpolant", "persistence": "persistence"}
x = np.arange(len(order))
ax.bar(x - 0.2, [g.loc[m, "none"] for m in order], 0.38, color=BLUE, label="as predicted")
ax.bar(x + 0.2, [max(abs(g.loc[m, "oracle_mean"]), 9e-4) for m in order], 0.38, color=LIGHT, hatch="///",
       edgecolor="white", label="after shifting onto the true mean")
for i, m in enumerate(order):
    ax.text(i - 0.2, g.loc[m, "none"] * 1.35, f"{g.loc[m, 'none']:.2f}", ha="center", fontsize=7.4)
    ax.text(i + 0.2, 1.6e-3, f"{g.loc[m, 'oracle_mean']:+.3f}", ha="center", fontsize=6.8, color=GREY)
ax.set_yscale("log")
ax.set_ylim(6e-4, 60)
ax.set_xticks(x)
ax.set_xticklabels([lab[m] for m in order], fontsize=7.4)
ax.set_ylabel("energy distance (log scale)")
ax.set_title("(b) correcting only the mean removes all of the error")
ax.legend(fontsize=6.9, loc="upper left", ncol=1)
ax.grid(axis="y", alpha=0.25, lw=0.5)

# (c) quadratic law
ax = axes[2]
pts = shape[(shape.correction == "none") & (shape.variant != "mean_curve")]
pts = pts.groupby("variant")[["mean_error", "energy_distance"]].mean()
extra = {"spline target (all knots)": (ceil.loc["ceiling_obs"], None)}
xx = np.linspace(0, 4.2, 100)
c = float((pts.energy_distance / pts.mean_error ** 2).mean())
ax.plot(xx, c * xx ** 2, color=GREY, lw=1.1, ls="--", label=f"{c:.3f} x (mean error)²")
for name, r in pts.iterrows():
    ax.scatter(r.mean_error, r.energy_distance, s=42, color={"spline": BLUE, "mccann": ORANGE, "persistence": RED}[name], zorder=3)
    ax.annotate(name, (r.mean_error, r.energy_distance), textcoords="offset points", xytext=(6, -3), fontsize=7.2)
ax.set_xlabel("error of the predicted mean (L2 over 40 genes)")
ax.set_ylabel("energy distance")
ax.set_title("(c) the metric is a function of the mean error alone")
ax.legend(fontsize=7.4, loc="upper left")
ax.grid(alpha=0.25, lw=0.5)

fig.tight_layout()
fig.savefig(OUT / "fig10_ceiling.png")
fig.savefig(OUT / "fig10_ceiling.pdf")
print("saved fig10_ceiling")
