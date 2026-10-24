"""Figure 11: the held-out window on real data, and the fact that real data scores shape, not only location.

Panel (a) is the benchmark on the embryoid-body time course. Panel (b) is the contrast that matters for the
method: on the synthetic simulator the energy distance is a function of the error of the predicted mean alone
(one quadratic curve, Figure 10c), while on the real data the ranking by mean error and the ranking by energy
distance disagree — the McCann/OT interpolant puts the cloud in almost the right place and still loses, because
its shape is wrong. The synthetic benchmark could not see that; the real one can.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/make_figure_eb.py
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

LABEL = {"flow_matching": "flow model\n(ours)", "mean_interp": "mean\ninterpolation", "ot_interp": "McCann / OT\ninterpolant",
         "linear": "linear map", "mlp": "MLP map", "persistence": "persistence", "freefm_replay": "velocity\nreplay",
         "flow_matching_shuffled_time": "ours, shuffled\ntime labels", "noise_floor": "noise floor"}
COLOR = {"flow_matching": NAVY, "flow_matching_shuffled_time": RED, "noise_floor": GREEN}

import argparse  # noqa: E402

_p = argparse.ArgumentParser()
_p.add_argument("--results", default="outputs/eb_interp/results.csv")
_a = _p.parse_args()

eb = per_seed(pd.read_csv(_a.results))
g = eb.groupby("model")[["energy_distance", "mean_error"]].agg(["mean", "std"])
null_q95 = (float(eb.loc[eb["model"] == "flow_matching", "null_q95"].mean())   # the threshold at the model's own
            if "null_q95" in eb.columns else None)                               # sample sizes

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1.0]})

# ---------------------------------------------------------------- (a) the benchmark
ax = axes[0]
order = [m for m in g.index if m != "noise_floor"]
order = sorted(order, key=lambda m: g.loc[m, ("energy_distance", "mean")])
vals = [g.loc[m, ("energy_distance", "mean")] for m in order]
errs = [g.loc[m, ("energy_distance", "std")] for m in order]
cols = [COLOR.get(m, LIGHT) for m in order]
ax.bar(range(len(order)), vals, yerr=errs, capsize=2.5, color=cols, width=0.68, edgecolor="white",
       error_kw={"lw": 0.9, "ecolor": GREY})
for i, v in enumerate(vals):
    ax.text(i, v + max(vals) * 0.028, f"{v:.3f}", ha="center", fontsize=7.2)
if null_q95 is not None:
    ax.axhline(null_q95, color=GREEN, ls="--", lw=1)
    ax.text(len(order) - 0.45, null_q95 + max(vals) * 0.02, f"permutation threshold {null_q95:.3f}",
            color=GREEN, fontsize=6.8, ha="right")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LABEL.get(m, m).replace("\n", " ") for m in order], fontsize=6.5, rotation=18,
                   ha="right", rotation_mode="anchor")
ax.set_ylabel("energy distance at the held-out window")
n_seeds = int(eb["seed"].nunique())
ax.set_title(f"(a) embryoid body, days 12–15 held out ({n_seeds} seed{'s' if n_seeds > 1 else ''})")
ax.grid(axis="y", alpha=0.25, lw=0.5)

# ---------------------------------------------------------------- (b) location is not the whole story
ax = axes[1]
by_mean = sorted(order, key=lambda m: g.loc[m, ("mean_error", "mean")])
by_ed = sorted(order, key=lambda m: g.loc[m, ("energy_distance", "mean")])
for m in order:
    y0, y1 = by_mean.index(m), by_ed.index(m)
    col = COLOR.get(m, ORANGE if m == "ot_interp" else LIGHT)
    lw = 1.9 if m in ("flow_matching", "ot_interp") else 1.0
    ax.plot([0, 1], [y0, y1], color=col, lw=lw, zorder=3 if lw > 1 else 2, alpha=1.0 if lw > 1 else 0.85)
    ax.scatter([0, 1], [y0, y1], s=30, color=col, zorder=4)
    ax.text(-0.045, y0, LABEL.get(m, m).replace("\n", " "), ha="right", va="center", fontsize=6.9,
            color=col if lw > 1 else "black")
    ax.text(1.045, y1, LABEL.get(m, m).replace("\n", " "), ha="left", va="center", fontsize=6.9,
            color=col if lw > 1 else "black")
ax.set_xlim(-0.72, 1.72)
ax.set_ylim(len(order) - 0.5, -0.9)
ax.set_xticks([0, 1])
ax.set_xticklabels(["ranked by the error\nof the predicted mean", "ranked by the\nenergy distance"], fontsize=7.4)
ax.set_yticks([])
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
ax.set_title("(b) on real data the two rankings disagree")
ax.text(0.5, -0.78, "the OT interpolant puts the cloud in almost the right place\nand still loses: its shape is wrong",
        ha="center", va="center", fontsize=6.9, color=NAVY)

fig.tight_layout()
fig.savefig(OUT / "fig11_real_data.png")
fig.savefig(OUT / "fig11_real_data.pdf")
print("saved fig11_real_data")
