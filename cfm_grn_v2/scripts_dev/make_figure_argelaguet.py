"""Figure 12: the second real data set, and where the real-data error lives.

(a) The held-out stage E8.0 of the gastrulation series (version 0.1.10, gene filter, five seeds) against every
    control, with the distance between the two E8.0 biological replicates and the permutation threshold.
(b, d) For each data set, at its held-out stage: mean interpolation, the path family the field is trained on
    (no network, linear centre) and the trained model. On the embryoid body the path is far better than mean
    interpolation and the network loses half of it; on the gastrulation series the network reproduces its path
    and the path itself is worse than mean interpolation.
(c) Why the path loses there: the spread of each construction relative to the held-out population.
The path-family numbers use the benchmark's own evaluation draws (scripts_dev/diag_blend.py), so they are directly
comparable with the benchmark rows; they are a diagnosis of a measured result and choose nothing.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/make_figure_argelaguet.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
from cfm_grn_v2.evaluation.report import per_seed  # noqa: E402

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, LIGHT, ORANGE, GREEN, RED, GREY, PURPLE = ("#1f4e79", "#2e75b6", "#9dc3e6", "#d9822b", "#3a7d44",
                                                       "#b23a48", "#6e6e6e", "#7b5ea7")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "figure.facecolor": "white", "legend.frameon": False})
LABEL = {"flow_matching": "flow model (ours)", "mean_interp": "mean interpolation", "ot_interp": "McCann / OT interpolant",
         "linear": "linear map", "mlp": "MLP map", "persistence": "persistence", "freefm_replay": "velocity replay",
         "flow_matching_shuffled_time": "ours, shuffled time"}
COLOR = {"flow_matching": NAVY, "flow_matching_shuffled_time": RED, "mean_interp": ORANGE}

bench = {ds: per_seed(pd.read_csv(f"outputs/{ds}_v0110/results.csv")) for ds in ("eb", "argelaguet")}
diag = {ds: pd.read_csv(f"outputs/diag_blend_{ds}/blend.csv") for ds in ("eb", "argelaguet")}
summ = json.loads(Path("outputs/argelaguet_v0110/summary.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), gridspec_kw={"width_ratios": [1.45, 1.0]})

# ---------------------------------------------------------------- (a) the gastrulation benchmark
ax = axes[0, 0]
ag = bench["argelaguet"]
g = ag.groupby("model")["energy_distance"].agg(["mean", "std"])
order = sorted([m for m in g.index if m != "noise_floor"], key=lambda m: g.loc[m, "mean"])
order = [m for m in order if g.loc[m, "mean"] < 6] + [m for m in order if g.loc[m, "mean"] >= 6]
vals = [g.loc[m, "mean"] for m in order]
ax.bar(range(len(order)), vals, yerr=[g.loc[m, "std"] for m in order], capsize=2.5, width=0.68, edgecolor="white",
       color=[COLOR.get(m, LIGHT) for m in order], error_kw={"lw": 0.9, "ecolor": GREY})
for i, v in enumerate(vals):
    ax.text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=7.0)
rep = summ["replicate_yardstick"]["E8.0"]["energy_distance"]
ax.axhline(rep, color=GREEN, ls="--", lw=1)
ax.text(len(order) - 0.45, rep + max(vals) * 0.025, f"two E8.0 embryo pools: {rep:.3f}", color=GREEN, fontsize=6.8,
        ha="right")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LABEL.get(m, m) for m in order], fontsize=6.6, rotation=20, ha="right", rotation_mode="anchor")
ax.set_ylabel("energy distance at E8.0")
ax.set_title("(a) mouse gastrulation, E8.0 held out, predicted from E7.75 (5 seeds)")
ax.grid(axis="y", alpha=0.25, lw=0.5)


# ---------------------------------------------------------------- (b, c) construction vs network
def split_bars(ax, ds, title):
    d = diag[ds][diag[ds].split.str.startswith("test")]
    rows = [("mean interpolation", d[d.construction == "mean_interp"].ED, ORANGE),
            ("path family\n(no network)", d[d.construction == "path_linear"].ED, PURPLE),
            ("trained model\n(v0.1.10)", bench[ds].loc[bench[ds].model == "flow_matching", "energy_distance"], NAVY)]
    for i, (lab, v, c) in enumerate(rows):
        ax.bar(i, v.mean(), yerr=v.std(), color=c, width=0.62, capsize=2.5, edgecolor="white",
               error_kw={"lw": 0.9, "ecolor": GREY})
        ax.text(i, v.mean() * 1.04 + 0.01 * max(r[1].mean() for r in rows), f"{v.mean():.3f}", ha="center",
                fontsize=7.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.0)
    ax.set_ylabel("energy distance, held-out stage")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, lw=0.5)


split_bars(axes[0, 1], "eb", "(b) embryoid body: the network loses half its path")
split_bars(axes[1, 1], "argelaguet", "(d) gastrulation: the path itself loses")

# ---------------------------------------------------------------- (d) spread
ax = axes[1, 0]
cons = [("mean_interp", "mean interpolation\n(real cells, shifted)", ORANGE),
        ("ot_interp", "McCann / OT interpolant\n(blends of two cells)", LIGHT),
        ("path_linear", "path family\n(spline through chained cells)", PURPLE)]
w = 0.36
for j, (ds, name, hatch) in enumerate((("eb", "embryoid body, days 12–15", ""), ("argelaguet", "gastrulation, E8.0", "//"))):
    d = diag[ds][diag[ds].split.str.startswith("test")]
    for i, (c, _, col) in enumerate(cons):
        v = d[d.construction == c].var_ratio.mean()
        ax.bar(i + (j - 0.5) * w, v, width=w, color=col, hatch=hatch, edgecolor="white" if not hatch else GREY,
               lw=0.4, label=name if i == 0 else None)
        ax.text(i + (j - 0.5) * w, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.8)
ax.axhline(1.0, color=GREEN, ls="--", lw=1)
ax.text(1.0, 1.04, "spread of the true population", color=GREEN, fontsize=6.8, ha="center")
ax.set_xticks(range(3))
ax.set_xticklabels([c[1] for c in cons], fontsize=6.9)
ax.set_ylabel("total variance, predicted / true")
ax.set_title("(c) spread of each construction at the held-out stage")
ax.legend(fontsize=6.8, loc="upper left")
ax.set_ylim(0, max(1.75, ax.get_ylim()[1]))
ax.grid(axis="y", alpha=0.25, lw=0.5)

fig.tight_layout()
fig.savefig(OUT / "fig12_gastrulation_and_diagnosis.png")
fig.savefig(OUT / "fig12_gastrulation_and_diagnosis.pdf")
print("saved fig12_gastrulation_and_diagnosis")
