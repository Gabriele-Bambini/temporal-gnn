"""Figures for the 19 September 2026 deliverable (pipeline, architecture, protocol, results, OT ablation).

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/make_figures.py
Writes outputs/figures/fig0N_*.png (and .pdf). Figure 9 trains two small flow models on the two-lineage toy
(about two minutes on CPU); their predictions are cached in outputs/figures/crossing_pred.npz.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

sys.path.insert(0, "src")
from cfm_grn_v2.data.schema import SnapshotSet  # noqa: E402
from cfm_grn_v2.data.synthetic_grn import TimeVaryingGRNSimulator  # noqa: E402
from cfm_grn_v2.evaluation.report import per_seed  # noqa: E402
from cfm_grn_v2.flows.coupling import chain_ot  # noqa: E402
from cfm_grn_v2.flows.path import SplinePath  # noqa: E402
from cfm_grn_v2.training.config import TrainConfig  # noqa: E402
from cfm_grn_v2.training.trainer import train_flow  # noqa: E402

torch.set_num_threads(6)
OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, LIGHT, ORANGE, GREEN, RED, GREY, PALE = "#1f4e79", "#2e75b6", "#9dc3e6", "#d9822b", "#3a7d44", "#b23a48", "#6e6e6e", "#eef3f9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "figure.facecolor": "white", "legend.frameon": False})
LABELS = {"flow_matching": "flow model (OT-spline, duration clock)", "spline_ot": "spline paths, OT chaining",
          "spline_random": "spline paths, random chaining", "linear_ot": "straight paths, OT coupling",
          "linear_indep": "straight paths, independent coupling", "mean_interp": "mean interpolation",
          "ot_interp": "McCann / OT interpolant", "linear": "linear map (same horizon)", "mlp": "MLP map (same horizon)",
          "persistence": "persistence", "freefm_replay": "velocity replay",
          "flow_matching_shuffled_time": "flow model, shuffled time labels",
          "spline_ot_shuffled_time": "spline OT, shuffled time labels", "noise_floor": "noise floor"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name, flush=True)


def canvas(w, h, y0=0.0):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(y0, y0 + h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", fc=PALE, ec=NAVY, fs=8.2, title_dy=0.13, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2,
                                linestyle=ls))
    ax.text(x + w / 2, y + h - title_dy, title, ha="center", va="top", fontsize=fs + 0.6, fontweight="bold", color=NAVY)
    if body:
        ax.text(x + w / 2, y + (h - title_dy - 0.18) / 2, body, ha="center", va="center", fontsize=fs - 0.4,
                color="#1a1a1a", linespacing=1.4)


def arrow(ax, p, q, color=NAVY, lw=1.3, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def energy_distance(a, b):
    n, m = len(a), len(b)
    return 2 * cdist(a, b).mean() - cdist(a, a).sum() / (n * (n - 1)) - cdist(b, b).sum() / (m * (m - 1))


TIMES5 = [0.0, 1.0, 2.0, 3.0, 4.0]


def crossing_lineages(n_per_cluster=200, seed=0, spread=0.35, replicates=("r1", "r2")):
    rng = np.random.default_rng(seed)
    xs, ts, reps = [], [], []
    for rep in replicates:
        for t in TIMES5:
            left = np.array([-3.0, t]) + spread * rng.normal(size=(n_per_cluster, 2))
            right = np.array([3.0, -t]) + spread * rng.normal(size=(n_per_cluster, 2))
            x = np.concatenate([left, right])
            xs.append(x)
            ts.append(np.full(len(x), t))
            reps.append(np.array([rep] * len(x)))
    x = np.concatenate(xs).astype(np.float32)
    return SnapshotSet(x, np.concatenate(ts), np.array(["ctrl"] * len(x)), np.concatenate(reps), ["x", "y"], name="crossing")


def small_box(ax, x, y, w, h, title, body="", fc=PALE, ec=NAVY, ls="-", fs_title=7.0, fs_body=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06", fc=fc, ec=ec, lw=1.0,
                                linestyle=ls))
    if body:
        ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=fs_title, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, y + (h - 0.24) / 2, body, ha="center", va="center", fontsize=fs_body, color="#1a1a1a", linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs_title, fontweight="bold", color=NAVY)


# ---------------------------------------------------------------- Figure 1: pipeline -------------------------------------------
def fig01_pipeline():
    W, H = 7.6, 5.75
    fig, ax = canvas(W, H)
    bw, bh, gap = 2.3, 2.15, 0.15
    xs = [0.2 + k * (bw + gap) for k in range(3)]
    y1, y2 = 3.05, 0.35
    pos = [(xs[0], y1), (xs[1], y1), (xs[2], y1), (xs[0], y2), (xs[1], y2)]
    titles = ["1  Unpaired snapshots", "2  Optimal transport", "3  Spline paths", "4  Autonomous field", "5  Predict and evaluate"]
    bodies = ["cells × genes at t₀ … t_m\nper (condition, replicate)\nno cell tracking, no PCA",
              "exact minibatch OT\n(Hungarian assignment)\nwithin one biological unit\nchained t₀→t₁→…→t_m\n= pseudo-trajectories",
              "natural cubic spline per\npseudo-trajectory\nin biological time\nstate x(u), velocity ẋ(u)",
              "v(x, c): MLP with SiLU\ncondition embedding, no time input\nloss ‖v(x(u), c) − ẋ(u)‖²\nAdam · EMA 0.999 · clip 1.0",
              "RK4 over Δt = t_b − t_a\n20 steps per time unit\nenergy distance (U-statistic)\ncontrols · permutation null\n5 seeds, paired CI95"]
    for (x, y), t, b in zip(pos, titles, bodies):
        box(ax, x, y, bw, bh, t, "", fs=7.4, title_dy=0.11)
        ax.text(x + bw / 2, y + 0.55, b, ha="center", va="center", fontsize=6.2, color="#1a1a1a", linespacing=1.4)
    for a, b in ((0, 1), (1, 2), (3, 4)):
        (xa, ya), (xb, _) = pos[a], pos[b]
        arrow(ax, (xa + bw + 0.02, ya + bh / 2), (xb - 0.02, ya + bh / 2), lw=1.5)
    xm, ym = xs[2] + bw / 2, y1 - 0.02
    xb, yb = xs[0] + bw / 2, y2 + bh + 0.02
    ymid = (ym + yb) / 2
    ax.plot([xm, xm, xb], [ym, ymid, ymid], color=NAVY, lw=1.5, solid_capstyle="round")
    arrow(ax, (xb, ymid), (xb, yb), lw=1.5)
    rng = np.random.default_rng(1)
    cmap = plt.get_cmap("viridis")
    dy = 1.45
    # 1: four little clouds
    x, y = pos[0]
    cx, cy = x + bw / 2, y + dy
    for k in range(4):
        c = np.array([cx + (k - 1.5) * 0.48, cy + 0.07 * np.sin(k)])
        pts = c + 0.08 * rng.normal(size=(30, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=cmap(k / 3), lw=0)
        ax.text(c[0], cy - 0.36, f"t{k}", ha="center", fontsize=5.8, color=GREY)
    # 2: OT matching between two columns
    x, y = pos[1]
    cx, cy = x + bw / 2, y + dy
    ya = np.linspace(cy - 0.28, cy + 0.28, 6)
    xa, xb2 = cx - 0.5, cx + 0.5
    for i, yy in enumerate(ya):
        ax.plot([xa, xb2], [yy, yy + 0.03 * (i - 2.5)], color=BLUE, lw=0.7)
    ax.scatter([xa] * 6, ya, s=8, color=cmap(0.0), zorder=3)
    ax.scatter([xb2] * 6, ya + 0.03 * (np.arange(6) - 2.5), s=8, color=cmap(0.5), zorder=3)
    # 3: spline through four points
    x, y = pos[2]
    cx, cy = x + bw / 2, y + dy
    px = cx + np.array([-0.8, -0.27, 0.27, 0.8])
    py = cy + np.array([0.0, 0.28, -0.14, 0.24])
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(px, py, bc_type="natural")
    xx = np.linspace(px[0], px[-1], 80)
    ax.plot(xx, cs(xx), color=ORANGE, lw=1.5)
    ax.scatter(px, py, s=12, color=[cmap(k / 3) for k in range(4)], zorder=3)
    # 4: tiny network
    x, y = pos[3]
    cx, cy = x + bw / 2, y + dy
    sizes = (3, 4, 4, 3)
    lx = [cx - 0.72, cx - 0.24, cx + 0.24, cx + 0.72]
    for j, (xl, n) in enumerate(zip(lx, sizes)):
        ys = cy + (np.arange(n) - (n - 1) / 2) * 0.16
        ax.scatter([xl] * n, ys, s=10, color=NAVY if j in (0, 3) else BLUE, zorder=3)
        if j < 3:
            ys2 = cy + (np.arange(sizes[j + 1]) - (sizes[j + 1] - 1) / 2) * 0.16
            for a in ys:
                for b in ys2:
                    ax.plot([xl, lx[j + 1]], [a, b], color="#b7c7db", lw=0.4, zorder=1)
    # 5: target vs prediction clouds
    x, y = pos[4]
    cx, cy = x + bw / 2, y + dy
    for ddx, col, lab in ((-0.42, GREY, "target"), (0.42, BLUE, "prediction")):
        c = np.array([cx + ddx, cy])
        ax.scatter(*(c + 0.09 * rng.normal(size=(40, 2))).T, s=3, color=col, lw=0)
        ax.text(c[0], cy - 0.36, lab, ha="center", fontsize=5.8, color=col)
    # design rules in the free slot
    box(ax, xs[2], y2, bw, bh, "Design rules", "", fc="white", ec=GREY, ls="--", fs=7.4, title_dy=0.11)
    ax.text(xs[2] + bw / 2, y2 + (bh - 0.3) / 2,
            "Two clocks, never mixed:\nthe flow coordinate s ∈ [0, 1]\nis a training device; biological\ntime (hours, days, stages) is data.\n"
            "The field is autonomous and is\nintegrated over the biological\nhorizon; time will enter only\nthrough regulatory gates (Phase 3).\n\n"
            "Biological units are never split:\ncouplings live inside one\n(condition, replicate) group; whole\ntime points and replicates are held out.",
            ha="center", va="center", fontsize=5.9, color="#333333", linespacing=1.32)
    ax.text(0.2, H - 0.1, "Optimal-transport conditional flow matching for unpaired single-cell snapshots", fontsize=8.6,
            fontweight="bold", color=NAVY, va="top")
    save(fig, "fig01_pipeline")


# ---------------------------------------------------------------- Figure 2: OT vs random paths --------------------------------
def fig02_paths():
    s = crossing_lineages(n_per_cluster=70, seed=3, replicates=("r1",))
    xs = [s.x[s.biological_time == t] for t in TIMES5]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True, sharey=True)
    for ax, mode, title in zip(axes, ("ot", "random"), ("(a) OT-chained pseudo-trajectories", "(b) random chaining (ablation)")):
        traj = chain_ot(xs, np.random.default_rng(0), max_cells=140, mode=mode)      # (5, n, 2)
        path = SplinePath(TIMES5, traj)
        u = np.linspace(0, 4, 120)
        for i in range(0, traj.shape[1], 4):
            x, _ = path.evaluate(u, np.full(len(u), i))
            ax.plot(x[:, 0], x[:, 1], color=NAVY if mode == "ot" else RED, lw=0.5, alpha=0.45, zorder=1)
        for k, (t, x) in enumerate(zip(TIMES5, xs)):
            if t == 2.0:
                ax.scatter(x[:, 0], x[:, 1], s=16, facecolors="none", edgecolors=cmap(k / 4), lw=0.9, zorder=3,
                           label="t = 2 (held out)")
            else:
                ax.scatter(x[:, 0], x[:, 1], s=9, color=cmap(k / 4), lw=0, zorder=3, label=f"t = {int(t)}")
        mid, _ = path.evaluate(np.full(traj.shape[1], 2.0), np.arange(traj.shape[1]))
        ax.scatter(mid[:, 0], mid[:, 1], s=9, marker="x", color=NAVY if mode == "ot" else RED, lw=0.7, zorder=4,
                   label="path state at u = 2")
        ax.set_title(title)
        ax.set_xlabel("gene 1 (toy coordinate)")
        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-5.4, 5.4)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("gene 2 (toy coordinate)")
    axes[1].legend(loc="upper right", fontsize=7.2, markerscale=1.2, handletextpad=0.3)
    axes[1].annotate("crossing paths:\nstates at u = 2 collapse\ntowards the centre", xy=(0.2, 0.1), xytext=(-4.9, -4.9),
                     fontsize=7.4, color=RED, arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8))
    fig.tight_layout()
    save(fig, "fig02_paths_ot_vs_random")


# ---------------------------------------------------------------- Figure 3: architecture ---------------------------------------
def fig03_architecture():
    W, H, Y0 = 7.6, 4.3, 0.35
    fig, ax = canvas(W, H, y0=Y0)
    top = Y0 + H
    ax.text(0.15, top - 0.08, "Velocity-field architecture (selected model: autonomous field, spline targets)", fontsize=8.6,
            fontweight="bold", color=NAVY, va="top")
    ax.text(0.15, top - 0.36, "parameters: 44,464 at width 128 (selected GRN configuration) · 154,416 at width 256", fontsize=6.6,
            color=GREY, va="top")
    small_box(ax, 0.15, 3.5, 1.4, 0.55, "state x ∈ ℝᴳ", "standardised expression", fs_title=6.8, fs_body=6.0)
    small_box(ax, 0.15, 2.9, 1.4, 0.55, "condition c", "Embedding(n_cond, 8)", fs_title=6.8, fs_body=6.0)
    small_box(ax, 0.15, 2.3, 1.4, 0.55, "flow time s (optional)", "sinusoidal (32), flow clock", fc="white", ec=GREY, ls="--",
              fs_title=6.2, fs_body=5.6)
    small_box(ax, 1.75, 2.3, 0.55, 1.75, "concat", "G + 8\n(+ 32)", fs_title=6.8, fs_body=6.0)
    for yc in (3.78, 3.18):
        arrow(ax, (1.56, yc), (1.74, yc), lw=1.1)
    arrow(ax, (1.56, 2.58), (1.74, 2.58), color=GREY, ls="--", lw=1.0)
    hx = [2.5, 3.85, 5.2]
    for k, x in enumerate(hx):
        small_box(ax, x, 2.55, 1.2, 1.25, "Linear → SiLU", "width 256\n(128 in the\nselected config.)", fs_title=7.0, fs_body=6.1)
        arrow(ax, (x - 0.14 if k else 2.31, 3.17), (x - 0.02, 3.17), lw=1.2)
    small_box(ax, 6.55, 2.55, 0.9, 1.25, "Linear → ℝᴳ", "velocity v(x, c)\nper unit of\nbiological time", fs_title=7.0, fs_body=5.8)
    arrow(ax, (6.41, 3.17), (6.53, 3.17), lw=1.2)
    small_box(ax, 0.15, 0.5, 3.55, 1.45, "Training (spline targets)",
              "sample (x(u), ẋ(u), c) from the OT-chained splines\nloss ‖v(x(u) + σ_min ε, c) − ẋ(u)‖²\nAdam 10⁻³ · gradient clip 1.0 · batch 256\nEMA 0.999 weights; early stop on the validation\nenergy distance; no refit", fs_title=7.2, fs_body=6.2)
    small_box(ax, 3.9, 0.5, 3.55, 1.45, "Inference (biological horizon)",
              "x(t_b) = x(t_a) + ∫ v(x, c) dt over Δt = t_b − t_a\nRK4, 20 steps per unit of biological time\nany horizon: interpolation and forecast use\nthe same field; controls are scored in the\nsame standardised gene space", fs_title=7.2, fs_body=6.2)
    save(fig, "fig03_architecture")


# ---------------------------------------------------------------- Figure 4: protocol -------------------------------------------
def fig04_protocol():
    W, H = 7.6, 4.75
    fig, ax = canvas(W, H)
    ax.text(0.15, H - 0.08, "Evaluation protocol: leave-one-inner-time-point-out interpolation with nested selection", fontsize=8.6,
            fontweight="bold", color=NAVY, va="top")
    y = 3.35
    xs = np.linspace(0.55, 5.75, 7)
    ax.plot([xs[0] - 0.25, xs[-1] + 0.25], [y, y], color=GREY, lw=1.1, zorder=1)
    for k, x in enumerate(xs):
        if k == 3:
            ax.scatter([x], [y], s=120, facecolors="white", edgecolors=RED, lw=1.8, zorder=3)
            ax.text(x, y - 0.32, "t = 3\nheld out (test)", ha="center", va="top", fontsize=6.4, color=RED)
        elif k in (1, 5):
            ax.scatter([x], [y], s=95, color=NAVY, zorder=3)
            ax.scatter([x], [y], s=230, facecolors="none", edgecolors=ORANGE, lw=1.4, zorder=2)
            ax.text(x, y - 0.32, f"t = {k}\nvalidation fold", ha="center", va="top", fontsize=6.4, color=ORANGE)
        else:
            ax.scatter([x], [y], s=95, color=NAVY, zorder=3)
            ax.text(x, y - 0.32, f"t = {k}", ha="center", va="top", fontsize=6.4, color=NAVY)
    arrow(ax, (xs[2], y + 0.2), (xs[3] - 0.06, y + 0.2), color=RED, lw=1.6)
    ax.text(xs[2] - 0.1, y + 0.26, "predict over Δt = 1\nfrom the cells at t = 2", ha="right", va="bottom", fontsize=6.3, color=RED)
    ax.add_patch(FancyArrowPatch((xs[2], y + 0.08), (xs[4], y + 0.08), connectionstyle="arc3,rad=-0.55", arrowstyle="-",
                                 color=GREEN, lw=1.3, linestyle="--"))
    ax.text(xs[3], y + 0.78, "observed bracketing transition 2 → 4", ha="center", fontsize=6.4, color=GREEN)
    ax.text(xs[6] + 0.3, y, "terminal forecast 5 → 6:\ndiagnostic only\n(pure extrapolation)", va="center", fontsize=6.1, color=GREY)
    ax.text(0.15, 2.05, "Controls fitted on the same observed cells, scored in the same standardised gene space:", fontsize=6.9,
            color="#1a1a1a", va="bottom")
    rows = [["persistence", "mean interpolation", "McCann / OT interpolant", "linear map, same horizon"],
            ["MLP map, same horizon", "velocity replay", "flow model, shuffled time labels"]]
    for r, ctr in enumerate(rows):
        x = 0.15
        yy = 1.55 - 0.47 * r
        for c in ctr:
            w = 0.055 * len(c) + 0.2
            ax.add_patch(FancyBboxPatch((x, yy), w, 0.36, boxstyle="round,pad=0.015,rounding_size=0.06", fc=PALE, ec=NAVY, lw=0.9))
            ax.text(x + w / 2, yy + 0.18, c, ha="center", va="center", fontsize=6.4, color=NAVY)
            x += w + 0.1
    lines = ["Metrics: energy distance (U-statistic, primary), unbiased MMD, assignment W₂, error of means, per-gene W₁;",
             "permutation null (50 splits); five seeds; paired 95% t-intervals.",
             "Selection: mean validation energy distance over the two validation folds (t = 1 and t = 5), two search seeds;",
             "the test point is never touched. Verdict: below every control, paired CI95 < 0 vs the best control, better than shuffled time."]
    for i, line in enumerate(lines):
        ax.text(0.15, 0.78 - 0.22 * i, line, fontsize=6.4, color="#1a1a1a", va="center")
    save(fig, "fig04_protocol")


# ---------------------------------------------------------------- Figure 5: toys -----------------------------------------------
def fig05_toys():
    r = pd.read_csv("outputs/phase1_v1_toys/results.csv")
    ps = per_seed(r)
    names = {"toy_translation": "translation", "toy_variance": "variance change", "toy_bimodal": "bimodal split",
             "toy_condition_shift": "condition-dependent shift"}
    pretty = {"mean_shift": "mean shift", "linear": "linear map", "freefm_replay": "velocity replay", "mlp": "MLP map",
              "persistence": "persistence"}
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ds = list(names)
    xpos = np.arange(len(ds))
    fm = [ps[(ps.dataset == d) & (ps.model == "flow_matching")].energy_distance for d in ds]
    thr = [r[(r.dataset == d) & (r.model == "flow_matching")].null_q95.mean() for d in ds]
    sh = [ps[(ps.dataset == d) & (ps.model == "flow_matching_shuffled_time")].energy_distance for d in ds]
    best = []
    for d in ds:
        sub = ps[(ps.dataset == d) & ~ps.model.isin(["flow_matching", "flow_matching_shuffled_time", "noise_floor", "train_mean"])]
        g = sub.groupby("model").energy_distance.mean()
        best.append((g.idxmin(), g.min()))
    w = 0.2
    ax.bar(xpos - 1.5 * w, [v.mean() for v in fm], w, yerr=[v.std() for v in fm], color=BLUE, label="flow model", capsize=2)
    ax.bar(xpos - 0.5 * w, thr, w, color=GREY, alpha=0.55, label="permutation threshold (95%)")
    ax.bar(xpos + 0.5 * w, [max(b[1], 1e-9) for b in best], w, color=ORANGE, alpha=0.85, label="best control")
    ax.bar(xpos + 1.5 * w, [v.mean() for v in sh], w, color=RED, alpha=0.8, label="flow model, shuffled time")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{names[d]}\nbest control:\n{pretty.get(m, m)} ({v:.4f})" for d, (m, v) in zip(ds, best)], fontsize=7.4)
    ax.set_ylabel("energy distance (log scale)")
    ax.set_title("Toy transformations, held-out replicate (5 seeds, mean ± sd)")
    ax.legend(ncol=4, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    save(fig, "fig05_toys")


# ---------------------------------------------------------------- Figure 6: simulator ------------------------------------------
def fig06_simulator():
    sim = TimeVaryingGRNSimulator(n_genes=40, n_regulators=10, seed=0)
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    s, truth = sim.snapshot_set(times, n_cells_per_cohort=150, replicates=("r1",), seed=0)
    grid = np.linspace(0, 6, 121)
    act = np.stack([sim.activity(t) for t in grid])                                   # (121, E)
    kinds = np.array([sc.kind for sc in sim.schedules])
    tsw = np.array([sc.t_switch for sc in sim.schedules])
    order = np.lexsort((tsw, np.array([{"switch_on": 0, "switch_off": 1, "constant": 2}[k] for k in kinds])))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})
    ax = axes[0]
    im = ax.imshow(act[:, order].T, aspect="auto", cmap="Blues", extent=[0, 6, len(order), 0], vmin=0, vmax=1)
    ax.set_xlabel("biological time")
    ax.set_ylabel("edge (switch-on | switch-off | constant)", fontsize=8)
    ax.set_title("(a) edge activity schedule (ground truth)")
    n_on, n_off = (kinds == "switch_on").sum(), (kinds == "switch_off").sum()
    ax.axhline(n_on, color=ORANGE, lw=0.8)
    ax.axhline(n_on + n_off, color=ORANGE, lw=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="activity")
    ax = axes[1]
    Wm = truth["W"][:10, :]
    lim = np.abs(Wm).max()
    im = ax.imshow(Wm, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xlabel("target gene")
    ax.set_ylabel("regulator")
    ax.set_title(f"(b) signed weights, {int((truth['W'] != 0).sum())} edges")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="weight")
    ax = axes[2]
    mean = np.stack([s.x[s.biological_time == t].mean(0) for t in times])            # (7, G)
    z = (mean - mean.mean(0)) / (mean.std(0) + 1e-6)
    peak = np.argmax(z, axis=0)
    im = ax.imshow(z[:, np.argsort(peak)].T, aspect="auto", cmap="viridis", extent=[-0.5, 6.5, 40, 0])
    ax.set_xlabel("biological time (snapshots)")
    ax.set_ylabel("gene (sorted by peak time)")
    ax.set_title("(c) mean expression per snapshot (z-scored)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="z")
    fig.tight_layout()
    save(fig, "fig06_grn_simulator")


# ---------------------------------------------------------------- Figure 7: GRN results + selection ----------------------------
def fig07_grn_results():
    r = pd.read_csv("outputs/phase1_v2_grn_interp/results.csv")
    ps = per_seed(r)
    ps = ps[ps.model != "noise_floor"]
    g = ps.groupby("model").energy_distance.agg(["mean", "std"]).sort_values("mean", ascending=False)
    thr = r[r.model == "flow_matching"].null_q95.mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    colors = [BLUE if m == "flow_matching" else (RED if "shuffled" in m else "#b0b7c0") for m in g.index]
    ax.barh(range(len(g)), g["mean"], xerr=g["std"], color=colors, capsize=2, height=0.62)
    for i, m in enumerate(g.index):
        vals = ps[ps.model == m].energy_distance
        ax.scatter(vals, np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(vals)), s=8, color="k", zorder=3)
        ax.text(g["mean"].iloc[i] + 0.06, i, f"{g['mean'].iloc[i]:.3f}", va="center", fontsize=7.2)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([LABELS.get(m, m) for m in g.index], fontsize=7.8)
    ax.axvline(thr, color=GREEN, ls="--", lw=1)
    ax.text(thr + 0.05, len(g) - 0.55, f"permutation threshold {thr:.3f}", color=GREEN, fontsize=7)
    ax.set_xlabel("energy distance to the held-out t = 3 population (mean ± sd over 5 seeds)")
    ax.set_title("(a) held-out inner time point, 40-gene regulatory simulator")
    ax.set_xlim(0, g["mean"].max() * 1.22)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax = axes[1]
    sc = pd.read_csv("outputs/phase1_v2_grn_interp/search.csv")
    sc["family"] = sc.apply(lambda q: f"{q.path} paths · {q.clock} clock · {'all pairs' if q.pair_mode == 'all' else q.pair_mode}", axis=1)
    fam = sc.groupby(["family", "config_id"]).val_energy_distance.mean().reset_index()
    fams = fam.groupby("family").val_energy_distance.agg(["mean", "min", "max", "count"]).sort_values("mean", ascending=False)
    ax.barh(range(len(fams)), fams["mean"], color=[BLUE if f.startswith("spline") else "#b0b7c0" for f in fams.index], height=0.6)
    for i, f in enumerate(fams.index):
        vals = fam[fam.family == f].val_energy_distance
        ax.scatter(vals, np.full(len(vals), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(vals)), s=9, color="k", zorder=3)
        ax.text(fams["mean"].iloc[i] + 0.04, i, f"{fams['mean'].iloc[i]:.2f}  (n = {int(fams['count'].iloc[i])} configs)", va="center", fontsize=7.2)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(list(fams.index), fontsize=7.8)
    ax.set_xlabel("validation energy distance (mean of two folds, two seeds)")
    ax.set_title("(b) nested selection: configuration families")
    ax.set_xlim(0, fams["mean"].max() * 1.6)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "fig07_grn_interpolation")


# ---------------------------------------------------------------- Figure 8: OT ablation ----------------------------------------
def fig08_ablation():
    panels = [("outputs/ot_ablation", "(a) regulatory simulator, held-out t = 3 (40 genes)"),
              ("outputs/ot_crossing", "(b) two lineages moving apart, held-out t = 2")]
    avail = [(p, t) for p, t in panels if Path(p, "results.csv").exists()]
    fig, axes = plt.subplots(1, len(avail), figsize=(4.7 * len(avail), 3.6), squeeze=False)
    order = ["spline_ot", "spline_random", "linear_ot", "linear_indep", "ot_interp", "mean_interp"]
    short = {"spline_ot": "spline\nOT chain", "spline_random": "spline\nrandom chain",
             "linear_ot": "straight\nOT pairs", "linear_indep": "straight\nrandom pairs",
             "ot_interp": "McCann\ninterpolant", "mean_interp": "mean\ninterpolation"}
    col = {"spline_ot": NAVY, "spline_random": LIGHT, "linear_ot": ORANGE, "linear_indep": "#f3c99a", "ot_interp": "#8c8c8c", "mean_interp": "#c4c4c4"}
    for ax, (folder, title) in zip(axes[0], avail):
        r = pd.read_csv(Path(folder, "results.csv"))
        ps = per_seed(r)
        rows = [m for m in order if m in set(ps.model)]
        means = np.array([ps[ps.model == m].energy_distance.mean() for m in rows])
        sds = np.array([ps[ps.model == m].energy_distance.std() for m in rows])
        top = means.max()
        ax.bar(range(len(rows)), means, yerr=sds, color=[col[m] for m in rows], capsize=3, width=0.66, edgecolor="white")
        ax.axhline(0, color="#444444", lw=0.8)
        for i, m in enumerate(rows):
            v = ps[ps.model == m].energy_distance
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=9, color="k", zorder=3)
            ax.text(i, max(means[i] + sds[i], 0) + 0.025 * top, f"{means[i]:.3f}", ha="center", fontsize=7.2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([short[m] for m in rows], fontsize=7.0)
        ax.set_ylabel("energy distance (mean ± sd, 5 seeds)")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(min(-0.04 * top, means.min() * 1.8), top * 1.28)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        sj = Path(folder, "summary.json")
        if sj.exists():
            pairs = json.loads(sj.read_text(encoding="utf-8"))["paired"]
            keys = [k for k in ("spline_ot_vs_spline_random", "linear_ot_vs_linear_indep") if k in pairs]
            txt = "\n".join(f"{k.replace('_vs_', ' − ').replace('_', ' ')}: {pairs[k]['mean']:+.3f} [{pairs[k]['ci_low']:.3f}, {pairs[k]['ci_high']:.3f}]" for k in keys)
            ax.text(0.99, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6.9, color="#1a1a1a",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.6))
    fig.tight_layout()
    save(fig, "fig08_ot_ablation")


# ---------------------------------------------------------------- Figure 9: crossing prediction + learned field -----------------
def fig09_crossing_prediction():
    cache = OUT / "crossing_pred.npz"
    s = crossing_lineages()
    x1 = s.x[s.biological_time == 1.0]
    x2 = s.x[s.biological_time == 2.0]
    if cache.exists():
        z = np.load(cache)
        preds = {"ot": z["ot"], "random": z["random"]}
        fields = {"ot": (z["gx"], z["gy"], z["ot_u"], z["ot_v"]), "random": (z["gx"], z["gy"], z["random_u"], z["random_v"])}
    else:
        preds, fields = {}, {}
        gx, gy = np.meshgrid(np.linspace(-5, 5, 17), np.linspace(-5, 5, 17))
        grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
        base = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=2500, eval_every=250,
                           patience=6, clock="duration", coupling="ot", pair_mode="consecutive", path="spline")
        fit_mask = s.biological_time != 2.0
        for mode in ("ot", "random"):
            t0 = time.time()
            model, _ = train_flow(s, fit_mask, [0.0, 1.0, 3.0, 4.0], base.replace(chain=mode, seed=0))
            preds[mode] = model.predict(x1, "ctrl", 1.0, 2.0)
            with torch.no_grad():
                zg = torch.as_tensor(model.standardizer.transform(grid), dtype=torch.float32)
                ctx = model._context("ctrl", 1.0, 2.0, len(zg))
                v = model.vector_field(zg, torch.zeros(len(zg)), ctx).numpy() * model.standardizer.std_
            fields[mode] = (gx, gy, v[:, 0].reshape(gx.shape), v[:, 1].reshape(gx.shape))
            print(f"trained {mode} in {time.time() - t0:.0f}s", flush=True)
        np.savez(cache, ot=preds["ot"], random=preds["random"], gx=fields["ot"][0], gy=fields["ot"][1],
                 ot_u=fields["ot"][2], ot_v=fields["ot"][3], random_u=fields["random"][2], random_v=fields["random"][3])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharex=True, sharey=True)
    for ax, mode, colr, title in zip(axes, ("ot", "random"), (NAVY, RED), ("(a) OT-chained spline paths", "(b) random chaining")):
        gx, gy, u, v = fields[mode]
        sp = np.hypot(u, v)
        ax.quiver(gx, gy, u / (sp + 1e-6), v / (sp + 1e-6), sp, cmap="Greys", scale=28, width=0.004, alpha=0.9, zorder=1)
        ax.scatter(x1[:, 0], x1[:, 1], s=6, color="#9dc3e6", lw=0, label="cells at t = 1 (start)", zorder=2)
        ax.scatter(x2[:, 0], x2[:, 1], s=14, facecolors="none", edgecolors="#444444", lw=0.7, label="truth at t = 2 (held out)", zorder=3)
        ax.scatter(preds[mode][:, 0], preds[mode][:, 1], s=7, color=colr, lw=0, label="prediction at t = 2", zorder=4)
        ed = energy_distance(preds[mode][:600], x2[:600])
        ax.set_title(f"{title}   energy distance {ed:.3f}")
        ax.set_xlabel("gene 1 (toy coordinate)")
        ax.set_xlim(-5.3, 5.3)
        ax.set_ylim(-5.3, 5.3)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("gene 2 (toy coordinate)")
    axes[1].legend(loc="lower left", fontsize=7.2, markerscale=1.5)
    fig.suptitle("Learned velocity field (arrows, grey = speed) and one-step prediction from t = 1 to t = 2", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, "fig09_crossing_prediction")


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in [("fig01", fig01_pipeline), ("fig02", fig02_paths), ("fig03", fig03_architecture), ("fig04", fig04_protocol),
                     ("fig05", fig05_toys), ("fig06", fig06_simulator), ("fig07", fig07_grn_results), ("fig08", fig08_ablation),
                     ("fig09", fig09_crossing_prediction)]:
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as exc:  # keep producing the other figures, report the failure
            import traceback
            traceback.print_exc()
            print(f"FAILED {name}: {exc}", flush=True)
