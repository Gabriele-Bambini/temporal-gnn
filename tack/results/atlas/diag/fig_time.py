"""Figure: modelling through time on the gastrulation atlas (real data; EXPLORATORY diagnosis figure).

A: all 9 stages in one PCA space (origin7.npz), cells coloured by stage, stage centroids joined in time.
B-D: the interpolation task at E7.75 (interp5.npz): the two neighbours, the 50/50 mix, OT pairs, against the truth.
E: embryo clock per biological pool (diag_oracle.json B). F: neighbour separation vs time gap (diag_gap.json).
G: OT / mix ratio vs time gap (diag_gap.json). Writes modeling_through_time.png next to this file.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = os.environ.get("DIAG_DATA", r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad")
STAGES = ["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
TRUTH, MIX, OT, NOISE = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue", ["#c9dcf5", "#2a78d6", "#0c2f5e"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(13, 11.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.28)

    # ---- A: the whole time course
    ax = fig.add_subplot(gs[0, :])
    d = np.load(os.path.join(SCR, "atlas", "origin7.npz"), allow_pickle=True)
    Z = np.vstack([d["Z_past"], d["Z_fut"]])[:, :2]
    st = np.r_[d["stage_past"], d["stage_fut"]]
    cents = []
    for s in range(9):
        i = np.where(st == s)[0]
        i = rng.choice(i, min(450, len(i)), replace=False)
        ax.scatter(Z[i, 0], Z[i, 1], s=4, color=SEQ(s / 8), alpha=0.45, linewidths=0)
        cents.append(Z[st == s].mean(0))
    cents = np.array(cents)
    lo, hi = np.percentile(Z, [1, 97], axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.plot(cents[:, 0], cents[:, 1], color=INK, lw=2, zorder=3)
    offs = [(-34, -4), (4, -14), (-42, -10), (-4, 10), (8, -14), (8, 4), (8, 6), (4, 10), (-36, 6)]
    for s, c in enumerate(cents):
        ax.scatter(*c, s=90, color=SEQ(s / 8), edgecolor=SURF, linewidth=2, zorder=4)
        ax.annotate(STAGES[s], c, xytext=offs[s], textcoords="offset points", fontsize=9, color=INK, zorder=5,
                    fontweight="bold")
    ax.set_title("A. L'embrione nel tempo: 9 stadi veri, 6 ore l'uno dall'altro (ogni punto = una cellula; "
                 "linea nera = centro di ogni stadio)", loc="left")
    ax.text(0.55, 0.3, "Le nuvole di stadi vicini si sovrappongono quasi del tutto:\nil centro si sposta poco "
            "rispetto alla larghezza della nuvola.", transform=ax.transAxes, color=INK2)
    clean(ax)

    # ---- B-D: the interpolation task at E7.75
    k = 5
    d = np.load(os.path.join(SCR, "atlas", f"interp{k}.npz"), allow_pickle=True)
    stf, Zf, Zh = d["stage_fit"], d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
    A, B = np.where(stf == k - 1)[0], np.where(stf == k + 1)[0]
    iA, iB = IB.sub(len(A), seed=1000), IB.sub(len(B), seed=2000)
    ZA, ZB = Zf[A[iA]], Zf[B[iB]]
    M = IB.methods(ZA, Zf[A[iA]], ZB, Zf[B[iB]], np.zeros(len(iA)), np.eye(Zf.shape[1]), 1500,
                   np.random.default_rng(1000 * k), which=("union", "ot_pairs"))
    Zu, Zo = M["union"][0], M["ot_pairs"][0]
    lim = np.percentile(np.vstack([ZA, ZB, Zh])[:, :2], [1, 99], axis=0)
    pad = 0.1 * (lim[1] - lim[0])

    def panel(pos, layers, title, note):
        ax = fig.add_subplot(gs[1, pos])
        for P, col, lab, a in layers:
            j = rng.choice(len(P), min(900, len(P)), replace=False)
            ax.scatter(P[j, 0], P[j, 1], s=5, color=col, alpha=a, linewidths=0, label=lab)
        ax.set_xlim(lim[0, 0] - pad[0], lim[1, 0] + pad[0])
        ax.set_ylim(lim[0, 1] - pad[1], lim[1, 1] + pad[1])
        ax.set_title(title, loc="left")
        ax.text(0.02, 0.02, note, transform=ax.transAxes, color=INK2, fontsize=8.5)
        leg = ax.legend(loc="upper right", frameon=False, markerscale=3, fontsize=8)
        for t in leg.get_texts():
            t.set_color(INK)
        clean(ax)

    panel(0, [(ZA, "#b5b3ad", "E7.5 (prima)", 0.5), (ZB, "#52514e", "E8.0 (dopo)", 0.5), (Zh, TRUTH, "E7.75 vero (nascosto)", 0.6)],
          "B. Il compito: ricostruire E7.75", "Il modello vede solo prima e dopo.")
    panel(1, [(Zh, TRUTH, "E7.75 vero", 0.35), (Zu, MIX, "metà e metà", 0.55)],
          "C. Metà e metà (miscela)", "Errore 0.248  (vince)")
    panel(2, [(Zh, TRUTH, "E7.75 vero", 0.35), (Zo, OT, "OT a coppie", 0.55)],
          "D. OT: ogni cellula va a metà strada", "Errore 0.403  (crea cellule a metà tra tipi diversi,\nche non esistono)")

    # ---- E: embryo clock
    o = json.load(open(os.path.join(HERE, "diag_oracle.json")))
    pools = o["B"]["pools"]
    ax = fig.add_subplot(gs[2, 0])
    nom = np.array([v["stage"] for v in pools.values()], float)
    age = np.array([v["age_mean"] for v in pools.values()])
    jit = np.array([(-1) ** i * 0.06 * (i % 3) for i in range(len(nom))])
    ax.plot([0, 8], [0, 8], color=INK2, lw=1, ls="--")
    ax.scatter(nom + jit, age, s=40, color=TRUTH, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.set_xticks(range(9))
    ax.set_xticklabels(STAGES, rotation=45, fontsize=7.5)
    ax.set_yticks(range(9))
    ax.set_yticklabels(STAGES, fontsize=7.5)
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("stadio scritto sull'etichetta")
    ax.set_ylabel("età stimata dalle cellule")
    ax.set_title("E. L'etichetta non è l'orologio", loc="left")
    ax.text(0.03, 0.9, "ogni punto = un gruppo di embrioni\n14 su 24 più vicini a un altro stadio",
            transform=ax.transAxes, color=INK2, fontsize=8)

    # ---- F, G: time gap
    g = json.load(open(os.path.join(HERE, "diag_gap.json")))["units"]
    hrs = np.array([u["g"] * 6 for u in g])
    sep = np.array([u["separation"] for u in g])
    ax = fig.add_subplot(gs[2, 1])
    ax.axhspan(1.18, 1.93, color=MIX, alpha=0.12, lw=0)
    ax.text(6.3, 1.55, "eritroidi a 6 ore\n(dove l'OT vinceva)", color=INK2, fontsize=8, va="center")
    ax.scatter(hrs + rng.uniform(-0.6, 0.6, len(hrs)), sep, s=34, color=TRUTH, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xticks([6, 12, 18, 24])
    ax.set_xlabel("salto di tempo tra prima e dopo (ore)")
    ax.set_ylabel("quanto si sposta / quanto è larga")
    ax.set_ylim(0, 2.1)
    ax.grid(color=GRID, lw=0.6, axis="y")
    ax.set_title("F. L'atlante si muove piano", loc="left")

    ax = fig.add_subplot(gs[2, 2])
    for key, col, lab in (("ot_pairs", OT, "OT a coppie"), ("sb_half", NOISE, "OT con rumore")):
        xs, ys = [], []
        for h in (6, 12, 18, 24):
            r = [u["mean"][key] / u["mean"]["union"] for u in g if u["g"] * 6 == h]
            xs.append(h)
            ys.append(np.mean(r))
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=lab)
        ax.annotate(lab, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    ax.axhline(1.0, color=MIX, lw=2, ls="--")
    ax.text(6, 1.005, "metà e metà", color=INK, fontsize=8, va="bottom")
    ax.set_xticks([6, 12, 18, 24])
    ax.set_xlim(4, 29)
    ax.set_xlabel("salto di tempo (ore)")
    ax.set_ylabel("errore OT / errore metà e metà")
    ax.grid(color=GRID, lw=0.6, axis="y")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("G. Sotto la linea = l'OT vince", loc="left")

    fig.suptitle("Modellare l'embrione nel tempo: perché l'OT non vince sull'atlante (dati veri, analisi esplorativa)",
                 x=0.07, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.savefig(os.path.join(HERE, "modeling_through_time.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
