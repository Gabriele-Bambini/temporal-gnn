"""Figures for tack/report/REPORT_OT_couplings_vs_snapshots.md, read only from the frozen result JSONs in tack/results.
  fig1_interpolation_vs_mix.png    ED_U of OT pairs and random pairs / ED_U of the 50/50 mix, per held-out unit
  fig2_enrichment_vs_spread.png    clonal enrichment vs spread (ot / knn / soft), cytokine Epo 2->4 and G-csf 2->4
  fig3_matched_spread_ratios.png   OT/kNN and OT/soft at OT's spread: in vitro, cytokine, in vivo (not evaluable)
Palette: dataviz reference slots 1-3 (blue, orange, aqua; validated all-pairs, light), shapes as secondary encoding.
Usage: python make_figures_2026-10-30.py <out_dir>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#a9a8a3"
ATLAS_STAGE = {k: f"E{6.5 + 0.25 * k:g}" for k in range(1, 8)}          # 9 stages E6.5-E8.5, k = 1..7 held out
ZF_HPF = {k: f"{18 + 2 * k}" for k in range(1, 10)}                      # 18-38 hpf every 2 h, k = 1..9 held out
UNIT = {"Epo_24": "Epo 2→4", "Gcsf_24": "G-csf 2→4", "Mcsf_24": "M-csf 2→4", "SCF_24": "SCF 2→4",
        "full_24": "Full 2→4", "Epo_46": "Epo 4→6", "Gcsf_46": "G-csf 4→6", "Mcsf_46": "M-csf 4→6",
        "full_46": "Full 4→6", "sca1pos_24": "Sca1+ 2→4", "sca1pos_46": "Sca1+ 4→6", "sca1neg_24": "Sca1− 2→4",
        "sca1neg_46": "Sca1− 4→6"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titlecolor": INK,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})


def load(p):
    with open(os.path.join(RES, p)) as fh:
        return json.load(fh)


def style(ax, logy=True):
    if logy:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig1(out):
    panels = []
    for title, p, lab in [("Mouse gastrulation atlas", "atlas/confirm.json", ATLAS_STAGE),
                          ("Zebrafish embryos (hpf)", "zf/confirm_zf.json", ZF_HPF)]:
        d = load(p)["interp"]
        ks = sorted(d, key=int)
        m = [d[k]["mean"] for k in ks]
        panels.append((title, [lab[int(k)] for k in ks], [x["ot_pairs"]["ED_U"] / x["union"]["ED_U"] for x in m],
                       [x["indep"]["ED_U"] / x["union"]["ED_U"] for x in m]))
    d = load("larry/larry_confirm.json")["interp"]
    lab, ot, rp = [], [], []
    for pop, name in (("sca1pos", "Sca1+"), ("sca1neg", "Sca1−")):
        pm = d[pop]["pool_mean"]
        for w in d[pop]["pools"]:
            lab.append(f"{name}\nw{w}")
            ot.append(pm["ot_pairs"][w] / pm["union"][w])
            rp.append(pm["indep"][w] / pm["union"][w])
    panels.append(("LARRY in vitro, day 4", lab, ot, rp))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [7, 9, 4.8]}, sharey=True)
    for ax, (title, lab, ot, rp) in zip(axes, panels):
        x = range(len(lab))
        ax.axhline(1, color=INK2, lw=0.9, ls="--")
        ax.plot(x, rp, ls="none", marker="s", ms=5.5, color=ORANGE, mec="white", mew=0.8, label="random pairs / mix")
        ax.plot(x, ot, ls="none", marker="o", ms=6, color=BLUE, mec="white", mew=0.8, label="OT pairs / mix")
        ax.set_xticks(list(x))
        ax.set_xticklabels(lab, fontsize=7)
        ax.set_title(title, loc="left")
        style(ax)
        ax.set_xlim(-0.6, len(lab) - 0.4)
    axes[0].set_ylabel("energy distance / mix's\n(below 1 = better than the mix)")
    axes[0].set_yticks([0.8, 1, 1.5, 2, 3, 5, 10, 20, 30])
    axes[0].set_yticklabels(["0.8", "1", "1.5", "2", "3", "5", "10", "20", "30"])
    axes[0].minorticks_off()
    for ax in axes:
        ax.minorticks_off()
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lb[::-1], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "fig1_interpolation_vs_mix.png"), dpi=220)
    plt.close(fig)


def fig2(out):
    r = load("larry_cyto/larry_cyto_confirm.json")["units"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, name in zip(axes, ("Epo_24", "Gcsf_24")):
        cur = r[name]["curves"]
        ax.axhline(1, color=INK2, lw=0.9, ls=":")
        for fam, col, mk, lab in (("knn", ORANGE, "s", "kNN (K)"), ("soft", AQUA, "^", "soft kernel (no mass constraint)"),
                                  ("ot", BLUE, "o", "OT")):
            xs = [c["spread"] for c in cur[fam]]
            ys = [c["enrichment"] for c in cur[fam]]
            ax.plot(xs, ys, color=col, lw=1.6, marker=mk, ms=5, mec="white", mew=0.7, label=lab)
        ot0 = r[name]["ot0"]
        ax.plot([ot0["spread"]], [ot0["enrichment"]], ls="none", marker="o", ms=11, mfc="none", mec=INK, mew=1.0,
                label="OT, pre-registered ε = 0.05")
        ax.set_xscale("log")
        ax.set_title(f"Cytokine perturbation, {UNIT[name]}", loc="left")
        ax.set_xlabel("spread (effective partners per cell)")
        style(ax, logy=False)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("clonal enrichment (1 = chance)")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lb[::-1], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "fig2_enrichment_vs_spread.png"), dpi=220)
    plt.close(fig)


def fig3(out):
    rb = load("larry/larry_robust.json")
    vitro = (list(rb["units"]), rb["reading"]["ratio_ot_knn"], rb["reading"]["ratio_ot_soft"])
    cy = load("larry_cyto/larry_cyto_confirm.json")["units"]
    order = ["Epo_24", "Gcsf_24", "Mcsf_24", "SCF_24", "full_24", "Epo_46", "Gcsf_46", "Mcsf_46", "full_46"]
    cyto = (order, [cy[n]["matched"]["knn"]["ratio"] for n in order], [cy[n]["matched"]["soft"]["ratio"] for n in order])
    vv = load("larry_vivo/larry_vivo_confirm.json")["units"]
    vn = sorted(vv)
    vivo = ([n.split("_")[1] for n in vn], [vv[n]["matched"]["knn"]["ratio"] for n in vn],
            [vv[n]["matched"]["soft"]["ratio"] for n in vn])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [5.6, 9, 7.4]}, sharey=True)
    specs = [("In vitro (exploratory)", vitro, False), ("Cytokine (pre-registered)", cyto, False),
             ("In vivo: not evaluable", vivo, True)]
    for ax, (title, (names, rk, rs), grey) in zip(axes, specs):
        x = list(range(len(names)))
        ax.axhline(1, color=INK2, lw=0.9, ls="--")
        ck, cs = (MUTED, MUTED) if grey else (ORANGE, AQUA)
        ax.plot([i - 0.12 for i in x], rk, ls="none", marker="s", ms=5.5, color=ck, mec="white", mew=0.8,
                label="OT / kNN at OT's spread")
        ax.plot([i + 0.12 for i in x], rs, ls="none", marker="^", ms=6, color=cs, mec="white", mew=0.8,
                label="OT / soft kernel at OT's spread")
        ax.set_xticks(x)
        ax.set_xticklabels([UNIT.get(n, n).replace(" ", "\n", 1) for n in names], fontsize=6.4)
        ax.set_title(title, loc="left", color=INK2 if grey else INK)
        style(ax)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.minorticks_off()
    axes[0].set_yticks([0.8, 1, 1.5, 2, 3, 4])
    axes[0].set_yticklabels(["0.8", "1", "1.5", "2", "3", "4"])
    axes[0].set_ylabel("ratio of clonal enrichment\n(above 1 = OT better)")
    h, lb = axes[1].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(out, "fig3_matched_spread_ratios.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    fig1(out)
    fig2(out)
    fig3(out)
    print("figures written to", out)
