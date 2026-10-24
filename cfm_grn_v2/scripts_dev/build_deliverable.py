"""Assemble the deliverable folder and zip: package code + tests, report (md/docx/pdf), key results, docs.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/build_deliverable.py
"""
import io
import re
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

root = Path(".").resolve()
proj = root.parent
name = "cfm_grn_v2_deliverable_2026-09-25"
dst = proj / "deliverables" / name


def clear_tree(path: Path, attempts: int = 6) -> None:
    """Remove a previous build. The sync client can hold a directory open, so retry before failing."""
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            for f in path.rglob("*"):
                try:
                    os.chmod(f, stat.S_IWRITE)
                except OSError:
                    pass
            time.sleep(1.5)


if dst.exists():
    clear_tree(dst)
dst.mkdir(parents=True)

# --- report: fill the ablation tables ------------------------------------------------------------------
md = io.open(root / "DELIVERABLE_2026-09-19_OT_flow_matching.md", encoding="utf-8").read()
LABELS = {"spline_ot": "spline paths, OT-chained (selected path family)", "spline_random": "spline paths, random chaining",
          "linear_ot": "straight pair paths, OT coupling", "linear_indep": "straight pair paths, independent coupling",
          "spline_ot_shuffled_time": "spline OT, shuffled time labels", "mean_interp": "mean interpolation (control)",
          "ot_interp": "McCann / OT interpolant (control)", "linear": "linear map (control)", "mlp": "MLP map (control)",
          "persistence": "persistence (control)", "freefm_replay": "velocity replay (control)"}


def minus(s: str) -> str:
    """Typographic minus on numbers only, so hyphenated words stay hyphenated."""
    return re.sub(r"-(?=\d)", "\u2212", s)


def ablation_table(folder: Path, placeholder: str, text: str) -> str:
    import json
    if not (folder / "summary.csv").exists():
        return text.replace(placeholder, f"_(experiment still running when this file was built; see {folder.name}/)_")
    s = pd.read_csv(folder / "summary.csv").set_index("model")
    rows = ["| arm | energy distance, mean over 5 seeds | sd |", "|---|---|---|"]
    arms = ["spline_ot", "spline_random", "linear_ot", "linear_indep"]
    order = [m for m in arms if m in s.index] + [m for m in s.index if m not in arms and m != "noise_floor"]
    for m in order:
        r = s.loc[m]
        rows.append(f"| {LABELS.get(m, m)} | {r['mean']:.3f} | {r['std']:.3f} |")
    pairs = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["paired"]
    arm = {"spline_ot": "OT-chained spline", "spline_random": "randomly chained spline", "linear_ot": "straight OT paths",
           "linear_indep": "straight independent paths", "ot_interp": "the McCann / OT interpolant",
           "mean_interp": "mean interpolation", "shuffled_time": "shuffled time labels"}

    def name(k):
        a, c = k.split("_vs_")
        return f"{arm.get(a, a)} vs {arm.get(c, c)}"

    txt = "\n".join(rows) + "\n\nPaired differences, first minus second (n = 5 seeds): " + "; ".join(
        minus(f"{name(k)} {v['mean']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]") for k, v in pairs.items()) + "."
    return text.replace(placeholder, txt)


md = ablation_table(root / "outputs" / "ot_ablation", "OT_ABLATION_TABLE", md)
md = ablation_table(root / "outputs" / "ot_crossing", "OT_CROSSING_TABLE", md)


def combo_table(text: str) -> str:
    f = root / "outputs" / "combo_scan" / "results.csv"
    if not f.exists():
        return text.replace("COMBO_TABLE", "_(selection run not available)_")
    d = pd.read_csv(f)
    label = {"base": "path loss on the fit knots (version 0.1.8)",
             "refit": "+ refit on every observed knot",
             "ft": "+ windowed path supervision and marginal matching",
             "ft_refit": "both corrections (selected)"}
    rows = ["| configuration | fold t = 1 | fold t = 5 | mean | change vs the published model |", "|---|---|---|---|---|"]
    piv = d.pivot_table(index="arm", columns="fold", values="energy_distance", aggfunc="mean")
    base = d[d.arm == "base"].set_index(["fold", "seed"]).energy_distance
    for arm in ("base", "refit", "ft", "ft_refit"):
        if arm not in piv.index:
            continue
        sub = d[d.arm == arm].set_index(["fold", "seed"]).energy_distance
        dl = (sub - base.reindex(sub.index)).dropna()
        chg = "reference" if arm == "base" else f"{dl.mean():+.4f}".replace("-", "\u2212")
        rows.append(f"| {label[arm]} | {piv.loc[arm, 1.0]:.4f} | {piv.loc[arm, 5.0]:.4f} | "
                    f"{piv.loc[arm].mean():.4f} | {chg} |")
    return text.replace("COMBO_TABLE", "\n".join(rows))


def final_table(text: str) -> str:
    import json
    f = root / "outputs" / "final_test" / "summary.json"
    if not f.exists():
        return text.replace("FINAL_TEST_TABLE", "_(final measurement not available)_")
    j = json.loads(f.read_text(encoding="utf-8"))
    nm = {"flow_v019": "flow model, version 0.1.9 (all knots + marginal matching)",
          "flow_v018": "flow model, version 0.1.8 (published)",
          "flow_v019_shuffled_time": "version 0.1.9, shuffled time labels",
          "flow_v018_shuffled_time": "version 0.1.8, shuffled time labels"}
    rows = ["| model | energy distance, mean over 5 seeds | sd |", "|---|---|---|"]
    for r in j["summary"]:
        if r["model"] == "noise_floor":
            continue
        rows.append(f"| {nm.get(r['model'], LABELS.get(r['model'], r['model']))} | {r['mean']:.3f} | {r['std']:.3f} |")
    pname = {"v019_vs_v018": "version 0.1.9 vs version 0.1.8",
             "v019_vs_mean_interp": "vs mean interpolation",
             "v019_vs_ot_interp": "vs the McCann / OT interpolant",
             "v019_vs_shuffled_time": "vs shuffled time labels"}
    p = j["paired"]
    txt = "\n".join(rows) + "\n\nPaired differences (n = 5 seeds): " + "; ".join(
        f"{pname.get(k, k)} {v['mean']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]".replace("-", "\u2212")
        for k, v in p.items()) + "."
    return text.replace("FINAL_TEST_TABLE", txt)


READABLE = {"flow_matching": "flow model, version 0.1.10", "flow_matching_shuffled_time": "the same model, shuffled time labels",
            "mean_interp": "mean interpolation", "ot_interp": "the McCann / OT interpolant", "linear": "linear map",
            "mlp": "MLP map", "persistence": "persistence", "freefm_replay": "velocity replay",
            "noise_floor": "noise floor (two halves of the true window)"}


def flow_null_q95(folder: Path) -> float:
    d = per_seed_local(pd.read_csv(folder / "results.csv"))
    return float(d.loc[d["model"] == "flow_matching", "null_q95"].mean())


def per_seed_local(df):
    sys.path.insert(0, str(root / "src"))
    from cfm_grn_v2.evaluation.report import per_seed
    return per_seed(df)


def bench_table(folder: Path, placeholder: str, text: str, unit: str, extra: str = "") -> str:
    """One real-data benchmark (summary.json + results.csv of eb_benchmark.py / argelaguet_benchmark.py)."""
    import json
    f = folder / "summary.json"
    if not f.exists():
        return text.replace(placeholder, f"_(run not available: {folder.name})_")
    j = json.loads(f.read_text(encoding="utf-8"))
    summ = [r for r in j["summary"] if r["model"] != "noise_floor"] + [r for r in j["summary"] if r["model"] == "noise_floor"]
    rows = [f"| model | energy distance, mean over {int(summ[0]['count'])} seeds | sd |", "|---|---|---|"]
    for r in summ:
        label = READABLE.get(r["model"], r["model"])
        if r["model"] not in ("flow_matching", "flow_matching_shuffled_time", "noise_floor"):
            label += " (control)"
        rows.append(minus(f"| {label} | {r['mean']:.3f} | {r['std']:.3f} |"))
    txt = "\n".join(rows) + "\n\nPaired differences, flow model minus the named model (n = 5 seeds): " + "; ".join(
        minus(f"vs {READABLE.get(k.replace('flow_vs_', ''), k)} {v['mean']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
        for k, v in j["paired"].items()) + ". "
    g = j["genes"]
    filt = (f"among the genes detected in at least {100 * j['min_detect']:.0f} % of the fit cells, "
            if j.get("min_detect", 0) > 0 else "")
    txt += (f"Gene space: {g['n_selected']} genes ({filt}the {g['n_hvg_requested']} most variable on the fit cells, "
            f"union the {g['n_prior_regulators']} prior regulators); permutation threshold at the flow model's sample "
            f"sizes {flow_null_q95(folder):.3f}; at most {j['max_cells_per_group']} cells per {unit}." + extra)
    return text.replace(placeholder, txt)


def replicate_sentence(folder: Path) -> str:
    import json
    f = folder / "summary.json"
    if not f.exists():
        return ""
    reps = json.loads(f.read_text(encoding="utf-8")).get("replicate_yardstick", {})
    if not reps:
        return ""
    return (" Energy distance between the two biological replicates of each stage, in the same space: "
            + ", ".join(f"{k} {v['energy_distance']:.3f}" for k, v in reps.items()) + ".")


def eb_table(text: str) -> str:
    return bench_table(root / "outputs" / "eb_v0110", "EB_TABLE", text, "window")


def arg_table(text: str) -> str:
    folder = root / "outputs" / "argelaguet_v0110"
    return bench_table(folder, "ARG_TABLE", text, "stage", replicate_sentence(folder))


md = combo_table(md)
md = final_table(md)
md = eb_table(md)
md = arg_table(md)


def eb_pca_sentence(text: str) -> str:
    import json
    f = root / "outputs" / "eb_v0110_pca50" / "summary.json"
    if not f.exists():
        return text.replace("EB_PCA_SENTENCE", "")
    j = json.loads(f.read_text(encoding="utf-8"))
    m = {r["model"]: r for r in j["summary"]}
    p = j["paired"]
    best = min((r for k, r in m.items() if k not in ("flow_matching", "noise_floor", "flow_matching_shuffled_time")),
               key=lambda r: r["mean"])
    key = f"flow_vs_{best['model']}"
    txt = (f"**Robustness, 50 principal components.** The same run in a 50-dimensional space fitted on the fit cells "
           f"only (a reduced space, closer to those the published methods work in but not the same: their benchmark "
           f"uses five whitened components and the Wasserstein-1 distance, MIOFlow 200 components, so these numbers "
           f"are still not comparable to theirs) gives the flow model "
           f"{m['flow_matching']['mean']:.3f} against {best['mean']:.3f} for the best control "
           f"({READABLE.get(best['model'], best['model'])}), paired difference {p[key]['mean']:+.3f} "
           f"[{p[key]['ci_low']:+.3f}, {p[key]['ci_high']:+.3f}], and {p['flow_vs_flow_matching_shuffled_time']['mean']:+.3f} "
           f"against shuffled time labels. ")
    lo, hi = p[key]["ci_low"], p[key]["ci_high"]
    if hi < 0:
        txt += "The ranking does not depend on working in genes."
    elif lo > 0:
        txt += ("In this space mean interpolation is better than the flow model, which still beats every other "
                "control. The gene-space win should not be read as independent of the representation (§7).")
    else:
        txt += ("In this space the flow model and mean interpolation are statistically tied, and the flow model "
                "still beats every other control, including the OT interpolant and the shuffled-time model. The "
                "gene-space win over mean interpolation should therefore not be read as independent of the "
                "representation (§7).")
    return text.replace("EB_PCA_SENTENCE", minus(txt))


md = eb_pca_sentence(md)

ARM_LABEL = {"base": "version 0.1.9 (hidden 128, spline centre)", "center": "straight-line centre",
             "wide": "hidden 512", "wide_center": "straight-line centre and hidden 512",
             "time": "0.1.10 + field sees biological time", "noft": "0.1.10 without the marginal fine-tune"}
STAGE = {"eb": {1.0: "days 6–9", 3.0: "days 18–21"}, "argelaguet": {1.0: "E7.75", 4.0: "E8.5"}}
DS_NAME = {"eb": "embryoid body", "argelaguet": "gastrulation"}


def selection_table(sets: dict, base_label: str) -> str | None:
    """sets = {dataset: output folder of select_config.py}; one column block per data set."""
    import json
    have = {ds: f for ds, f in sets.items() if (f / "choice.json").exists()}
    if not have:
        return None
    blocks = {}
    for ds, f in have.items():
        d = pd.read_csv(f / "results.csv")
        c = json.loads((f / "choice.json").read_text(encoding="utf-8"))
        piv = d.pivot_table(index="arm", columns="fold", values="energy_distance", aggfunc="mean")
        blocks[ds] = (piv, c)
    arms = list(next(iter(blocks.values()))[0].index)
    arms = ["base"] + [a for a in arms if a != "base"]
    head, sep = ["configuration"], ["---"]
    for ds, (piv, c) in blocks.items():
        for fold in piv.columns:
            head.append(f"{DS_NAME[ds]}, {STAGE[ds].get(fold, fold)}")
            sep.append("---")
        head.append(f"{DS_NAME[ds]}: change (pairs better)")
        sep.append("---")
    rows = ["| " + " | ".join(head) + " |", "|" + "|".join(sep) + "|"]
    for a in arms:
        cells = [ARM_LABEL.get(a, a) if a != "base" else base_label]
        for ds, (piv, c) in blocks.items():
            cells += [f"{piv.loc[a, fold]:.3f}" for fold in piv.columns]
            if a == "base":
                cells.append("reference")
            else:
                pa = c["per_arm"][a]
                win = " **chosen**" if c["winner"] == a else ""
                cells.append(minus(f"{pa['mean_change']:+.3f} ({pa['n_better']}/{pa['n']}){win}"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def select_v0110_table(text: str) -> str:
    t = selection_table({"eb": root / "outputs" / "eb_select", "argelaguet": root / "outputs" / "argelaguet_select"},
                        ARM_LABEL["base"])
    return text.replace("SELECT_V0110_TABLE", t or "_(selection runs not available)_")


def history_table(text: str) -> str:
    import json
    runs = [("embryoid body", "first run: no gene filter, version 0.1.9", "eb_interp"),
            ("embryoid body", "gene filter, version 0.1.9", "eb_filter_base"),
            ("embryoid body", "gene filter, version 0.1.10", "eb_v0110"),
            ("embryoid body, 50 PCs", "first run: no gene filter, version 0.1.9", "eb_interp_pca50"),
            ("embryoid body, 50 PCs", "gene filter, version 0.1.10", "eb_v0110_pca50"),
            ("gastrulation", "first run: no gene filter, version 0.1.9", "argelaguet_interp"),
            ("gastrulation", "gene filter, version 0.1.9", "argelaguet_filter_base"),
            ("gastrulation", "gene filter, version 0.1.10", "argelaguet_v0110")]
    rows = ["| data set | run | genes | flow model | mean interpolation | flow − mean interpolation | "
            "flow − shuffled time |", "|---|---|---|---|---|---|---|"]
    for ds, label, run in runs:
        f = root / "outputs" / run / "summary.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        m = {r["model"]: r["mean"] for r in j["summary"]}
        p = j["paired"]
        genes = str(j["genes"]["n_selected"]) if not j.get("pca") else f"50 PCs of {j['genes']['n_selected']}"
        pm, ps = p["flow_vs_mean_interp"], p["flow_vs_flow_matching_shuffled_time"]
        rows.append(minus(f"| {ds} | {label} | {genes} | {m['flow_matching']:.3f} | {m['mean_interp']:.3f} | "
                          f"{pm['mean']:+.3f} [{pm['ci_low']:+.3f}, {pm['ci_high']:+.3f}] | "
                          f"{ps['mean']:+.3f} [{ps['ci_low']:+.3f}, {ps['ci_high']:+.3f}] |"))
    txt = "\n".join(rows)
    eff = []
    for ds, a, b in (("embryoid body", "eb_v0110", "eb_filter_base"), ("gastrulation", "argelaguet_v0110",
                                                                      "argelaguet_filter_base")):
        fa, fb = root / "outputs" / a / "results.csv", root / "outputs" / b / "results.csv"
        if fa.exists() and fb.exists():
            A, B = per_seed_local(pd.read_csv(fa)), per_seed_local(pd.read_csv(fb))
            dd = (A[A.model == "flow_matching"].set_index("seed").energy_distance
                  - B[B.model == "flow_matching"].set_index("seed").energy_distance).dropna()
            from scipy import stats as st
            h = st.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / len(dd) ** 0.5
            eff.append(minus(f"{ds} {dd.mean():+.3f} [{dd.mean() - h:+.3f}, {dd.mean() + h:+.3f}]"))
    if eff:
        txt += ("\n\nEffect of the straight-line centre on the held-out stage, same gene space and seeds (version 0.1.10 "
                "minus version 0.1.9, paired by seed): " + "; ".join(eff) + ".")
    return text.replace("HISTORY_TABLE", txt)


def diag_table(text: str) -> str:
    import json
    rows = ["| data set | construction | energy distance | spread | distance to data |", "|---|---|---|---|---|"]
    names = [("mean_interp", "mean interpolation (real cells, shifted)"),
             ("ot_interp", "McCann / OT interpolant (two-cell averages)"),
             ("path_spline", "path family, spline centre (version 0.1.9)"),
             ("path_linear", "path family, straight-line centre (version 0.1.10)")]
    scale_parts = []
    for ds, bench in (("eb", "eb_v0110"), ("argelaguet", "argelaguet_v0110")):
        f = root / "outputs" / f"diag_blend_{ds}" / "blend.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        t = d[d.split.str.startswith("test")].groupby("construction")[["ED", "var_ratio", "nn_ratio"]].mean()
        for c, lab in names:
            rows.append(f"| {DS_NAME[ds]} | {lab} | {t.loc[c, 'ED']:.3f} | {t.loc[c, 'var_ratio']:.2f} | "
                        f"{t.loc[c, 'nn_ratio']:.2f} |")
        b = root / "outputs" / bench / "summary.json"
        if b.exists():
            m = {r["model"]: r["mean"] for r in json.loads(b.read_text(encoding="utf-8"))["summary"]}
            rows.append(f"| {DS_NAME[ds]} | trained model, version 0.1.10 (benchmark) | {m['flow_matching']:.3f} | — | — |")
        v = d[d.split == "validation"].groupby(["fold", "construction"]).ED.mean()
        if ds == "argelaguet":
            scale_parts = [f"{STAGE[ds].get(fold, fold)} fold {v.loc[(fold, 'path_linear')]:.3f} → "
                           f"{v.loc[(fold, 'path_linear+scale')]:.3f}" for fold in sorted(set(v.index.get_level_values(0)))]
    text = text.replace("DIAG_TABLE", "\n".join(rows))
    return text.replace("DIAG_SCALE_SENTENCE", "; ".join(scale_parts) if scale_parts else "see outputs/diag_blend_argelaguet")


def v0111_section(text: str) -> str:
    import json
    sets = {"eb": root / "outputs" / "eb_select_v0111", "argelaguet": root / "outputs" / "argelaguet_select_v0111"}
    t = selection_table(sets, "version 0.1.10 (straight-line centre)")
    if t is None:
        return text.replace("V0111_SECTION", "**Version 0.1.11.** The selection of the non-autonomous field (§3.5) on "
                            "the validation folds of both data sets, with a third arm that drops the marginal fine-tune "
                            "stage as a diagnostic of that stage, was running on the HPC when this report was built "
                            "(`scripts_dev/hpc_v0111.sbatch`). Its choice and, if an arm qualifies, the single "
                            "measurement on the test stages will be added.")
    out = ["**Version 0.1.11.** Two arms against version 0.1.10: the non-autonomous field (§3.5), with the marginal "
           "fine-tune stage kept, and, as a diagnostic, version 0.1.10 itself (autonomous field) without the marginal "
           "fine-tune stage; both were selected by the same pre-registered rule on the same validation folds.", "",
           "Table: Table 13. Selection of version 0.1.11 on validation folds, test stage removed (energy distance, two "
           "seeds per fold; change = mean paired change against version 0.1.10, pairs better out of four).", "", t, ""]
    for ds in ("eb", "argelaguet"):
        cf = sets[ds] / "choice.json"
        if not cf.exists():
            out.append(f"{DS_NAME[ds].capitalize()}: selection still running when this report was built.")
            continue
        c = json.loads(cf.read_text(encoding="utf-8"))
        if c["winner"] == "base":
            out.append(f"{DS_NAME[ds].capitalize()}: no arm met the rule; version 0.1.10 stays and nothing new was "
                       f"measured on the test stage.")
            continue
        f = root / "outputs" / f"{ds}_v0111" / "summary.json"
        if not f.exists():
            out.append(f"{DS_NAME[ds].capitalize()}: {ARM_LABEL.get(c['winner'], c['winner'])} met the rule; its "
                       f"measurement on the test stage was still running when this report was built.")
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        m = {r["model"]: r["mean"] for r in j["summary"]}
        p = j["paired"]
        A = per_seed_local(pd.read_csv(root / "outputs" / f"{ds}_v0111" / "results.csv"))
        B = per_seed_local(pd.read_csv(root / "outputs" / f"{ds}_v0110" / "results.csv"))
        dd = (A[A.model == "flow_matching"].set_index("seed").energy_distance
              - B[B.model == "flow_matching"].set_index("seed").energy_distance).dropna()
        from scipy import stats as st
        h = st.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / len(dd) ** 0.5
        pm, ps = p["flow_vs_mean_interp"], p["flow_vs_flow_matching_shuffled_time"]
        prev = float(B[B.model == "flow_matching"].energy_distance.mean())
        out.append(minus(
            f"{DS_NAME[ds].capitalize()}: {ARM_LABEL.get(c['winner'], c['winner'])} met the rule and was measured once "
            f"on the test stage: flow model {m['flow_matching']:.3f} (version 0.1.10: {prev:.3f}), "
            f"paired change {dd.mean():+.3f} [{dd.mean() - h:+.3f}, {dd.mean() + h:+.3f}]; against mean interpolation "
            f"{pm['mean']:+.3f} [{pm['ci_low']:+.3f}, {pm['ci_high']:+.3f}]; against shuffled time labels "
            f"{ps['mean']:+.3f} [{ps['ci_low']:+.3f}, {ps['ci_high']:+.3f}]."))
        out.append("")
    return text.replace("V0111_SECTION", "\n".join(out))


md = select_v0110_table(md)
md = history_table(md)
md = diag_table(md)
md = v0111_section(md)
left = [k for k in ("EB_TABLE", "ARG_TABLE", "SELECT_V0110_TABLE", "HISTORY_TABLE", "DIAG_TABLE", "DIAG_SCALE_SENTENCE",
                    "V0111_SECTION", "EB_PCA_SENTENCE", "COMBO_TABLE", "FINAL_TEST_TABLE") if k in md]
assert not left, f"unfilled placeholders: {left}"
report_md = dst / "REPORT_OT_flow_matching.md"
report_md.write_text(md, encoding="utf-8")
figs = dst / "figures"
figs.mkdir()
for f in sorted((root / "outputs" / "figures").glob("fig*.png")):
    shutil.copy(f, figs / f.name)

# --- code, tests, docs ---------------------------------------------------------------------------------------
pkg = dst / "cfm_grn_v2"
shutil.copytree(root / "src", pkg / "src", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(root / "tests", pkg / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
shutil.copytree(root / "scripts_dev", pkg / "scripts_dev", ignore=shutil.ignore_patterns("__pycache__"))
for f in ("pyproject.toml", "README.md"):
    shutil.copy(root / f, pkg / f)
cards = root / "data_cards"
if cards.exists():
    shutil.copytree(cards, pkg / "data_cards")
docs = dst / "docs"
docs.mkdir()
shutil.copy(proj / "docs" / "superpowers" / "decisions.md", docs / "decisions.md")
shutil.copy(proj / "docs" / "superpowers" / "plans" / "2026-09-18-rafm-phase1-cfm-engine.md", docs / "plan_phase1_engine.md")
shutil.copy(proj / "docs" / "superpowers" / "plans" / "2026-09-18-rafm-phase2-real-data.md", docs / "plan_phase2_real_data.md")
shutil.copy(proj / "docs" / "superpowers" / "specs" / "2026-09-18-rafm-phase3-gated-grn-design.md", docs / "phase3_gated_grn_design.md")
shutil.copy(proj / "scripts" / "materialize_plan.py", docs / "materialize_plan.py")

# --- results -------------------------------------------------------------------------------------------------
res = dst / "results"
res.mkdir()
for run in ("phase1_report", "phase1_v2_grn_interp", "phase1_v1_toys", "phase1_v1_grn", "ot_ablation",
            "ot_crossing", "diag_ceiling", "combo_scan", "knots_scan", "marginal_scan", "final_test",
            "eb_interp", "eb_interp_pca50", "eb_diagnose", "eb_select", "eb_v0110", "eb_filter_base", "eb_v0110_pca50",
            "argelaguet_interp", "argelaguet_check_scale", "argelaguet_select", "argelaguet_v0110",
            "argelaguet_filter_base", "diag_blend_eb", "diag_blend_argelaguet", "eb_select_v0111",
            "argelaguet_select_v0111", "eb_v0111", "argelaguet_v0111"):
    src = root / "outputs" / run
    if src.exists():
        d = res / run
        d.mkdir()
        for f in src.iterdir():
            if f.suffix in (".md", ".csv", ".json", ".npz") and f.stat().st_size < 50_000_000:
                shutil.copy(f, d / f.name)

# --- docx / pdf via pandoc (styled reference) + Word --------------------------------------------------------
sys.path.insert(0, str(root / "scripts_dev"))
from docx_style import make_reference_docx, polish_docx  # noqa: E402

ref = make_reference_docx(dst / "_reference.docx")
docx = dst / "REPORT_OT_flow_matching.docx"
subprocess.run(["pandoc", report_md.name, "-o", docx.name, "--from", "markdown", "--to", "docx", "--reference-doc", ref.name,
                "--toc", "--toc-depth", "2", "-M", "lang=en-GB"], check=True, cwd=str(dst))
ref.unlink()
polish_docx(docx, "Optimal-transport conditional flow matching for time-resolved single-cell populations  ·  G. Bambini  ·  25 September 2026")
pdf = dst / "REPORT_OT_flow_matching.pdf"
ps = f'''$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
$d = $w.Documents.Open("{docx}"); $d.Fields.Update() | Out-Null
foreach ($t in $d.TablesOfContents) {{ $t.Update() | Out-Null }}
$d.Save(); $d.SaveAs2("{pdf}", 17); $d.Close(); $w.Quit()'''
r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
print("pdf:", "ok" if pdf.exists() else f"failed: {r.stderr[:300]}")

# --- zip -----------------------------------------------------------------------------------------------------
zip_path = shutil.make_archive(str(proj / "deliverables" / name), "zip", root_dir=dst.parent, base_dir=dst.name)
print("deliverable folder:", dst)
print("zip:", zip_path, round(Path(zip_path).stat().st_size / 1e6, 1), "MB")
