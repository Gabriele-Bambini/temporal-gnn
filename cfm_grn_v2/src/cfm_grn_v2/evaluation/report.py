"""Aggregation over seeds, confidence intervals, paired differences and the gate verdict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["energy_distance", "mmd", "w2", "mean_error", "per_gene_w1"]
CONTROLS = ["persistence", "train_mean", "mean_shift", "mean_interp", "ot_interp", "linear", "mlp", "freefm_replay"]


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Average the groups of each (dataset, model, seed) first: seeds are the replication unit."""
    cols = [c for c in METRICS + ["null_q95"] if c in df.columns]
    return df.groupby(["dataset", "model", "seed"], as_index=False)[cols].mean()


def _ci95(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, model), g in per_seed(df).groupby(["dataset", "model"]):
        for m in METRICS:
            rows.append({"dataset": ds, "model": model, "metric": m, "mean": float(g[m].mean()),
                         "std": float(g[m].std(ddof=1)) if len(g) > 1 else 0.0, "ci95": _ci95(g[m]), "n": int(len(g))})
    return pd.DataFrame(rows)


def paired_difference(df: pd.DataFrame, dataset: str, model: str, reference: str, metric: str = "energy_distance") -> dict:
    ps = per_seed(df)
    ps = ps[ps["dataset"] == dataset]
    a = ps[ps["model"] == model].set_index("seed")[metric]
    b = ps[ps["model"] == reference].set_index("seed")[metric]
    seeds = sorted(set(a.index) & set(b.index))
    if not seeds:
        raise ValueError(f"no common seed between {model!r} and {reference!r} on {dataset!r}")
    d = np.array([a[s] - b[s] for s in seeds], dtype=np.float64)
    ci = _ci95(d)
    return {"model": model, "reference": reference, "metric": metric, "mean": float(d.mean()),
            "ci_low": float(d.mean() - ci), "ci_high": float(d.mean() + ci), "n": int(len(d))}


def verdict(df: pd.DataFrame, dataset: str, fm: str = "flow_matching", shuffled: str = "flow_matching_shuffled_time",
            controls=CONTROLS, metric: str = "energy_distance", mode: str = "benchmark") -> dict:
    """mode="benchmark": FM must beat every control (paired CI95 < 0 against the best one) and use time.
    mode="sanity": FM must be indistinguishable from the target under the energy-distance permutation
    test (mean ED <= mean 95% null quantile) and use time (toys whose optimum is a linear map cannot be
    beaten, only matched)."""
    if mode not in ("benchmark", "sanity"):
        raise ValueError(f"unknown verdict mode {mode!r}")
    ps = per_seed(df)
    ps = ps[ps["dataset"] == dataset]
    means = ps.groupby("model")[metric].mean()
    present = [c for c in controls if c in means.index]
    if fm not in means.index or not present:
        raise ValueError(f"dataset {dataset!r} lacks {fm!r} or every control")
    best_control = min(present, key=lambda c: float(means[c]))
    d_best = paired_difference(df, dataset, fm, best_control, metric)
    beats_all = all(float(means[fm]) < float(means[c]) for c in present)
    ci_ok = d_best["ci_high"] < 0 if not np.isnan(d_best["ci_high"]) else d_best["mean"] < 0
    uses_time, d_shuf = None, None
    if shuffled in means.index:
        d_shuf = paired_difference(df, dataset, fm, shuffled, metric)
        uses_time = bool(d_shuf["mean"] < 0 and (np.isnan(d_shuf["ci_high"]) or d_shuf["ci_high"] < 0))
    floor = float(means["noise_floor"]) if "noise_floor" in means.index else float("nan")
    if "null_q95" in ps.columns:
        null_q95 = float(ps[ps["model"] == fm]["null_q95"].mean())
        matches_target = bool(float(means[fm]) <= null_q95)
    else:
        null_q95, matches_target = float("nan"), False
    core = (beats_all and ci_ok) if mode == "benchmark" else matches_target
    return {"dataset": dataset, "metric": metric, "mode": mode, "pass": bool(core and (uses_time is None or uses_time)),
            "beats_all_controls": bool(beats_all), "best_control": best_control, "paired_vs_best_control": d_best,
            "uses_time": uses_time, "paired_vs_shuffled_time": d_shuf, "fm_mean": float(means[fm]),
            "noise_floor": floor, "null_q95": null_q95, "matches_target": matches_target}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def write_report(df: pd.DataFrame, md_path, json_path, modes: dict | None = None) -> dict:
    """modes: dataset -> "benchmark" | "sanity" (default benchmark)."""
    modes = modes or {}
    summ = summarize(df)
    verdicts = {ds: verdict(df, ds, mode=modes.get(ds, "benchmark")) for ds in sorted(df["dataset"].unique())}
    lines = ["# Forecast benchmark report", "", "Metrics in the evaluation space standardised on observed cells; "
             "mean ± CI95 over seeds (groups averaged within seed). Lower is better.", ""]
    for ds in sorted(df["dataset"].unique()):
        s = summ[summ["dataset"] == ds]
        lines += [f"## {ds}", "", "| model | " + " | ".join(METRICS) + " | n seeds |", "|---|" + "---|" * (len(METRICS) + 1)]
        order = sorted(s["model"].unique(), key=lambda m: float(s[(s["model"] == m) & (s["metric"] == "energy_distance")]["mean"].iloc[0]))
        for m in order:
            cells = []
            for met in METRICS:
                r = s[(s["model"] == m) & (s["metric"] == met)].iloc[0]
                cells.append(f"{r['mean']:.3f}±{r['ci95']:.3f}" if not np.isnan(r["ci95"]) else f"{r['mean']:.3f}")
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {int(s[s['model'] == m]['n'].iloc[0])} |")
        v = verdicts[ds]
        pb = v["paired_vs_best_control"]
        lines += ["", f"Verdict ({v['mode']}): **{'PASS' if v['pass'] else 'FAIL'}** — best control `{v['best_control']}`, paired ΔED = "
                  f"{pb['mean']:.3f} [{pb['ci_low']:.3f}, {pb['ci_high']:.3f}] (n={pb['n']}), uses_time = {v['uses_time']}, "
                  f"noise floor = {v['noise_floor']:.3f}, null q95 = {v['null_q95']:.3f}, matches target = {v['matches_target']}", ""]
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(json_path).write_text(json.dumps(verdicts, indent=1, default=_json_default), encoding="utf-8")
    return verdicts
