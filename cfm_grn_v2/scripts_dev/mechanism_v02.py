"""Pre-registration v0.2, Step 2 (Job A): why does the refit's unguarded second stage help on some folds and hurt on
others? Data exactly as select_config.py (test stage REMOVED); folds = the inner remaining times; seeds 0, 1, 2.
Non-confirmatory: these folds were seen by the v0.1.11 selection.

Per (dataset, fold, seed), three trainings, each a nested fit (early stopping on the design's validation task) and a
refit on every observed time for the transferred first-stage count, the refit keeping its stage-1 weights and a
stage-2 snapshot every 25 iterations:
  R-a  v0.1.10 (linear centre)        -> base (final), noft (stage 1), gated (stage 1 if h_rel < 1.5 else final),
                                         transfer (stage-2 snapshot at the nested fit's accepted iteration)
  R-b  + field_time="bio"             -> time (final), time_noft (stage 1)
  R-c  spline centre, no second stage -> spline_noft
Non-voting diagnostics from the same models: C2 (the held-out stage also predicted from the first time, R-a),
C3 (R-a refit without the first knot, long folds only), predictions at 8 RK4 steps per unit (H5), var_ratio and the
mean/trace split of the energy distance, the nested fit's stage-2 verdict and validation scores, and the path
family's own energy distance (no network) for each knot set.

The rules (reproducibility gate, Rule B, admission to Step 6) are those of
docs/superpowers/prereg/2026-09-25-prereg-v02.md and are applied by code at the end (summary.json).

Run from cfm_grn_v2/:
  PYTHONPATH=src python scripts_dev/mechanism_v02.py --dataset eb --cache <npz> --prior <tsv> --out outputs/mech_eb \
      --reference outputs/eb_select_v0111/results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance, median_bandwidth
from cfm_grn_v2.flows.coupling import chain_ot
from cfm_grn_v2.flows.path import SplinePath
from cfm_grn_v2.training.search import interpolation_design
from cfm_grn_v2.training.trainer import task_groups, time_task, train_flow
from select_config import load_dataset

RECIPES = {"R-a": {"path_center": "linear"},
           "R-b": {"path_center": "linear", "field_time": "bio"},
           "R-c": {"path_center": "spline", "marginal_iters": 0}}
SHORT = 1.5                      # h_rel below this is a short horizon (pre-registered)


def grid_gap(times) -> float:
    """Median consecutive gap of the data set's FULL time grid (test stage included)."""
    return float(np.median(np.diff(sorted(times))))


def path_ed(x_eval, tt, knots, target_time, seed, center):
    rng = np.random.default_rng(seed)
    traj = chain_ot([x_eval[tt == w] for w in knots], rng, 1000, mode="ot")
    path = SplinePath(knots, traj, center=center)
    xp, _ = path.evaluate(np.full(path.n, target_time), np.arange(path.n))
    truth = x_eval[tt == target_time]
    er = np.random.default_rng(100 + seed)
    return float(energy_distance(xp[er.choice(len(xp), min(1000, len(xp)), replace=False)],
                                 truth[er.choice(len(truth), min(1000, len(truth)), replace=False)]))


def rule_b(df: pd.DataFrame, long_folds: list) -> dict:
    """g = (ED_noft - ED_M) / (ED_noft - ED_base) on the long folds x seeds, only where ED_noft - ED_base > 0.05; a pair
    failing that condition counts against both verdicts. Repairs iff g >= 0.5 in >= 5/6 pairs; does not repair iff
    g < 0.25 in >= 5/6 pairs; otherwise inconclusive."""
    main = df[(df.source == "t_prev") & (df.steps == 20) & (df.variant == "main")]
    main = main[[(ds, f) in long_folds for ds, f in zip(main.dataset, main.fold)]]
    ed = main.pivot_table(index=["dataset", "fold", "seed"], columns="arm", values="energy_distance")
    out = {}
    for m in ("time_noft", "spline_noft"):
        gs, n_rep, n_not = [], 0, 0
        for idx, r in ed.iterrows():
            gap = r["noft"] - r["base"]
            if not np.isfinite(r.get(m, np.nan)) or gap <= 0.05:
                gs.append(None)
                continue
            g = float((r["noft"] - r[m]) / gap)
            gs.append(g)
            n_rep += g >= 0.5
            n_not += g < 0.25
        n = len(gs)
        need = int(np.ceil(5 / 6 * n)) if n else 1
        verdict = "repairs" if n_rep >= need else ("does not repair" if n_not >= need else "inconclusive")
        out[m] = {"g": gs, "n_pairs": n, "n_ge_0.5": int(n_rep), "n_lt_0.25": int(n_not), "verdict": verdict,
                  "median_g": float(np.median([g for g in gs if g is not None])) if any(g is not None for g in gs)
                  else None}
    return out


def decide(df: pd.DataFrame, long_folds: list, references: dict) -> dict:
    """The pre-registered rules over every data set in df: reproducibility gate (seeds 0 and 1 of base, noft and time
    within 1e-3 of the v0.1.11 selection), Rule B, admission to Step 6. long_folds: [(dataset, fold)]."""
    main_rows = df[(df.variant == "main") & (df.source == "t_prev") & (df.steps == 20)]
    diffs = {}
    for ds, ref_path in references.items():
        ref = pd.read_csv(ref_path).set_index(["fold", "seed", "arm"]).energy_distance
        sub = main_rows[main_rows.dataset == ds]
        for arm in ("base", "noft", "time"):
            for (fold, seed), v in sub[sub.arm == arm].set_index(["fold", "seed"]).energy_distance.items():
                if seed in (0, 1) and (fold, seed, arm) in ref.index:
                    diffs[f"{ds}|{arm}|{fold}|{seed}"] = float(v - ref[(fold, seed, arm)])
    gate = {"max_abs_diff": max(abs(v) for v in diffs.values()) if diffs else None, "n": len(diffs),
            "passes": bool(diffs) and max(abs(v) for v in diffs.values()) <= 1e-3, "diffs": diffs}
    out = {"reproducibility_gate": gate, "long_folds": [list(x) for x in long_folds]}
    if not gate["passes"]:
        out["rule_b"] = "not applied: reproducibility gate failed (stop and debug)"
        return out
    rb = rule_b(df, long_folds)
    ed = main_rows.pivot_table(index=["dataset", "fold", "seed"], columns="arm", values="energy_distance")
    admitted = [arm for arm in ("base", "noft", "gated", "transfer")
                if not ((~np.isfinite(ed[arm])).any() or (ed[arm] > 3 * ed["base"]).any())]
    reps = [m for m in ("time_noft", "spline_noft") if rb[m]["verdict"] == "repairs"]
    if reps:
        admitted.append(max(reps, key=lambda m: (rb[m]["median_g"], m == "spline_noft")))
    out.update({"rule_b": rb, "admitted_to_step6": admitted})
    return out


def merge_decision(summary: dict, decision: dict) -> None:
    """Add decide()'s verdicts to a per-data-set summary without replacing its own fields ('long_folds' stays the
    plain fold list that combine() reads)."""
    summary.update({k: v for k, v in decision.items() if k not in summary})


def combine(dirs: list, out_json: str) -> None:
    dfs, long_folds, refs = [], [], {}
    for d in dirs:
        summ = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
        dfs.append(pd.read_csv(Path(d) / "results.csv"))
        # plain folds, or the [dataset, fold] pairs that job 670008's summaries carry (decide() overwrote the field)
        long_folds += [(summ["dataset"], float(f[1] if isinstance(f, (list, tuple)) else f))
                       for f in summ["long_folds"]]
        refs[summ["dataset"]] = summ["reference"]
    res = {"prereg": "docs/superpowers/prereg/2026-09-25-prereg-v02.md", "dirs": dirs,
           **decide(pd.concat(dfs, ignore_index=True), long_folds, refs)}
    Path(out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("eb", "argelaguet"))
    p.add_argument("--cache")
    p.add_argument("--prior")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-hvg", type=int, default=1000)
    p.add_argument("--min-detect", type=float, default=0.01)
    p.add_argument("--out")
    p.add_argument("--reference", default=None, help="v0.1.11 selection results.csv for the reproducibility gate")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--combine", nargs="+", default=None, help="output folders of both data sets: apply the rules")
    p.add_argument("--decision", default="outputs/mech_v02_decision.json")
    a = p.parse_args()
    if a.combine:
        combine(a.combine, a.decision)
        return
    if not (a.dataset and a.cache and a.prior and a.out):
        p.error("--dataset, --cache, --prior and --out are required (or --combine)")
    X, obs, symbols, regs, test_time, max_cells, base_cfg = load_dataset(a.dataset, a.cache, a.prior)
    g_full = grid_gap(np.unique(obs["time"]))
    smoke = dict(n_iters=300, eval_every=100, patience=2, marginal_iters=50, marginal_eval_every=25) if a.smoke else {}
    if a.smoke:
        a.n_hvg, a.seeds, max_cells = 150, 1, 400
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    keep = (obs["time"] != test_time).to_numpy()
    X, obs = X[keep], obs.loc[keep].reset_index(drop=True)
    times = sorted(float(v) for v in np.unique(obs["time"]))
    folds = times[1:-1]
    print(f"{a.dataset}: test time {test_time} removed; times {times}; folds {folds}; grid gap {g_full}", flush=True)
    Xn = normalize_log1p(X, keep_sparse=True)
    rows, path_rows, long_folds = [], [], []

    for fold in folds:
        d = interpolation_design(times, fold)
        h_rel = (d["test_time"] - d["t_prev"]) / g_full
        if h_rel >= SHORT:
            long_folds.append(fold)
        fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
        sel = select_genes(Xn[fit_cells], symbols, n_hvg=a.n_hvg, must_keep=regs, min_detect=a.min_detect)
        s = snapshot_set_from_arrays(Xn, obs, symbols, time_key="time", replicate_key="replicate",
                                     condition_key="condition", gene_idx=sel, max_cells_per_group=max_cells,
                                     min_cells=200, rng=np.random.default_rng(0), name=f"{a.dataset}_mech")
        fit_mask, all_mask = s.mask(times=d["fit_times"]), s.mask(times=d["obs_times"])
        val_task = time_task(s, d["v_prev"], d["val_time"])
        eval_std = Standardizer().fit(s.x[all_mask])
        x_eval = eval_std.transform(s.x)
        proto = EvalProtocol(n_samples=1000, seed=0, bandwidth=median_bandwidth(x_eval[all_mask]))
        tt = s.biological_time
        first = d["obs_times"][0]
        sources = {"t_prev": d["t_prev"]}
        if d["t_prev"] != first:
            sources["t0"] = first                                  # C2
        c3_knots = [t for t in d["obs_times"] if t != first] if fold in long_folds else None
        print(f"\n=== fold {fold}: from {d['t_prev']} (h_rel {h_rel:g}), bracketing {d['t_prev']}->{d['t_next']}; "
              f"validation {d['v_prev']}->{d['val_time']}; fit {d['fit_times']}; {len(sel)} genes; "
              f"C3 knots {c3_knots} ({time.time() - t0:.0f}s) ===", flush=True)
        for seed in range(a.seeds):
            for center in ("spline", "linear"):
                path_rows.append({"fold": fold, "seed": seed, "knots": "observed", "center": center,
                                  "path_energy_distance": path_ed(x_eval, tt, d["obs_times"], fold, seed, center)})
            if c3_knots:
                path_rows.append({"fold": fold, "seed": seed, "knots": "without first", "center": "linear",
                                  "path_energy_distance": path_ed(x_eval, tt, c3_knots, fold, seed, "linear")})

        def score(model, arm, recipe, seed, stage, source, steps, variant, nested, extra=None):
            task = time_task(s, sources[source], d["test_time"])
            res = []
            for cond, rep, m0, m1 in task_groups(s, task):
                pred = eval_std.transform(model.predict(s.x[m0], cond, sources[source], d["test_time"],
                                                        stage=stage, n_steps=steps))
                res.append(proto.evaluate(pred, x_eval[m1], seed_offset=seed))
            keys = ("energy_distance", "mean_error", "var_ratio", "ed_approx", "ed_mean_part", "ed_trace_part")
            r = {k: float(np.mean([x[k] for x in res])) for k in keys}
            rows.append({"dataset": a.dataset, "fold": fold, "seed": seed, "recipe": recipe, "arm": arm,
                         "variant": variant, "source": source, "source_time": sources[source], "steps": steps,
                         "h_rel": (d["test_time"] - sources[source]) / g_full, **r,
                         "nested_best_iter": nested.best_iter, "nested_marginal_iter": nested.marginal_iter,
                         "nested_stage1_val": nested.stage1_best_val, "nested_final_val": nested.best_val,
                         "refit_marginal_iter": model.marginal_iter, "refit_guarded": model.marginal_guarded,
                         "n_genes": int(len(sel)), **(extra or {})})
            print(f"  seed {seed} {recipe} {arm:12s} {variant:5s} from {source:6s} steps {steps:2d}  "
                  f"ED {r['energy_distance']:.4f}  mean_err {r['mean_error']:.3f}  var {r['var_ratio']:.2f}", flush=True)

        for seed in range(a.seeds):
            for recipe, over in RECIPES.items():
                cfg = base_cfg.replace(seed=seed, **{**smoke, **over})
                t1 = time.time()
                nested, _ = train_flow(s, fit_mask, d["fit_times"], cfg, val_task, proto)
                n_iters = max(int(nested.best_iter), cfg.eval_every)
                model, _ = train_flow(s, all_mask, d["obs_times"], cfg.replace(n_iters=n_iters), None, proto,
                                      keep_snapshots=True)
                secs = round(time.time() - t1, 1)
                if recipe == "R-a":
                    k = int(nested.marginal_iter or 0)
                    for arm, stage in (("base", None), ("noft", "stage1"),
                                       ("gated", "stage1" if h_rel < SHORT else None), ("transfer", k)):
                        score(model, arm, recipe, seed, stage, "t_prev", 20, "main", nested, {"seconds": secs})
                    for arm, stage in (("base", None), ("noft", "stage1")):
                        score(model, arm, recipe, seed, stage, "t_prev", 8, "main", nested)          # H5
                        if "t0" in sources:
                            score(model, arm, recipe, seed, stage, "t0", 20, "main", nested)         # C2
                    if c3_knots:                                                                     # C3
                        c3_mask = s.mask(times=c3_knots)
                        m3, _ = train_flow(s, c3_mask, c3_knots, cfg.replace(n_iters=n_iters), None, proto)
                        for arm, stage in (("base", None), ("noft", "stage1")):
                            score(m3, arm, recipe, seed, stage, "t_prev", 20, "C3", nested)
                elif recipe == "R-b":
                    score(model, "time", recipe, seed, None, "t_prev", 20, "main", nested, {"seconds": secs})
                    score(model, "time_noft", recipe, seed, "stage1", "t_prev", 20, "main", nested)
                else:
                    score(model, "spline_noft", recipe, seed, None, "t_prev", 20, "main", nested, {"seconds": secs})
                print(f"  seed {seed} {recipe} done in {time.time() - t1:.0f}s (total {(time.time() - t0) / 60:.0f} min)",
                      flush=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            pd.DataFrame(path_rows).to_csv(out / "path_ceiling.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(path_rows).to_csv(out / "path_ceiling.csv", index=False)
    main_rows = df[(df.variant == "main") & (df.source == "t_prev") & (df.steps == 20)]
    summary = {"dataset": a.dataset, "smoke": bool(a.smoke), "test_time_removed": test_time, "folds": folds,
               "long_folds": long_folds, "grid_gap": g_full, "reference": a.reference,
               "prereg": "docs/superpowers/prereg/2026-09-25-prereg-v02.md",
               "note": "per-data-set view only; the pre-registered rules pool both data sets (--combine)"}
    if a.reference and Path(a.reference).exists() and not a.smoke:
        merge_decision(summary, decide(df, [(a.dataset, f) for f in long_folds], {a.dataset: a.reference}))
    summary["mean_by_fold_arm"] = {f"{k[0]}|{k[1]}": float(v) for k, v in
                                   main_rows.groupby(["fold", "arm"]).energy_distance.mean().items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "mean_by_fold_arm"}, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
