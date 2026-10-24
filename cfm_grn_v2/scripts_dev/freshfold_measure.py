"""Pre-registration v0.2, Step 7 (Job E): the one measurement on the fresh test stages.

Tests (designs from cfm_grn_v2.training.designs on the full grid, test stage included):
  PRIMARY    veres_day4             Veres day 4 predicted from day 3 (bracket 3 -> 5)
  PRIMARY    ps_E8.0                Pijuan-Sala E8.0 (quarter day 6) from E7.75 (5), bracket 5 -> 7
  SECONDARY  ps_E8.0_without_E8.25  E8.0 from E7.75 with E8.25 (7) also removed: bracket E7.75 -> E8.5, as Argelaguet
Models: the Step-6 winner (choice.json, written by code in freshfold_select.py --decide; no edit in between) plus base,
noft and gated from the same checkpoints, 5 seeds; every control; the shuffled-time twin (the winner's recipe and
weights, trained on shuffled time labels). Gene selection on the fit cells, caps, QC gate, nested fit and refit as in
Job D (freshfold_select.py). Scores: EvalProtocol.evaluate, groups averaged within a seed.
Pijuan-Sala trains, couples and scores with replicate = 'all': a documented deviation from the literal Step-5 text
"replicate = pool" (decisions.md, 2026-09-26 afternoon). Reason: every embryo pool exists at one stage only, so with
replicate = pool no transition stays inside one replicate and there would be no training pair at all. The pool label
is kept, unchanged, for the per-pool ROBUST check and the replicate floor (ROBUST_KEY).

No human edit between selection and measurement: before loading any data, R-SEL is applied again to the Job-D outputs
choice.json names (freshfold_select.verify_choice) and the run stops unless it gives the file's winner; summary.json
records the file's sha256. Only --smoke accepts --force-winner (to exercise a non-base winner end to end). A non-smoke
run also refuses to start with other than the pre-registered parameters: it would look at the test stage for nothing.

R-CONF (applied by code in --decide, which pools both data sets): Delta = ED(winner) - ED(best control), the best
control being the lowest seed mean, chosen afterwards; paired t-test over seeds; Holm over the 2 primaries. POSITIVE iff
Delta < 0 with p_Holm < 0.05 and the upper 95% CI of (winner - shuffled) < 0; NEGATIVE iff Delta > 0 with p_Holm < 0.05;
otherwise TIE. A positive is ROBUST iff also Delta < 0 against each biological replicate target separately (Veres
differentiation 1 and 2; each E8.0 pool) and the 95% interval of a 200-resample cell bootstrap of Delta lies below 0.
The cell bootstrap resamples, with replacement, the cells of each (seed, group) evaluation subsample (predictions and
target; winner and control share the resampled positions) and averages Delta over seeds and groups. The secondary gets
the same verdict from its unadjusted p (non-confirmatory).
Also reported: the replicate floor (ED between biological replicates at equal n at the test stage; the two E8.0 pools
are in different sequencing batches), the fraction of the gap closed (best control - winner)/(best control - floor),
var_ratio and the mean/trace split of every model, and the pre-stated E8.0 prediction (the path family over-disperses,
var_ratio > 1.2, and the winner does not beat mean interpolation).
A non-smoke --decide refuses unless both data sets have exactly one finished run (complete true), every planned test
was measured or dropped by the QC gate, every run used the pre-registered parameters (seeds 0-4, 200 bootstrap
resamples, 1000 HVG, min_detect 0.01, the caps, frozen v0.1.10) and both read the same choice.json, which still has the
recorded sha256 and still passes verify_choice.

Run from cfm_grn_v2/ (two jobs, then the verdict):
  PYTHONPATH=src python scripts_dev/freshfold_measure.py --dataset veres --data <path> \
      --choice outputs/freshfold/choice.json --out outputs/freshfold_measure_veres
  PYTHONPATH=src python scripts_dev/freshfold_measure.py --dataset pijuan_sala --data <path> \
      --choice outputs/freshfold/choice.json --out outputs/freshfold_measure_ps
  PYTHONPATH=src python scripts_dev/freshfold_measure.py --decide outputs/freshfold_measure_veres outputs/freshfold_measure_ps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
import freshfold_select as fs
from cfm_grn_v2.data.real import normalize_log1p
from cfm_grn_v2.evaluation.metrics import energy_distance, noise_floor
from cfm_grn_v2.training.designs import CONTROLS, interpolation_controls, multi_removal_design
from cfm_grn_v2.training.trainer import task_groups, time_task

TESTS = {"veres": [{"name": "veres_day4", "role": "primary", "removed": (4.0,), "targets": (4.0,), "source": 3.0}],
         "pijuan_sala": [{"name": "ps_E8.0", "role": "primary", "removed": (6.0,), "targets": (6.0,), "source": 5.0},
                         {"name": "ps_E8.0_without_E8.25", "role": "secondary", "removed": (6.0, 7.0),
                          "targets": (6.0,), "source": 5.0}]}
PRIMARIES = ("veres_day4", "ps_E8.0")
ROBUST_KEY = {"veres": "replicate", "pijuan_sala": "pool"}
ALPHA, N_BOOT, SEEDS = 0.05, 200, [0, 1, 2, 3, 4]
FLOOR_NOTE = {"pijuan_sala": "the E8.0 pools are in different sequencing batches: their distance mixes biological and "
                             "batch differences"}
PRESTATED = ("on E8.0 the path will over-disperse (var_ratio > 1.2) and the winner will not beat mean interpolation; a "
             "positive there is reported as 'the Argelaguet negative did not replicate'")


# ---------------------------------------------------------------- statistics (pure, tested on hand-made tables)

def _f(x) -> float:
    return float("nan") if x is None else float(x)


def paired(a, b) -> dict:
    """Per-seed difference a - b: mean, 95% t-interval, two-sided paired t-test p."""
    a, b = pd.Series(a), pd.Series(b)
    common = a.index.intersection(b.index)
    d = (a[common] - b[common]).to_numpy(dtype=float)
    n = len(d)
    out = {"mean": float(d.mean()) if n else np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": n,
           "diffs": d.tolist()}
    if n >= 2:
        sd = d.std(ddof=1)
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.update(ci_low=float(d.mean() - half), ci_high=float(d.mean() + half),
                   p=float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (0.0 if d.mean() != 0 else 1.0))
    return out


def holm(pvals: dict, m=None) -> dict:
    """Holm-adjusted p values; m (default: the number given) counts tests that were planned but are missing."""
    items = sorted(((k, 1.0 if not np.isfinite(_f(v)) else _f(v)) for k, v in pvals.items()), key=lambda kv: kv[1])
    m = len(items) if m is None else int(m)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def conf_verdict(delta_mean, p, shuffled_ci_high) -> str:
    p, sh = _f(p), _f(shuffled_ci_high)
    sig = np.isfinite(p) and p < ALPHA
    if _f(delta_mean) < 0 and sig and np.isfinite(sh) and sh < 0:
        return "POSITIVE"
    if _f(delta_mean) > 0 and sig:
        return "NEGATIVE"
    return "TIE"


def is_robust(verdict: str, rep_deltas: dict, boot_ci_high) -> bool:
    return bool(verdict == "POSITIVE" and rep_deltas and all(_f(v) < 0 for v in rep_deltas.values())
                and np.isfinite(_f(boot_ci_high)) and _f(boot_ci_high) < 0)


def conf_stats(ps: pd.DataFrame, winner: str) -> dict:
    """ps: per-seed energy distance (columns seed, model, energy_distance)."""
    wide = ps.pivot(index="seed", columns="model", values="energy_distance")
    means = wide.mean()
    ctrls = [c for c in CONTROLS if c in wide.columns]
    best = min(ctrls, key=lambda c: float(means[c]))
    nan = {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "n": 0}
    return {"winner": winner, "best_control": best, "seed_means": {k: float(v) for k, v in means.items()},
            "delta": paired(wide[winner], wide[best]),
            "vs_shuffled": paired(wide[winner], wide["shuffled_twin"]) if "shuffled_twin" in wide else nan,
            "vs_mean_interp": paired(wide[winner], wide["mean_interp"]) if "mean_interp" in wide else nan}


def decide_conf(tests: dict, primaries=PRIMARIES) -> dict:
    """tests: name -> {"role", "stats" (conf_stats), "rep_deltas", "boot"}. Holm over the planned primaries (a missing
    one counts as p = 1); the secondary uses its unadjusted p."""
    adj = holm({n: t["stats"]["delta"]["p"] for n, t in tests.items() if t["role"] == "primary"}, m=len(primaries))
    out = {}
    for n, t in tests.items():
        st = t["stats"]
        p_holm = adj[n] if t["role"] == "primary" else None
        v = conf_verdict(st["delta"]["mean"], p_holm if t["role"] == "primary" else st["delta"]["p"],
                         st["vs_shuffled"]["ci_high"])
        out[n] = {"role": t["role"], "winner": st["winner"], "best_control": st["best_control"],
                  "delta": st["delta"], "p_raw": st["delta"]["p"], "p_holm": p_holm, "vs_shuffled": st["vs_shuffled"],
                  "verdict": v, "robust": is_robust(v, t.get("rep_deltas") or {}, (t.get("boot") or {}).get("ci_high")),
                  "rep_deltas": t.get("rep_deltas"), "boot": t.get("boot")}
    return out


def weighted_ed(d_ab, d_aa, d_bb, wa, wb) -> float:
    """Energy distance (U-statistic) of samples a, b whose points carry multiplicities wa, wb (a bootstrap draw):
    copies of one point are distinct sample points at distance 0."""
    na, nb = wa.sum(), wb.sum()
    return float(2.0 * wa @ d_ab @ wb / (na * nb) - wa @ d_aa @ wa / (na * (na - 1)) - wb @ d_bb @ wb / (nb * (nb - 1)))


def boot_delta(items: list, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """items: [(winner subsample, its target subsample, control subsample, its target subsample)], one per (seed,
    group). Delta_b = mean over items of ED_w - ED_c on a resample of the cells (positions shared by winner and
    control when the sizes agree); 95% percentile interval over n_boot resamples."""
    draws, point = np.zeros(n_boot), 0.0
    for k, (pw, tw, pc, tc) in enumerate(items):
        rng = np.random.default_rng([seed, k])
        dw = (cdist(pw, tw), cdist(pw, pw), cdist(tw, tw))
        dc = (cdist(pc, tc), cdist(pc, pc), cdist(tc, tc))
        point += weighted_ed(*dw, np.ones(len(pw)), np.ones(len(tw))) - weighted_ed(*dc, np.ones(len(pc)), np.ones(len(tc)))

        def draw(n):
            return np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)

        for b in range(n_boot):
            wp, wt = draw(len(pw)), draw(len(tw))
            wpc = wp if len(pc) == len(pw) else draw(len(pc))
            wtc = wt if len(tc) == len(tw) else draw(len(tc))
            draws[b] += weighted_ed(*dw, wp, wt) - weighted_ed(*dc, wpc, wtc)
    draws /= max(len(items), 1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point / max(len(items), 1), "ci_low": float(lo), "ci_high": float(hi), "n_boot": int(n_boot),
            "n_items": len(items)}


def eval_subsample(proto, pred, target, seed_offset: int):
    """The prediction subsample and target indices EvalProtocol.evaluate(pred, target, seed_offset) scores."""
    rng = np.random.default_rng(proto.seed + seed_offset)
    p = np.asarray(proto.subsample(pred, rng))
    n = len(target)
    return p, (rng.choice(n, proto.n_samples, replace=False) if n > proto.n_samples else np.arange(n))


def replicate_floor(x, labels, n_max: int = 1000, n_rep: int = 5, seed: int = 0) -> dict:
    """Energy distance between every pair of biological replicates at equal n (mean of n_rep subsample draws)."""
    x, labels = np.asarray(x), np.asarray(labels).astype(str)
    vals = sorted(set(labels.tolist()))
    pairs = {}
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            xa, xb = x[labels == a], x[labels == b]
            n = min(len(xa), len(xb), n_max)
            if n < 2:
                continue
            eds = []
            for r in range(n_rep):
                rng = np.random.default_rng(seed + r)
                eds.append(energy_distance(xa[rng.choice(len(xa), n, replace=False)],
                                           xb[rng.choice(len(xb), n, replace=False)]))
            pairs[f"{a}|{b}"] = {"energy_distance": float(np.mean(eds)), "n": int(n)}
    floor = float(np.mean([v["energy_distance"] for v in pairs.values()])) if pairs else None
    return {"pairs": pairs, "floor": floor, "n_rep": n_rep, "labels": vals}


# ---------------------------------------------------------------- the measurement

def run_test(ds, test, Xn, obs, keep, symbols, regs, grid, winner, cfg0, seeds, n_hvg, min_detect, mlp, n_boot, t0):
    spec = fs.DATASETS[ds]
    d = multi_removal_design(grid, test["removed"], spec["full_grid"], test["targets"])
    if d["source"] != test["source"]:
        raise ValueError(f"{test['name']}: source {d['source']:g} differs from the pre-registered {test['source']:g}")
    (p,) = d["predictions"]
    F = fs.prepare_fold(Xn, obs, keep, symbols, regs, d, n_hvg, min_detect, f"{ds}_{test['name']}")
    task = time_task(F.s, p["source"], p["target"])
    if (F.all_mask & task.target_mask).any():
        raise ValueError("test target cells overlap the observed cells")
    groups = task_groups(F.s, task)
    target_of = {(c, r): m1 for c, r, _, m1 in groups}
    key = ROBUST_KEY[ds]
    labels = F.obs[key].astype(str).to_numpy()
    arms = ["base", "noft", "gated"] + ([winner] if winner not in ("base", "noft", "gated") else [])
    print(f"\n=== {test['name']} ({test['role']}): {p['target']:g} from {p['source']:g} (h_rel {p['h_rel']:g}, h_span "
          f"{p['h_span']:.3f}, frac {p['frac']:.3f}); bracket {p['source']:g}->{p['t_next']:g}; validation {d['v_prev']:g}->{d['val_time']:g}; "
          f"fit {d['fit_times']}; {len(F.sel)} genes; winner {winner}; models {arms} ({time.time() - t0:.0f}s) ===",
          flush=True)
    rows, rep_rows, path_rows, subs = [], [], [], {}
    for seed in seeds:
        t1 = time.time()
        cfg = cfg0.replace(seed=seed)
        tr = {"v0110": fs.fit_pair(F, d, cfg)}
        if winner in fs.M_RECIPES:
            tr[winner] = fs.fit_pair(F, d, cfg.replace(**fs.M_RECIPES[winner]))
        tr["shuffled"] = fs.fit_pair(F, d, cfg.replace(**fs.M_RECIPES.get(winner, {}), shuffle_biological_time=True))
        secs = round(time.time() - t1, 1)
        preds = []
        for name, trained, rule in [(a, tr[a] if a in fs.M_RECIPES else tr["v0110"], a) for a in arms] + \
                [("shuffled_twin", tr["shuffled"], winner)]:
            nested, model = trained
            stage, weights = fs.arm_stage(rule, p["h_rel"], nested.marginal_iter, model.marginal_iter)
            for cond, rep, m0, _ in groups:
                pred = F.eval_std.transform(model.predict(F.s.x[m0], cond, p["source"], p["target"], stage=stage))
                preds.append((name, "flow", cond, rep, pred, {"weights": weights, "train_seconds": secs,
                                                             **fs.nested_info(nested, model)}))
        for name, cond, rep, pred in interpolation_controls(F.s, F.x_eval, F.all_mask, d["obs_times"], p["source"],
                                                            p["target"], p["t_next"], groups, seed, mlp_kwargs=mlp):
            preds.append((name, "control", cond, rep, pred, {}))
        for name, family, cond, rep, pred, extra in preds:
            m1 = target_of[(cond, rep)]
            target = F.x_eval[m1]
            common = {"dataset": ds, "test": test["name"], "role": test["role"], "seed": seed, "group": f"{cond}|{rep}",
                      "condition": cond, "replicate": rep, "model": name, "family": family}
            rows.append({**common, **F.proto.evaluate(pred, target, seed_offset=seed), **extra})
            for v in sorted(set(labels[m1].tolist())):
                sub = m1 & (labels == v)
                if sub.sum() >= 2:
                    rep_rows.append({**common, "rep_key": key, "rep_value": v, "n_target": int(sub.sum()),
                                     **fs.light_evaluate(F.proto, pred, F.x_eval[sub], seed)})
            if name == winner or family == "control":
                subs[(seed, name, cond, rep)] = eval_subsample(F.proto, pred, target, seed)
        for cond, rep, _, m1 in groups:
            rows.append({"dataset": ds, "test": test["name"], "role": test["role"], "seed": seed,
                         "group": f"{cond}|{rep}", "condition": cond, "replicate": rep, "model": "noise_floor",
                         "family": "floor", **noise_floor(F.x_eval[m1], F.proto)})
        path_rows.append({"seed": seed, **fs.path_family(F, d, p["source"], p["target"], seed)})
        ed = pd.DataFrame([r for r in rows if r["seed"] == seed]).groupby("model")["energy_distance"].mean()
        print(f"  seed {seed}: " + "  ".join(f"{k} {ed[k]:.4f}" for k in [*arms, "shuffled_twin", *CONTROLS] if k in ed)
              + f"  ({time.time() - t1:.0f}s, total {(time.time() - t0) / 60:.0f} min)", flush=True)

    df = pd.DataFrame(rows)
    ps = df[df["family"] != "floor"].groupby(["seed", "model"], as_index=False)["energy_distance"].mean()
    st = conf_stats(ps, winner)
    best = st["best_control"]
    rr = pd.DataFrame(rep_rows)
    idx = ["seed", "group", "rep_value"]
    dr = (rr[rr["model"] == winner].set_index(idx)["energy_distance"]
          - rr[rr["model"] == best].set_index(idx)["energy_distance"])
    rep_deltas = {str(k): float(v) for k, v in
                  dr.groupby(level=["seed", "rep_value"]).mean().groupby(level="rep_value").mean().items()}
    items = []
    for (seed, name, cond, rep), (pw, tw_idx) in subs.items():
        if name == winner:
            pc, tc_idx = subs[(seed, best, cond, rep)]
            tgt = F.x_eval[target_of[(cond, rep)]]
            items.append((pw, tgt[tw_idx], pc, tgt[tc_idx]))
    boot = boot_delta(items, n_boot=n_boot)
    tgt_cells = F.s.mask(times=[p["target"]])
    floor = replicate_floor(F.x_eval[tgt_cells], labels[tgt_cells])
    if "batch" in F.obs.columns:
        floor["batches"] = {v: sorted(set(F.obs.loc[tgt_cells & (labels == v), "batch"].astype(str)))
                            for v in floor["labels"]}
    means = df[df["family"] != "floor"].groupby(["seed", "model"]).mean(numeric_only=True).groupby("model").mean()
    cols = [c for c in ("energy_distance", "mean_error", "var_ratio", "ed_approx", "ed_mean_part", "ed_trace_part",
                        "mmd", "w2", "null_q95") if c in means.columns]
    mb, mw = st["seed_means"][best], st["seed_means"][winner]
    gap = (mb - mw) / (mb - floor["floor"]) if floor["floor"] is not None and mb - floor["floor"] > 0 else None
    pth = pd.DataFrame(path_rows).mean(numeric_only=True)
    out = {"name": test["name"], "role": test["role"], "design": d, "n_genes": int(len(F.sel)), "seeds": list(seeds),
           "horizon": {k: p[k] for k in ("h_rel", "h_span", "frac")},
           "models": arms, "stats": st, "rep_key": key, "rep_deltas": rep_deltas, "boot": boot,
           "replicate_floor": floor, "floor_note": FLOOR_NOTE.get(ds), "gap_closed": gap,
           "model_means": {m: {c: float(means.loc[m, c]) for c in cols} for m in means.index},
           "path_family": {k: float(v) for k, v in pth.items() if k != "seed"}}
    if ds == "pijuan_sala":
        pvr, beats = out["path_family"]["path_var_ratio"], _f(st["vs_mean_interp"]["ci_high"]) < 0
        out["prestated_E8.0"] = {"statement": PRESTATED, "path_var_ratio": pvr,
                                 "winner_var_ratio": out["model_means"][winner]["var_ratio"],
                                 "over_disperses": bool(pvr > 1.2), "winner_beats_mean_interp": bool(beats),
                                 "vs_mean_interp": st["vs_mean_interp"], "held": bool(pvr > 1.2 and not beats)}
    return out, rows, rep_rows


def prereg_params(dataset: str) -> dict:
    """The Step-7 run parameters (Job D's plus 5 seeds and the 200-resample bootstrap)."""
    return {**fs.prereg_params(dataset, SEEDS), "n_boot": N_BOOT}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_problems(summaries: list) -> list:
    """Why a non-smoke set of Job-E runs cannot give the confirmatory verdict (empty: it can)."""
    probs = []
    datasets = sorted(s["dataset"] for s in summaries)
    if datasets != sorted(TESTS):
        probs.append(f"exactly one run per data set is needed; got {datasets}")
    for s in summaries:
        ds = s["dataset"]
        if s.get("complete") is not True:
            probs.append(f"{ds}: the run did not finish (complete is not true)")
        if s.get("winner_forced"):
            probs.append(f"{ds}: the winner was forced (--force-winner is for --smoke only)")
        gone = [t["name"] for t in TESTS[ds] if t["name"] not in s["tests"] and t["name"] not in s["dropped_tests"]]
        if gone:
            probs.append(f"{ds}: tests {gone} were neither measured nor dropped by the QC gate")
        probs += fs.param_problems(s, prereg_params(ds))
    if len({(s.get("choice"), s.get("choice_sha256")) for s in summaries}) != 1:
        probs.append("the runs read different choice.json files: "
                     f"{sorted({(s.get('choice'), s.get('choice_sha256')) for s in summaries}, key=str)}")
    return probs


def check_choice_again(summary: dict, winner: str) -> None:
    """The choice.json both runs read still has the recorded sha256 and still passes verify_choice with this winner."""
    path = Path(summary["choice"])
    if not path.exists() or sha256(path) != summary.get("choice_sha256"):
        raise SystemExit(f"{path} is missing or changed since the measurement read it (sha256 differs)")
    w = fs.verify_choice(json.loads(path.read_text(encoding="utf-8")))
    if w != winner:
        raise SystemExit(f"R-SEL now gives {w!r}, the runs measured {winner!r}")


def decide(dirs: list, verdict_path: str) -> dict:
    summaries = [json.loads((Path(d) / "summary.json").read_text(encoding="utf-8")) for d in dirs]
    smoke = any(s["smoke"] for s in summaries)
    winners = {s["winner"] for s in summaries}
    if len(winners) != 1:
        raise SystemExit(f"the data sets were measured with different winners {winners}")
    if not smoke:
        problems = run_problems(summaries)
        if problems:
            raise SystemExit(f"--decide needs both finished, pre-registered Job-E runs; problems: {problems[:10]}")
        check_choice_again(summaries[0], next(iter(winners)))
    tests, dropped = {}, {}
    for s in summaries:
        tests.update(s["tests"])
        dropped.update(s["dropped_tests"])
    res = decide_conf(tests)
    for n, t in res.items():
        src = tests[n]
        t.update({k: src.get(k) for k in ("horizon", "gap_closed", "replicate_floor", "floor_note", "model_means",
                                          "path_family", "prestated_E8.0", "rep_key", "seeds", "n_genes")})
        t["seed_means"] = src["stats"]["seed_means"]
    for n, why in dropped.items():
        res[n] = {"verdict": "DROPPED", "reason": why}
    if res.get("ps_E8.0", {}).get("verdict") == "POSITIVE":
        res["ps_E8.0"]["reading"] = "the Argelaguet negative did not replicate"
    out = {"prereg": fs.PREREG, "step": "7 (Job E)", "inputs": [str(d) for d in dirs], "smoke": smoke,
           "winner": winners.pop(), "choice": summaries[0].get("choice"),
           "choice_sha256": summaries[0].get("choice_sha256"),
           "winner_forced": any(s.get("winner_forced") for s in summaries),
           "parameters": {s["dataset"]: {k: s.get(k) for k in prereg_params(s["dataset"])} for s in summaries},
           "rule": "R-CONF (see scripts_dev/freshfold_measure.py)", "tests": res}
    fs.write_json(verdict_path, out)
    for n, t in res.items():
        print(f"{n:24s} {t['verdict']:9s} robust {t.get('robust')}  Delta {_f((t.get('delta') or {}).get('mean')):+.4f} "
              f"vs {t.get('best_control')}  p_raw {_f(t.get('p_raw')):.4g}  p_holm {_f(t.get('p_holm')):.4g}", flush=True)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(TESTS))
    p.add_argument("--data", default=None, help="passed to the data set's loader")
    p.add_argument("--cache", default=None, help="generic .npz cache (freshfold_select.write_npz_cache)")
    p.add_argument("--prior", default=None)
    p.add_argument("--choice", default="outputs/freshfold/choice.json", help="written by freshfold_select.py --decide")
    p.add_argument("--seeds", type=int, default=len(SEEDS))
    p.add_argument("--n-hvg", type=int, default=fs.N_HVG)
    p.add_argument("--min-detect", type=float, default=fs.MIN_DETECT)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out")
    p.add_argument("--smoke", action="store_true", help="tiny budget, two seeds: checks the script end to end")
    p.add_argument("--force-winner", default=None,
                   help="--smoke only: measure this arm instead of choice.json's winner (exercises non-base winners)")
    p.add_argument("--decide", nargs="+", default=None, help="output folders of both data sets: apply R-CONF")
    p.add_argument("--verdict", default="outputs/freshfold/verdict.json")
    a = p.parse_args(argv)
    if a.decide:
        decide(a.decide, a.verdict)
        return
    if not (a.dataset and a.out):
        p.error("--dataset and --out are required (or --decide)")
    if a.force_winner and not a.smoke:
        p.error("--force-winner is accepted only with --smoke (no human edit between selection and measurement)")
    raw = Path(a.choice).read_bytes()
    choice, choice_sha = json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
    if choice.get("smoke") and not a.smoke:
        raise SystemExit(f"{a.choice} comes from a smoke run")
    winner = fs.verify_choice(choice)                      # R-SEL again on the Job-D outputs, before any data
    if a.force_winner:
        winner = a.force_winner
    if winner not in fs.FLOW_ARMS and winner not in fs.M_RECIPES:
        raise SystemExit(f"unknown winner {winner!r}")
    spec = fs.DATASETS[a.dataset]
    cfg0, cap, n_hvg, seeds, mlp, n_boot = fs.measured_config(), spec["cap"], a.n_hvg, list(range(a.seeds)), \
        dict(fs.MLP), a.n_boot
    if a.smoke:
        cfg0, cap, n_hvg, seeds, mlp, n_boot = cfg0.replace(**fs.SMOKE), fs.SMOKE_CAP, fs.SMOKE_HVG, [0, 1], \
            fs.SMOKE_MLP, 50
    summary = {"dataset": a.dataset, "smoke": bool(a.smoke), "prereg": fs.PREREG, "choice": a.choice,
               "choice_sha256": choice_sha, "winner": winner, "winner_forced": bool(a.force_winner),
               "winner_recipe": fs.winner_recipe(winner), "config": cfg0.to_dict(), "seeds": seeds, "cap": cap,
               "min_group_cells": fs.MIN_GROUP_CELLS, "n_hvg": n_hvg, "min_detect": a.min_detect, "mlp": mlp,
               "n_boot": n_boot, "tests": {}, "dropped_tests": {}}
    bad = [] if a.smoke else fs.param_problems(summary, prereg_params(a.dataset))
    if bad:
        raise SystemExit(f"not the pre-registered parameters (the test stage is not opened for this): {bad}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, obs, symbols, qc = fs.load_data(a.dataset, a.data, a.cache)
    obs = obs.reset_index(drop=True)
    fs.check_contract(obs, symbols, a.dataset, need=("time", "replicate", "condition", ROBUST_KEY[a.dataset]))
    gated, gate = fs.qc_gate(obs)
    X, obs = X[obs.index.isin(gated.index)], gated.reset_index(drop=True)
    grid = sorted(float(t) for t in np.unique(obs["time"]))
    Xn = normalize_log1p(X, keep_sparse=True)
    regs = fs.prior_regulators(a.prior or spec["prior"])
    keep = fs.subsample_mask(obs, cap, fs.MIN_GROUP_CELLS)
    print(f"{a.dataset}: {X.shape[0]} cells; grid {grid}; QC gate {gate['dropped_stages'] or 'passed'}; winner "
          f"{winner} ({a.choice})", flush=True)
    summary.update(grid=grid, qc_loader=qc, qc_gate=gate)
    rows, rep_rows = [], []
    for test in TESTS[a.dataset]:
        if any(t not in grid for t in test["removed"]):
            summary["dropped_tests"][test["name"]] = "a removed stage failed the QC gate (dropped, never replaced)"
            continue
        res, r, rr = run_test(a.dataset, test, Xn, obs, keep, symbols, regs, grid, winner, cfg0, seeds, n_hvg,
                              a.min_detect, mlp, n_boot, t0)
        summary["tests"][test["name"]] = res
        rows += r
        rep_rows += rr
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        pd.DataFrame(rep_rows).to_csv(out / "replicate_targets.csv", index=False)
        fs.write_json(out / "summary.json", {**summary, "seconds": round(time.time() - t0, 1), "complete": False})
    fs.write_json(out / "summary.json", {**summary, "seconds": round(time.time() - t0, 1), "complete": True,
                                         "note": "per-data-set run; R-CONF pools both data sets (--decide)"})
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
