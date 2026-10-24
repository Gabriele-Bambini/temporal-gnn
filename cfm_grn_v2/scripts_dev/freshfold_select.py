"""Pre-registration v0.2, Step 6 (Job D): fresh-fold selection and the horizon test on Veres et al. (2019) Stage 5 and
Pijuan-Sala et al. (2019).

Why. On the folds seen so far the refit's unguarded second stage helped on some folds and hurt on others (Job A).
Which arm predicts best, and whether the second stage's effect follows the horizon, is decided HERE, on folds of two
data sets no earlier step has looked at, with the test stages deleted before anything else (Veres day 4; Pijuan-Sala
E8.0 = quarter day 6 from E6.5).

Folds: the pre-registered tables (FOLDS below; designs from cfm_grn_v2.training.designs: every removed target predicted
from the last observed time before the block, the bracketing transition observed, validation = the observed inner time
farthest from the centre of the block). Seeds 0 and 1. Each fold has its own gene selection on its fit cells (1000 HVG
union the CollecTRI regulators, min_detect 0.01), cell caps (Veres 1500 per (day, differentiation), Pijuan-Sala 3000 per
stage) and QC gate (a stage enters only with >= 500 cells). One training per (fold, seed): v0.1.10, a nested fit
early-stopped on the validation task, then a refit on every observed stage for the transferred first-stage count that
keeps its stage-1 weights and a stage-2 snapshot every 25 iterations. Arms from it: base (final), noft (stage 1), gated
(stage 1 if h_rel < 1.5, else final), transfer (the stage-2 snapshot at the nested fit's accepted iteration; stage 1 if
0). At most one M arm, the one Job A admitted (outputs/mech_v02_decision.json, 'admitted_to_step6'), is a separate
training: time_noft (field_time 'bio', stage-1 weights; trained with marginal_iters=0, whose weights are bit-identical to
the stage-1 snapshot of a run with the second stage, tests/test_trainer.py) or spline_noft (spline centre, no second
stage). Every control of run_interpolation_benchmark is scored per prediction (designs.interpolation_controls, same
random stream); interior-source folds also predict their targets from the first time (diagnostic, non-voting); base
and noft are also integrated at 8 RK4 steps per unit (H5); the path family's own energy distance and var_ratio at each
target (linear centre, no network) feed H4 and the least-squares fit.

Scores: EvalProtocol's energy distance (n_samples 1000, the same subsampling, hence the same value; averaged over the
(condition, replicate) groups within a seed), with mean_error, var_ratio and the mean/trace split. EvalProtocol's
permutation null and other metrics are skipped here: each costs about 50 energy distances and no Step-6 rule reads them.

Rules, pooled over both data sets and applied by code in --decide, which writes choice.json (read by
freshfold_measure.py; no human edit in between): R-SEL (the arm), R-HOR (the horizon sentence), and the non-deciding
contrasts H2-H5, the h/span secondary, the t=0 diagnostic and a least-squares fit with a fold bootstrap. A non-smoke
--decide refuses a run that is incomplete or was launched with other than the pre-registered parameters (seeds 0 and
1, 1000 HVG, min_detect 0.01, the caps, frozen v0.1.10); freshfold_measure.py re-applies R-SEL to the same results
(verify_choice) and refuses a choice.json whose winner the code did not write.

Data: --data is passed to veres_prepare.load_veres / pijuan_sala_prepare.load_pijuan_sala; --cache reads the generic
.npz written by write_npz_cache (data, indices, indptr, shape, time, replicate, condition, sample, pool, gene_symbols,
qc_json). Either gives (X raw counts, obs with time/replicate/condition/sample/pool, gene symbols, qc). Pijuan-Sala's
loader sets replicate = 'all' (training, couplings and energy-distance groups pool the embryo pools): a documented
deviation from the literal Step-5 text "replicate = pool" (decisions.md, 2026-09-26 afternoon). Reason: every pool is
different embryos at a single stage, so with replicate = pool no transition stays inside one replicate and there would
be no training pair at all. The pool label is kept for Step 7's replicate floor and per-pool ROBUST check.

Run from cfm_grn_v2/ (one data set per job, then the decision):
  PYTHONPATH=src python scripts_dev/freshfold_select.py --dataset veres --data <path> --out outputs/freshfold_veres
  PYTHONPATH=src python scripts_dev/freshfold_select.py --dataset pijuan_sala --data <path> --out outputs/freshfold_ps
  PYTHONPATH=src python scripts_dev/freshfold_select.py --decide outputs/freshfold_veres outputs/freshfold_ps
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.controls import persistence
from cfm_grn_v2.evaluation.metrics import (EvalProtocol, ed_mean_trace_split, energy_distance, mean_error,
                                           median_bandwidth, var_ratio)
from cfm_grn_v2.flows.coupling import chain_ot
from cfm_grn_v2.flows.path import SplinePath
from cfm_grn_v2.training.designs import SHORT, interpolation_controls, multi_removal_design
from cfm_grn_v2.training.trainer import task_groups, time_task, train_flow
from eb_benchmark import BASE, FT

PREREG = "docs/superpowers/prereg/2026-09-25-prereg-v02.md"
MIN_STAGE_CELLS = 500            # QC gate (pre-registered)
MIN_GROUP_CELLS = 200            # a (condition, replicate, stage) group below this is dropped, as in every earlier run
FLOW_ARMS = ("base", "noft", "gated", "transfer")
M_RECIPES = {"time_noft": {"field_time": "bio", "marginal_iters": 0},
             "spline_noft": {"path_center": "spline", "marginal_iters": 0}}
SIMPLICITY = ("noft", "gated", "transfer")          # then the M arm: the tie-break order of R-SEL
TIE, STEPS, H5_STEPS = 0.005, 20, 8
SMOKE = dict(n_iters=200, eval_every=100, patience=2, marginal_iters=50, marginal_eval_every=25, hidden=64)
SMOKE_CAP, SMOKE_HVG, SMOKE_MLP = 200, 150, {"hidden": 32, "n_iters": 200}
MLP, N_HVG, MIN_DETECT, SEEDS = {"hidden": 128, "n_iters": 2000}, 1000, 0.01, [0, 1]     # pre-registered (Step 6)
KEY = ["dataset", "fold", "prediction", "seed"]
H2_LISTED = {("veres", "V23", 2.0), ("pijuan_sala", "P234", 2.0), ("veres", "V3", 3.0), ("pijuan_sala", "P5", 5.0)}


def _folds(table: dict) -> dict:
    return {k: {"removed": tuple(float(t) for t in rem), "targets": tuple(float(t) for t in (tg or rem)),
                "source": float(src)} for k, (rem, tg, src) in table.items()}


DATASETS = {
    "veres": {"test_time": 4.0, "full_grid": [float(t) for t in range(8)], "cap": 1500,
              "prior": "~/rafm_data/priors/collectri_human.tsv", "loader": ("veres_prepare", "load_veres"),
              "folds": _folds({"V1": ((1,), None, 0), "V2": ((2,), None, 1), "V3": ((3,), None, 2),
                               "V5": ((5,), None, 3), "V6": ((6,), None, 5), "V12": ((1, 2), None, 0),
                               "V23": ((2, 3), None, 1), "V56": ((5, 6), None, 3)})},
    "pijuan_sala": {"test_time": 6.0, "full_grid": [float(t) for t in range(9)], "cap": 3000,
                    "prior": "~/rafm_data/priors/collectri_mouse.tsv",
                    "loader": ("pijuan_sala_prepare", "load_pijuan_sala"),
                    "folds": _folds({"P1": ((1,), None, 0), "P2": ((2,), None, 1), "P3": ((3,), None, 2),
                                     "P4": ((4,), None, 3), "P5": ((5,), None, 4), "P7": ((7,), None, 5),
                                     "P12": ((1, 2), None, 0), "P34": ((3, 4), None, 2),
                                     "P234": ((2, 3, 4), (2, 4), 1)})},
}


def measured_config():
    """Frozen v0.1.10: TrainConfig defaults + FT overrides + choice.json {path_center: linear}."""
    return BASE.replace(**FT).replace(path_center="linear")


def prereg_params(dataset: str, seeds=SEEDS) -> dict:
    """The run parameters the pre-registration fixes; a non-smoke run's summary.json must carry exactly these."""
    return {"seeds": list(seeds), "n_hvg": N_HVG, "min_detect": MIN_DETECT, "cap": DATASETS[dataset]["cap"],
            "min_group_cells": MIN_GROUP_CELLS, "mlp": MLP, "config": measured_config().to_dict()}


def param_problems(summary: dict, expected: dict) -> list:
    """Every summary.json value that differs from the expected one (compared after the JSON round trip)."""
    exp = json.loads(json.dumps(jsonable(expected)))
    out = []
    for k, v in exp.items():
        got = summary.get(k)
        if got == v:
            continue
        if isinstance(v, dict) and isinstance(got, dict):
            diff = sorted(x for x in set(v) | set(got) if v.get(x) != got.get(x))
            out.append(f"{summary.get('dataset')}: {k} differs from the pre-registered one in {diff}")
        else:
            out.append(f"{summary.get('dataset')}: {k} = {got!r}, pre-registered {v!r}")
    return out


# ---------------------------------------------------------------- data

def write_npz_cache(path, X, obs, symbols, qc) -> None:
    X = sp.csr_matrix(X)
    np.savez_compressed(path, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                        time=obs["time"].to_numpy(dtype=float),
                        **{k: np.array(obs[k].astype(str).tolist(), dtype=str)
                           for k in ("replicate", "condition", "sample", "pool")},
                        gene_symbols=np.array(list(symbols), dtype=str), qc_json=json.dumps(qc))


def load_npz_cache(path):
    z = np.load(path, allow_pickle=False)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    obs = pd.DataFrame({"time": z["time"].astype(float),
                        **{k: z[k].astype(str) for k in ("replicate", "condition", "sample", "pool")}})
    return X, obs, z["gene_symbols"].astype(str).tolist(), json.loads(str(z["qc_json"]))


def load_data(dataset: str, data=None, cache=None):
    """(X csr raw counts, obs, gene symbols, qc) from the data set's loader (imported only when needed) or a cache."""
    if cache:
        return load_npz_cache(Path(cache).expanduser())
    module, fn = DATASETS[dataset]["loader"]
    loader = getattr(importlib.import_module(module), fn)
    return loader(str(Path(data).expanduser())) if data else loader()


def check_contract(obs: pd.DataFrame, symbols, dataset: str, need=("time", "replicate", "condition")) -> None:
    missing = [c for c in need if c not in obs.columns]
    if missing:
        raise ValueError(f"obs lacks columns {missing}")
    extra = sorted(set(np.unique(obs["time"].to_numpy(dtype=float))) - set(DATASETS[dataset]["full_grid"]))
    if extra:
        raise ValueError(f"times {extra} are not on the pre-registered grid {DATASETS[dataset]['full_grid']}")
    if len(set(symbols)) != len(symbols):
        raise ValueError("gene symbols must be unique")


def at(times, values) -> np.ndarray:
    return np.isin(np.asarray(times, dtype=float), np.asarray(list(values), dtype=float))


def qc_gate(obs: pd.DataFrame, min_cells: int = MIN_STAGE_CELLS):
    """Stages with fewer than min_cells cells leave the data (the rows keep their index)."""
    counts = obs.groupby("time").size()
    ok = [float(t) for t, n in counts.items() if n >= min_cells]
    report = {"min_cells": min_cells, "stage_counts": {str(float(t)): int(n) for t, n in counts.items()},
              "dropped_stages": {str(float(t)): int(n) for t, n in counts.items() if n < min_cells}}
    return obs.loc[at(obs["time"], ok)], report


def prior_regulators(path) -> set:
    return set(pd.read_csv(Path(path).expanduser(), sep="\t")["source_genesymbol"].astype(str))


def subsample_mask(obs: pd.DataFrame, cap, min_cells: int = MIN_GROUP_CELLS, seed: int = 0) -> np.ndarray:
    """The cells snapshot_set_from_arrays(max_cells_per_group=cap, min_cells, rng=default_rng(seed)) keeps, as a mask
    over obs rows, so metadata beyond time/replicate/condition (the pool) stays attached to the kept cells."""
    rng = np.random.default_rng(seed)
    obs = obs.reset_index(drop=True)
    df = pd.DataFrame({"t": obs["time"].to_numpy(dtype=float), "r": obs["replicate"].astype(str).to_numpy(),
                       "c": obs["condition"].astype(str).to_numpy()})
    keep = np.zeros(len(obs), dtype=bool)
    for _, idx in df.groupby(["c", "r", "t"], sort=False).indices.items():
        idx = np.asarray(idx)
        if len(idx) < min_cells:
            continue
        if cap is not None and len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep[idx] = True
    return keep


# ---------------------------------------------------------------- designs, training, scoring

def fold_plan(dataset: str, grid=None):
    """Designs of the pre-registered folds on the QC-gated grid (default: the pre-registered grid). A fold whose
    removed stage is missing, or whose source differs from the table, is dropped and reported, never replaced; a fold
    whose bracket or validation moved is kept and flagged (changed_by_qc)."""
    spec = DATASETS[dataset]
    prereg_grid = [t for t in spec["full_grid"] if t != spec["test_time"]]
    grid = prereg_grid if grid is None else sorted(float(t) for t in grid)
    designs, dropped = {}, {}
    for name, f in spec["folds"].items():
        if any(t not in grid for t in f["removed"]):
            dropped[name] = "a removed stage failed the QC gate"
            continue
        try:
            d = multi_removal_design(grid, f["removed"], spec["full_grid"], f["targets"])
        except ValueError as exc:
            dropped[name] = f"no design on the QC-gated grid: {exc}"
            continue
        if d["source"] != f["source"]:
            dropped[name] = f"source changed by the QC gate ({d['source']:g} instead of {f['source']:g})"
            continue
        ref = multi_removal_design(prereg_grid, f["removed"], spec["full_grid"], f["targets"])
        d["changed_by_qc"] = any(d[k] != ref[k] for k in ("t_next", "val_time", "v_prev"))
        designs[name] = d
    return designs, dropped


def prepare_fold(Xn, obs, keep, symbols, regs, design, n_hvg, min_detect, name):
    """Genes chosen on the fold's fit cells; evaluation space standardised on its observed cells."""
    fit_cells = at(obs["time"], design["fit_times"])
    sel = select_genes(Xn[fit_cells], symbols, n_hvg=n_hvg, must_keep=regs, min_detect=min_detect)
    kept = obs.loc[keep].reset_index(drop=True)
    s = snapshot_set_from_arrays(Xn[keep], kept, symbols, time_key="time", replicate_key="replicate",
                                 condition_key="condition", gene_idx=sel, min_cells=0, name=name)
    fit_mask, all_mask = s.mask(times=design["fit_times"]), s.mask(times=design["obs_times"])
    eval_std = Standardizer().fit(s.x[all_mask])
    x_eval = eval_std.transform(s.x)
    return SimpleNamespace(s=s, sel=sel, obs=kept, fit_mask=fit_mask, all_mask=all_mask, eval_std=eval_std,
                           x_eval=x_eval, val_task=time_task(s, design["v_prev"], design["val_time"]),
                           proto=EvalProtocol(n_samples=1000, seed=0, bandwidth=median_bandwidth(x_eval[all_mask])),
                           n_fit_cells=int(fit_cells.sum()))


def fit_pair(F, design, cfg):
    """Nested fit (early stopping on the validation task), then the refit on every observed stage for the transferred
    first-stage count, keeping stage-1 weights and stage-2 snapshots."""
    nested, _ = train_flow(F.s, F.fit_mask, design["fit_times"], cfg, F.val_task, F.proto)
    n_iters = max(int(nested.best_iter), cfg.eval_every)
    model, _ = train_flow(F.s, F.all_mask, design["obs_times"], cfg.replace(n_iters=n_iters), None, F.proto,
                          keep_snapshots=True)
    return nested, model


def arm_stage(arm: str, h_rel: float, k_nested=None, k_refit=None):
    """(stage for FlowModel.predict, which weights that is). k_nested: the nested fit's accepted stage-2 iteration;
    k_refit: the refit's last stage-2 iteration (its snapshot there IS the final weights)."""
    if arm in M_RECIPES:
        return None, "separate"
    if arm == "base" or (arm == "gated" and h_rel >= SHORT):
        return None, "final"
    if arm in ("noft", "gated"):
        return "stage1", "stage1"
    if arm == "transfer":
        k = int(k_nested or 0)
        if k == 0:
            return "stage1", "stage1"
        return k, "final" if k_refit is not None and k == int(k_refit) else f"stage2@{k}"
    raise ValueError(f"unknown arm {arm!r}")


def light_evaluate(proto: EvalProtocol, pred, target, seed_offset: int = 0) -> dict:
    """EvalProtocol.evaluate's energy distance, mean_error, var_ratio and mean/trace split (same subsamples)."""
    rng = np.random.default_rng(proto.seed + seed_offset)
    p, t = proto.subsample(pred, rng), proto.subsample(target, rng)
    return {"energy_distance": energy_distance(p, t), "mean_error": mean_error(p, t), "var_ratio": var_ratio(p, t),
            **ed_mean_trace_split(p, t)}


def mean_dicts(ds: list) -> dict:
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}


def light_noise_floor(target, proto: EvalProtocol, n_rep: int = 5) -> dict:
    """metrics.noise_floor with light_evaluate: two disjoint halves of the target."""
    target = np.asarray(target)
    out = []
    for r in range(n_rep):
        perm = np.random.default_rng(proto.seed + 1000 + r).permutation(len(target))
        h = len(target) // 2
        out.append(light_evaluate(proto, target[perm[:h]], target[perm[h:2 * h]], r))
    return mean_dicts(out)


def path_family(F, design, source, target, seed) -> dict:
    """The path family itself at the target (no network): linear centre through OT-chained observed cells of each
    group, the refit's construction in the evaluation space (both are standardised on the observed cells)."""
    res = []
    for cond, rep, _, m1 in task_groups(F.s, time_task(F.s, source, target)):
        g = F.all_mask & F.s.mask(conditions=[cond], replicates=[rep])
        knots = [t for t in design["obs_times"] if (g & F.s.mask(times=[t])).sum() >= 2]
        traj = chain_ot([F.x_eval[g & F.s.mask(times=[t])] for t in knots], np.random.default_rng(seed), 1000, mode="ot")
        path = SplinePath(knots, traj, center="linear")
        xp, _ = path.evaluate(np.full(path.n, float(target)), np.arange(path.n))
        res.append(light_evaluate(F.proto, xp, F.x_eval[m1], seed))
    r = mean_dicts(res)
    return {"path_energy_distance": r["energy_distance"], "path_var_ratio": r["var_ratio"]}


def flow_eval(F, model, stage, source, target, groups, seed, steps) -> dict:
    res = []
    for cond, rep, m0, m1 in groups:
        if cond in model.registry.names:
            pred = F.eval_std.transform(model.predict(F.s.x[m0], cond, source, target, stage=stage, n_steps=steps))
            res.append(light_evaluate(F.proto, pred, F.x_eval[m1], seed))
    return mean_dicts(res)


def nested_info(nested, model) -> dict:
    return {"nested_best_iter": nested.best_iter, "nested_marginal_iter": nested.marginal_iter,
            "nested_stage1_val": nested.stage1_best_val, "nested_final_val": nested.best_val,
            "refit_iters": model.best_iter, "refit_marginal_iter": model.marginal_iter}


def run_fold(ds, name, d, F, seeds, cfg0, m_arm, mlp_kwargs, t_start) -> list:
    first = d["obs_times"][0]
    tasks = []
    for p in d["predictions"]:
        tasks.append({"source_role": "design", "prediction": p["target"], "kind": p["kind"], "source": p["source"],
                      "t_next": p["t_next"], "h_rel": p["h_rel"], "h_span": p["h_span"], "frac": p["frac"],
                      "boundary": d["boundary_source"]})
        if not d["boundary_source"]:                                                    # diagnostic, non-voting
            tasks.append({"source_role": "t0", "prediction": p["target"], "kind": p["kind"], "source": first,
                          "t_next": p["t_next"], "h_rel": (p["target"] - first) / d["g"],
                          "h_span": (p["target"] - first) / d["span"], "frac": np.nan, "boundary": True})
    for tk in tasks:
        tk["groups"] = task_groups(F.s, time_task(F.s, tk["source"], tk["prediction"]))
        if not tk["groups"]:
            raise ValueError(f"{name}: no group has cells at both {tk['source']} and {tk['prediction']}")
    common = {"dataset": ds, "fold": name, "removed": "|".join(f"{r:g}" for r in d["removed"]),
              "n_genes": int(len(F.sel)), "changed_by_qc": bool(d.get("changed_by_qc", False))}
    arms = FLOW_ARMS + ((m_arm,) if m_arm else ())
    rows = []
    for seed in seeds:
        t1 = time.time()
        path = {tk["prediction"]: path_family(F, d, tk["source"], tk["prediction"], seed)
                for tk in tasks if tk["source_role"] == "design"}

        def add(tk, model_name, family, metrics, **extra):
            rows.append({**common, "seed": seed, **{k: v for k, v in tk.items() if k != "groups"}, "model": model_name,
                         "family": family, **metrics, **(path[tk["prediction"]] if tk["source_role"] == "design" else {}),
                         **extra})

        cfg = cfg0.replace(seed=seed)
        trainings = {"v0110": fit_pair(F, d, cfg)}
        if m_arm:
            trainings[m_arm] = fit_pair(F, d, cfg.replace(**M_RECIPES[m_arm]))
        secs = round(time.time() - t1, 1)
        for tk in tasks:
            for arm in arms:
                nested, model = trainings[m_arm if arm == m_arm else "v0110"]
                stage, weights = arm_stage(arm, tk["h_rel"], nested.marginal_iter, model.marginal_iter)
                steps_list = (STEPS, H5_STEPS) if arm in ("base", "noft") and tk["source_role"] == "design" else (STEPS,)
                for steps in steps_list:
                    add(tk, arm, "flow", flow_eval(F, model, stage, tk["source"], tk["prediction"], tk["groups"], seed,
                                                   steps),
                        steps=steps, weights=weights, same_as_base=weights == "final", train_seconds=secs,
                        **nested_info(nested, model))
            target = {(c, r): F.x_eval[m1] for c, r, _, m1 in tk["groups"]}
            if tk["source_role"] == "t0":
                add(tk, "persistence", "control",
                    mean_dicts([light_evaluate(F.proto, persistence(F.x_eval[m0]), F.x_eval[m1], seed)
                                for _, _, m0, m1 in tk["groups"]]))
                continue
            per = {}
            for ctrl, cond, rep, pred in interpolation_controls(F.s, F.x_eval, F.all_mask, d["obs_times"], tk["source"],
                                                                tk["prediction"], tk["t_next"], tk["groups"], seed,
                                                                mlp_kwargs=mlp_kwargs):
                per.setdefault(ctrl, []).append(light_evaluate(F.proto, pred, target[(cond, rep)], seed))
            for ctrl, res in per.items():
                add(tk, ctrl, "control", mean_dicts(res))
            add(tk, "noise_floor", "floor", mean_dicts([light_noise_floor(t, F.proto) for t in target.values()]))
        for tk in (t for t in tasks if t["source_role"] == "design"):
            ed = {r["model"]: r["energy_distance"] for r in rows if r["seed"] == seed and r["source_role"] == "design"
                  and r["prediction"] == tk["prediction"] and r.get("steps", STEPS) == STEPS}
            print(f"  seed {seed} {name} {tk['kind']:4s} {tk['prediction']:g} from {tk['source']:g} "
                  f"(h_rel {tk['h_rel']:g}): " + "  ".join(f"{k} {ed[k]:.4f}" for k in (*arms, "persistence",
                                                                                       "mean_interp") if k in ed)
                  + f"  ({time.time() - t1:.0f}s, total {(time.time() - t_start) / 60:.0f} min)", flush=True)
    return rows


# ---------------------------------------------------------------- rules (applied by code)

def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0")


def voting_table(df: pd.DataFrame):
    """The voting pairs p = (dataset, fold, prediction, seed): design source, 20 steps per unit. Returns the wide ED
    table (pairs x models), same_as_base flags, ED_persistence(p) and the pair metadata."""
    main = df[df["source_role"] == "design"]
    flows = main[(main["family"] == "flow") & (main["steps"] == STEPS)]
    if flows.duplicated(KEY + ["model"]).any():
        raise ValueError("duplicate (pair, model) rows")
    ed = flows.set_index(KEY + ["model"])["energy_distance"].unstack("model")
    same = flows.set_index(KEY + ["model"])["same_as_base"].unstack("model")
    pers = main[main["model"] == "persistence"].set_index(KEY)["energy_distance"].reindex(ed.index)
    cols = [c for c in ("h_rel", "frac", "boundary", "path_var_ratio", "h_span") if c in flows.columns]
    meta = flows[flows["model"] == "base"].set_index(KEY)[cols]
    meta = meta.reindex(ed.index).copy()
    meta["boundary"] = meta["boundary"].map(_truthy)
    return ed, same, pers, meta


def _rank(arm: str) -> int:
    return SIMPLICITY.index(arm) if arm in SIMPLICITY else len(SIMPLICITY)


R_SEL_TEXT = ("r_a(p) = (ED_a - ED_base)/ED_persistence(p) over pairs p = (dataset, fold, prediction, seed); S_a = mean "
              "per data set; P_a = mean of S_a over the data sets. a qualifies iff P_a < 0; r_a < 0 in >= 2/3 of the pairs "
              "where its prediction differs from base; S_a <= +0.02 in each data set; no non-finite value and no ED_a > "
              "3 x ED_base. Winner: most negative P_a; if the best two differ by < 0.005 the simpler wins (noft < gated < "
              "transfer < M); if nothing qualifies, base stays.")


def r_sel(df: pd.DataFrame, eligible) -> dict:
    ed, same, pers, _ = voting_table(df)
    base = ed["base"]
    per_arm = {}
    for arm in eligible:
        if arm not in ed.columns:
            per_arm[arm] = {"qualifies": False, "reason": "not run"}
            continue
        a = ed[arm]
        r = (a - base) / pers
        S = r.groupby(level="dataset").mean()
        P = float(S.mean())
        differs = ~same[arm].map(_truthy) if arm in same.columns else (a != base)
        n_diff, n_neg = int(differs.sum()), int((r[differs] < 0).sum())
        c = {"1_P_below_0": bool(np.isfinite(P) and P < 0),
             "2_r_below_0_in_2_3_of_differing": bool(n_diff > 0 and 3 * n_neg >= 2 * n_diff),
             "3_S_at_most_0.02_each_dataset": bool(len(S) > 0 and np.isfinite(S).all() and (S <= 0.02).all()),
             "4_finite_and_within_3x_base": bool(np.isfinite(r).all() and np.isfinite(a).all()
                                                 and not (a > 3 * base).any())}
        per_arm[arm] = {"P": P, "S": {k: float(v) for k, v in S.items()}, "n_pairs": int(len(r)),
                        "n_differing": n_diff, "n_negative_differing": n_neg, "criteria": c,
                        "qualifies": all(c.values()),
                        "r": {"|".join(f"{x:g}" if isinstance(x, float) else str(x) for x in k): float(v)
                              for k, v in r.items()}}
    qual = sorted((a for a, v in per_arm.items() if v["qualifies"]), key=lambda a: per_arm[a]["P"])
    if not qual:
        winner, how = "base", "nothing qualifies: base stays"
    elif len(qual) == 1 or per_arm[qual[1]]["P"] - per_arm[qual[0]]["P"] >= TIE:
        winner, how = qual[0], "most negative P_a"
    else:
        winner = min(qual[:2], key=_rank)
        how = f"best two ({qual[0]}, {qual[1]}) differ by < {TIE}: the simpler wins"
    return {"per_arm": per_arm, "qualifying": qual, "winner": winner, "how": how}


def r_hor(df: pd.DataFrame) -> dict:
    """d(p) = ED_noft - ED_base. CONFIRMED iff d < 0 in >= 2/3 of short pairs, d > 0 in >= 2/3 of long pairs and the class
    medians have the predicted sign in both data sets; PARTIAL if exactly one class meets its 2/3 criterion; REFUTED
    otherwise (as written, also when both 2/3 criteria hold but a data set's median has the wrong sign)."""
    ed, _, _, meta = voting_table(df)
    d = ed["noft"] - ed["base"]
    short = (meta["h_rel"] < SHORT).to_numpy()
    ds = ed.index.get_level_values("dataset").to_numpy()

    def cls(mask, sign):
        x = d[mask]
        hits = int((x < 0).sum()) if sign < 0 else int((x > 0).sum())
        return {"n": int(len(x)), "hits": hits, "meets_2_3": bool(len(x) > 0 and 3 * hits >= 2 * len(x)),
                "median": float(x.median()) if len(x) else None}

    s, lg = cls(short, -1), cls(~short, +1)
    med = {}
    for name in sorted(set(ds)):
        ms, ml = d[(ds == name) & short].median(), d[(ds == name) & ~short].median()
        med[name] = {"short_median": float(ms), "long_median": float(ml), "ok": bool(ms < 0 and ml > 0)}
    medians_ok = bool(med) and all(v["ok"] for v in med.values())
    if s["meets_2_3"] and lg["meets_2_3"] and medians_ok:
        verdict, half = "CONFIRMED", None
    elif s["meets_2_3"] != lg["meets_2_3"]:
        verdict, half = "PARTIAL", "short" if s["meets_2_3"] else "long"
    else:
        verdict, half = "REFUTED", None
    note = ("both 2/3 criteria hold but a data set's class median has the wrong sign (or is undefined): REFUTED as "
            "written" if s["meets_2_3"] and lg["meets_2_3"] and not medians_ok else None)
    return {"d": "ED_noft - ED_base (design source, 20 steps per unit)", "short": s, "long": lg,
            "medians_by_dataset": med, "medians_ok": medians_ok, "verdict": verdict, "replicated_half": half,
            "note": note}


def _summ(x) -> dict:
    x = pd.Series(np.asarray(x, dtype=float)).dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(x.median()), "mean": float(x.mean()),
            "frac_positive": float((x > 0).mean()), "frac_nonpositive": float((x <= 0).mean())}


def _label(idx) -> list:
    return [f"{k[0]}|{k[1]}|{k[2]:g}|s{k[3]}" for k in idx]


def _spearman(a, b) -> float:
    """Spearman's rho, nan below 3 pairs or when either side is constant (e.g. h_span within the short pairs)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def contrasts(df: pd.DataFrame) -> dict:
    """H3, H2, H4, H5 and the t=0 diagnostic: reported, never deciding."""
    ed, _, pers, meta = voting_table(df)
    d = ed["noft"] - ed["base"]
    h, frac = meta["h_rel"].to_numpy(dtype=float), meta["frac"].to_numpy(dtype=float)
    bnd = meta["boundary"].to_numpy(dtype=bool)
    long_ = h >= SHORT
    h1, interior = np.isclose(h, 1.0), ~bnd
    listed = np.array([(k[0], k[1], float(k[2])) in H2_LISTED for k in ed.index])
    crit = h1 & interior & (frac >= 0.25 - 1e-9) & (frac <= 1 / 3 + 1e-9)
    half = h1 & interior & np.isclose(frac, 0.5)
    out = {"d": "ED_noft - ED_base (design source, 20 steps per unit)",
           "H3": {"long_boundary_source": _summ(d[long_ & bnd]), "long_interior_source": _summ(d[long_ & interior]),
                  "pairs_boundary": _label(ed.index[long_ & bnd]),
                  "prediction": "H3: d <= 0 for boundary sources; H1: d > 0 for both"},
           "H2": {"listed_frac_quarter_third": _summ(d[listed]), "listed_pairs": _label(ed.index[listed]),
                  "by_criterion_frac_quarter_third": _summ(d[crit]),
                  "criterion_pairs_not_listed": _label(ed.index[crit & ~listed]),
                  "frac_half": _summ(d[half]), "frac_half_pairs": _label(ed.index[half]),
                  "definition": "h_rel = 1, interior source; frac 1/4-1/3 against frac 1/2"}}
    h4 = {}
    pvr = meta["path_var_ratio"].to_numpy(dtype=float)
    for name, m in (("all", np.ones(len(d), dtype=bool)), ("short", ~long_), ("long", long_)):
        ok = m & np.isfinite(pvr) & np.isfinite(d.to_numpy(dtype=float))
        h4[name] = {"n": int(ok.sum()), "spearman_d_vs_path_var_ratio": _spearman(d.to_numpy(dtype=float)[ok], pvr[ok])}
    out["H4"] = h4
    if "h_span" in meta.columns:
        hs = meta["h_span"].to_numpy(dtype=float)
        dv = d.to_numpy(dtype=float)
        sec = {}
        for name, m in (("all", np.ones(len(d), dtype=bool)), ("short", ~long_), ("long", long_)):
            ok = m & np.isfinite(hs) & np.isfinite(dv)
            sec[name] = {"n": int(ok.sum()), "spearman_d_vs_h_span": _spearman(dv[ok], hs[ok]),
                         "spearman_d_vs_h_rel": _spearman(dv[ok], h[ok])}
        out["h_span_secondary"] = {**sec, "h_span": "(target - source)/span of the full grid (Veres 7 d, "
                                                    "Pijuan-Sala 8 quarter days): the pre-registered secondary "
                                                    "horizon measure, reported next to h_rel, not deciding",
                                   "d_by_h_span": {f"{v:.4g}": _summ(d[np.isclose(hs, v)]) for v in
                                                   sorted(set(np.round(hs[np.isfinite(hs)], 6)))}}
    flows = df[(df["source_role"] == "design") & (df["family"] == "flow")]
    e8 = flows[flows["steps"] == H5_STEPS].set_index(KEY + ["model"])["energy_distance"].unstack("model")
    if {"base", "noft"} <= set(e8.columns):
        d8 = e8["noft"] - e8["base"]
        s8 = (meta["h_rel"].reindex(e8.index) < SHORT).to_numpy()
        gain = ed["base"].reindex(e8.index) - e8["base"]
        out["H5"] = {"d8": _summ(d8), "d20": _summ(d.reindex(e8.index)), "d8_minus_d20": _summ(d8 - d.reindex(e8.index)),
                     "base_ed20_minus_ed8_short": _summ(gain[s8]), "base_ed20_minus_ed8_long": _summ(gain[~s8]),
                     "reading": "the 8-vs-20-step mismatch explains part of the short-pair hurt if base improves by "
                                "> 0.05 on short pairs at 8 steps"}
    t0 = df[(df["source_role"] == "t0") & (df["family"] == "flow") & (df["steps"] == STEPS)]
    if len(t0):
        e0 = t0.set_index(KEY + ["model"])["energy_distance"].unstack("model")
        d0 = e0["noft"] - e0["base"]
        dd = d.reindex(e0.index)
        out["t0_diagnostic"] = {"d_from_t0": _summ(d0), "d_from_design_source": _summ(dd),
                                "d_t0_minus_d_source": _summ(d0 - dd),
                                "reading": "H1 predicts d grows from t=0; H3 predicts it shrinks"}
    return out


def ls_fit(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """d/ED_persistence on log h_rel + frac + boundary + path var_ratio + data set, least squares with a 95% fold
    bootstrap interval (folds resampled with replacement). Reported, not deciding: no p-value decides anything."""
    ed, _, pers, meta = voting_table(df)
    y = ((ed["noft"] - ed["base"]) / pers).to_numpy(dtype=float)
    ds = ed.index.get_level_values("dataset").to_numpy()
    cols = {"intercept": np.ones(len(y)), "log_h_rel": np.log(meta["h_rel"].to_numpy(dtype=float)),
            "frac": meta["frac"].to_numpy(dtype=float), "boundary": meta["boundary"].to_numpy(dtype=float),
            "var_ratio": meta["path_var_ratio"].to_numpy(dtype=float),
            "dataset_pijuan_sala": (ds == "pijuan_sala").astype(float)}
    X = np.column_stack(list(cols.values()))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    folds = np.array([f"{a}|{b}" for a, b in zip(ds, ed.index.get_level_values("fold"))])[ok]
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    uf = np.unique(folds)
    rows_of = {f: np.flatnonzero(folds == f) for f in uf}
    rng = np.random.default_rng(seed)
    boots = np.array([np.linalg.lstsq(X[idx], y[idx], rcond=None)[0] for idx in
                      (np.concatenate([rows_of[uf[i]] for i in rng.integers(0, len(uf), len(uf))])
                       for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = list(cols)
    return {"y": "(ED_noft - ED_base)/ED_persistence", "coef": {k: float(c) for k, c in zip(names, coef)},
            "ci95": {k: [float(a), float(b)] for k, a, b in zip(names, lo, hi)}, "rank": int(rank),
            "n_columns": len(names), "n_pairs": int(ok.sum()), "n_folds": int(len(uf)), "n_boot": int(n_boot),
            "note": "fold bootstrap; minimum-norm least squares when the design is rank-deficient; not deciding"}


def winner_recipe(winner: str) -> dict:
    weights = {"base": "final", "noft": "stage1", "gated": "stage1 if h_rel < 1.5 else final",
               "transfer": "the refit's stage-2 snapshot at the nested fit's accepted iteration (stage1 if 0)"}
    return {"arm": winner, "training": "separate" if winner in M_RECIPES else "v0.1.10",
            "overrides": {"path_center": "linear", **M_RECIPES.get(winner, {})},
            "weights": weights.get(winner, "final weights of the separate training")}


def completeness(df: pd.DataFrame, summaries: list, seeds) -> list:
    problems = []
    for ds in sorted({s["dataset"] for s in summaries}):
        dropped = {k for s in summaries if s["dataset"] == ds for k in s["dropped_folds"]}
        got = set(map(tuple, df[(df["dataset"] == ds) & (df["source_role"] == "design") & (df["model"] == "base")
                                & (df["steps"] == STEPS)][["fold", "prediction", "seed"]].to_numpy().tolist()))
        for name, f in DATASETS[ds]["folds"].items():
            if name in dropped:
                continue
            for t in f["targets"]:
                for s in seeds:
                    if (name, t, s) not in got:
                        problems.append(f"{ds} {name} target {t:g} seed {s} missing")
    return problems


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(obj), indent=1), encoding="utf-8")


def load_selection(dirs: list) -> SimpleNamespace:
    """The Job-D outputs in dirs, pooled. A non-smoke set is refused (SystemExit) unless both data sets are there, every
    run finished, every pre-registered (fold, target, seed) was scored, no other seed was, and every run used the
    pre-registered parameters (prereg_params)."""
    summaries = [json.loads((Path(d) / "summary.json").read_text(encoding="utf-8")) for d in dirs]
    df = pd.concat([pd.read_csv(Path(d) / "results.csv") for d in dirs], ignore_index=True)
    smoke = any(s["smoke"] for s in summaries)
    datasets = sorted({s["dataset"] for s in summaries})
    if len({json.dumps(s["admitted_to_step6"]) for s in summaries}) != 1:
        raise SystemExit("the runs used different Job A admissions")
    admitted, m_arm = summaries[0]["admitted_to_step6"], summaries[0]["m_arm"]
    if smoke:
        problems = completeness(df, summaries, sorted({int(s) for x in summaries for s in x["seeds"]}))
    else:
        problems = completeness(df, summaries, SEEDS)
        problems += [f"{s['dataset']} run {d} did not finish (complete is not true)"
                     for d, s in zip(dirs, summaries) if s.get("complete") is not True]
        problems += [q for s in summaries for q in param_problems(s, prereg_params(s["dataset"]))]
        extra = sorted(set(pd.to_numeric(df["seed"]).astype(int)) - set(SEEDS))
        if extra:
            problems.append(f"rows with seeds {extra}: the pre-registered seeds are {SEEDS}")
        if datasets != ["pijuan_sala", "veres"] or problems:
            raise SystemExit(f"--decide pools both complete, pre-registered data sets; got {datasets}; problems: "
                             f"{problems[:10]}")
    eligible = [a for a in SIMPLICITY if a in admitted] + ([m_arm] if m_arm else [])
    return SimpleNamespace(summaries=summaries, df=df, smoke=smoke, datasets=datasets, admitted=admitted, m_arm=m_arm,
                           eligible=eligible, problems=problems)


def verify_choice(choice: dict) -> str:
    """Job E's guard that choice.json is what --decide wrote: R-SEL is applied again, by code, to the Job-D outputs the
    file names (same checks as --decide), and its winner must equal both 'winner' and 'r_sel.winner'. Returns it."""
    if not choice.get("smoke") and choice.get("missing"):
        raise SystemExit(f"choice.json lists missing Job-D results: {choice['missing'][:10]}")
    if not isinstance(choice.get("inputs"), list) or not choice["inputs"]:
        raise SystemExit("choice.json names no Job-D output folders ('inputs'): it was not written by --decide")
    sel = load_selection(choice["inputs"])
    if sel.smoke != bool(choice.get("smoke")):
        raise SystemExit("choice.json's smoke flag differs from the Job-D runs it names")
    if sel.eligible != choice.get("eligible"):
        raise SystemExit(f"choice.json's eligible arms {choice.get('eligible')} differ from the Job-D runs' {sel.eligible}")
    w = r_sel(sel.df, sel.eligible)["winner"]
    recipe = json.loads(json.dumps(jsonable(winner_recipe(w))))
    if not (w == choice.get("winner") == (choice.get("r_sel") or {}).get("winner")) or choice.get("winner_recipe") != recipe:
        raise SystemExit(f"choice.json names winner {choice.get('winner')!r} (r_sel: "
                         f"{(choice.get('r_sel') or {}).get('winner')!r}) but R-SEL on {choice['inputs']} gives {w!r}: "
                         "the file was not written by the code as it stands (no human edit is allowed)")
    return w


def decide(dirs: list, choice_path: str) -> dict:
    sel_in = load_selection(dirs)
    df, summaries, eligible = sel_in.df, sel_in.summaries, sel_in.eligible
    sel = r_sel(df, eligible)
    choice = {"prereg": PREREG, "step": "6 (Job D)", "inputs": [str(d) for d in dirs], "smoke": sel_in.smoke,
              "datasets": sel_in.datasets, "missing": sel_in.problems, "admitted_to_step6": sel_in.admitted,
              "m_arm": sel_in.m_arm, "eligible": eligible, "rule": R_SEL_TEXT, "r_sel": sel, "winner": sel["winner"],
              "winner_recipe": winner_recipe(sel["winner"]), "r_hor": r_hor(df), "contrasts": contrasts(df),
              "ls_fit": ls_fit(df), "parameters": {s["dataset"]: {k: s.get(k) for k in prereg_params(s["dataset"])}
                                                   for s in summaries},
              "dropped_folds": {s["dataset"]: s["dropped_folds"] for s in summaries}}
    write_json(choice_path, choice)
    print(json.dumps(jsonable({k: choice[k] for k in ("winner", "eligible", "missing")}), indent=1))
    print({a: (round(v.get("P", np.nan), 4), v["qualifies"]) for a, v in sel["per_arm"].items()})
    print(f"R-HOR: {choice['r_hor']['verdict']} -> {choice_path}", flush=True)
    return choice


def admitted_arms(path, smoke: bool = False):
    """Job A's admission list (base, noft, gated, transfer unless they failed sanity; at most one M arm)."""
    p = Path(path)
    if not p.exists():
        if smoke:
            return list(FLOW_ARMS), None
        raise SystemExit(f"{path} not found: Job A (pre-registration Step 2) must be decided before Job D")
    dec = json.loads(p.read_text(encoding="utf-8"))
    if "admitted_to_step6" not in dec:
        raise SystemExit(f"{path} has no admission list (Job A's reproducibility gate failed?): stop and debug")
    adm = [str(a) for a in dec["admitted_to_step6"]]
    unknown = set(adm) - set(FLOW_ARMS) - set(M_RECIPES)
    ms = [a for a in adm if a in M_RECIPES]
    if unknown or len(ms) > 1:
        raise SystemExit(f"unexpected admission list {adm}")
    return adm, (ms[0] if ms else None)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS))
    p.add_argument("--data", default=None, help="passed to the data set's loader")
    p.add_argument("--cache", default=None, help="generic .npz cache (write_npz_cache) instead of the loader")
    p.add_argument("--prior", default=None, help="CollecTRI TSV (default: the data set's)")
    p.add_argument("--mech-decision", default="outputs/mech_v02_decision.json")
    p.add_argument("--seeds", type=int, default=len(SEEDS))
    p.add_argument("--folds", nargs="+", default=None, help="subset of fold names (split jobs; --decide checks the union)")
    p.add_argument("--n-hvg", type=int, default=N_HVG)
    p.add_argument("--min-detect", type=float, default=MIN_DETECT)
    p.add_argument("--out")
    p.add_argument("--smoke", action="store_true", help="tiny budget, one seed: checks the script end to end")
    p.add_argument("--decide", nargs="+", default=None, help="output folders of both data sets: apply the rules")
    p.add_argument("--choice", default="outputs/freshfold/choice.json")
    a = p.parse_args(argv)
    if a.decide:
        decide(a.decide, a.choice)
        return
    if not (a.dataset and a.out):
        p.error("--dataset and --out are required (or --decide)")
    spec = DATASETS[a.dataset]
    admitted, m_arm = admitted_arms(a.mech_decision, a.smoke)
    cfg0, cap, n_hvg, seeds, mlp = measured_config(), spec["cap"], a.n_hvg, list(range(a.seeds)), dict(MLP)
    if a.smoke:
        cfg0, cap, n_hvg, seeds, mlp = cfg0.replace(**SMOKE), SMOKE_CAP, SMOKE_HVG, [0], SMOKE_MLP
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, obs, symbols, qc = load_data(a.dataset, a.data, a.cache)
    obs = obs.reset_index(drop=True)
    check_contract(obs, symbols, a.dataset)
    test = at(obs["time"], [spec["test_time"]])
    X, obs = X[~test], obs.loc[~test].reset_index(drop=True)             # the test stage leaves before anything else
    gated, gate = qc_gate(obs)
    X, obs = X[obs.index.isin(gated.index)], gated.reset_index(drop=True)
    grid = sorted(float(t) for t in np.unique(obs["time"]))
    designs, dropped = fold_plan(a.dataset, grid)
    if a.folds:
        unknown = set(a.folds) - set(spec["folds"])
        if unknown:
            p.error(f"unknown folds {sorted(unknown)}")
        designs = {k: v for k, v in designs.items() if k in a.folds}
    print(f"{a.dataset}: test time {spec['test_time']:g} removed; {X.shape[0]} cells; grid {grid}; QC gate "
          f"{gate['dropped_stages'] or 'passed'}; folds {list(designs)}; dropped {dropped or 'none'}; M arm {m_arm}",
          flush=True)
    Xn = normalize_log1p(X, keep_sparse=True)
    regs = prior_regulators(a.prior or spec["prior"])
    keep = subsample_mask(obs, cap, MIN_GROUP_CELLS)
    summary = {"dataset": a.dataset, "smoke": bool(a.smoke), "prereg": PREREG, "test_time_removed": spec["test_time"],
               "full_grid": spec["full_grid"], "grid": grid, "qc_loader": qc, "qc_gate": gate,
               "folds_requested": a.folds, "dropped_folds": dropped, "designs": designs, "genes": {},
               "admitted_to_step6": admitted, "m_arm": m_arm, "m_recipe": M_RECIPES.get(m_arm),
               "config": cfg0.to_dict(), "seeds": seeds, "cap": cap, "min_group_cells": MIN_GROUP_CELLS,
               "n_hvg": n_hvg, "min_detect": a.min_detect, "mlp": mlp, "n_kept_cells": int(keep.sum())}
    bad = [] if a.smoke else param_problems(summary, prereg_params(a.dataset))
    if bad:
        print(f"WARNING: not the pre-registered parameters, --decide will refuse this run: {bad}", flush=True)
    rows = []
    for name, d in designs.items():
        F = prepare_fold(Xn, obs, keep, symbols, regs, d, n_hvg, a.min_detect, f"{a.dataset}_{name}")
        summary["genes"][name] = {"n_selected": int(len(F.sel)), "n_fit_cells": F.n_fit_cells,
                                  "n_prior_regulators": int(sum(symbols[i] in regs for i in F.sel))}
        print(f"\n=== {name}: removed {d['removed']}, predictions "
              f"{[(pr['kind'], pr['target'], round(pr['h_rel'], 3), round(pr['frac'], 3)) for pr in d['predictions']]} "
              f"from {d['source']:g}; bracket {d['source']:g}->{d['t_next']:g}; validation {d['v_prev']:g}->"
              f"{d['val_time']:g}; fit {d['fit_times']}; {len(F.sel)} genes ({time.time() - t0:.0f}s) ===", flush=True)
        rows += run_fold(a.dataset, name, d, F, seeds, cfg0, m_arm, mlp, t0)
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        write_json(out / "summary.json", {**summary, "seconds": round(time.time() - t0, 1), "complete": False})
    write_json(out / "summary.json", {**summary, "seconds": round(time.time() - t0, 1), "complete": True,
                                      "note": "per-data-set run; the rules pool both data sets (--decide)"})
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
