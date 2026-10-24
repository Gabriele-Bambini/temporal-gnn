"""Pre-registration v0.2, Step 4 (Job C): the like-for-like published embryoid-body benchmark.

Why. Our embryoid-body result is an energy distance in ~1300 genes; the flow-matching literature scores the same time
course in 5 whitened principal components with the exact Wasserstein-1 distance (TrajectoryNet, Tong et al. 2020,
Table 3; OT-CFM, Tong et al. 2024: 0.790 +- 0.068). This job measures the frozen model in THAT space and metric, next
to our controls, the published copy baselines and an in-package OT-CFM equivalent, all on identical cells; the numbers
may sit next to the published ones only if the data and the metric first reproduce the published baselines.

Data. TrajectoryNet's processed file eb_velocity_v5.npz: 'pcs'[:, :5] and 'sample_labels' 0..4 as biological time at
unit spacing (one unit = six days); one condition (EB), one replicate, EVERY cell (no cap, no gene selection); the
other keys are ignored. The file's SHA-256 must equal the one recorded at download (pre-registration), else the run
stops. Scalers (sklearn StandardScaler), both reported:
  primary    fitted on ALL cells, the published convention (TorchCFM's CustomTrajectoryDataModule whitens before any
             split, the held-out window included: a known leak, reproduced for comparability; it decides the rules);
  secondary  fitted per fold on the four observed windows only.
Every model is trained, and every metric computed, inside each scaler's space.

Folds t = 1, 2, 3; seeds 0-4. search.interpolation_design on the other four windows (nested validation on the observed
inner window farthest from t, ties to the later one), early stopping on that validation task, refit on the four
observed windows for the transferred first-stage count (its 1000 stage-2 iterations unguarded), exactly as
run_interpolation_benchmark(refit_on_all=True); EVERY cell of window t-1 is pushed over one unit to t.

Configurations. C_a = frozen v0.1.10 (eb_benchmark.BASE + FT + path_center 'linear'), final weights only: the
pre-registration forbids stage-1, gated or transfer predictions on the EB days 12-15 window until a rule qualifies on
fresh folds, so none is computed. --choice (choice.json written by freshfold_select.py --decide) measures the Step-6
winner, C_b, with the same rules (arm semantics from freshfold_select.arm_stage; its shuffled twin uses the winner's
recipe and weights); a 'base' winner is refused: C_b would be C_a and nothing new is measured (Step 8).

Metrics, against ALL cells of window t. Exact W1: POT ot.emd2, uniform weights, Euclidean cost, numItermax 1e7 (as in
TorchCFM's distribution_distances), retried once at 1e8 when the simplex stops before optimality (a stopped emd2 is the
cost of a feasible plan, >= W1, not the exact W1); convergence recorded per row, and a rule fed by a row that still did
not converge is 'not decidable' (neither pass nor fail). A full run with POT first checks it on W1(X, X + a) = |a|
(preflight, before any training). EvalProtocol.evaluate (energy distance, mean_error, var_ratio, the mean/trace split,
...); a secondary W1 on 2000-vs-2000 seeded subsamples (the target subsample is shared by every model of a seed). Without
POT the pre-registered fallback (Step 1e) applies: the exact assignment (scipy linear_sum_assignment) on equal-size
subsamples, 2000 per side, mean of 10 seeded repeats (one assignment on everything when both sets already have the same
size <= 2000); the backend is recorded per row and in summary.json. Flow and twin rows also carry the prediction at 8 RK4
steps per unit (Step 1c: w1_8steps, energy_distance_8steps; non-voting). Also per fold: the W1 and energy distance
between two random halves of window t (the finite-sample floor, non-voting).

Controls (same cells, same space): persistence ('prev'), mean_interp, ot_interp (McCann, frac 1/2 along the bracket's
OT pairs), linear, mlp, freefm_replay (designs.interpolation_controls = search.run_interpolation_benchmark's controls,
same random stream); 'next' (a copy of window t+1); the shuffled-time twin (a separate comparator); and:

In-package OT-CFM equivalent: straight paths between OT-coupled consecutive observed windows, a time input, no second
stage. Hyper-parameters READ with WebFetch on 2026-09-26 (re-checked on the raw files the same day) from the TorchCFM
runner (github.com/atong01/conditional-flow-matching, main, runner/; composition experiment=cfm model=otcfm
datamodule=time_dist) and mapped onto TrainConfig, not tuned (TORCHCFM below is that record; were it marked unread,
otcfm_config would fall back to TrainConfig's defaults with the same structure, and summary.json says so):
  configs/model/otcfm.yaml  VelocityNet hidden_dims [64, 64, 64], selu -> hidden 64, depth 3 (our MLP is SiLU)
                      AdamW lr 1e-3, weight_decay 1e-5                -> lr 1e-3, weight_decay 1e-5 (Adam, L2 form)
                      sigma_min 0.1 (constant), ot_sampler 'exact'     -> sigma_min 0.1, coupling 'ot' (exact, per batch)
                      euler; the test integrates linspace(0, 1, 101)   -> ode_method 'euler', 100 steps per unit
  configs/experiment/cfm.yaml  datamodule.batch_size 512 (merged after time_dist's 128), trainer.max_epochs 1000,
                      check_val_every_n_epoch 10                        -> batch_size 512, n_iters = 1000 x batches/epoch
  configs/datamodule/time_dist.yaml  CustomTrajectoryDataModule, split [0.8, 0.1, 0.1], max_dim 5, whiten True
                      (StandardScaler on every cell before the split); an epoch is a CombinedLoader(mode='min_size') over
                      the time points' 80 % training splits with drop_last, i.e. min_t floor(n_train(t) / 512) batches
                      (n_train: torch random_split lengths; here over the four observed windows, which reproduces the
                      published count unless the held-out window is the smallest; recorded per fold)
  src/models/cfm_module.py  leave-out: the bracket pair (t-1, t+1) is OT-coupled and trained with the flow time doubled
                      and the velocity halved, the net fed the absolute time t_select + s; test: EVERY cell of the
                      observed previous window (test loader load_full) integrated over one unit and scored against EVERY
                      cell of the window, as here               -> clock 'flow', time_conditioning 'absolute' (the context
                      carries t0, t1: the absolute time t0 + s (t1 - t0) is a function of the inputs), path 'linear',
                      consecutive observed pairs; window t = the bracket transition integrated from the window t-1 cells
                      to flow time 1/2 (one biological unit, 100 Euler steps)
  configs/callbacks/default.yaml + src/train.py  early stopping on val/loss (patience 100 checks, one check every 10
                      epochs, i.e. never within 1000 epochs) and the best val/loss checkpoint tested; with batch 512 and
                      drop_last the 10 % validation split of every window (< 512 cells) holds no batch, so by our reading
                      neither acts and the final weights are tested -> final weights, no early stopping, no EMA
                      (ema_decay 0), no clipping (grad_clip 0), no second stage (marginal_iters 0); one training on every
                      cell of the four observed windows.
  Not reproduced: the runner's 80 % cell split, SELU, AdamW's decoupled decay, its raw scalar time input, the per-sample
  mixing of pairs inside a batch (a pair drawn uniformly per sample; we draw one pair per step, weighted by cell count),
  pairs sampled from the exact plan (we take the assignment of equal-size batches) and seed 42 (we use seeds 0-4).

Rules (pre-registered, applied by code in decide(), every outcome in summary.json; the primary scaler decides):
  calibration gate  W1 of persistence and 'next' within +-0.05 of TrajectoryNet Table 3 at every fold
                    (prev 1.715 / 1.400 / 0.814, next 1.400 / 0.814 / 1.694 for t = 1 / 2 / 3);
  model check       the OT-CFM equivalent's 3-fold mean W1 (averaged over seeds) within 0.790 +- 0.068;
  P2a               the literal text: the flow's per-seed 3-fold mean W1 below every control's IN EVERY SEED (the
                    shuffled twin is a separate comparator); upper paired 95% t-CI of (flow - best control) < 0, the
                    best control being the lowest seed mean, chosen afterwards; upper CI of (flow - shuffled twin) < 0.
                    The looser seed-mean reading is reported next to it (passes_seed_mean), never deciding;
  P2b               descriptive, only if gate and check both pass: 'below published OT-CFM' iff mean + 1.96 sd/sqrt(5)
                    < 0.790, 'within the published range' iff mean < 0.848, otherwise 'above' (no statement unless
                    the 5 pre-registered seeds are all there); never a significance claim against published numbers.
                    If a check fails, only within-run comparisons are claimed; a flow that loses to mean interpolation
                    is flagged as a finding about the benchmark.
Also per-fold tables (t = 2 is the window of our gene-space result), and P2a in the energy distance and in the
secondary scaler (descriptive). A full run refuses any seed set other than 0-4. summary.json is first written as soon
as the deciding primary scaler is done ('complete': false) and rewritten at the end ('complete': true). A C_b summary
carries the Step-8 label (STEP8_LABEL), also printed before its verdict line.

Run from cfm_grn_v2/ (--out defaults to outputs/eb_published, or outputs/eb_published_cb with --choice; a folder that
already holds another configuration's results is refused unless --force; --threads defaults to 1, faster than 8 for
these 5-D inputs and small networks):
  PYTHONPATH=src python scripts_dev/eb_published_protocol.py --npz ~/rafm_data/eb_published/eb_velocity_v5.npz \
      [--choice outputs/freshfold/choice.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.evaluation.metrics import EvalProtocol, median_bandwidth, noise_floor
from cfm_grn_v2.flows.solver import integrate
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.designs import CONTROLS, grid_gap, interpolation_controls
from cfm_grn_v2.training.search import interpolation_design
from cfm_grn_v2.training.trainer import task_groups, time_task, train_flow
from eb_benchmark import BASE, FT, WINDOWS

PREREG = "docs/superpowers/prereg/2026-09-25-prereg-v02.md"
NPZ = "~/rafm_data/eb_published/eb_velocity_v5.npz"
NPZ_SHA256 = "17623d16dd4fe7b679aad80916d130cb2863a4cd4ee742f139a3c2eed91ba230"
N_PCS = 5
TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]
FOLDS = (1.0, 2.0, 3.0)
SCALERS = ("primary", "secondary")
COND, REP = "EB", "eb1"
FLOW, SHUFFLED, OTCFM = "flow_matching", "flow_matching_shuffled_time", "otcfm_equivalent"
RULE_CONTROLS = tuple(CONTROLS) + ("next", OTCFM)
TABLE3 = {"persistence": {1.0: 1.715, 2.0: 1.400, 3.0: 0.814}, "next": {1.0: 1.400, 2.0: 0.814, 3.0: 1.694}}
CAL_TOL, OTCFM_PUBLISHED, OTCFM_TOL, PUBLISHED_UPPER = 0.05, 0.790, 0.068, 0.848
NUM_ITER_MAX, NUM_ITER_RETRY, SUB_N, FALLBACK_REPEATS = 10_000_000, 100_000_000, 2000, 10
SEEDS_PREREG, STEPS_EXTRA = [0, 1, 2, 3, 4], 8
OUT_CA, OUT_CB = "outputs/eb_published", "outputs/eb_published_cb"
STEP8_LABEL = "rule adopted on fresh folds, then measured once on stages inspected earlier by the diagnosis"
MLP = {"hidden": 128, "n_iters": 2000}
SMOKE = dict(n_iters=40, eval_every=20, patience=1, marginal_iters=10, marginal_eval_every=5, hidden=16,
             marginal_steps=2, n_ode_steps=4)
SMOKE_OTCFM, SMOKE_MLP, SMOKE_CAP = {"n_iters": 30, "hidden": 16}, {"hidden": 16, "n_iters": 30}, 300
TORCHCFM = {
    "read": True, "date": "2026-09-26", "how": "WebFetch of the GitHub pages",
    "rechecked": "2026-09-26, raw files: model/otcfm.yaml, experiment/cfm.yaml, datamodule/time_dist.yaml, "
                 "src/models/cfm_module.py (leave-out, linspace(0, 1, 101), sigma_min), "
                 "src/datamodules/distribution_datamodule.py (whiten on all cells, random_split, min_size, drop_last)",
    "source": "github.com/atong01/conditional-flow-matching (main): runner/configs/model/otcfm.yaml, "
              "runner/configs/experiment/cfm.yaml, runner/configs/datamodule/time_dist.yaml, runner/configs/train.yaml, "
              "runner/configs/callbacks/default.yaml, runner/configs/trainer/default.yaml, runner/src/train.py, "
              "runner/src/models/cfm_module.py, runner/src/datamodules/distribution_datamodule.py",
    "composition": "experiment=cfm model=otcfm datamodule=time_dist",
    "hidden_dims": [64, 64, 64], "activation": "selu", "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-5,
    "sigma_min": 0.1, "ot_sampler": "exact", "ode_solver": "euler", "steps_per_unit": 100, "batch_size": 512,
    "batch_size_time_dist": 128, "max_epochs": 1000, "check_val_every_n_epoch": 10,
    "train_val_test_split": [0.8, 0.1, 0.1], "loader_mode": "min_size", "drop_last": True, "max_dim": 5,
    "whiten": "StandardScaler on every cell, before the split", "test_cells": "every cell of every window (load_full)",
    "early_stopping": {"monitor": "val/loss", "patience": 100}, "checkpoint": "best val/loss tested (train.py)",
    "ema": None, "grad_clip": None, "seed": 42,
    "ambiguity": "time_dist.yaml sets batch_size 128; experiment/cfm.yaml sets datamodule.batch_size 512 and is merged "
                 "after the datamodule group, so 512 holds for the composition above (with 128: 14 000 iterations). "
                 "With 512 and drop_last the validation splits hold no batch: by our reading no val/loss is logged, "
                 "early stopping and checkpoint selection cannot act and the final weights are tested",
}


# ---------------------------------------------------------------- data, designs, configurations

def load_eb(path, n_pcs: int = N_PCS):
    """('pcs'[:, :n_pcs] float64, 'sample_labels' as float times); every other key is ignored."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as z:
        pcs = np.asarray(z["pcs"], dtype=np.float64)
        labels = np.asarray(z["sample_labels"]).astype(np.float64).ravel()
    if pcs.ndim != 2 or pcs.shape[1] < n_pcs or len(pcs) != len(labels):
        raise ValueError(f"pcs {pcs.shape} and sample_labels {labels.shape} do not match (need >= {n_pcs} columns)")
    if sorted(set(labels.tolist())) != TIMES:
        raise ValueError(f"sample_labels must be {TIMES}, got {sorted(set(labels.tolist()))}")
    if not np.isfinite(pcs[:, :n_pcs]).all():
        raise ValueError("pcs contains NaN or inf")
    return pcs[:, :n_pcs].copy(), labels


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(Path(path).expanduser(), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_card(path, labels, digest: str) -> dict:
    """What the file holds and what the run uses: every key's shape, the windows' cell counts and days."""
    keys = {}
    with np.load(Path(path).expanduser(), allow_pickle=False) as z:
        for k in z.files:
            try:
                keys[k] = list(z[k].shape)
            except ValueError:
                keys[k] = "object array (not loaded)"
    return {"source": "TrajectoryNet (Tong et al. 2020) eb_velocity_v5.npz, the Moon et al. 2019 embryoid body",
            "file": str(path), "sha256": digest, "sha256_matches_prereg": digest == NPZ_SHA256, "keys": keys,
            "keys_used": [f"pcs[:, :{N_PCS}]", "sample_labels"], "n_cells": int(len(labels)),
            "window_counts": {f"{t:g}": int((labels == t).sum()) for t in TIMES},
            "windows": {f"{t:g}": WINDOWS[t] for t in TIMES}, "time": "sample_labels at unit spacing (1 = six days)",
            "conditions": [COND], "replicates": [REP], "cells_dropped": 0}


def cap_windows(pcs, labels, cap: int, seed: int = 0):
    """Smoke only: at most cap cells per window (seeded), in the original order."""
    rng = np.random.default_rng(seed)
    keep = np.sort(np.concatenate([rng.choice(idx, min(cap, len(idx)), replace=False)
                                   for idx in (np.flatnonzero(labels == t) for t in TIMES)]))
    return pcs[keep], labels[keep]


def fold_design(fold: float) -> dict:
    return interpolation_design(TIMES, fold)


def scaler_space(pcs, labels, scaler: str, fold: float):
    """(every cell in the scaler's space, float32; the fitted StandardScaler). primary: fitted on all cells; secondary:
    on the four observed windows of the fold."""
    if scaler not in SCALERS:
        raise ValueError(f"unknown scaler {scaler!r}")
    rows = np.ones(len(pcs), dtype=bool) if scaler == "primary" else labels != float(fold)
    sc = StandardScaler().fit(pcs[rows])
    return sc.transform(pcs).astype(np.float32), sc


def make_snapshots(x, labels) -> SnapshotSet:
    n = len(x)
    return SnapshotSet(x, labels, np.full(n, COND), np.full(n, REP), [f"PC{i + 1}" for i in range(x.shape[1])],
                       name="eb_published")


def frozen_config() -> TrainConfig:
    """C_a: frozen v0.1.10 = eb_benchmark.BASE + FT + choice.json {path_center: linear}."""
    return BASE.replace(**FT).replace(path_center="linear")


def split_lengths(n: int, fracs=tuple(TORCHCFM["train_val_test_split"])) -> list:
    """torch.utils.data.random_split lengths for fractions: floors, then the remainder one by one from the first."""
    out = [int(math.floor(n * f)) for f in fracs]
    for i in range(n - sum(out)):
        out[i % len(out)] += 1
    return out


def published_iters(counts, batch: int = TORCHCFM["batch_size"], epochs: int = TORCHCFM["max_epochs"]) -> dict:
    """TorchCFM's optimiser steps: max_epochs x batches per epoch, an epoch being the shortest time point's training
    split in whole batches (CombinedLoader 'min_size', drop_last)."""
    per_epoch = min(split_lengths(int(n))[0] // int(batch) for n in counts)
    return {"batches_per_epoch": int(per_epoch), "n_iters": int(epochs * per_epoch), "batch_size": int(batch),
            "max_epochs": int(epochs)}


def otcfm_config(n_iters: int, record: dict | None = None) -> TrainConfig:
    """The TorchCFM single-cell configuration mapped onto TrainConfig (module docstring); everything else default. An
    unread record keeps TrainConfig's own hyper-parameters (and n_iters) with the same structure."""
    t = TORCHCFM if record is None else record
    structure = dict(coupling="ot", clock="flow", time_conditioning="absolute", path="linear", pair_mode="consecutive",
                     marginal_iters=0)
    if not t.get("read"):
        return TrainConfig(**structure)
    return TrainConfig(hidden=t["hidden_dims"][0], depth=len(t["hidden_dims"]), lr=t["lr"], weight_decay=t["weight_decay"],
                       batch_size=t["batch_size"], n_iters=int(n_iters), sigma_min=t["sigma_min"],
                       n_ode_steps=t["steps_per_unit"], ode_method=t["ode_solver"], ema_decay=0.0, grad_clip=0.0,
                       **structure)


def otcfm_predict(model, x0, cond: str, t_prev: float, t_next: float, target: float, n_steps=None) -> np.ndarray:
    """The held-out window as TorchCFM reaches it: the bracket transition t_prev -> t_next integrated from the observed
    t_prev cells over target - t_prev biological units, i.e. to flow time (target - t_prev)/(t_next - t_prev)."""
    frac = (float(target) - float(t_prev)) / (float(t_next) - float(t_prev))
    z = torch.as_tensor(model.standardizer.transform(x0))
    with torch.no_grad():
        ctx = model._context(cond, t_prev, t_next, z.shape[0])
        out = integrate(model.field(t_prev), z, ctx, int(n_steps or model.config.n_ode_steps), model.config.ode_method,
                        t_end=frac)
    return model.standardizer.inverse_transform(out.numpy())


def load_choice(path, smoke: bool):
    """(winner, its training recipe, the choice dict) from freshfold_select.py --decide. As in Job E, the winner is
    re-derived by code before anything runs (freshfold_select.verify_choice: R-SEL applied again to the Job-D outputs
    the file names): a choice.json the code did not write is refused (no human edit between selection and C_b)."""
    import freshfold_select as fs
    choice = json.loads(Path(path).read_text(encoding="utf-8"))
    if choice.get("smoke") and not smoke:
        raise SystemExit(f"{path} comes from a smoke run")
    winner = fs.verify_choice(choice)
    if winner not in fs.FLOW_ARMS and winner not in fs.M_RECIPES:
        raise SystemExit(f"unknown winner {winner!r} in {path}")
    if winner == "base":
        raise SystemExit("the Step-6 winner is base: C_b equals C_a and nothing new is measured (Step 8)")
    return winner, dict(fs.M_RECIPES.get(winner, {})), choice


def arm_weights(winner, h_rel: float, nested, model):
    """(stage for FlowModel.predict, which weights). C_a: the final weights, nothing else."""
    if winner is None:
        return None, "final"
    import freshfold_select as fs
    return fs.arm_stage(winner, h_rel, nested.marginal_iter, model.marginal_iter)


# ---------------------------------------------------------------- training and predictions

def fit_pair(s, d, cfg, proto, keep_snapshots: bool = False):
    """Nested fit early-stopped on the design's validation task, then the refit on every observed window for the
    transferred first-stage count (run_interpolation_benchmark with refit_on_all=True)."""
    nested, _ = train_flow(s, s.mask(times=d["fit_times"]), d["fit_times"], cfg,
                           time_task(s, d["v_prev"], d["val_time"]), proto)
    n_iters = max(int(nested.best_iter), cfg.eval_every)
    model, _ = train_flow(s, s.mask(times=d["obs_times"]), d["obs_times"], cfg.replace(n_iters=n_iters), None, proto,
                          keep_snapshots=keep_snapshots)
    return nested, model


def fold_predictions(pcs, labels, fold, scaler, seeds, cfg, mlp_kwargs, winner=None, recipe=None, otcfm_over=None,
                     verbose=True) -> SimpleNamespace:
    """Every model's prediction of window `fold` from window fold-1 in the scaler's space; nothing but the scoring target
    (and, for the primary scaler, the published whitening) sees the held-out window."""
    d = fold_design(fold)
    x, sc = scaler_space(pcs, labels, scaler, fold)
    s = make_snapshots(x, labels)
    all_mask = s.mask(times=d["obs_times"])
    task = time_task(s, d["t_prev"], d["test_time"])
    if (all_mask & task.target_mask).any():
        raise ValueError("test target cells overlap the observed cells")
    proto = EvalProtocol(n_samples=1000, seed=0, bandwidth=median_bandwidth(s.x[all_mask]))
    groups = task_groups(s, task)
    h_rel = d["horizon"] / grid_gap(TIMES)
    iters = published_iters([int((labels == t).sum()) for t in d["obs_times"]])
    oc = otcfm_config(iters["n_iters"]).replace(**(otcfm_over or {}))
    if oc.n_iters < 1:
        raise ValueError(f"the published epoch holds no whole batch of {oc.batch_size} cells ({iters})")
    preds, t0 = [], time.time()

    def add(seed, model, family, cond, rep, pred, **extra):
        preds.append({"seed": seed, "model": model, "family": family, "group": f"{cond}|{rep}", "pred": pred, **extra})

    for seed in seeds:
        c = cfg.replace(seed=seed, **(recipe or {}))
        for name, family, cv in ((FLOW, "flow", c), (SHUFFLED, "twin", c.replace(shuffle_biological_time=True))):
            t1 = time.time()
            nested, model = fit_pair(s, d, cv, proto, keep_snapshots=winner == "transfer")
            stage, weights = arm_weights(winner, h_rel, nested, model)
            info = {"weights": weights, "train_seconds": round(time.time() - t1, 1), "n_params": model.n_params,
                    "nested_best_iter": nested.best_iter, "nested_marginal_iter": nested.marginal_iter,
                    "nested_stage1_val": nested.stage1_best_val, "nested_final_val": nested.best_val,
                    "refit_iters": model.best_iter, "refit_marginal_iter": model.marginal_iter}
            for cond, rep, m0, _ in groups:          # pred8: Step 1(c)'s 8 RK4 steps per unit, non-voting
                add(seed, name, family, cond, rep,
                    model.predict(s.x[m0], cond, d["t_prev"], d["test_time"], stage=stage),
                    pred8=model.predict(s.x[m0], cond, d["t_prev"], d["test_time"], n_steps=STEPS_EXTRA, stage=stage),
                    **info)
        t1 = time.time()
        om, _ = train_flow(s, all_mask, d["obs_times"], oc.replace(seed=seed), None, proto)
        for cond, rep, m0, _ in groups:
            add(seed, OTCFM, "otcfm", cond, rep, otcfm_predict(om, s.x[m0], cond, d["t_prev"], d["t_next"], d["test_time"]),
                weights="final", train_seconds=round(time.time() - t1, 1), n_params=om.n_params, refit_iters=oc.n_iters)
        for name, cond, rep, pred in interpolation_controls(s, s.x, all_mask, d["obs_times"], d["t_prev"],
                                                            d["test_time"], d["t_next"], groups, seed,
                                                            mlp_kwargs=mlp_kwargs):
            add(seed, name, "control", cond, rep, pred)
        for cond, rep, _, _ in groups:
            add(seed, "next", "control", cond, rep,
                s.x[all_mask & s.mask(times=[d["t_next"]], conditions=[cond], replicates=[rep])].copy())
        if verbose:
            print(f"  [{scaler} t={fold:g}] seed {seed} trained ({time.time() - t0:.0f}s)", flush=True)
    return SimpleNamespace(s=s, design=d, scaler=sc, proto=proto, preds=preds, otcfm=iters, otcfm_config=oc,
                           targets={f"{c}|{r}": s.x[m1] for c, r, _, m1 in groups})


# ---------------------------------------------------------------- metrics

def have_pot() -> bool:
    """True iff POT imports; any import failure (a broken compiled part raises more than ImportError) counts as absent."""
    try:
        import ot  # noqa: F401
    except Exception:
        return False
    return True


def pot_preflight(seed: int = 0) -> dict:
    """Before any training: ot.emd2 must run and give W1(X, X + a) = |a| (translation: exact, the identity plan).
    Stops the run (SystemExit) otherwise; returns the check and POT's version."""
    rng = np.random.default_rng(seed)
    x, a = rng.normal(size=(64, N_PCS)), rng.normal(size=N_PCS)
    try:
        import ot
        r = w1_exact(x, x + a)
    except Exception as e:
        raise SystemExit(f"POT preflight failed: ot.emd2 raised {type(e).__name__}: {e}")
    want = float(np.linalg.norm(a))
    ok = r["w1_backend"] == "pot" and r["w1_converged"] and abs(r["w1"] - want) < 1e-9
    out = {"pot_version": str(getattr(ot, "__version__", "unknown")), "w1": r["w1"], "expected": want,
           "converged": bool(r["w1_converged"]), "backend": r["w1_backend"], "passes": bool(ok)}
    if not ok:
        raise SystemExit(f"POT preflight failed: W1(X, X + a) = {r['w1']!r} ({r['w1_backend']}, converged "
                         f"{r['w1_converged']}), expected |a| = {want!r}")
    return out


def _assignment_w1(a, b) -> float:
    M = cdist(a, b)
    r, c = linear_sum_assignment(M)
    return float(M[r, c].mean())


def _emd2(a, b, M, num_iter_max):
    import ot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cost, log = ot.emd2(np.full(len(a), 1.0 / len(a)), np.full(len(b), 1.0 / len(b)), M,
                            numItermax=int(num_iter_max), log=True)
    return float(cost), bool(log.get("warning") is None and int(log.get("result_code", 1)) == 1)


def w1_exact(a, b, num_iter_max: int = NUM_ITER_MAX, sub_n: int = SUB_N, repeats: int = FALLBACK_REPEATS,
             seed: int = 0, retry_iter_max: int = NUM_ITER_RETRY) -> dict:
    """Exact W1 between the uniform empirical measures of a and b on the Euclidean cost: POT's network simplex (ot.emd2,
    as TorchCFM's distribution_distances) when installed, retried once at retry_iter_max when it stops before optimality
    (w1_converged False if it still does); otherwise the pre-registered fallback (Step 1e), the exact assignment on
    equal-size subsamples of at most sub_n cells per side, mean of `repeats` seeded draws (a single assignment on
    everything when both sets have the same size <= sub_n: every draw is then the whole set)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if have_pot():
        M = cdist(a, b)
        cost, ok = _emd2(a, b, M, num_iter_max)
        retried = not ok and int(retry_iter_max) > int(num_iter_max)
        if retried:
            cost, ok = _emd2(a, b, M, retry_iter_max)
        return {"w1": cost, "w1_converged": ok, "w1_backend": "pot", "w1_retried": bool(retried)}
    if len(a) == len(b) <= sub_n:
        return {"w1": _assignment_w1(a, b), "w1_converged": True, "w1_backend": "assignment", "w1_retried": False}
    if len(a) > len(b):                  # the same draws either way round: W1(a, b) = W1(b, a)
        a, b = b, a
    n, rng = min(len(a), len(b), int(sub_n)), np.random.default_rng(seed)
    vals = [_assignment_w1(a[rng.choice(len(a), n, replace=False)], b[rng.choice(len(b), n, replace=False)])
            for _ in range(int(repeats))]
    return {"w1": float(np.mean(vals)), "w1_converged": True, "w1_backend": f"assignment_sub{n}x{int(repeats)}",
            "w1_retried": False}


def w1_subsample(pred, target, seed: int, n: int = SUB_N) -> dict:
    """W1 on seeded subsamples of at most n cells per side; the target subsample depends on the seed only."""
    ti = np.random.default_rng(40_000 + seed).permutation(len(target))[:n]
    pi = np.random.default_rng(50_000 + seed).permutation(len(pred))[:n]
    r = w1_exact(np.asarray(pred)[pi], np.asarray(target)[ti])
    return {"w1_sub2000": r["w1"], "w1_sub2000_converged": r["w1_converged"]}


def target_floor(target, proto) -> dict:
    perm = np.random.default_rng(0).permutation(len(target))
    h = len(target) // 2
    w = w1_exact(target[perm[:h]], target[perm[h:2 * h]])
    return {"w1_split_half": w["w1"], "w1_split_half_converged": w["w1_converged"],
            "ed_noise_floor": noise_floor(target, proto)["energy_distance"]}


def score_fold(F, fold: float, scaler: str, config_label: str) -> list:
    """One row per (seed, model, group); the exact W1 of a prediction seen before (deterministic controls) is reused.
    Flow and twin rows add the 8-steps-per-unit prediction's W1 and energy distance (non-voting)."""
    rows, cache = [], {}

    def exact(group, pred):
        key = (group, pred.shape, hashlib.sha1(pred.tobytes()).hexdigest())
        if key not in cache:
            cache[key] = w1_exact(pred, F.targets[group])
        return cache[key]

    for p in F.preds:
        pred, target = np.ascontiguousarray(p["pred"]), F.targets[p["group"]]
        extra = {k: v for k, v in p.items() if k not in ("pred", "pred8", "seed", "model", "family", "group")}
        if p.get("pred8") is not None:
            p8 = np.ascontiguousarray(p["pred8"])
            r8 = exact(p["group"], p8)
            extra.update(w1_8steps=r8["w1"], w1_8steps_converged=r8["w1_converged"],
                         energy_distance_8steps=F.proto.evaluate(p8, target, seed_offset=p["seed"])["energy_distance"])
        rows.append({"fold": float(fold), "seed": p["seed"], "scaler": scaler, "config": config_label,
                     "model": p["model"], "family": p["family"], "group": p["group"], **exact(p["group"], pred),
                     **w1_subsample(pred, target, p["seed"]), "n_pred": int(len(pred)), "n_target": int(len(target)),
                     **F.proto.evaluate(pred, target, seed_offset=p["seed"]), **extra})
    return rows


# ---------------------------------------------------------------- rules (pure functions of the results table)

def paired(a, b) -> dict:
    """Per-seed difference a - b: mean and 95% t-interval."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    common = a.index.intersection(b.index)
    dd = (a[common] - b[common]).to_numpy(dtype=float)
    n = len(dd)
    out = {"mean": float(dd.mean()) if n else float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    if n >= 2:
        half = float(stats.t.ppf(0.975, n - 1) * dd.std(ddof=1) / np.sqrt(n))
        out.update(ci_low=out["mean"] - half, ci_high=out["mean"] + half)
    return out


def _fin(x) -> bool:
    return x is not None and np.isfinite(float(x))


def best_control(means, controls):
    """The control with the lowest finite seed mean (chosen after the fact, which favours the controls)."""
    return min(controls, key=lambda c: float(means[c]) if _fin(means[c]) else np.inf)


def calibration_gate(w1: dict) -> dict:
    """w1[(model, fold)]: W1 of persistence ('prev') and 'next' in the primary scaler (deterministic)."""
    checks = {}
    for m, ref in TABLE3.items():
        for f, r in ref.items():
            v = w1.get((m, f))
            ok = _fin(v) and abs(float(v) - r) <= CAL_TOL + 1e-12
            checks[f"{m}|t={f:g}"] = {"w1": None if v is None else float(v), "published": r,
                                      "diff": float(v) - r if _fin(v) else None, "within_0.05": bool(ok)}
    return {"rule": "W1 of prev (persistence) and next within +-0.05 of TrajectoryNet Table 3 at every fold",
            "checks": checks, "passes": all(c["within_0.05"] for c in checks.values())}


def model_check(per_seed) -> dict:
    v = pd.Series(per_seed, dtype=float)
    m = float(v.mean()) if v.notna().all() and len(v) else float("nan")
    return {"rule": "in-package OT-CFM 3-fold mean W1 within 0.790 +- 0.068", "mean": m,
            "per_seed": [float(x) for x in v], "published": OTCFM_PUBLISHED, "tolerance": OTCFM_TOL,
            "passes": bool(_fin(m) and abs(m - OTCFM_PUBLISHED) <= OTCFM_TOL + 1e-12)}


def p2a(ps: pd.DataFrame, flow: str = FLOW, controls=RULE_CONTROLS, shuffled: str = SHUFFLED) -> dict:
    """ps: seed x model table of the per-seed 3-fold mean. Deciding: the literal per-seed reading (below every control
    in every seed); the seed-mean reading is reported as passes_seed_mean and decides nothing."""
    means = ps.mean()
    present = [c for c in controls if c in ps.columns]
    missing = [c for c in controls if c not in ps.columns]
    best = best_control(means, present)
    below = bool(_fin(means[flow]) and all(_fin(means[c]) and means[flow] < means[c] for c in present))
    every_seed = bool(len(ps) and all((ps[flow] < ps[c]).all() for c in present))       # NaN compares False
    vb = paired(ps[flow], ps[best])
    vs = paired(ps[flow], ps[shuffled]) if shuffled in ps.columns else paired(pd.Series(dtype=float), pd.Series(dtype=float))
    cis = (not missing and _fin(vb["ci_high"]) and vb["ci_high"] < 0 and _fin(vs["ci_high"]) and vs["ci_high"] < 0)
    return {"rule": "the flow's per-seed 3-fold mean below every control in every seed; upper paired CI (flow - best "
                    "control) < 0; upper paired CI (flow - shuffled) < 0",
            "seed_means": {k: float(v) for k, v in means.items()}, "best_control": best, "missing_controls": missing,
            "below_every_control_in_every_seed": every_seed, "below_every_control": below,
            "vs_best_control": vb, "vs_shuffled": vs, "passes": bool(every_seed and cis),
            "passes_seed_mean": bool(below and cis),
            "passes_seed_mean_note": "the looser reading (seed means below every control's), descriptive only"}


def p2b(flow_per_seed, gate_passes: bool, check_passes: bool, n_required: int | None = None) -> dict:
    """n_required: the pre-registered seed count (5; None in smoke runs); with fewer finite seeds no statement."""
    v = pd.Series(flow_per_seed, dtype=float).dropna().to_numpy()
    n = len(v)
    m = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    upper = m + 1.96 * sd / np.sqrt(n) if n > 1 else float("nan")
    out = {"rule": "only if gate and check pass: 'below published OT-CFM' iff mean + 1.96 sd/sqrt(5) < 0.790; "
                   "'within the published range' iff mean < 0.848; otherwise 'above'; never a significance claim",
           "mean": m, "sd": sd, "n": n, "n_required": n_required, "upper": float(upper),
           "applies": bool(gate_passes and check_passes), "statement": None}
    if not out["applies"]:
        out["reason"] = "descriptive only if the calibration gate and the model check both pass"
    elif n_required is not None and n != int(n_required):
        out["reason"] = f"needs the {n_required} pre-registered seeds with a finite 3-fold mean, got {n}"
    elif _fin(upper) and upper < OTCFM_PUBLISHED:
        out["statement"] = "below published OT-CFM"
    elif _fin(m) and m < PUBLISHED_UPPER:
        out["statement"] = "within the published range"
    else:
        out["statement"] = "above"
    return out


def seed_fold_means(df: pd.DataFrame, scaler: str, metric: str = "w1") -> pd.DataFrame:
    """seed x model: the mean over the three folds (NaN unless every fold is present)."""
    sub = df[df["scaler"] == scaler]
    g = sub.groupby(["seed", "model", "fold"])[metric].mean().unstack("fold")
    full = g.reindex(columns=list(FOLDS))
    return full.mean(axis=1).where(full.notna().all(axis=1)).unstack("model")


def per_fold(df: pd.DataFrame, scaler: str, metric: str) -> dict:
    out = {}
    for fold, g in df[df["scaler"] == scaler].groupby("fold"):
        wide = g.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        means, sds = wide.mean(), wide.std(ddof=1)
        best = best_control(means, [c for c in RULE_CONTROLS if c in wide.columns])
        vs = {f"flow_minus_{k}": paired(wide[FLOW], wide[c]) for k, c in
              (("best_control", best), ("shuffled", SHUFFLED), ("mean_interp", "mean_interp"), ("otcfm", OTCFM))
              if FLOW in wide.columns and c in wide.columns}
        out[f"t={fold:g}"] = {"seed_mean": {k: float(v) for k, v in means.items()},
                              "seed_sd": {k: float(v) for k, v in sds.fillna(0.0).items()}, "best_control": best, **vs}
    return out


def unconverged(df: pd.DataFrame, scaler: str, models) -> list:
    """The (model, fold, seed) rows of `models` whose exact W1 did not converge (a table without the column: none)."""
    if "w1_converged" not in df.columns:
        return []
    sub = df[(df["scaler"] == scaler) & df["model"].isin(list(models))]
    bad = sub[~sub["w1_converged"].astype(bool)]
    return [{"model": m, "fold": float(f), "seed": int(s)}
            for m, f, s in sorted(set(zip(bad["model"], bad["fold"], bad["seed"])))]


def settle(rule: dict, bad: list) -> dict:
    """A rule fed by a non-converged W1 is neither passed nor failed: passes None, verdict 'not decidable'."""
    if bad:
        rule.update(passes=None, decidable=False, verdict="not decidable", unconverged_rows=bad)
    else:
        rule.update(decidable=True, verdict="pass" if rule["passes"] else "fail")
    return rule


def decide(df: pd.DataFrame, smoke: bool = False) -> dict:
    """The pre-registered rules on the primary scaler; the same comparisons, descriptive, on the secondary scaler."""
    seeds = sorted(int(s) for s in df["seed"].unique())
    out = {"prereg": PREREG, "step": "4 (Job C)", "smoke": bool(smoke), "deciding_scaler": "primary",
           "controls": list(RULE_CONTROLS), "separate_comparator": SHUFFLED, "seeds": seeds,
           "seeds_match_prereg": seeds == SEEDS_PREREG, "binding": bool(not smoke and seeds == SEEDS_PREREG)}
    for scaler in [s for s in SCALERS if s in set(df["scaler"])]:
        ps = seed_fold_means(df, scaler, "w1")
        cal = {(m, f): float(df[(df["scaler"] == scaler) & (df["model"] == m) & (df["fold"] == f)]["w1"].mean())
               for m in TABLE3 for f in FOLDS}
        res = {"p2a": settle(p2a(ps), unconverged(df, scaler, (FLOW, SHUFFLED) + RULE_CONTROLS)),
               "p2a_energy_distance": p2a(seed_fold_means(df, scaler, "energy_distance")),
               "flow_minus_mean_interp": paired(ps[FLOW], ps["mean_interp"]),
               "seed_means_w1": {k: float(v) for k, v in ps.mean().items()},
               "w1_unconverged_rows": unconverged(df, scaler, sorted(set(df["model"])))}
        res["loses_to_mean_interp"] = bool(res["seed_means_w1"][FLOW] >= res["seed_means_w1"]["mean_interp"])
        if scaler == "primary":
            gate = settle(calibration_gate(cal), unconverged(df, scaler, tuple(TABLE3)))
            check = settle(model_check(ps[OTCFM]), unconverged(df, scaler, (OTCFM,)))
            b = p2b(ps[FLOW], gate["passes"], check["passes"], None if smoke else len(SEEDS_PREREG))
            bad = unconverged(df, scaler, (FLOW,))
            if bad and b["applies"]:
                b.update(statement=None, reason="not decidable: a flow W1 did not converge", unconverged_rows=bad)
            res.update({"role": "deciding", "calibration_gate": gate, "model_check": check, "p2b": b})
            res["reading"] = ("published comparison allowed (calibration gate and model check pass): P2b applies"
                              if gate["passes"] and check["passes"] else
                              "only within-run comparisons are claimed (a check failed or is not decidable)")
            if res["loses_to_mean_interp"]:
                res["mean_interp_note"] = ("the model loses to mean interpolation: a finding about the benchmark (the "
                                           "published tables omit mean interpolation; 'next' alone scores 0.814 at t=2)")
        else:
            res.update({"role": "descriptive (the published numbers refer to the primary scaler)",
                        "calibration_w1": {f"{m}|t={f:g}": v for (m, f), v in cal.items()},
                        "otcfm_3fold_mean_w1": float(ps[OTCFM].mean())})
        out[scaler] = res
    out["per_fold"] = {s: {m: per_fold(df, s, m) for m in ("w1", "energy_distance", "w1_sub2000") if m in df.columns}
                       for s in SCALERS if s in set(df["scaler"])}
    return out


# ---------------------------------------------------------------- run

def jsonable(o):
    """Plain JSON types; non-finite floats become null (strict JSON)."""
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
    Path(path).write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False), encoding="utf-8")


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    """Seed mean and sd of every metric per (scaler, fold, model): the per-fold tables (t = 2 included)."""
    cols = [c for c in ("w1", "w1_sub2000", "energy_distance", "mean_error", "var_ratio") if c in df.columns]
    per_seed = df.groupby(["scaler", "fold", "model", "seed"])[cols].mean()
    g = per_seed.groupby(["scaler", "fold", "model"])
    out = g.mean().join(g.std(ddof=1), rsuffix="_sd").join(g.size().rename("n_seeds"))
    return out.reset_index().sort_values(["scaler", "fold", "w1"])


DEVIATIONS = [      # each needs a dated decisions.md row before Job C is submitted (rule discipline, prereg line 46)
    "the OT-CFM equivalent reproduces the runner's hyper-parameters, not its implementation (module docstring: "
    "'Not reproduced')",
    "the OT-CFM iteration count uses the four observed windows' training splits (the runner's loader also holds the "
    "held-out window; the counts agree unless that window is the smallest, recorded per fold)",
    "the OT-CFM batch size is 512 (experiment/cfm.yaml merged after datamodule/time_dist.yaml's 128), giving 3000 "
    "iterations on the full data; with 128 it would be 14 000 (TORCHCFM['ambiguity'])",
]
READINGS = [        # not deviations: how the frozen text is applied
    "P2a decides on the literal per-seed reading (below every control in every seed); the seed-mean reading is "
    "reported as passes_seed_mean and decides nothing",
    "a stopped ot.emd2 is retried once at numItermax 1e8; a rule fed by a W1 that still did not converge is 'not "
    "decidable' (the exact W1 is the pre-registered metric)",
]


def config_labels(winner) -> tuple:
    """(config_label, Step-8 label or None): C_a for the frozen model, C_b:<winner> with Step 8's label."""
    return ("C_a", None) if winner is None else (f"C_b:{winner}", STEP8_LABEL)


def resolve_out(out, choice) -> Path:
    """--out, or the configuration's own default folder (C_b never lands on C_a's folder by default)."""
    return Path(out if out is not None else (OUT_CB if choice else OUT_CA))


def guard_out(out: Path, label: str, force: bool = False) -> None:
    """Refuse a folder that already holds another configuration's results (C_a's record must survive a C_b run)."""
    if force:
        return
    found = set()
    try:
        found.add(json.loads((out / "summary.json").read_text(encoding="utf-8")).get("config_label"))
    except FileNotFoundError:
        pass
    except (ValueError, OSError) as e:
        raise SystemExit(f"{out / 'summary.json'} exists but cannot be read ({e}); use another --out or --force")
    if (out / "results.csv").exists():
        try:
            found |= set(pd.read_csv(out / "results.csv", usecols=["config"])["config"].astype(str))
        except (ValueError, OSError) as e:          # no 'config' column, empty or unreadable: not ours to overwrite
            raise SystemExit(f"{out / 'results.csv'} exists but cannot be read ({e}); use another --out or --force")
    other = sorted(str(x) for x in found - {label})
    if other:
        raise SystemExit(f"{out} already holds results of {other}, not {label}: use another --out (or --force)")


def run_folds(pcs, labels, seeds, cfg, mlp, winner, recipe, label, otcfm_over, out, t0, on_scaler_done=None):
    """Every (scaler, fold): predictions, scores (results.csv rewritten after each) and the fold's record;
    on_scaler_done(rows, folds) after each scaler but the last (the deciding primary scaler first; the caller writes
    the complete summary after the last)."""
    rows, folds = [], {}
    all_counts = [int((labels == t).sum()) for t in TIMES]
    for scaler in SCALERS:
        for fold in FOLDS:
            d = fold_design(fold)
            print(f"\n=== {scaler} scaler, t={fold:g}: from {d['t_prev']:g}, bracket {d['t_prev']:g}->{d['t_next']:g}; "
                  f"validation {d['v_prev']:g}->{d['val_time']:g}; fit {d['fit_times']} ({time.time() - t0:.0f}s) ===",
                  flush=True)
            F = fold_predictions(pcs, labels, fold, scaler, seeds, cfg, mlp, winner, recipe, otcfm_over)
            rows += score_fold(F, fold, scaler, label)
            folds[f"{scaler}|t={fold:g}"] = {
                "design": d, "scaler_mean": F.scaler.mean_.tolist(), "scaler_scale": F.scaler.scale_.tolist(),
                "scaler_cells": int(len(labels)) if scaler == "primary" else int((labels != fold).sum()),
                "otcfm_iters": F.otcfm, "otcfm_n_iters_used": int(F.otcfm_config.n_iters),
                "otcfm_config": F.otcfm_config.to_dict(),
                "otcfm_count_rule_matches_published": bool(published_iters(all_counts)["n_iters"] == F.otcfm["n_iters"]),
                "target_floor": target_floor(F.targets[f"{COND}|{REP}"], F.proto)}
            df = pd.DataFrame(rows)
            df.to_csv(out / "results.csv", index=False)
            tab = df[(df["scaler"] == scaler) & (df["fold"] == fold)].groupby("model")[["w1", "energy_distance"]]
            print(tab.mean().sort_values("w1").round(4).to_string(), flush=True)
        if on_scaler_done is not None and scaler != SCALERS[-1]:
            on_scaler_done(rows, folds)
    return rows, folds


def verdict(rule: dict) -> str:
    return {"pass": "PASS", "fail": "FAIL"}.get(rule.get("verdict"), "NOT DECIDABLE")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Pre-registration v0.2 Step 4 (Job C): published EB benchmark, 5 PCs, W1")
    p.add_argument("--npz", default=NPZ, help="TrajectoryNet's eb_velocity_v5.npz")
    p.add_argument("--seeds", type=int, default=len(SEEDS_PREREG), help="must be 5 (seeds 0-4) in a full run")
    p.add_argument("--out", default=None, help=f"default {OUT_CA}, or {OUT_CB} with --choice")
    p.add_argument("--force", action="store_true", help="write into a folder holding another configuration's results")
    p.add_argument("--threads", type=int, default=1, help="torch threads (1 is fastest for 5-D inputs, small nets)")
    p.add_argument("--choice", default=None, help="choice.json of freshfold_select.py --decide: measure its winner (C_b)")
    p.add_argument("--smoke", action="store_true",
                   help=f"tiny budgets, seeds 0 and 1, at most {SMOKE_CAP} cells per window: checks the script end to end")
    a = p.parse_args(argv)
    t0 = time.time()
    if not a.smoke and list(range(a.seeds)) != SEEDS_PREREG:
        raise SystemExit(f"a full run uses the pre-registered seeds {SEEDS_PREREG} (Step 4), not {a.seeds} seeds")
    path = Path(a.npz).expanduser()
    digest = sha256(path)
    if digest != NPZ_SHA256 and not a.smoke:
        raise SystemExit(f"{path} has SHA-256 {digest}, not the pre-registered {NPZ_SHA256}")
    pcs, labels = load_eb(path)
    card = data_card(path, labels, digest)
    pot = have_pot()
    preflight = pot_preflight() if pot else None                 # before any training: a broken POT stops here
    if not pot:
        print(f"WARNING: POT (import ot) is not installed: every W1 uses the pre-registered fallback (exact assignment "
              f"on {SUB_N}-per-side subsamples, {FALLBACK_REPEATS} repeats), not the all-cell ot.emd2", flush=True)
    if a.smoke:
        pcs, labels = cap_windows(pcs, labels, SMOKE_CAP)
        card["cells_dropped"] = card["n_cells"] - len(labels)
    cfg, winner, recipe, choice = frozen_config(), None, {}, None
    if a.choice:
        winner, recipe, choice = load_choice(a.choice, a.smoke)
    label, step8 = config_labels(winner)
    seeds, mlp = ([0, 1], SMOKE_MLP) if a.smoke else (list(SEEDS_PREREG), MLP)
    if a.smoke:
        cfg = cfg.replace(**SMOKE)
    out = resolve_out(a.out, a.choice)
    guard_out(out, label, a.force)
    out.mkdir(parents=True, exist_ok=True)
    counts = {f"{t:g}": int((labels == t).sum()) for t in TIMES}
    print(f"eb_published: {len(labels)} cells, {pcs.shape[1]} PCs, windows {counts}; sha256 {digest[:12]}... "
          f"({'matches' if digest == NPZ_SHA256 else 'DIFFERS from'} the pre-registration); configuration {label}; "
          f"seeds {seeds}; W1 backend {'POT ' + preflight['pot_version'] if pot else 'fallback (no POT)'}; "
          f"{a.threads} torch thread(s) -> {out}", flush=True)
    summary = {}

    def summarize(rows, folds, complete: bool) -> None:
        df = pd.DataFrame(rows)
        df.to_csv(out / "results.csv", index=False)
        fold_table(df).to_csv(out / "per_fold.csv", index=False)
        summary.clear()
        summary.update({
            "prereg": PREREG, "step": "4 (Job C)", "smoke": bool(a.smoke), "complete": bool(complete),
            "scalers_done": [s for s in SCALERS if s in set(df["scaler"])], "npz": str(path), "npz_sha256": digest,
            "npz_matches_prereg": digest == NPZ_SHA256, "data_card": card, "n_pcs": N_PCS, "window_counts": counts,
            "config_label": label, "step8_label": step8, "config": cfg.to_dict(), "winner": winner, "recipe": recipe,
            "choice": choice, "choice_path": a.choice, "seeds": seeds, "mlp": mlp, "torch_threads": a.threads,
            "scalers": {"primary": "StandardScaler on all cells (published convention, known leak)",
                        "secondary": "StandardScaler on the four observed windows of each fold"},
            "otcfm": {"torchcfm": TORCHCFM, "config": otcfm_config(0).to_dict(),
                      "n_iters_by_fold": {k: v["otcfm_n_iters_used"] for k, v in folds.items()},
                      "prediction": "bracket t-1 -> t+1 integrated from the window t-1 cells to flow time 1/2"},
            "w1": {"backend": sorted(set(df["w1_backend"])), "pot_installed": pot,
                   "pot_version": preflight["pot_version"] if pot else None, "pot_preflight": preflight,
                   "numItermax": NUM_ITER_MAX, "numItermax_retry": NUM_ITER_RETRY,
                   "fallback": f"exact assignment on {SUB_N}-per-side subsamples, mean of {FALLBACK_REPEATS} "
                               "repeats (pre-registration Step 1e), used only without POT",
                   "all_converged": bool(df["w1_converged"].all()),
                   "n_not_converged": int((~df["w1_converged"].astype(bool)).sum()),
                   "n_retried": int(df["w1_retried"].astype(bool).sum()),
                   "sub2000_all_converged": bool(df["w1_sub2000_converged"].all())},
            "extra_steps_per_unit": {"steps": STEPS_EXTRA, "columns": ["w1_8steps", "energy_distance_8steps"],
                                     "rows": "flow and shuffled twin", "role": "non-voting (Step 1c)"},
            "folds": folds, "rules": decide(df, a.smoke), "deviations": DEVIATIONS, "readings": READINGS,
            "seconds": round(time.time() - t0, 1)})
        write_json(out / "summary.json", summary)
        print(f"summary.json written ({'complete' if complete else 'partial: ' + ', '.join(summary['scalers_done'])})",
              flush=True)

    threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(a.threads)))
    try:
        rows, folds = run_folds(pcs, labels, seeds, cfg, mlp, winner, recipe, label, SMOKE_OTCFM if a.smoke else None,
                                out, t0, on_scaler_done=lambda r, f: summarize(r, f, False))
    finally:
        torch.set_num_threads(threads)
    summarize(rows, folds, True)
    rules = summary["rules"]
    pr = rules["primary"]
    if step8:
        print(f"\n[{step8}]", flush=True)
    vb = pr["p2a"]["vs_best_control"]
    print(f"{'' if step8 else chr(10)}calibration gate {verdict(pr['calibration_gate'])}; model check "
          f"{verdict(pr['model_check'])} (OT-CFM equivalent {pr['model_check']['mean']:.4f}); "
          f"P2a {verdict(pr['p2a'])} (best control {pr['p2a']['best_control']}, {vb['mean']:+.4f} "
          f"[{vb['ci_low']:+.4f}, {vb['ci_high']:+.4f}]; every seed {pr['p2a']['below_every_control_in_every_seed']}, "
          f"seed-mean reading {'PASS' if pr['p2a']['passes_seed_mean'] else 'FAIL'}); P2b {pr['p2b']['statement']}",
          flush=True)
    t2 = rules["per_fold"]["primary"]["w1"].get("t=2", {})
    if t2:
        print("t=2 (primary, W1 seed means): " + "  ".join(f"{k} {v:.4f}" for k, v in
                                                          sorted(t2["seed_mean"].items(), key=lambda kv: kv[1])))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
