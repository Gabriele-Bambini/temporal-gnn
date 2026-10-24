"""Nested grid search (validation timepoint only), the forecast benchmark and the leave-one-inner-timepoint-out
interpolation benchmark, both with mandatory controls fitted on the same observed cells."""
from __future__ import annotations

import time
from itertools import zip_longest

import numpy as np
import pandas as pd

from ..data.preprocessing import Standardizer
from ..evaluation.controls import LinearMap, MLPMap, freefm_replay, mean_shift, ot_pairs, persistence, train_mean
from ..evaluation.metrics import EvalProtocol, median_bandwidth, noise_floor
from ..flows.coupling import ot_coupling
from .config import expand_grid, random_subset
from .trainer import build_transitions, task_groups, time_task, train_flow


def nested_grid_search(snapshots, fit_mask, fit_times, val_task, base_config, grid, seeds, protocol,
                       max_configs=None, seed=0, verbose=False):
    """Select the configuration with the lowest mean validation energy distance. The test is never touched."""
    configs = list(grid) if isinstance(grid, list) else expand_grid(grid)
    if max_configs is not None:
        configs = random_subset(configs, max_configs, seed)
    rows = []
    for k, over in enumerate(configs):
        for s in seeds:
            cfg = base_config.replace(**over, seed=s)
            t0 = time.time()
            model, _ = train_flow(snapshots, fit_mask, fit_times, cfg, val_task, protocol)
            rows.append({"config_id": k, **over, "seed": s, "val_energy_distance": model.best_val,
                         "best_iter": model.best_iter, "seconds": time.time() - t0})
            if verbose:
                print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    return select_best(table, base_config), table


_TABLE_COLS = {"config_id", "seed", "val_energy_distance", "best_iter", "seconds", "dataset", "val_time"}


def select_best(table: pd.DataFrame, base_config):
    """Configuration with the lowest mean validation energy distance over seeds (and, if the table stacks
    several validation folds, over folds)."""
    best_id = table.groupby("config_id")["val_energy_distance"].mean().idxmin()
    sub = table[table["config_id"] == best_id]
    over = {k: sub[k].iloc[0] for k in table.columns if k not in _TABLE_COLS}     # column-wise: dtypes preserved
    return base_config.replace(**{k: (v.item() if hasattr(v, "item") else v) for k, v in over.items()})


def interpolation_val_folds(times, test_time, n_folds: int = 2) -> list:
    """Inner observed timepoints usable for nested validation of an interpolation fold: farthest from
    test_time first, alternating sides when possible, at most n_folds."""
    times = sorted(float(t) for t in times)
    test_time = float(test_time)
    inner = [t for t in times if t != test_time][1:-1]
    before = sorted(t for t in inner if t < test_time)
    after = sorted((t for t in inner if t > test_time), reverse=True)
    picks = [t for pair in zip_longest(before, after) for t in pair if t is not None]
    return picks[:n_folds]


def interpolation_design(times, test_time, val_time=None) -> dict:
    """Leave-one-inner-timepoint-out design: the held-out timepoint is predicted from the previous observed
    one, and the transition that brackets the gap (t_prev -> t_next) is observed. Nested validation holds
    out a second inner timepoint (default: the observed inner timepoint farthest from test_time)."""
    times = sorted(float(t) for t in times)
    test_time = float(test_time)
    if test_time not in times[1:-1]:
        raise ValueError(f"test_time {test_time} must be an inner timepoint of {times}")
    k = times.index(test_time)
    t_prev, t_next = times[k - 1], times[k + 1]
    obs_times = [t for t in times if t != test_time]
    inner = obs_times[1:-1]
    if not inner:
        raise ValueError("interpolation with nested validation needs at least four timepoints")
    val_time = max(inner, key=lambda t: (abs(t - test_time), t)) if val_time is None else float(val_time)
    if val_time not in inner:
        raise ValueError(f"val_time {val_time} must be an inner observed timepoint, one of {inner}")
    fit_times = [t for t in obs_times if t != val_time]
    v_prev = max(t for t in fit_times if t < val_time)
    return {"times": times, "test_time": test_time, "t_prev": t_prev, "t_next": t_next, "obs_times": obs_times,
            "val_time": val_time, "v_prev": v_prev, "fit_times": fit_times, "horizon": test_time - t_prev,
            "frac": (test_time - t_prev) / (t_next - t_prev)}


def _horizon_pairs(snapshots, x_eval, all_mask, obs_times, cond, horizon, rng, max_pairs=1000):
    """Pooled OT pairs over the observed pairs of timepoints of one condition whose horizon equals the test
    horizon; if none exists, every consecutive transition with its displacement rescaled to that horizon."""
    sub = snapshots.subset(all_mask)
    x_sub = x_eval[all_mask]
    same = [tr for tr in build_transitions(sub, obs_times, x_sub, pair_mode="all")
            if tr.condition == cond and abs((tr.bio_t1 - tr.bio_t0) - horizon) < 1e-9]
    scaled = not same
    trs = same or [tr for tr in build_transitions(sub, obs_times, x_sub) if tr.condition == cond]
    p0, p1 = [], []
    for tr in trs:
        a, b = ot_pairs(tr.x0, tr.x1, rng, max_size=max_pairs)
        if scaled:
            b = a + (b - a) * (horizon / (tr.bio_t1 - tr.bio_t0))
        p0.append(a)
        p1.append(b)
    if not p0:
        raise ValueError(f"no observed transition for condition {cond!r}")
    return np.concatenate(p0), np.concatenate(p1)


def run_interpolation_benchmark(snapshots, times, test_time, config, seeds, protocol, dataset_name, val_time=None,
                                refit_on_all=False, shuffled_control=True, mlp_kwargs=None, linear_alpha=1.0,
                                verbose=False):
    """One row per (model, seed, condition|replicate group) for a held-out inner timepoint.

    The flow model is early-stopped on the nested validation timepoint and, if refit_on_all, retrained on
    every observed timepoint for the selected number of iterations; it predicts test_time from the cells at
    t_prev over the horizon test_time - t_prev. Controls (same observed cells, same evaluation space):
    persistence; mean_interp = x0 + frac * (mean x(t_next) - mean x(t_prev)); ot_interp = displacement
    (McCann) interpolation along the OT pairs of the bracketing transition; linear/mlp maps fitted on the
    observed pairs with the test horizon; freefm_replay = the velocity of the transition arriving at t_prev
    replayed at the nearest arrival position (the bracketing pairs when t_prev is the first timepoint).
    """
    d = interpolation_design(times, test_time, val_time)
    fit_mask = snapshots.mask(times=d["fit_times"])
    all_mask = snapshots.mask(times=d["obs_times"])
    val_task = time_task(snapshots, d["v_prev"], d["val_time"])
    test_task = time_task(snapshots, d["t_prev"], d["test_time"])
    if (all_mask & test_task.target_mask).any():
        raise ValueError("test target cells overlap the observed cells")
    eval_std = Standardizer().fit(snapshots.x[all_mask])
    x_eval = eval_std.transform(snapshots.x)
    protocol = EvalProtocol(n_samples=protocol.n_samples, seed=protocol.seed,
                            bandwidth=median_bandwidth(x_eval[all_mask]), w2_max=protocol.w2_max,
                            n_perm=protocol.n_perm)
    groups = task_groups(snapshots, test_task)
    if not groups:
        raise ValueError("test task has no (condition, replicate) group with cells on both sides")
    horizon, frac = d["horizon"], d["frac"]
    rows = []

    def add(model_name, seed, cond, rep, pred, target, extra=None):
        rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "group": f"{cond}|{rep}",
                     "condition": cond, "replicate": rep, **protocol.evaluate(pred, target, seed_offset=seed),
                     **(extra or {})})

    for seed in seeds:
        variants = [("flow_matching", config.replace(seed=seed))]
        if shuffled_control:
            variants.append(("flow_matching_shuffled_time", config.replace(seed=seed, shuffle_biological_time=True)))
        for name, cfg in variants:
            t0 = time.time()
            model, _ = train_flow(snapshots, fit_mask, d["fit_times"], cfg, val_task, protocol)
            if refit_on_all:
                n_iters = max(int(model.best_iter), cfg.eval_every)
                model, _ = train_flow(snapshots, all_mask, d["obs_times"], cfg.replace(n_iters=n_iters), None, protocol)
            secs = time.time() - t0
            for cond, rep, m0, m1 in groups:
                if cond not in model.registry.names:
                    continue
                pred = eval_std.transform(model.predict(snapshots.x[m0], cond, d["t_prev"], d["test_time"]))
                add(name, seed, cond, rep, pred, x_eval[m1],
                    {"train_seconds": secs, "n_params": model.n_params, "best_iter": model.best_iter})
            if verbose:
                print(f"[{dataset_name}] {name} seed={seed} done in {secs:.0f}s", flush=True)
        rng = np.random.default_rng(20_000 + seed)
        for cond, rep, m0, m1 in groups:
            x0, target = x_eval[m0], x_eval[m1]
            x_next = x_eval[all_mask & snapshots.mask(times=[d["t_next"]], conditions=[cond], replicates=[rep])]
            add("persistence", seed, cond, rep, persistence(x0), target)
            add("mean_interp", seed, cond, rep, x0 + frac * (x_next.mean(0) - x0.mean(0)), target)
            i0, i1 = ot_coupling(x0, x_next, rng, max_size=len(x0))
            add("ot_interp", seed, cond, rep, x0[i0] + frac * (x_next[i1] - x0[i0]), target)
            p0, p1 = _horizon_pairs(snapshots, x_eval, all_mask, d["obs_times"], cond, horizon, rng)
            add("linear", seed, cond, rep, LinearMap(alpha=linear_alpha).fit(p0, p1).predict(x0), target)
            add("mlp", seed, cond, rep, MLPMap(seed=seed, **(mlp_kwargs or {})).fit(p0, p1).predict(x0), target)
            before = [t for t in d["obs_times"] if t < d["t_prev"]]
            if before:
                t_pp = before[-1]
                x_pp = x_eval[all_mask & snapshots.mask(times=[t_pp], conditions=[cond], replicates=[rep])]
                q0, q1 = ot_pairs(x_pp, x0, rng, max_size=1000)
                arrival = q1 + (q1 - q0) * (horizon / (d["t_prev"] - t_pp))
                add("freefm_replay", seed, cond, rep, freefm_replay(x0, q1, arrival, k=1), target)
            else:
                q0, q1 = ot_pairs(x0, x_next, rng, max_size=1000)
                add("freefm_replay", seed, cond, rep, freefm_replay(x0, q0, q0 + frac * (q1 - q0), k=1), target)
            rows.append({"dataset": dataset_name, "model": "noise_floor", "seed": seed, "group": f"{cond}|{rep}",
                         "condition": cond, "replicate": rep, **noise_floor(target, protocol)})
    return pd.DataFrame(rows)


def _control_pairs(snapshots, x_eval, all_mask, all_times, cond, rng, max_pairs=1000):
    """Pooled OT pairs over every observed transition of one condition (fit + validation cells only)."""
    sub = snapshots.subset(all_mask)
    p0, p1 = [], []
    for tr in build_transitions(sub, all_times, x_eval[all_mask]):
        if tr.condition == cond:
            a, b = ot_pairs(tr.x0, tr.x1, rng, max_size=max_pairs)
            p0.append(a)
            p1.append(b)
    if not p0:
        raise ValueError(f"no observed transition for condition {cond!r}")
    return np.concatenate(p0), np.concatenate(p1)


def run_forecast_benchmark(snapshots, fit_mask, fit_times, val_task, test_task, config, seeds, protocol, dataset_name,
                           refit_on_all=False, shuffled_control=True, mlp_kwargs=None, linear_alpha=1.0, verbose=False):
    """One row per (model, seed, condition|replicate group) on test_task.

    Evaluation space: standardised on the observed cells (fit + validation). The flow model is
    early-stopped on the validation task and, if refit_on_all, retrained on fit + validation for the
    selected number of iterations. Controls are fitted on the same observed cells. Test cells are
    used only as the forecast source (bio_t0) and as the target to score.
    """
    fit_mask = np.asarray(fit_mask, dtype=bool)
    all_mask = fit_mask | val_task.source_mask | val_task.target_mask
    if (all_mask & test_task.target_mask).any():
        raise ValueError("test target cells overlap the observed (fit + validation) cells")
    all_times = sorted({float(t) for t in fit_times} | {val_task.bio_t0, val_task.bio_t1})
    eval_std = Standardizer().fit(snapshots.x[all_mask])
    x_eval = eval_std.transform(snapshots.x)
    protocol = EvalProtocol(n_samples=protocol.n_samples, seed=protocol.seed,
                            bandwidth=median_bandwidth(x_eval[all_mask]), w2_max=protocol.w2_max)
    groups = task_groups(snapshots, test_task)
    if not groups:
        raise ValueError("test task has no (condition, replicate) group with cells on both sides")
    rows = []

    def add(model_name, seed, cond, rep, pred, target, extra=None):
        rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "group": f"{cond}|{rep}",
                     "condition": cond, "replicate": rep, **protocol.evaluate(pred, target, seed_offset=seed),
                     **(extra or {})})

    for seed in seeds:
        variants = [("flow_matching", config.replace(seed=seed))]
        if shuffled_control:
            variants.append(("flow_matching_shuffled_time", config.replace(seed=seed, shuffle_biological_time=True)))
        for name, cfg in variants:
            t0 = time.time()
            model, _ = train_flow(snapshots, fit_mask, fit_times, cfg, val_task, protocol)
            if refit_on_all:
                n_iters = max(int(model.best_iter), cfg.eval_every)
                model, _ = train_flow(snapshots, all_mask, all_times, cfg.replace(n_iters=n_iters), None, protocol)
            secs = time.time() - t0
            for cond, rep, m0, m1 in groups:
                if cond not in model.registry.names:
                    continue
                pred = eval_std.transform(model.predict(snapshots.x[m0], cond, test_task.bio_t0, test_task.bio_t1))
                add(name, seed, cond, rep, pred, x_eval[m1],
                    {"train_seconds": secs, "n_params": model.n_params, "best_iter": model.best_iter})
            if verbose:
                print(f"[{dataset_name}] {name} seed={seed} done in {secs:.0f}s", flush=True)
        rng = np.random.default_rng(10_000 + seed)
        prev_t, last_t = all_times[-2], all_times[-1]
        for cond, rep, m0, m1 in groups:
            x0, target = x_eval[m0], x_eval[m1]
            obs_prev = x_eval[all_mask & snapshots.mask(times=[prev_t], conditions=[cond])]
            obs_last = x_eval[all_mask & snapshots.mask(times=[last_t], conditions=[cond])]
            add("persistence", seed, cond, rep, persistence(x0), target)
            add("train_mean", seed, cond, rep, train_mean(obs_last, len(x0)), target)
            add("mean_shift", seed, cond, rep, mean_shift(x0, obs_prev, obs_last), target)
            p0, p1 = _control_pairs(snapshots, x_eval, all_mask, all_times, cond, rng)
            add("linear", seed, cond, rep, LinearMap(alpha=linear_alpha).fit(p0, p1).predict(x0), target)
            add("mlp", seed, cond, rep, MLPMap(seed=seed, **(mlp_kwargs or {})).fit(p0, p1).predict(x0), target)
            add("freefm_replay", seed, cond, rep, freefm_replay(x0, p0, p1, k=1), target)
            rows.append({"dataset": dataset_name, "model": "noise_floor", "seed": seed, "group": f"{cond}|{rep}",
                         "condition": cond, "replicate": rep, **noise_floor(target, protocol)})
    return pd.DataFrame(rows)
