"""Multi-removal interpolation designs (pre-registration v0.2, Steps 1d, 6 and 7).

A block R of consecutive inner time points is removed from a grid. Every removed time is predicted from the last
observed time before min(R) (near = min(R), far = the others), and the transition that brackets the block (source ->
first observed time after max(R)) stays observed. Nested validation holds out the observed inner time (first and last
observed excluded) farthest from the centre of R (the mean of R; ties to the later time), predicted from the previous
fit time. With one removed time this is exactly search.interpolation_design.

Horizons are relative to g, the median consecutive gap of the data set's FULL time grid (test stage included):
h_rel = (target - source)/g, short iff h_rel < 1.5; frac = (target - source)/(bracket end - source) is the position
of the target inside the bracket; h_span = (target - source)/(span of the full grid) is the pre-registered secondary
horizon measure (reported, never deciding).

interpolation_controls gives the controls of search.run_interpolation_benchmark for one such prediction and draws its
random numbers in the same order, so a single-removal design reproduces that function's control predictions.
"""
from __future__ import annotations

import numpy as np

from ..evaluation.controls import LinearMap, MLPMap, freefm_replay, ot_pairs, persistence
from ..flows.coupling import ot_coupling
from .search import _horizon_pairs

SHORT = 1.5
CONTROLS = ("persistence", "mean_interp", "ot_interp", "linear", "mlp", "freefm_replay")


def grid_gap(times) -> float:
    """Median consecutive gap of a time grid."""
    t = np.unique(np.asarray(list(times), dtype=np.float64))
    if len(t) < 2:
        raise ValueError("a time grid needs at least two distinct times")
    return float(np.median(np.diff(t)))


def multi_removal_design(times, removed, full_times=None, targets=None) -> dict:
    """times: the grid the design works on (test stage already removed); removed: the block R; full_times: the data
    set's full grid for g (default: times); targets: the removed times to predict (default: all of R)."""
    times = sorted({float(t) for t in times})
    rem = sorted({float(t) for t in removed})
    if not rem:
        raise ValueError("the removed set is empty")
    if any(r not in times for r in rem):
        raise ValueError(f"removed times {rem} must lie on the grid {times}")
    obs = [t for t in times if t not in rem]
    before, after = [t for t in obs if t < rem[0]], [t for t in obs if t > rem[-1]]
    if not before or not after:
        raise ValueError(f"removed times {rem} must lie strictly inside the grid {times}")
    source, t_next = before[-1], after[0]
    if any(source < t < t_next for t in obs):
        raise ValueError(f"removed times {rem} must be consecutive on the grid {times}")
    tg = rem if targets is None else sorted({float(t) for t in targets})
    if not tg or any(t not in rem for t in tg):
        raise ValueError(f"targets {tg} must be a non-empty subset of the removed times {rem}")
    inner = obs[1:-1]
    if not inner:
        raise ValueError("nested validation needs an observed inner time")
    centre = float(np.mean(rem))
    val_time = max(inner, key=lambda t: (abs(t - centre), t))
    fit_times = [t for t in obs if t != val_time]
    v_prev = max(t for t in fit_times if t < val_time)
    full = times if full_times is None else sorted({float(t) for t in full_times})
    g, span = grid_gap(full), float(max(full) - min(full))
    preds = [{"target": t, "source": source, "t_next": t_next, "kind": "near" if t == rem[0] else "far",
              "horizon": t - source, "h_rel": (t - source) / g, "h_span": (t - source) / span,
              "frac": (t - source) / (t_next - source)} for t in tg]
    return {"times": times, "removed": rem, "obs_times": obs, "source": source, "t_next": t_next,
            "boundary_source": source == obs[0], "centre": centre, "val_time": val_time, "v_prev": v_prev,
            "fit_times": fit_times, "g": g, "span": span, "predictions": preds}


def interpolation_controls(snapshots, x_eval, all_mask, obs_times, source, target, t_next, groups, seed,
                           mlp_kwargs=None, linear_alpha=1.0) -> list:
    """[(control, condition, replicate, prediction)] for predicting target from source with the bracket source -> t_next
    observed, in the evaluation space x_eval, every control fitted on the observed cells (all_mask) only:
    persistence; mean_interp = x0 + frac (mean x(t_next) - mean x0); ot_interp = McCann interpolation at frac along the
    OT pairs of the bracket; linear/mlp maps fitted on observed pairs at the prediction's horizon; freefm_replay = the
    velocity arriving at the source replayed over the horizon (the bracket pairs when the source is the first time)."""
    horizon = float(target) - float(source)
    frac = horizon / (float(t_next) - float(source))
    before = [t for t in obs_times if t < source]
    rng = np.random.default_rng(20_000 + seed)
    out = []
    for cond, rep, m0, _ in groups:
        x0 = x_eval[m0]
        x_next = x_eval[all_mask & snapshots.mask(times=[t_next], conditions=[cond], replicates=[rep])]
        out.append(("persistence", cond, rep, persistence(x0)))
        out.append(("mean_interp", cond, rep, x0 + frac * (x_next.mean(0) - x0.mean(0))))
        i0, i1 = ot_coupling(x0, x_next, rng, max_size=len(x0))
        out.append(("ot_interp", cond, rep, x0[i0] + frac * (x_next[i1] - x0[i0])))
        p0, p1 = _horizon_pairs(snapshots, x_eval, all_mask, obs_times, cond, horizon, rng)
        out.append(("linear", cond, rep, LinearMap(alpha=linear_alpha).fit(p0, p1).predict(x0)))
        out.append(("mlp", cond, rep, MLPMap(seed=seed, **(mlp_kwargs or {})).fit(p0, p1).predict(x0)))
        if before:
            t_pp = before[-1]
            x_pp = x_eval[all_mask & snapshots.mask(times=[t_pp], conditions=[cond], replicates=[rep])]
            q0, q1 = ot_pairs(x_pp, x0, rng, max_size=1000)
            out.append(("freefm_replay", cond, rep,
                        freefm_replay(x0, q1, q1 + (q1 - q0) * (horizon / (source - t_pp)), k=1)))
        else:
            q0, q1 = ot_pairs(x0, x_next, rng, max_size=1000)
            out.append(("freefm_replay", cond, rep, freefm_replay(x0, q0, q0 + frac * (q1 - q0), k=1)))
    return out
