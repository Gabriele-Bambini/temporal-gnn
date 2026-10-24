"""Training loop of the conditional flow matching engine (no graph, no mask).

``train_flow`` fits on the cells in ``fit_mask`` over consecutive ``fit_times``; an optional
``val_task`` drives early stopping on the validation energy distance. Validation cells never
touch the standardizer or the loss. ``FlowModel.predict`` maps observed cells at bio_t0 to a
population at bio_t1 by integrating the learned field over flow_time in [0, 1] (clock="flow") or,
with clock="duration", by integrating an autonomous field v(x, condition) trained on the velocity
(x1 - x0)/dt over the biological horizon bio_t1 - bio_t0 (dt-independent, like a neural ODE).
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .. import __version__
from ..data.preprocessing import Standardizer, fit_on_train
from ..data.schema import SnapshotSet
from ..evaluation.metrics import EvalProtocol, energy_distance
from ..flows.coupling import chain_ot, independent_coupling, ot_coupling
from ..flows.marginal import marginal_loss
from ..flows.path import SplinePath, cfm_loss, sample_flow_time, spline_loss
from ..flows.solver import integrate
from ..models.conditioning import ConditionRegistry, ContextEncoder
from ..models.vector_field import MLPVectorField
from .config import TrainConfig


@dataclass
class Transition:
    condition: str
    replicate: str
    bio_t0: float
    bio_t1: float
    x0: np.ndarray
    x1: np.ndarray


@dataclass
class ForecastTask:
    """Predict the cells in target_mask (at bio_t1) from the cells in source_mask (at bio_t0)."""
    source_mask: np.ndarray
    target_mask: np.ndarray
    bio_t0: float
    bio_t1: float
    name: str = "task"


def time_task(snapshots: SnapshotSet, t0: float, t1: float, replicates=None) -> ForecastTask:
    return ForecastTask(snapshots.mask(times=[t0], replicates=replicates),
                        snapshots.mask(times=[t1], replicates=replicates), float(t0), float(t1), f"t{t0}->t{t1}")


def replicate_task(snapshots: SnapshotSet, replicate: str, t0: float, t1: float) -> ForecastTask:
    return ForecastTask(snapshots.mask(times=[t0], replicates=[replicate]),
                        snapshots.mask(times=[t1], replicates=[replicate]), float(t0), float(t1),
                        f"{replicate}:t{t0}->t{t1}")


def build_transitions(snapshots: SnapshotSet, times, x=None, min_cells: int = 2, pair_mode: str = "consecutive") -> list:
    """Transitions within each (condition, replicate) group: consecutive (t_k -> t_k+1) pairs, or every
    ordered pair (t_i -> t_j, i < j) with pair_mode="all" (the delta-t context then carries the horizon)."""
    x = snapshots.x if x is None else np.asarray(x)
    if len(x) != snapshots.n_cells:
        raise ValueError("x must be aligned with snapshots (one row per cell)")
    times = np.unique(np.asarray(list(times), dtype=np.float64))
    if pair_mode == "consecutive":
        pairs = list(zip(times[:-1], times[1:]))
    elif pair_mode == "all":
        pairs = [(t0, t1) for i, t0 in enumerate(times) for t1 in times[i + 1:]]
    else:
        raise ValueError(f"unknown pair_mode {pair_mode!r} (expected 'consecutive' or 'all')")
    out = []
    for cond in snapshots.conditions():
        for rep in snapshots.replicates():
            for t0, t1 in pairs:
                m0 = snapshots.mask(times=[t0], conditions=[cond], replicates=[rep])
                m1 = snapshots.mask(times=[t1], conditions=[cond], replicates=[rep])
                if m0.sum() >= min_cells and m1.sum() >= min_cells:
                    out.append(Transition(cond, rep, float(t0), float(t1), x[m0], x[m1]))
    if not out:
        raise ValueError("no transition has enough cells on both sides; check times, conditions and replicates")
    return out


def task_groups(snapshots: SnapshotSet, task: ForecastTask, min_cells: int = 2) -> list:
    """(condition, replicate, source_mask, target_mask) for every group present on both sides of the task."""
    out = []
    for cond in snapshots.conditions():
        for rep in snapshots.replicates():
            g = snapshots.mask(conditions=[cond], replicates=[rep])
            m0, m1 = task.source_mask & g, task.target_mask & g
            if m0.sum() >= min_cells and m1.sum() >= min_cells:
                out.append((cond, rep, m0, m1))
    return out


class WeightEMA:
    """Exponential moving average of module weights, with the usual warm-up d_n = min(decay, (1+n)/(10+n))
    so the average is not dominated by the initial weights. Used for validation and for the final model."""

    def __init__(self, modules: dict, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.modules, self.decay, self.n_updates = dict(modules), float(decay), 0
        self.shadow = {name: {k: v.detach().clone() for k, v in m.state_dict().items()}
                       for name, m in self.modules.items()}

    def update(self) -> None:
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        with torch.no_grad():
            for name, m in self.modules.items():
                for k, v in m.state_dict().items():
                    if v.dtype.is_floating_point:
                        self.shadow[name][k].mul_(d).add_(v, alpha=1.0 - d)
                    else:
                        self.shadow[name][k].copy_(v)
        self.n_updates += 1

    def state(self) -> dict:
        return copy.deepcopy(self.shadow)


def _context_mode(config: TrainConfig) -> str:
    """clock="duration" trains an autonomous field: the context carries the condition only."""
    if config.clock == "flow":
        return config.time_conditioning
    if config.clock == "duration":
        return "none"
    raise ValueError(f"unknown clock {config.clock!r} (expected 'flow' or 'duration')")


def _bio_time(config: TrainConfig) -> bool:
    """field_time="bio": the duration-clock field receives the biological time (scaled by time_scale)."""
    if config.field_time not in ("none", "bio"):
        raise ValueError(f"unknown field_time {config.field_time!r} (expected 'none' or 'bio')")
    if config.field_time == "bio" and config.clock != "duration":
        raise ValueError("field_time='bio' is defined for the duration clock only")
    return config.field_time == "bio"


def build_networks(config: TrainConfig, registry: ConditionRegistry, n_genes: int, time_scale: float):
    if config.path not in ("linear", "spline"):
        raise ValueError(f"unknown path {config.path!r} (expected 'linear' or 'spline')")
    if config.path == "spline" and config.clock != "duration":
        raise ValueError("path='spline' regresses biological velocities: it needs clock='duration'")
    bio = _bio_time(config)
    ctx = ContextEncoder(registry, config.cond_dim, time_scale, _context_mode(config))
    vf = MLPVectorField(n_genes, ctx.out_dim, config.hidden, config.depth, config.time_embed_dim,
                        use_time=config.clock == "flow" or bio, time_embed="scalar" if bio else "sinusoidal")
    return ctx, vf


def spline_paths(fit: SnapshotSet, fit_times, x_fit, rng: np.random.Generator, max_cells: int = 1000,
                 chain: str = "ot", bc_type: str = "natural", center: str = "spline") -> list:
    """One (condition, SplinePath) per (condition, replicate) group, through OT-chained pseudo-trajectories
    over every fit timepoint with at least two cells in that group."""
    times = np.unique(np.asarray(list(fit_times), dtype=np.float64))
    out = []
    for cond in fit.conditions():
        for rep in fit.replicates():
            keep = [(t, m) for t in times if (m := fit.mask(times=[t], conditions=[cond], replicates=[rep])).sum() >= 2]
            if len(keep) < 2:
                continue
            traj = chain_ot([x_fit[m] for _, m in keep], rng, max_cells, mode=chain)
            out.append((cond, SplinePath([t for t, _ in keep], traj, bc_type=bc_type, center=center)))
    if not out:
        raise ValueError("no (condition, replicate) group has two timepoints with enough cells")
    return out


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class FlowModel:
    def __init__(self, registry, standardizer, context_encoder, vector_field, config, gene_names, time_scale):
        self.registry = registry
        self.standardizer = standardizer
        self.context_encoder = context_encoder
        self.vector_field = vector_field
        self.config = config
        self.gene_names = list(gene_names)
        self.time_scale = float(time_scale)
        self.best_iter = None
        self.best_val = None
        self.marginal_iter = None
        self.train_seconds = None
        self.stage1_best_val = None     # validation score at the end of stage 1 (best_val is overwritten by stage 2)
        self.marginal_guarded = None    # True: stage 2 was early-stopped on a validation task; False: ran unguarded
        self.stage_states = {}          # "stage1": the weights at the end of stage 1 (EMA, before stage 2)
        self.marginal_snapshots = {}    # stage-2 iteration -> weights, kept only when train_flow(keep_snapshots=True)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in list(self.context_encoder.parameters()) + list(self.vector_field.parameters()))

    def field(self, bio_t0: float = 0.0):
        """What the ODE solver integrates. The solver's clock s starts at 0; with field_time="bio" it is turned
        into the scaled biological time (bio_t0 + s) / time_scale the field was trained on."""
        if self.config.clock == "duration" and self.config.field_time == "bio":
            vf, t0, sc = self.vector_field, float(bio_t0), self.time_scale
            return lambda x, s, ctx: vf(x, (t0 + s) / sc, ctx)
        return self.vector_field

    def path_time(self, u) -> torch.Tensor | None:
        """Scaled biological times of path samples for a non-autonomous field (None for the autonomous one)."""
        if self.config.clock == "duration" and self.config.field_time == "bio":
            return torch.as_tensor(np.asarray(u) / self.time_scale, dtype=torch.float32)
        return None

    def _context(self, condition: str, bio_t0: float, bio_t1: float, b: int) -> torch.Tensor:
        idx = self.registry.encode([condition] * b)
        return self.context_encoder(idx, torch.full((b,), float(bio_t0)), torch.full((b,), float(bio_t1)))

    def _stage_state(self, stage):
        """Weights of a training stage: None/"final" (current), "stage1" or 0 (end of stage 1), or a stage-2 iteration
        kept by train_flow(keep_snapshots=True)."""
        if stage is None or stage == "final":
            return None
        if stage == "stage1" or (isinstance(stage, (int, np.integer)) and int(stage) == 0):
            if "stage1" not in self.stage_states:
                raise ValueError("this model kept no stage-1 weights")
            return self.stage_states["stage1"]
        if isinstance(stage, (int, np.integer)) and int(stage) in self.marginal_snapshots:
            return self.marginal_snapshots[int(stage)]
        raise ValueError(f"unknown stage {stage!r}: use 'final', 'stage1' or one of {sorted(self.marginal_snapshots)}")

    def predict_standardized(self, x0, condition: str, bio_t0: float, bio_t1: float, n_steps=None, method=None,
                             stage=None) -> np.ndarray:
        state = self._stage_state(stage)
        if state is not None:
            current = self.snapshot_state()
            self.load_snapshot(state)
            try:
                return self.predict_standardized(x0, condition, bio_t0, bio_t1, n_steps, method)
            finally:
                self.load_snapshot(current)
        self.vector_field.eval()
        self.context_encoder.eval()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(x0), dtype=torch.float32)
            ctx = self._context(condition, bio_t0, bio_t1, x.shape[0])
            steps = int(n_steps or self.config.n_ode_steps)
            horizon = 1.0
            if self.config.clock == "duration":
                horizon = float(bio_t1) - float(bio_t0)
                if horizon <= 0:
                    raise ValueError("bio_t1 must exceed bio_t0 for the duration clock")
                steps = max(steps, int(math.ceil(steps * horizon)))     # keep the step size of a unit horizon
            out = integrate(self.field(bio_t0), x, ctx, steps, method or self.config.ode_method, t_end=horizon)
        self.vector_field.train()
        self.context_encoder.train()
        return out.numpy()

    def predict(self, x0_raw, condition: str, bio_t0: float, bio_t1: float, **kw) -> np.ndarray:
        """kw: n_steps, method, stage (see _stage_state)."""
        z = self.standardizer.transform(x0_raw)
        return self.standardizer.inverse_transform(self.predict_standardized(z, condition, bio_t0, bio_t1, **kw))

    def snapshot_state(self) -> dict:
        return {"ctx": copy.deepcopy(self.context_encoder.state_dict()),
                "vf": copy.deepcopy(self.vector_field.state_dict())}

    def load_snapshot(self, state: dict) -> None:
        self.context_encoder.load_state_dict(state["ctx"])
        self.vector_field.load_state_dict(state["vf"])

    def save(self, path) -> None:
        torch.save({"version": __version__, "config": self.config.to_dict(), "gene_names": self.gene_names,
                    "standardizer": self.standardizer.to_dict(), "registry": self.registry.to_dict(),
                    "time_scale": self.time_scale, "state": self.snapshot_state(),
                    "best_iter": self.best_iter, "best_val": self.best_val, "stage_states": self.stage_states,
                    "marginal_iter": self.marginal_iter, "marginal_guarded": self.marginal_guarded,
                    "stage1_best_val": self.stage1_best_val}, path)

    @classmethod
    def load(cls, path, expected_gene_names=None) -> "FlowModel":
        ck = torch.load(path, weights_only=False)
        if ck.get("version") != __version__:
            raise ValueError(f"checkpoint version {ck.get('version')!r} differs from code version {__version__!r}")
        if expected_gene_names is not None and list(expected_gene_names) != list(ck["gene_names"]):
            raise ValueError("checkpoint gene names differ from the data gene names")
        config = TrainConfig.from_dict(ck["config"])
        registry = ConditionRegistry.from_dict(ck["registry"])
        ctx, vf = build_networks(config, registry, len(ck["gene_names"]), ck["time_scale"])
        model = cls(registry, Standardizer.from_dict(ck["standardizer"]), ctx, vf, config, ck["gene_names"],
                    ck["time_scale"])
        model.load_snapshot(ck["state"])
        model.best_iter, model.best_val = ck.get("best_iter"), ck.get("best_val")
        model.stage_states = ck.get("stage_states", {})
        model.marginal_iter, model.marginal_guarded = ck.get("marginal_iter"), ck.get("marginal_guarded")
        model.stage1_best_val = ck.get("stage1_best_val")
        return model


def _validate(model: FlowModel, snapshots: SnapshotSet, x_std: np.ndarray, task: ForecastTask,
              protocol: EvalProtocol) -> float:
    vals = []
    for cond, rep, m0, m1 in task_groups(snapshots, task):
        if cond not in model.registry.names:
            continue
        rng = np.random.default_rng(protocol.seed)
        x0 = protocol.subsample(x_std[m0], rng)
        x1 = protocol.subsample(x_std[m1], rng)
        vals.append(energy_distance(model.predict_standardized(x0, cond, task.bio_t0, task.bio_t1), x1))
    if not vals:
        raise ValueError("validation task shares no (condition, replicate) group with the training conditions")
    return float(np.mean(vals))


def _finetune_marginal(model: FlowModel, snapshots: SnapshotSet, x_std: np.ndarray, transitions: list, paths,
                       config: TrainConfig, val_task: ForecastTask | None, protocol: EvalProtocol,
                       rng: np.random.Generator, history: list, verbose: bool = False,
                       keep_snapshots: bool = False) -> None:
    """Second stage (v0.1.9): match the OBSERVED marginals instead of a constructed path.

    A path loss can only be as good as its path. On the synthetic regulatory benchmark the spline family's own
    marginal at the held-out timepoint sits at energy distance 0.636 while the field trained on it reaches
    0.641: the construction, not the network, is the ceiling (scripts_dev/diag_ceiling.py). This stage asks the
    field to push each observed snapshot onto the next one, scored by a differentiable two-sample divergence,
    so the interior of the transition is left to the field's own smoothness and to consistency with the other
    observed transitions. The path loss is kept with weight ``marginal_prior`` as a regulariser. When a
    validation task is given, early stopping still uses it and the stage can only replace the pre-trained weights
    by improving it (``marginal_iter`` = the kept iteration, 0 = stage 2 rejected). Without one (the refit on every
    observed time point, which makes the reported predictions) the stage runs all ``marginal_iters`` iterations
    unguarded (``marginal_guarded`` False, ``marginal_iter`` = iterations run). keep_snapshots stores the weights
    that would be kept (the EMA) at every evaluation point, so an earlier stopping point can be used afterwards;
    storing copies draws no random number.
    """
    model.marginal_guarded = val_task is not None
    params = list(model.context_encoder.parameters()) + list(model.vector_field.parameters())
    opt = torch.optim.Adam(params, lr=config.marginal_lr, weight_decay=config.weight_decay)
    ema = WeightEMA({"ctx": model.context_encoder, "vf": model.vector_field}, config.ema_decay) if config.ema_decay > 0 else None
    gen = torch.Generator().manual_seed(int(config.seed) + 9973)
    w = np.array([min(len(t.x0), len(t.x1)) for t in transitions], dtype=np.float64)
    w /= w.sum()
    pw = None
    if paths:
        pw = np.array([p.n for _, p in paths], dtype=np.float64)
        pw /= pw.sum()
    best_val = model.best_val if (val_task is not None and model.best_val is not None) else float("inf")
    best_state = model.snapshot_state() if val_task is not None else None
    best_iter, bad = 0, 0
    for it in range(1, int(config.marginal_iters) + 1):
        tr = transitions[rng.choice(len(transitions), p=w)]
        b = min(int(config.marginal_batch), len(tr.x0), len(tr.x1))
        if b < 2:
            continue
        x0 = torch.as_tensor(tr.x0[rng.choice(len(tr.x0), b, replace=False)], dtype=torch.float32)
        x1 = torch.as_tensor(tr.x1[rng.choice(len(tr.x1), b, replace=False)], dtype=torch.float32)
        horizon = (tr.bio_t1 - tr.bio_t0) if config.clock == "duration" else 1.0
        steps = max(int(config.marginal_steps), int(math.ceil(config.marginal_steps * horizon)))
        loss = config.marginal_weight * marginal_loss(model.field(tr.bio_t0), x0, x1,
                                                      model._context(tr.condition, tr.bio_t0, tr.bio_t1, b),
                                                      horizon, steps, config.ode_method, config.marginal_metric, gen)
        if paths and config.marginal_prior > 0:
            cond, path = paths[rng.choice(len(paths), p=pw)]
            pb = min(config.batch_size, path.n)
            xs, vs, us = path.sample(rng, pb, window=config.marginal_window)
            loss = loss + config.marginal_prior * spline_loss(
                model.vector_field, torch.as_tensor(xs, dtype=torch.float32),
                torch.as_tensor(vs, dtype=torch.float32), model._context(cond, 0.0, 1.0, pb), config.sigma_min,
                time=model.path_time(us))
        opt.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        opt.step()
        if ema is not None:
            ema.update()
        if it % config.marginal_eval_every == 0 or it == config.marginal_iters:
            rec = {"iter": it, "stage": "marginal", "loss": float(loss.item())}
            if keep_snapshots:
                model.marginal_snapshots[it] = ema.state() if ema is not None else model.snapshot_state()
            if val_task is not None:
                raw_state = model.snapshot_state()
                if ema is not None:
                    model.load_snapshot(ema.state())          # validate, and keep, the averaged weights
                v = _validate(model, snapshots, x_std, val_task, protocol)
                rec["val_energy_distance"] = v
                if v < best_val - 1e-9:
                    best_val, best_iter, best_state, bad = v, it, model.snapshot_state(), 0
                else:
                    bad += 1
                model.load_snapshot(raw_state)
            history.append(rec)
            if verbose:
                print(rec, flush=True)
            if val_task is not None and bad >= config.marginal_patience:
                break
    if best_state is not None:
        model.load_snapshot(best_state)
        model.best_val = best_val
    elif ema is not None:
        model.load_snapshot(ema.state())
    model.marginal_iter = best_iter if val_task is not None else int(config.marginal_iters)


def train_flow(snapshots: SnapshotSet, fit_mask, fit_times, config: TrainConfig, val_task: ForecastTask | None = None,
               protocol: EvalProtocol | None = None, verbose: bool = False, keep_snapshots: bool = False):
    """Stage 1 (flow matching on the path, early-stopped on val_task when given), then, if marginal_iters > 0, stage 2
    (_finetune_marginal). The weights at the end of stage 1 are kept in model.stage_states["stage1"];
    keep_snapshots also keeps stage-2 weights at every marginal_eval_every iterations (model.marginal_snapshots)."""
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    gen = torch.Generator().manual_seed(config.seed)
    protocol = EvalProtocol() if protocol is None else protocol
    fit_mask = np.asarray(fit_mask, dtype=bool)
    fit_times = sorted(float(t) for t in fit_times)
    if val_task is not None and (fit_mask & val_task.target_mask).any():
        raise ValueError("validation target cells overlap the fit cells")
    standardizer = fit_on_train(snapshots.x, fit_mask)
    x_std = standardizer.transform(snapshots.x)
    fit = snapshots.subset(fit_mask)
    x_fit = x_std[fit_mask]
    if config.shuffle_biological_time:          # control: destroy temporal order within each condition
        bt = fit.biological_time.copy()
        for cond in fit.conditions():
            idx = np.flatnonzero(fit.condition == cond)
            bt[idx] = rng.permutation(bt[idx])
        fit = SnapshotSet(fit.x, bt, fit.condition, fit.replicate, fit.gene_names, name=fit.name,
                          require_two_timepoints=False)
    transitions = build_transitions(fit, fit_times, x_fit, pair_mode=config.pair_mode)
    paths = (spline_paths(fit, fit_times, x_fit, rng, chain=config.chain, bc_type=config.spline_kind,
                          center=config.path_center)
             if config.path == "spline" else None)
    registry = ConditionRegistry(fit.conditions())
    t_all = list(fit_times) + ([val_task.bio_t1] if val_task is not None else [])
    time_scale = max(max(abs(t) for t in t_all), 1e-6)
    ctx, vf = build_networks(config, registry, snapshots.n_genes, time_scale)
    model = FlowModel(registry, standardizer, ctx, vf, config, snapshots.gene_names, time_scale)
    params = list(ctx.parameters()) + list(vf.parameters())
    opt = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    ema = WeightEMA({"ctx": ctx, "vf": vf}, config.ema_decay) if config.ema_decay > 0 else None
    weights = np.array([min(len(t.x0), len(t.x1)) for t in transitions], dtype=np.float64)
    weights /= weights.sum()
    if paths is not None:
        path_weights = np.array([p.n for _, p in paths], dtype=np.float64)
        path_weights /= path_weights.sum()
    history, best_state, best_val, best_iter, bad = [], None, float("inf"), config.n_iters, 0
    t_start = time.time()
    for it in range(1, config.n_iters + 1):
        if paths is not None:                        # curved multi-marginal paths, autonomous velocity target
            cond, path = paths[rng.choice(len(paths), p=path_weights)]
            b = min(config.batch_size, path.n)
            xs, vs, us = path.sample(rng, b)
            context = model._context(cond, 0.0, 1.0, b)
            loss = spline_loss(vf, torch.as_tensor(xs, dtype=torch.float32), torch.as_tensor(vs, dtype=torch.float32),
                               context, config.sigma_min, time=model.path_time(us))
        else:
            tr = transitions[rng.choice(len(transitions), p=weights)]
            b = min(config.batch_size, len(tr.x0), len(tr.x1))
            if config.coupling == "ot":
                i0, i1 = ot_coupling(tr.x0, tr.x1, rng, max_size=b)
            elif config.coupling == "independent":
                i0, i1 = independent_coupling(len(tr.x0), len(tr.x1), rng, size=b)
            else:
                raise ValueError(f"unknown coupling {config.coupling!r}")
            x0 = torch.as_tensor(tr.x0[i0], dtype=torch.float32)
            x1 = torch.as_tensor(tr.x1[i1], dtype=torch.float32)
            context = model._context(tr.condition, tr.bio_t0, tr.bio_t1, b)
            scale = 1.0 / (tr.bio_t1 - tr.bio_t0) if config.clock == "duration" else 1.0
            loss = cfm_loss(vf, x0, x1, context, sample_flow_time(b, gen), config.sigma_min, target_scale=scale)
        opt.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        opt.step()
        if ema is not None:
            ema.update()
        if it % config.eval_every == 0 or it == config.n_iters:
            rec = {"iter": it, "loss": loss.item(), "seconds": time.time() - t_start}
            if val_task is not None:
                raw_state = model.snapshot_state()
                if ema is not None:
                    model.load_snapshot(ema.state())        # validate, and keep, the averaged weights
                v = _validate(model, snapshots, x_std, val_task, protocol)
                rec["val_energy_distance"] = v
                if v < best_val - 1e-9:
                    best_val, best_iter, best_state, bad = v, it, model.snapshot_state(), 0
                else:
                    bad += 1
                model.load_snapshot(raw_state)
            history.append(rec)
            if verbose:
                print(rec, flush=True)
            if val_task is not None and bad >= config.patience:
                break
    if best_state is not None:
        model.load_snapshot(best_state)
    elif ema is not None:
        model.load_snapshot(ema.state())
    model.best_iter = best_iter
    model.best_val = best_val if val_task is not None else None
    model.stage1_best_val = model.best_val
    model.stage_states["stage1"] = model.snapshot_state()
    if config.marginal_iters > 0:
        _finetune_marginal(model, snapshots, x_std, transitions, paths, config, val_task, protocol, rng, history, verbose,
                           keep_snapshots)
    model.train_seconds = time.time() - t_start
    return model, history
