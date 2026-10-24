"""Synthetic gene-regulatory dynamics with KNOWN time-varying edge activity.

Ground truth exists for (i) the signed regulator -> target support W and (ii) the per-edge
activity m_e(t) in [0, 1] as a function of biological_time, so their recovery can be measured
exactly (Phase 3). Cells are sampled destructively: every (time, condition, replicate) cohort
is an independent set of trajectories, hence snapshots are unpaired, like real scRNA-seq.

dx_i = [ clip(basal + sum_j m_ij(t) W_ji h(x_j), 0) - decay x_i ] dt + noise sqrt(x_i) dW
with h(x) = x^n / (K^n + x^n). A knockout clamps x_k = 0 for the whole trajectory.
Switch times lie in [0.25, 0.7] t_max, so the last transitions (validation, test) are autonomous
relaxation after the last switch: forecastable in principle, never seen as a context in training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import SnapshotSet


@dataclass(frozen=True)
class EdgeSchedule:
    kind: str          # "constant" | "switch_on" | "switch_off"
    t_switch: float
    width: float


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TimeVaryingGRNSimulator:
    def __init__(self, n_genes: int = 30, n_regulators: int = 8, density: float = 0.2, frac_dynamic: float = 0.5,
                 frac_repressive: float = 0.3, hill: float = 2.0, decay: float = 0.7, basal: float = 0.3,
                 noise: float = 0.05, dt: float = 0.01, t_max: float = 6.0, seed: int = 0):
        if n_regulators > n_genes:
            raise ValueError("n_regulators cannot exceed n_genes")
        rng = np.random.default_rng(seed)
        self.G, self.R = int(n_genes), int(n_regulators)
        self.hill, self.decay, self.basal, self.noise = float(hill), float(decay), float(basal), float(noise)
        self.dt, self.t_max, self.K = float(dt), float(t_max), 1.0
        W = np.zeros((self.G, self.G))
        for j in range(self.R):
            candidates = [i for i in range(self.G) if i != j]
            targets = [i for i in candidates if rng.random() < density] or [int(rng.choice(candidates))]
            for i in targets:
                sign = -1.0 if rng.random() < frac_repressive else 1.0
                W[j, i] = sign * rng.uniform(0.8, 2.0)
        self.W = W
        self.edges = np.argwhere(W != 0)          # rows (j, i): regulator j -> target i
        self.schedules = []
        for _ in range(len(self.edges)):
            if rng.random() < frac_dynamic:
                kind = "switch_on" if rng.random() < 0.5 else "switch_off"
                self.schedules.append(EdgeSchedule(kind, float(rng.uniform(0.25, 0.7) * self.t_max), 0.05 * self.t_max))
            else:
                self.schedules.append(EdgeSchedule("constant", 0.0, 1.0))
        self.gene_names = [f"g{i}" for i in range(self.G)]
        self.regulators = list(range(self.R))

    def activity(self, t: float) -> np.ndarray:
        out = np.ones(len(self.edges))
        for e, sch in enumerate(self.schedules):
            if sch.kind == "switch_on":
                out[e] = _sigmoid((t - sch.t_switch) / sch.width)
            elif sch.kind == "switch_off":
                out[e] = 1.0 - _sigmoid((t - sch.t_switch) / sch.width)
        return out

    def effective_weights(self, t: float) -> np.ndarray:
        w = np.zeros_like(self.W)
        j, i = self.edges[:, 0], self.edges[:, 1]
        w[j, i] = self.W[j, i] * self.activity(t)
        return w

    def drift(self, x: np.ndarray, t: float, knockout=None) -> np.ndarray:
        xc = np.clip(x, 0.0, None)
        h = xc ** self.hill / (self.K ** self.hill + xc ** self.hill)
        production = np.clip(self.basal + h @ self.effective_weights(t), 0.0, None)
        dx = production - self.decay * x
        if knockout is not None:
            dx[:, knockout] = 0.0
        return dx

    def simulate_cohort(self, n_cells: int, t_end: float, rng: np.random.Generator, knockout=None) -> np.ndarray:
        x = self.basal / self.decay + 0.1 * np.abs(rng.normal(size=(n_cells, self.G)))
        if knockout is not None:
            x[:, knockout] = 0.0
        t = 0.0
        for _ in range(int(round(t_end / self.dt))):
            dx = self.drift(x, t, knockout)
            x = x + dx * self.dt + self.noise * np.sqrt(self.dt) * np.sqrt(np.clip(x, 0.0, None)) * rng.normal(size=x.shape)
            x = np.clip(x, 0.0, None)
            if knockout is not None:
                x[:, knockout] = 0.0
            t += self.dt
        return x.astype(np.float32)

    def snapshot_set(self, times, n_cells_per_cohort: int = 400, replicates=("r1", "r2"), knockouts=(), seed: int = 0):
        times = [float(t) for t in times]
        if any(t < 0.0 or t > self.t_max for t in times):
            raise ValueError(f"times must lie in [0, {self.t_max}]")
        conditions = [("ctrl", None)] + [(f"ko_{self.gene_names[int(k)]}", int(k)) for k in knockouts]
        xs, ts, cs, rs = [], [], [], []
        for ri, rep in enumerate(replicates):
            for ci, (cname, ko) in enumerate(conditions):
                for ti, t in enumerate(times):
                    rng = np.random.default_rng([seed, ri, ci, ti])
                    x = self.simulate_cohort(n_cells_per_cohort, t, rng, ko)
                    xs.append(x)
                    ts.append(np.full(len(x), t))
                    cs.append(np.array([cname] * len(x)))
                    rs.append(np.array([rep] * len(x)))
        snaps = SnapshotSet(np.concatenate(xs), np.concatenate(ts), np.concatenate(cs), np.concatenate(rs),
                            self.gene_names, name="synthetic_grn")
        truth = {"W": self.W.copy(), "edges": self.edges.copy(), "times": np.array(times),
                 "activity": np.stack([self.activity(t) for t in times]),
                 "schedules": [s.__dict__ for s in self.schedules], "regulators": list(self.regulators)}
        return snaps, truth
