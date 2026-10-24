"""Snapshot data schema: unpaired cells observed at biological times under conditions."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """Raised when a SnapshotSet is malformed."""


@dataclass
class SnapshotSet:
    x: np.ndarray                  # [n_cells, n_genes] float32
    biological_time: np.ndarray    # [n_cells] measured hours / days / stages
    condition: np.ndarray          # [n_cells] str, e.g. "ctrl", "ko_GENE", "stim"
    replicate: np.ndarray          # [n_cells] str, biological unit used for splits
    gene_names: list
    name: str = "unnamed"
    require_two_timepoints: bool = True

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32)
        if self.x.ndim != 2:
            raise SchemaError(f"x must be 2-D [cells, genes], got shape {self.x.shape}")
        n = self.x.shape[0]
        self.biological_time = np.asarray(self.biological_time, dtype=np.float64).reshape(-1)
        self.condition = np.asarray(self.condition).astype(str).reshape(-1)
        self.replicate = np.asarray(self.replicate).astype(str).reshape(-1)
        self.gene_names = [str(g) for g in self.gene_names]
        for field_name, arr in (("biological_time", self.biological_time), ("condition", self.condition),
                                ("replicate", self.replicate)):
            if arr.shape[0] != n:
                raise SchemaError(f"{field_name} has {arr.shape[0]} entries, x has {n} cells")
        if len(self.gene_names) != self.x.shape[1]:
            raise SchemaError(f"{len(self.gene_names)} gene names for {self.x.shape[1]} columns")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise SchemaError("gene names must be unique")
        if not np.isfinite(self.x).all():
            raise SchemaError("x contains NaN or inf")
        if not np.isfinite(self.biological_time).all():
            raise SchemaError("biological_time contains NaN or inf")
        if self.require_two_timepoints and len(self.timepoints()) < 2:
            raise SchemaError("a SnapshotSet needs at least two biological timepoints")

    @property
    def n_cells(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.x.shape[1])

    def timepoints(self) -> np.ndarray:
        return np.unique(self.biological_time)

    def conditions(self) -> list:
        return sorted(set(self.condition.tolist()))

    def replicates(self) -> list:
        return sorted(set(self.replicate.tolist()))

    def mask(self, times=None, conditions=None, replicates=None) -> np.ndarray:
        m = np.ones(self.n_cells, dtype=bool)
        if times is not None:
            t = np.asarray(list(times), dtype=np.float64)
            m &= np.isclose(self.biological_time[:, None], t[None, :]).any(axis=1)
        if conditions is not None:
            m &= np.isin(self.condition, [str(c) for c in conditions])
        if replicates is not None:
            m &= np.isin(self.replicate, [str(r) for r in replicates])
        return m

    def subset(self, mask: np.ndarray) -> "SnapshotSet":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n_cells,):
            raise SchemaError("mask must have one boolean entry per cell")
        return replace(self, x=self.x[mask], biological_time=self.biological_time[mask],
                       condition=self.condition[mask], replicate=self.replicate[mask],
                       require_two_timepoints=False)

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame({"biological_time": self.biological_time, "condition": self.condition,
                           "replicate": self.replicate})
        out = df.value_counts().rename("n_cells").reset_index()
        return out.sort_values(["condition", "replicate", "biological_time"]).reset_index(drop=True)
