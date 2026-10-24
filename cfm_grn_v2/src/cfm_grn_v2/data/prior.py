"""Prior regulatory support (regulator -> target) with signs, and its degree-preserving randomisation."""
from __future__ import annotations

import numpy as np


class GRNPrior:
    """support[j, i] = True if regulator j may regulate target i; signs[j, i] in {-1, 0, +1} (0 = unknown)."""

    def __init__(self, support, signs=None, gene_names=None):
        support = np.asarray(support, dtype=bool)
        if support.ndim != 2 or support.shape[0] != support.shape[1]:
            raise ValueError("support must be a square (genes x genes) boolean matrix")
        if support.diagonal().any():
            raise ValueError("self-regulation is not part of the support")
        if not support.any():
            raise ValueError("the support has no edge")
        self.support = support
        signs = np.zeros(support.shape) if signs is None else np.asarray(signs, dtype=np.float64)
        if signs.shape != support.shape or not np.isin(np.unique(signs), [-1.0, 0.0, 1.0]).all():
            raise ValueError("signs must be a genes x genes matrix with entries in {-1, 0, 1}")
        self.signs = np.where(support, signs, 0.0)
        self.gene_names = None if gene_names is None else list(gene_names)

    @classmethod
    def from_weights(cls, W, gene_names=None) -> "GRNPrior":
        W = np.asarray(W, dtype=np.float64)
        return cls(W != 0, np.sign(W), gene_names)

    @property
    def n_genes(self) -> int:
        return int(self.support.shape[0])

    @property
    def edges(self) -> np.ndarray:
        """(n_edges, 2) rows (regulator j, target i), row-major order."""
        return np.argwhere(self.support)

    @property
    def n_edges(self) -> int:
        return int(self.support.sum())

    @property
    def regulators(self) -> np.ndarray:
        return np.flatnonzero(self.support.any(1))

    def randomized(self, rng: np.random.Generator, n_swaps: int | None = None) -> "GRNPrior":
        """Degree-preserving rewiring (double-edge swaps): every regulator keeps its out-degree, every target
        its in-degree, every edge keeps its sign; the edge set changes. The audit's lesson: a control that
        breaks the biology but not the degrees."""
        edges = [tuple(int(v) for v in e) for e in self.edges]
        signs = {e: float(self.signs[e]) for e in edges}
        present = set(edges)
        n_swaps = 10 * len(edges) if n_swaps is None else int(n_swaps)
        done = 0
        for _ in range(50 * n_swaps):
            if done >= n_swaps:
                break
            a, b = rng.choice(len(edges), 2, replace=False)
            (j1, i1), (j2, i2) = edges[a], edges[b]
            if j1 == j2 or i1 == i2 or j1 == i2 or j2 == i1:
                continue
            if (j1, i2) in present or (j2, i1) in present:
                continue
            present -= {(j1, i1), (j2, i2)}
            present |= {(j1, i2), (j2, i1)}
            s1, s2 = signs.pop((j1, i1)), signs.pop((j2, i2))
            signs[(j1, i2)], signs[(j2, i1)] = s1, s2
            edges[a], edges[b] = (j1, i2), (j2, i1)
            done += 1
        support = np.zeros_like(self.support)
        sgn = np.zeros(self.support.shape)
        for (j, i), s in signs.items():
            support[j, i] = True
            sgn[j, i] = s
        return GRNPrior(support, sgn, self.gene_names)

    def to_dict(self) -> dict:
        return {"support": self.support.astype(np.uint8).tolist(), "signs": self.signs.tolist(),
                "gene_names": self.gene_names}

    @classmethod
    def from_dict(cls, d: dict) -> "GRNPrior":
        return cls(np.asarray(d["support"], dtype=bool), np.asarray(d["signs"], dtype=np.float64), d.get("gene_names"))
