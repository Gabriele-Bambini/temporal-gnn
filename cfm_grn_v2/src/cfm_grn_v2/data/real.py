"""Real single-cell time series -> SnapshotSet, with choices that keep the later phases open:
genes = highly variable genes UNION the prior's transcription factors, biological time = the real stage,
replicates and conditions preserved, no PCA (the network phase needs genes)."""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .prior import GRNPrior
from .schema import SnapshotSet


def normalize_log1p(X, target_sum: float = 1e4, keep_sparse: bool = False):
    """Per-cell library-size normalisation to target_sum followed by log1p.

    Returns dense float32 by default. With keep_sparse and a sparse input the result stays CSR: log1p
    fixes zero, so the sparsity pattern is unchanged and a real count matrix never has to be densified.
    The operation is per cell, so it cannot move information between cells or between timepoints.
    """
    if sp.issparse(X):
        Y = sp.csr_matrix(X, dtype=np.float64, copy=True)
        lib = np.asarray(Y.sum(1)).ravel()
        lib[lib == 0] = 1.0
        Y.data *= np.repeat(target_sum / lib, np.diff(Y.indptr))
        Y.data = np.log1p(Y.data)
        Y = Y.astype(np.float32)
        return Y if keep_sparse else Y.toarray()
    X = np.asarray(X, dtype=np.float64)
    lib = X.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * target_sum).astype(np.float32)


def highly_variable_genes(X_norm, n_top: int = 2000, n_bins: int = 20, min_mean: float = 1e-3) -> np.ndarray:
    """Indices of the n_top genes with the highest normalised dispersion (var/mean of the expm1 values,
    z-scored within bins of mean expression): the classic single-cell recipe, fit cells only.
    Accepts a dense array or a CSR matrix; the sparse path never densifies."""
    if sp.issparse(X_norm):
        Y = sp.csr_matrix(X_norm, dtype=np.float64, copy=True)
        Y.data = np.expm1(Y.data)
        n = Y.shape[0]
        mean = np.asarray(Y.sum(0)).ravel() / n
        var = np.asarray(Y.multiply(Y).sum(0)).ravel() / n - mean ** 2
        var = np.maximum(var, 0.0)
    else:
        Y = np.expm1(np.asarray(X_norm, dtype=np.float64))
        mean, var = Y.mean(0), Y.var(0)
    ok = mean > min_mean
    disp = np.zeros_like(mean)
    disp[ok] = var[ok] / mean[ok]
    score = np.full(mean.shape, -np.inf)
    if ok.sum() == 0:
        return np.array([], dtype=int)
    bins = pd.qcut(np.log1p(mean[ok]), q=min(n_bins, max(int(ok.sum()) // 2, 1)), labels=False, duplicates="drop")
    d_ok = np.log1p(disp[ok])
    z = np.zeros_like(d_ok)
    for b in np.unique(bins):
        m = bins == b
        s = d_ok[m].std()
        z[m] = (d_ok[m] - d_ok[m].mean()) / (s if s > 0 else 1.0)
    score[ok] = z
    order = np.argsort(-score)
    return np.sort(order[:min(n_top, int(ok.sum()))])


def detection_rate(X) -> np.ndarray:
    """Fraction of the cells (rows) in which each gene (column) is detected, i.e. non-zero."""
    if sp.issparse(X):
        X = sp.csr_matrix(X)
        return np.bincount(X.indices[X.data != 0], minlength=X.shape[1]) / X.shape[0]
    return (np.asarray(X) != 0).mean(0)


def select_genes(X_norm_fit, gene_names, n_hvg: int = 2000, must_keep=(), min_detect: float = 0.0) -> np.ndarray:
    """HVG on the fit cells, union the genes in must_keep (e.g. the prior's regulators); returns sorted gene indices.

    Only genes detected in at least a fraction min_detect of the fit cells are candidates for either list; with
    the default 0 a gene qualifies if it is detected in one fit cell. A gene seen in a handful of cells has a high
    dispersion score and, once standardised, is a spike (one cell hundreds of units away, every other cell at
    zero); min_detect=0.01 is the SCENIC default (Aibar et al. 2017) and removes them."""
    names = np.asarray(gene_names)
    rate = detection_rate(X_norm_fit)
    if min_detect > 0:
        cand = np.flatnonzero(rate >= min_detect)
        hvg = {int(cand[i]) for i in highly_variable_genes(X_norm_fit[:, cand], n_hvg)}
    else:
        cand = np.flatnonzero(rate > 0)
        hvg = set(highly_variable_genes(X_norm_fit, n_hvg).tolist())
    keep = set(must_keep)
    hvg |= {int(i) for i in cand if names[i] in keep}
    return np.array(sorted(hvg), dtype=int)


def qc_table(obs: pd.DataFrame, time_key: str, replicate_key: str, condition_key: str | None = None) -> pd.DataFrame:
    """Cells per (condition, replicate, time) — the data-card table."""
    keys = [k for k in (condition_key, replicate_key, time_key) if k is not None]
    return obs.groupby(keys, observed=True).size().rename("n_cells").reset_index()


def snapshot_set_from_arrays(X_norm: np.ndarray, obs: pd.DataFrame, gene_names, time_key: str, replicate_key: str,
                             condition_key: str | None = None, time_map: dict | None = None, gene_idx=None,
                             max_cells_per_group: int | None = None, min_cells: int = 200, rng=None,
                             name: str = "real") -> SnapshotSet:
    """Build the SnapshotSet: biological_time = time_map[obs[time_key]] (or the numeric value), groups with
    fewer than min_cells cells are dropped (documented in the data card), groups larger than
    max_cells_per_group are subsampled (compute only)."""
    rng = np.random.default_rng(0) if rng is None else rng
    obs = obs.reset_index(drop=True)
    t_raw = obs[time_key].astype(str) if time_map is not None else obs[time_key]
    bio_t = np.array([float(time_map[v]) for v in t_raw]) if time_map is not None else obs[time_key].to_numpy(dtype=float)
    rep = obs[replicate_key].astype(str).to_numpy()
    cond = obs[condition_key].astype(str).to_numpy() if condition_key is not None else np.array(["ctrl"] * len(obs))
    keep = np.zeros(len(obs), dtype=bool)
    df = pd.DataFrame({"t": bio_t, "r": rep, "c": cond})
    for _, idx in df.groupby(["c", "r", "t"], sort=False).indices.items():
        idx = np.asarray(idx)
        if len(idx) < min_cells:
            continue
        if max_cells_per_group is not None and len(idx) > max_cells_per_group:
            idx = rng.choice(idx, max_cells_per_group, replace=False)
        keep[idx] = True
    if not keep.any():
        raise ValueError("no (condition, replicate, time) group reaches min_cells")
    names = list(gene_names)
    if sp.issparse(X_norm):
        X = sp.csr_matrix(X_norm)[keep]
        if gene_idx is not None:
            gene_idx = np.asarray(gene_idx, dtype=int)
            X, names = X[:, gene_idx], [names[i] for i in gene_idx]
        X = X.toarray()                                   # dense only for the genes that were selected
    else:
        X = np.asarray(X_norm)[keep]
        if gene_idx is not None:
            gene_idx = np.asarray(gene_idx, dtype=int)
            X, names = X[:, gene_idx], [names[i] for i in gene_idx]
    return SnapshotSet(X.astype(np.float32), bio_t[keep], cond[keep], rep[keep], names, name=name)


def load_collectri(path, gene_names, min_curation: int = 0) -> GRNPrior:
    """CollecTRI TSV from OmniPath (columns source_genesymbol, target_genesymbol, is_stimulation,
    is_inhibition, ...) restricted to gene_names; sign +1 stimulation, -1 inhibition, 0 both/unknown."""
    path = Path(path)
    raw = gzip.open(path, "rt", encoding="utf-8", errors="replace").read() if path.suffix == ".gz" else \
        io.open(path, encoding="utf-8", errors="replace").read()
    df = pd.read_csv(io.StringIO(raw), sep="\t")
    need = {"source_genesymbol", "target_genesymbol", "is_stimulation", "is_inhibition"}
    if not need <= set(df.columns):
        raise ValueError(f"CollecTRI table lacks columns {sorted(need - set(df.columns))}")
    if "curation_effort" in df.columns and min_curation > 0:
        df = df[df["curation_effort"] >= min_curation]
    names = list(gene_names)
    pos = {g: i for i, g in enumerate(names)}
    G = len(names)
    support = np.zeros((G, G), dtype=bool)
    signs = np.zeros((G, G))
    for s, t, st, ih in zip(df["source_genesymbol"], df["target_genesymbol"], df["is_stimulation"], df["is_inhibition"]):
        if s in pos and t in pos and s != t:
            j, i = pos[s], pos[t]
            support[j, i] = True
            st, ih = str(st).lower() == "true", str(ih).lower() == "true"
            signs[j, i] = 1.0 if st and not ih else (-1.0 if ih and not st else 0.0)
    if not support.any():
        raise ValueError("no CollecTRI edge falls within the given genes")
    return GRNPrior(support, signs, names)
