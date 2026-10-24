# RA-FM Fase 2 — dati reali (piano, 2026-09-18)

> Esecuzione inline (skill `executing-plans`); i blocchi `# file:` sono la sorgente di verità e vengono materializzati con `python scripts/materialize_plan.py <questo piano> --root .`; le modifiche a file di Fase 1 passano da script di patch sul piano di Fase 1. Versione pacchetto: 0.1.8 (loader), poi 0.2.x con i gate.

**Obiettivo:** consolidare il passo 1 (ricostruzione di un time point interno nascosto) su dati reali, con scelte che NON chiudono i passi 2 (attività degli archi ancorata alla cromatina) e 3 (interventi): geni = HVG ∪ fattori di trascrizione del prior, tempo = stadio reale, repliche e condizioni conservate, valutazione nello spazio dei geni.

**Vincoli (dal committente, 2026-09-18):** niente condizioni "comode" che bloccano gli step successivi; provare più modi per costruire la rete prior (CollecTRI/DoRothEA firmati; motivi+ATAC tipo SCENIC+/CellOracle; GRNBoost2 solo come confronto).

## A.1 Dataset (schede in `cfm_grn_v2/data_cards/`)

| dataset | perché | serve ai passi | file | dimensione |
|---|---|---|---|---|
| Argelaguet et al. 2022, gastrulazione topo, 10x Multiome (GEO GSE205117; processati nella cartella Dropbox `Argelaguet2022_mouse_organogenesis` di R. Argelaguet) | 5 stadi E7.5, E7.75, E8.0, E8.5, E8.75 con repliche biologiche (2 per stadio tranne E7.75), RNA e cromatina sulle stesse cellule, più un knockout CRISPR di T (Brachyury) a E8.5 con controllo WT | 1 (serie temporale con repliche), 2 (ATAC per ancorare i gate), 3 (KO di un TF) | `data/processed/rna/anndata.h5ad` (2.12 GB), `data/processed/rna/metadata.txt.gz` (1.68 MB), `data/sample_metadata.txt.gz` (3.68 MB); per il passo 2 `data/processed/atac/anndata/PeakMatrix_anndata.h5ad` (1.53 GB) | ~3.7 GB |
| Moon et al. 2019, corpi embrioidi umani (Mendeley `v6n743h5ng`, `scRNAseq.zip`) | benchmark standard della letteratura per il giorno nascosto (TrajectoryNet, MIOFlow, MMFM): confronto diretto con i metodi pubblicati; 5 finestre temporali (giorni 0–3, 6–9, 12–15, 18–21, 24–27) | 1 | `scRNAseq.zip` (161.7 MB) | 0.16 GB |
| RENGE, GSE213069 (KO temporizzati in iPSC umane) | passo 3, più avanti | 3 | da auditare | — |

Archiviazione: `C:\Users\Gabriele\data\rafm\` (fuori da Google Drive) o beegfs su Bocconi. Prior: `C:\Users\Gabriele\data\rafm\priors\collectri_mouse.tsv` (13.223 archi firmati, 733 TF, da OmniPath); la versione umana scaricata è troncata (4.342 archi): da riscaricare.

## A.2 Protocollo (uguale alla Fase 1, v0.1.7)

- Task primario: interpolazione leave-one-inner-timepoint-out (Argelaguet: stadio escluso E8.0 → predetto da E7.75; validazione annidata su E7.75/E8.5 non escluso... con 5 stadi la validazione ha un solo fold interno per lato: `interpolation_val_folds`). Controlli: persistenza, interpolazione delle medie, interpolante OT, lineare/MLP, replay, tempo rimescolato.
- Unità di split: stadi interi; repliche come gruppi separati nel coupling e nelle transizioni (mai mescolate).
- Preprocessing solo sulle cellule di fit: normalizzazione per cellula + log1p, HVG (dispersione normalizzata a bin di media) ∪ TF del prior presenti, standardizzazione. Nessuna PCA nel task primario; PCA-50 congelata solo come controllo di robustezza.
- Sottocampionamento per gruppo (max cellule per stadio×replica) solo per il calcolo, dichiarato nel report.
- Gate G2 (dal piano di Fase 1): FM batte i controlli con IC95 sul time point interno escluso e batte il tempo rimescolato; risultati stabili in geni e in PCA-50. Fallback: crescita/sbilanciamento, spazio latente, coupling sbilanciato, σ.

## A.3 Prior della rete (passo 2, preparato ora)

1. CollecTRI (OmniPath): archi TF→bersaglio firmati e curati; `load_collectri(path, gene_names)` → `GRNPrior`.
2. Alternativa data-driven: motivi + picchi ATAC del multiome (SCENIC+/CellOracle-like) — Fase 3.
3. GRNBoost2 solo come confronto (non firmato, bias di grado noto).
Copertura richiesta nella scheda dati: numero di TF del prior espressi, archi entro i geni selezionati.

## Parte B — file

- `cfm_grn_v2/src/cfm_grn_v2/data/prior.py` (nuovo, anticipato dalla spec di Fase 3): `GRNPrior`.
- `cfm_grn_v2/src/cfm_grn_v2/data/real.py` (nuovo): normalizzazione, selezione geni, `SnapshotSet` da AnnData, `load_collectri`, `qc_table`.
- `cfm_grn_v2/tests/test_prior.py`, `cfm_grn_v2/tests/test_real.py` (nuovi).
- `cfm_grn_v2/data_cards/argelaguet2022_gastrulation.md`, `cfm_grn_v2/data_cards/moon2019_embryoid_body.md` (schede, completate dopo lo scaricamento).

## Parte C — Task

### Task 1: prior firmato con randomizzazione che conserva i gradi

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/prior.py
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
```

```python
# file: cfm_grn_v2/tests/test_prior.py
import numpy as np
import pytest

from cfm_grn_v2.data.prior import GRNPrior


def _prior():
    rng = np.random.default_rng(0)
    W = np.zeros((12, 12))
    for j in range(4):
        for i in rng.choice(np.arange(4, 12), 4, replace=False):
            W[j, i] = rng.choice([-1.5, 2.0])
    return GRNPrior.from_weights(W, [f"g{i}" for i in range(12)]), W


def test_prior_from_weights_and_edges():
    p, W = _prior()
    assert p.n_genes == 12 and p.n_edges == 16 and p.regulators.tolist() == [0, 1, 2, 3]
    e = p.edges
    assert np.all(W[e[:, 0], e[:, 1]] != 0) and np.all(p.signs[e[:, 0], e[:, 1]] == np.sign(W[e[:, 0], e[:, 1]]))
    assert GRNPrior.from_dict(p.to_dict()).n_edges == 16
    with pytest.raises(ValueError):
        GRNPrior(np.eye(3, dtype=bool))
    with pytest.raises(ValueError):
        GRNPrior(np.zeros((3, 3), dtype=bool))


def test_randomized_prior_preserves_degrees_and_signs_but_not_edges():
    p, _ = _prior()
    q = p.randomized(np.random.default_rng(1))
    assert q.n_edges == p.n_edges
    assert np.array_equal(q.support.sum(1), p.support.sum(1)) and np.array_equal(q.support.sum(0), p.support.sum(0))
    assert sorted(q.signs[q.support].tolist()) == sorted(p.signs[p.support].tolist())
    assert (q.support != p.support).sum() > 0 and not q.support.diagonal().any()
```

### Task 2: dati reali → `SnapshotSet`, geni = HVG ∪ TF del prior, prior CollecTRI

```python
# file: cfm_grn_v2/src/cfm_grn_v2/data/real.py
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
```

```python
# file: cfm_grn_v2/tests/test_real.py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cfm_grn_v2.data.real import (highly_variable_genes, load_collectri, normalize_log1p, qc_table, select_genes,
                                  snapshot_set_from_arrays)


def _counts(n_cells=900, n_genes=60, seed=0):
    rng = np.random.default_rng(seed)
    rates = np.exp(rng.normal(0.0, 1.0, size=n_genes))
    rates[:5] *= 20.0                                        # five highly variable, highly expressed genes
    X = rng.poisson(rates[None, :] * rng.uniform(0.5, 2.0, size=(n_cells, 1)))
    X[:, 5:10] = rng.poisson(rates[None, 5:10] * (1 + 10 * (np.arange(n_cells) % 3 == 0))[:, None])
    obs = pd.DataFrame({"stage": np.repeat(["E7.5", "E8.0", "E8.5"], n_cells // 3),
                        "sample": np.tile(np.repeat(["rep1", "rep2"], n_cells // 6), 3),
                        "genotype": ["WT"] * n_cells})
    return X, obs, [f"g{i}" for i in range(n_genes)]


def test_normalize_and_hvg_and_gene_selection():
    X, obs, genes = _counts()
    Xn = normalize_log1p(X)
    assert Xn.shape == X.shape and Xn.dtype == np.float32 and np.isfinite(Xn).all()
    hv = highly_variable_genes(Xn, n_top=10)
    assert len(hv) == 10 and set(range(5, 10)) <= set(hv.tolist())          # the bursty genes are variable
    sel = select_genes(Xn, genes, n_hvg=10, must_keep=("g50", "g51", "not_a_gene"))
    assert set(hv.tolist()) <= set(sel.tolist()) and {50, 51} <= set(sel.tolist()) and len(sel) <= 12


def test_snapshot_set_from_arrays_groups_and_subsampling():
    X, obs, genes = _counts()
    Xn = normalize_log1p(X)
    tmap = {"E7.5": 7.5, "E8.0": 8.0, "E8.5": 8.5}
    s = snapshot_set_from_arrays(Xn, obs, genes, "stage", "sample", "genotype", tmap, gene_idx=[0, 1, 2],
                                 max_cells_per_group=100, min_cells=50, rng=np.random.default_rng(0))
    assert s.timepoints().tolist() == [7.5, 8.0, 8.5] and s.replicates() == ["rep1", "rep2"] and s.conditions() == ["WT"]
    assert s.n_cells == 600 and s.n_genes == 3 and s.gene_names == ["g0", "g1", "g2"]
    q = qc_table(obs, "stage", "sample", "genotype")
    assert q["n_cells"].sum() == 900 and len(q) == 6
    with pytest.raises(ValueError):
        snapshot_set_from_arrays(Xn, obs, genes, "stage", "sample", "genotype", tmap, min_cells=10_000)


def test_load_collectri_signs_and_restriction(tmp_path):
    tsv = ("source\ttarget\tsource_genesymbol\ttarget_genesymbol\tis_directed\tis_stimulation\tis_inhibition\n"
           "a\tb\tTfA\tGene1\tTrue\tTrue\tFalse\n"
           "a\tc\tTfA\tGene2\tTrue\tFalse\tTrue\n"
           "d\te\tTfB\tGene1\tTrue\tTrue\tTrue\n"
           "d\tf\tTfB\tMissing\tTrue\tTrue\tFalse\n"
           "a\ta\tTfA\tTfA\tTrue\tTrue\tFalse\n")
    p = tmp_path / "collectri.tsv"
    p.write_text(tsv, encoding="utf-8")
    prior = load_collectri(p, ["Gene1", "TfA", "Gene2", "TfB"])
    j_a, j_b, g1, g2 = 1, 3, 0, 2
    assert prior.n_edges == 3 and prior.signs[j_a, g1] == 1.0 and prior.signs[j_a, g2] == -1.0 and prior.signs[j_b, g1] == 0.0
    assert prior.regulators.tolist() == [1, 3]
    with pytest.raises(ValueError):
        load_collectri(p, ["x", "y"])


def test_sparse_path_matches_dense_path():
    X, obs, genes = _counts()
    Xs = sp.csr_matrix(X)
    dense = normalize_log1p(X)
    sparse = normalize_log1p(Xs, keep_sparse=True)
    assert sp.issparse(sparse) and sparse.dtype == np.float32
    assert np.allclose(sparse.toarray(), dense, atol=1e-6)
    assert np.array_equal(highly_variable_genes(sparse, n_top=10), highly_variable_genes(dense, n_top=10))
    assert np.array_equal(select_genes(sparse, genes, n_hvg=10, must_keep=("g50",)),
                          select_genes(dense, genes, n_hvg=10, must_keep=("g50",)))
    tmap = {"E7.5": 7.5, "E8.0": 8.0, "E8.5": 8.5}
    kw = dict(time_key="stage", replicate_key="sample", condition_key="genotype", time_map=tmap,
              gene_idx=[0, 1, 2], max_cells_per_group=100, min_cells=50)
    a = snapshot_set_from_arrays(sparse, obs, genes, rng=np.random.default_rng(0), **kw)
    b = snapshot_set_from_arrays(dense, obs, genes, rng=np.random.default_rng(0), **kw)
    assert np.allclose(a.x, b.x, atol=1e-6) and a.gene_names == b.gene_names and a.n_cells == b.n_cells


def test_select_genes_min_detect_excludes_rarely_detected_genes():
    """min_detect: a gene detected in fewer than that fraction of the fit cells is a candidate neither for the
    variable genes nor for the must-keep list; min_detect=0 keeps the old rule (detected in one cell)."""
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(1000, 30)).astype(float)
    X[:, 0] = 0.0
    X[:3, 0] = 50.0                                          # detected in 0.3% of the cells
    X[:, 1] = 0.0
    X[:5, 1] = 20.0                                          # a regulator detected in 0.5% of the cells
    Xn = normalize_log1p(X)
    names = [f"g{i}" for i in range(30)]
    old = select_genes(Xn, names, n_hvg=30, must_keep={"g1"})
    new = select_genes(Xn, names, n_hvg=30, must_keep={"g1"}, min_detect=0.01)
    assert {0, 1} <= set(old.tolist()) and len(old) == 30
    assert not ({0, 1} & set(new.tolist())) and len(new) == 28
    assert 1 in select_genes(Xn, names, n_hvg=0, must_keep={"g1"})
    assert 1 not in select_genes(Xn, names, n_hvg=0, must_keep={"g1"}, min_detect=0.01)
    assert np.array_equal(new, select_genes(sp.csr_matrix(Xn), names, n_hvg=30, must_keep={"g1"}, min_detect=0.01))
```

### Task 3: schede dati (dopo lo scaricamento) e run del task di interpolazione sul dataset Argelaguet

- [ ] Scaricare i file (con il via libera del committente) in `C:\Users\Gabriele\data\rafm\argelaguet2022\`, checksum e dimensioni nella scheda.
- [ ] `scripts_dev/audit_argelaguet.py`: tabella cellule per stadio × replica × genotipo, geni, copertura del prior (TF di CollecTRI espressi, archi entro HVG ∪ TF).
- [ ] `experiments/run_real.py`: stesso protocollo di `run_synthetic` (interpolazione, due fold, controlli, report) con `--dataset argelaguet --hold E8.0`.
- [ ] Verdetto G2 e voce in `docs/superpowers/decisions.md`.
