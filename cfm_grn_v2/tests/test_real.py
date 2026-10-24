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

