"""Shared preparation for the Chronocell erythroid baseline and the TACK step-0 adapter probe.

Gene selection and read depth follow FGP_2024/notebooks/Erythoid.ipynb (Fang, Gorin, Pachter 2025,
commit 907ba1f). Chronocell itself (commit a827f89) is used only as an external baseline engine.

Deviation: the notebook's data/cell_cycle_genes.txt is not in the repository; the Tirosh et al. 2016
S + G2/M list (97 genes, mouse capitalisation) is used instead. The number of genes it removes is
reported next to the notebook's 6.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pathlib
import sys

import numpy as np

TACK = pathlib.Path.home() / "tack"
sys.path.insert(0, str(TACK / "ext" / "Chronocell"))

STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
STAGE_HOURS = {s: (float(s[1:]) - 7.0) * 24.0 for s in STAGES}
CELL_TYPES = ["Blood progenitors 1", "Blood progenitors 2", "Erythroid1", "Erythroid2", "Erythroid3"]
TOPO = np.array([[0, 1, 2]])
TAU = (0, 1, 2)
M_GRID = 100

TIROSH_S = """Mcm5 Pcna Tyms Fen1 Mcm2 Mcm4 Rrm1 Ung Gins2 Mcm6 Cdca7 Dtl Prim1 Uhrf1 Mlf1ip Cenpu Hells
Rfc2 Rpa2 Nasp Rad51ap1 Gmnn Wdr76 Slbp Ccne2 Ubr7 Pold3 Msh2 Atad2 Rad51 Rrm2 Cdc45 Cdc6 Exo1 Tipin
Dscc1 Blm Casp8ap2 Usp1 Clspn Pola1 Chaf1b Brip1 E2f8""".split()
TIROSH_G2M = """Hmgb2 Cdk1 Nusap1 Ube2c Birc5 Tpx2 Top2a Ndc80 Cks2 Nuf2 Cks1b Mki67 Tmpo Cenpf Tacc3 Fam64a
Pimreg Smc4 Ccnb2 Ckap2l Ckap2 Aurkb Bub1 Kif11 Anp32e Tubb4b Gtse1 Kif20b Hjurp Cdca3 Hn1 Jpt1 Cdc20
Ttk Cdc25c Kif2c Rangap1 Ncapd2 Dlgap5 Cdca2 Cdca8 Ect2 Kif23 Hmmr Aurka Psrc1 Anln Lbr Ckap5 Cenpe
Ctcf Nek2 G2e3 Gas2l3 Cbx5 Cenpa""".split()
CELL_CYCLE = set(TIROSH_S + TIROSH_G2M)


def load_erythroid(path=None):
    import anndata

    path = path or TACK / "data" / "erythroid" / "erythroid_lineage.h5ad"
    return anndata.read_h5ad(path)


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def normalized_cov_offdiag(Xmat):
    """(sum_{j!=k} cov_jk / (m_j m_k)) / (p (p-1)), the notebook's estimate, computed in O(n p).

    With Y_ij = X_ij / m_j: sum_{j,k} cov(Y_j, Y_k) = var(sum_j Y_j); the diagonal is sum_j var(Y_j).
    ddof=1 matches np.cov.
    """
    m = Xmat.mean(0)
    Y = Xmat / m[None, :]
    p = Xmat.shape[1]
    total = np.var(Y.sum(1), ddof=1)
    diag = np.var(Y, axis=0, ddof=1).sum()
    return (total - diag) / ((p - 1) * p)


def poissonian_depth(U, S, n_iter=5, a=1.0):
    """Notebook cells 21-24: extra-Poisson CV^2 'sp' and read depth from near-Poissonian genes."""
    keep = (S.mean(0) > 0) & (U.mean(0) > 0)
    Xs = (S + U)[:, keep].astype(np.float64)
    sp = normalized_cov_offdiag(Xs)
    x_mean = Xs.mean(0)
    x_var = Xs.var(0)
    for _ in range(n_iter):
        pmask = (x_mean > 0.01) & (x_var / (a * x_mean + sp * x_mean**2) < 1.2)
        sp = normalized_cov_offdiag(Xs[:, pmask])
    depth_raw = Xs[:, pmask].mean(1)
    return sp, depth_raw, int(pmask.sum())


def select_genes(adata, sp, a=1.0, var_threshold=1.2):
    """Notebook cell 26 (+ cell 27 with the Tirosh list)."""
    U = adata.layers["unspliced"]
    S = adata.layers["spliced"]

    def mean_var(Z):
        mu = np.asarray(Z.mean(0)).ravel()
        z2 = Z.multiply(Z) if hasattr(Z, "multiply") else Z**2
        ex2 = np.asarray(z2.mean(0)).ravel()
        return mu, ex2 - mu**2

    u_mean, u_var = mean_var(U)
    s_mean, s_var = mean_var(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = (
            (u_mean > 0.02)
            & (s_mean > 0.1)
            & (u_var / (a * u_mean + sp * u_mean**2) > var_threshold)
            & (s_var / (a * s_mean + sp * s_mean**2) > var_threshold)
            & (np.abs(np.log(s_mean / u_mean)) < 4)
        )
    genes = np.asarray(adata.var_names[idx])
    n_before = len(genes)
    genes = np.array([g for g in genes if g not in CELL_CYCLE])
    return genes, n_before


def prepare(out_npz=None):
    adata = load_erythroid()
    U_all = adata.layers["unspliced"]
    S_all = adata.layers["spliced"]
    keep = (np.asarray(S_all.mean(0)).ravel() > 0) & (np.asarray(U_all.mean(0)).ravel() > 0)
    U = _dense(U_all[:, keep]).astype(np.float32)
    S = _dense(S_all[:, keep]).astype(np.float32)
    sp, depth_raw, n_pois = poissonian_depth(U, S)
    del U, S
    genes, n_before_cc = select_genes(adata, sp)
    gidx = np.array([np.where(adata.var_names == g)[0][0] for g in genes])
    X = np.zeros((adata.n_obs, len(genes), 2))
    X[:, :, 0] = _dense(U_all[:, gidx])
    X[:, :, 1] = _dense(S_all[:, gidx])
    stage = np.asarray(adata.obs["stage"]).astype(str)
    celltype = np.asarray(adata.obs["celltype"]).astype(str)
    batch = np.asarray(adata.obs["sequencing.batch"]).astype(str)
    info = dict(
        n_cells=int(adata.n_obs),
        n_genes_total=int(adata.n_vars),
        sp=float(sp),
        n_poissonian_genes=n_pois,
        n_genes_before_cc=int(n_before_cc),
        n_cc_removed=int(n_before_cc - len(genes)),
        n_genes=int(len(genes)),
        notebook_reference=dict(n_genes_before_cc=57, n_genes_after_cc=51, sp_final=0.11504),
    )
    if out_npz:
        np.savez_compressed(out_npz, X=X, depth_raw=depth_raw, genes=genes, stage=stage,
                            celltype=celltype, batch=batch)
    return X, depth_raw, genes, stage, celltype, batch, info


def load_prep(npz):
    d = np.load(npz, allow_pickle=True)
    return d["X"], d["depth_raw"], d["genes"], d["stage"], d["celltype"], d["batch"]


def celltype_warm_q(celltype, m=M_GRID):
    """Notebook cell 43: celltype k gets uniform weight over the k-th fifth of the process-time grid."""
    w = np.zeros((len(celltype), m))
    for i, ct in enumerate(CELL_TYPES):
        sel = celltype == ct
        w[sel, int(i * m / len(CELL_TYPES)):int((i + 1) * m / len(CELL_TYPES))] += 1
    w = w / w.sum(1, keepdims=True)
    return w[:, None, :]
