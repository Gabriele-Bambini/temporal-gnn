"""Strict per-origin data products for forecasting the erythroid lineage over REAL embryonic stages.

Cells: scVelo gastrulation_erythroid (~/tack/data/erythroid/erythroid_lineage.h5ad, 9815 cells, spliced/unspliced,
obs 'stage' E7.0 .. E8.5 = 7 real time points). Origin s (s = 1..5) forecasts stage s+1 from stages 0..s. For every
origin, everything is computed from the PAST cells only (stage <= s):
  gene filter  min_shared_counts = 20 on past cells (scVelo's rule: counts of each layer in cells where both > 0);
  normalisation per cell by its own spliced (unspliced) total over all genes, scaled to the past median total, log1p;
  2000 highly variable genes (scanpy, seurat flavour) on past log expression; PCA (30) fitted on past cells;
  30-nn graph on past PCA; first moments Ms, Mu (scVelo moments on that graph);
  velocity: (a) our own steady-state model v = Mu - gamma Ms, gamma per gene by least squares through the origin on
  the cells in the 5 % / 95 % extremes of (Ms/max + Mu/max) (original code); (b) scVelo dynamical model (library,
  used as an external baseline only);
  proliferation score per cell from the Tirosh 2016 S + G2M genes shipped with scVelo (mean centred log
  expression, past centring).
The stage s+1 cells are only TRANSFORMED with the origin's rules (gene set, normalisation constant, PCA) and saved
apart, for evaluation; they never enter a fit. Output: ~/tack/out/ery/origin<s>.npz + meta.json.
"""
import json
import os
import time

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import scvelo as scv

H5 = os.path.expanduser("~/tack/data/erythroid/erythroid_lineage.h5ad")
OUT = os.path.expanduser("~/tack/out/ery")
STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
DAYS = [7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5]
N_HVG, N_PC, N_NB = 2000, 30, 30


def shared_filter(S, U, min_shared=20):
    nz = (S > 0).multiply(U > 0)
    cs = np.asarray(S.multiply(nz).sum(0)).ravel()
    cu = np.asarray(U.multiply(nz).sum(0)).ravel()
    return (cs >= min_shared) & (cu >= min_shared)


def normalise(M, size, target):
    return sp.diags(target / np.maximum(size, 1.0)) @ M


def steady_state_velocity(Ms, Mu, perc=5):
    """gamma_g from u ~ gamma s through the origin on extreme cells; v = Mu - gamma Ms."""
    s_n = Ms / np.maximum(Ms.max(0), 1e-12)
    u_n = Mu / np.maximum(Mu.max(0), 1e-12)
    w = s_n + u_n
    lo, hi = np.percentile(w, perc, axis=0), np.percentile(w, 100 - perc, axis=0)
    sel = (w <= lo) | (w >= hi)
    num = (Mu * Ms * sel).sum(0)
    den = (Ms * Ms * sel).sum(0)
    gamma = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return Mu - gamma * Ms, gamma


def main():
    os.makedirs(OUT, exist_ok=True)
    a = ad.read_h5ad(H5)
    stage = np.array([STAGES.index(s) for s in a.obs["stage"].astype(str)])
    # Tirosh et al. 2016 S / G2M markers as shipped with scVelo (the local Regev-lab file is a failed download)
    import importlib
    probe = ad.AnnData(X=np.zeros((2, a.n_vars), np.float32))
    probe.var_names = a.var_names
    s_genes, g2m_genes = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc_genes = set(str(g).capitalize() for g in list(s_genes) + list(g2m_genes))
    S_all = a.layers["spliced"].tocsr().astype(np.float64)
    U_all = a.layers["unspliced"].tocsr().astype(np.float64)
    size_s = np.asarray(S_all.sum(1)).ravel()
    size_u = np.asarray(U_all.sum(1)).ravel()
    names = np.asarray(a.var_names).astype(str)
    murk = a.var["MURK_gene"].astype(bool).to_numpy()
    meta = {"stages": STAGES, "days": DAYS, "n_cells_per_stage": np.bincount(stage).tolist(), "origins": {}}
    for s in range(1, 6):
        t0 = time.time()
        past, fut = stage <= s, stage == s + 1
        keep = shared_filter(S_all[past], U_all[past])
        tgt_s, tgt_u = np.median(size_s[past]), np.median(size_u[past])
        Sn = normalise(S_all[:, keep], size_s, tgt_s).tocsr()
        Un = normalise(U_all[:, keep], size_u, tgt_u).tocsr()
        Xl = Sn.copy()
        Xl.data = np.log1p(Xl.data)
        p = ad.AnnData(X=Xl[past].copy())
        p.var_names = names[keep]
        sc.pp.highly_variable_genes(p, n_top_genes=N_HVG, flavor="seurat")
        hv = p.var["highly_variable"].to_numpy()
        gidx = np.where(keep)[0][hv]
        Xp = np.asarray(Xl[past][:, hv].todense(), np.float32)
        Xf = np.asarray(Xl[fut][:, hv].todense(), np.float32)
        mu = Xp.mean(0)
        U_, S_, Vt = np.linalg.svd(Xp - mu, full_matrices=False)
        comps = Vt[:N_PC]
        Zp, Zf = (Xp - mu) @ comps.T, (Xf - mu) @ comps.T
        q = ad.AnnData(X=Xp, layers={"spliced": np.asarray(Sn[past][:, hv].todense(), np.float32),
                                     "unspliced": np.asarray(Un[past][:, hv].todense(), np.float32)})
        q.var_names = names[gidx]
        q.obsm["X_pca"] = Zp.astype(np.float32)
        sc.pp.neighbors(q, n_neighbors=N_NB, use_rep="X_pca")
        scv.pp.moments(q, n_pcs=None, n_neighbors=None)
        Ms, Mu = np.asarray(q.layers["Ms"], np.float64), np.asarray(q.layers["Mu"], np.float64)
        v_ss, gamma = steady_state_velocity(Ms, Mu)
        try:
            scv.tl.recover_dynamics(q, n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), show_progress_bar=False)
            scv.tl.velocity(q, mode="dynamical")
            v_dyn = np.nan_to_num(np.asarray(q.layers["velocity"], np.float64))
            dyn_ok = True
        except Exception as e:  # noqa: BLE001
            v_dyn, dyn_ok = np.zeros_like(v_ss), f"failed: {e!r}"
        # proliferation: mean of centred log expression over cell-cycle genes present after the filter (all kept genes)
        ccm = np.array([n.capitalize() in cc_genes for n in names[keep]])
        Xcc = np.asarray(Xl[:, ccm].todense())
        cc_score = (Xcc - Xcc[past].mean(0)).mean(1)
        np.savez_compressed(
            os.path.join(OUT, f"origin{s}.npz"),
            genes=names[gidx], murk=murk[gidx], stage_past=stage[past], stage_fut=stage[fut],
            celltype_past=a.obs["celltype"].astype(str).to_numpy()[past],
            celltype_fut=a.obs["celltype"].astype(str).to_numpy()[fut],
            X_past=Xp.astype(np.float16), X_fut=Xf.astype(np.float16), pca_mean=mu, pca_comps=comps,
            Z_past=Zp.astype(np.float32), Z_fut=Zf.astype(np.float32),
            Ms=Ms.astype(np.float32), Mu=Mu.astype(np.float32), v_ss=v_ss.astype(np.float32),
            v_dyn=v_dyn.astype(np.float32), gamma=gamma, cc_past=cc_score[past], cc_fut=cc_score[fut],
            size_s_past=size_s[past], size_s_fut=size_s[fut])
        meta["origins"][s] = {"n_past": int(past.sum()), "n_fut": int(fut.sum()), "n_genes_kept": int(keep.sum()),
                              "n_hvg": int(hv.sum()), "murk_in_hvg": int(murk[gidx].sum()), "dynamical": dyn_ok,
                              "n_cc_genes": int(ccm.sum()), "seconds": round(time.time() - t0, 1)}
        print(s, meta["origins"][s], flush=True)
        with open(os.path.join(OUT, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
