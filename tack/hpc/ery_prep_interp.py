"""Strict data products for INTERPOLATING a held-out real embryonic stage of the erythroid lineage.

For each held-out stage k in {2, 3, 4, 5} (E7.5 .. E8.25), the gene filter, normalisation, 2000 HVGs, PCA (30),
neighbour graph, moments and velocities (own steady-state + scVelo dynamical) are fitted on the two neighbouring
stages k-1 and k+1 ONLY (same rules as ery_prep.py); the held-out stage is only transformed. Sample labels are
kept so that distances between samples of the same stage give a replicate floor.
Output: ~/tack/out/ery/interp<k>.npz.
"""
import os
import time

import anndata as ad
import numpy as np
import scanpy as sc
import scvelo as scv

import ery_prep as ep

OUT = os.path.expanduser("~/tack/out/ery")


def main():
    a = ad.read_h5ad(ep.H5)
    stage = np.array([ep.STAGES.index(s) for s in a.obs["stage"].astype(str)])
    sample = a.obs["sample"].astype(str).to_numpy()
    S_all = a.layers["spliced"].tocsr().astype(np.float64)
    U_all = a.layers["unspliced"].tocsr().astype(np.float64)
    size_s = np.asarray(S_all.sum(1)).ravel()
    size_u = np.asarray(U_all.sum(1)).ravel()
    names = np.asarray(a.var_names).astype(str)
    murk = a.var["MURK_gene"].astype(bool).to_numpy()
    for k in (2, 3, 4, 5):
        t0 = time.time()
        fit = (stage == k - 1) | (stage == k + 1)
        held = stage == k
        keep = ep.shared_filter(S_all[fit], U_all[fit])
        Sn = ep.normalise(S_all[:, keep], size_s, np.median(size_s[fit])).tocsr()
        Un = ep.normalise(U_all[:, keep], size_u, np.median(size_u[fit])).tocsr()
        Xl = Sn.copy()
        Xl.data = np.log1p(Xl.data)
        p = ad.AnnData(X=Xl[fit].copy())
        sc.pp.highly_variable_genes(p, n_top_genes=ep.N_HVG, flavor="seurat")
        hv = p.var["highly_variable"].to_numpy()
        gidx = np.where(keep)[0][hv]
        Xf_, Xh = np.asarray(Xl[fit][:, hv].todense(), np.float32), np.asarray(Xl[held][:, hv].todense(), np.float32)
        mu = Xf_.mean(0)
        _, _, Vt = np.linalg.svd(Xf_ - mu, full_matrices=False)
        comps = Vt[:ep.N_PC]
        q = ad.AnnData(X=Xf_, layers={"spliced": np.asarray(Sn[fit][:, hv].todense(), np.float32),
                                      "unspliced": np.asarray(Un[fit][:, hv].todense(), np.float32)})
        q.var_names = names[gidx]
        q.obsm["X_pca"] = ((Xf_ - mu) @ comps.T).astype(np.float32)
        sc.pp.neighbors(q, n_neighbors=ep.N_NB, use_rep="X_pca")
        scv.pp.moments(q, n_pcs=None, n_neighbors=None)
        Ms, Mu = np.asarray(q.layers["Ms"], np.float64), np.asarray(q.layers["Mu"], np.float64)
        v_ss, _ = ep.steady_state_velocity(Ms, Mu)
        try:
            scv.tl.recover_dynamics(q, n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), show_progress_bar=False)
            scv.tl.velocity(q, mode="dynamical")
            v_dyn = np.nan_to_num(np.asarray(q.layers["velocity"], np.float64))
        except Exception:  # noqa: BLE001
            v_dyn = np.zeros_like(v_ss)
        np.savez_compressed(
            os.path.join(OUT, f"interp{k}.npz"), genes=names[gidx], murk=murk[gidx],
            stage_fit=stage[fit], sample_fit=sample[fit], sample_held=sample[held],
            X_fit=Xf_.astype(np.float16), X_held=Xh.astype(np.float16), pca_mean=mu, pca_comps=comps,
            Z_fit=q.obsm["X_pca"], Z_held=((Xh - mu) @ comps.T).astype(np.float32),
            Ms=Ms.astype(np.float32), v_ss=v_ss.astype(np.float32), v_dyn=v_dyn.astype(np.float32),
            celltype_fit=a.obs["celltype"].astype(str).to_numpy()[fit],
            celltype_held=a.obs["celltype"].astype(str).to_numpy()[held])
        print(k, {"n_fit": int(fit.sum()), "n_held": int(held.sum()), "n_hvg": int(hv.sum()),
                  "seconds": round(time.time() - t0, 1)}, flush=True)


if __name__ == "__main__":
    main()
