"""Calibration gate D1: reproduce Chronocell on the gastrulation erythroid lineage.

Published reference (Fang, Gorin, Pachter, PLoS Comput Biol 2025; FGP_2024 notebook Erythoid.ipynb):
AP of random initialisations ~0.79, about 21-24 of 49 genes kept by the trajectory-vs-cluster test,
per-stage mean process time flat until E7.5 then increasing.

Usage: python chrono_repro.py --out ~/tack/out/repro [--quick] [--procs 8] [--n-random 100]
"""
import argparse
import json
import time

import chrono_common as cc  # sets BLAS threads before numpy
import numpy as np


def fit_random(args):
    seed, X, rd, epoch = args
    from Chronocell.inference import Trajectory

    t0 = time.time()
    traj = Trajectory(cc.TOPO, cc.TAU, model="two_species_ss", verbose=0)
    params = {"r": rd, "lambda_tau": 0, "lambda_a": 0}
    traj = traj.fit(X, m=cc.M_GRID, n_init=1, params=params, perm_theta=False, epoch=epoch,
                    parallel=False, n_threads=1, seed=seed)
    t_hat = traj.Q[:, 0] @ traj.t
    return dict(seed=seed, t_hat=t_hat.astype(np.float32), aic=float(traj.compute_AIC(traj.X, standard=True)),
                elbo=float(traj.elbos[0][-1]), theta=traj.theta, tau=traj.tau.copy(), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=100)
    a = ap.parse_args()

    from multiprocessing import Pool
    import pathlib

    from Chronocell.inference import Trajectory
    from Chronocell.mixtures import PoissonMixtureSS
    from scipy.stats import pearsonr
    from sklearn.metrics import average_precision_score

    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    epoch = 5 if a.quick else 100
    n_random = 4 if a.quick else a.n_random
    res = dict(quick=a.quick, epoch=epoch, n_random=n_random)
    T0 = time.time()

    X, depth_raw, genes, stage, celltype, batch, info = cc.prepare(out / "erythroid_prep.npz")
    res["prep"] = info
    rd = depth_raw / depth_raw.mean()
    print("prep", json.dumps(info), flush=True)

    # 3-cluster Poisson mixture (notebook cell 34) = reference for the gene test
    pm = PoissonMixtureSS(n_components=3, verbose=0)
    q_pm, elbo_pm = pm.fit(X, warm_start=False, read_depth=rd, epoch=5 if a.quick else 10, n_init=2 if a.quick else 5)
    cluster_gene_logl, cluster_kl = pm.compute_gene_logL(X, q_pm)
    res["pm"] = dict(weights=np.asarray(pm.weights).tolist(), elbo=float(np.ravel(elbo_pm)[-1]))

    # warm start from cell types (notebook cell 45)
    q0 = cc.celltype_warm_q(celltype)
    t1 = time.time()
    ws = Trajectory(cc.TOPO, cc.TAU, model="two_species_ss", verbose=0)
    ws = ws.fit(X, warm_start=True, fit_tau=True, params={"r": rd / rd.mean()}, Q=q0, parallel=True,
                n_threads=a.procs, epoch=epoch)
    gene_logl, kl = ws.compute_gene_logL(ws.X, ws.Q)
    p = len(gene_logl)
    dlogl = gene_logl - cluster_gene_logl
    dh = (cluster_kl - kl) / p
    good = (dlogl > dh) & (ws.theta[:, -2] < 500) & (ws.theta[:, -1] < 100)
    t_ws = ws.Q[:, 0] @ ws.t
    res["warm_start"] = dict(
        seconds=time.time() - t1, elbo=float(ws.elbos[-1]), tau=ws.tau.tolist(),
        n_good=int(good.sum()), n_genes=int(p), good_genes=[str(g) for g in genes[good]],
        stage_mean_t={s: float(t_ws[stage == s].mean()) for s in cc.STAGES},
    )
    print("warm start", json.dumps(res["warm_start"]), flush=True)

    # random initialisations (notebook cells 66-68)
    right_t = q0[:, 0] @ ws.t
    t2 = time.time()
    with Pool(a.procs) as pool:
        fits = pool.map(fit_random, [(s, X, rd, epoch) for s in range(n_random)], chunksize=1)
    corrs = np.array([pearsonr(right_t, f["t_hat"])[0] for f in fits])
    aics = np.array([f["aic"] for f in fits])
    correct = corrs > 0.8
    ap_score = float(average_precision_score(correct, -aics)) if 0 < correct.sum() < len(correct) else None
    best = fits[int(np.argmin(aics))]
    res["random_init"] = dict(
        seconds=time.time() - t2, mean_seconds_per_fit=float(np.mean([f["seconds"] for f in fits])),
        frac_correct=float(correct.mean()), average_precision=ap_score,
        corr_best_aic=float(corrs[int(np.argmin(aics))]), aic_best=float(aics.min()),
        aic_warm_start=float(ws.compute_AIC(X, standard=True)), aic_3_clusters=float(pm.compute_AIC(X, standard=True)),
        stage_mean_t_best={s: float(best["t_hat"][stage == s].mean()) for s in cc.STAGES},
        tau_best=np.asarray(best["tau"]).tolist(),
        corrs=corrs.tolist(), aics=aics.tolist(),
    )
    res["reference"] = dict(average_precision=0.79, n_good_of_49="21-24", stage_mean_t_notebook=dict(
        zip(cc.STAGES, [0.19, 0.15, 0.16, 0.43, 0.65, 0.99, 1.57])))
    res["total_seconds"] = time.time() - T0
    (out / "repro.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "repro_fits.npz", t_ws=t_ws, theta_ws=ws.theta, tau_ws=ws.tau,
                        t_best=best["t_hat"], theta_best=best["theta"], corrs=corrs, aics=aics)
    print("random init", json.dumps({k: v for k, v in res["random_init"].items() if k not in ("corrs", "aics")}), flush=True)
    print("DONE", res["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
