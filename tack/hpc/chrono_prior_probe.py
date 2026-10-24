"""TACK step 0, adapter probe (NOT TACK): which clock prior predicts a held-out erythroid stage best?

Engine: Chronocell (external baseline, commit a827f89), used through its per-cell `prior` hook and get_logL.
Our code: the clock priors (uniform, Gaussian capture-time, age-structured population), the held-out
evaluation and the controls. No clock claims are made from this data set (open reservoir, embryo staging
noise, one sequencing batch per stage).

For each held-out stage h (interior E7.25..E8.25, terminal E8.5):
 1. decoder theta_U = Chronocell fit with the uniform prior on the other stages (warm start from cell types);
 2. frozen-decoder comparison: grid log-likelihood L_i(t) of the held-out cells under theta_U, and the
    held-out log-likelihood of each predicted mixing distribution q(t | T_h):
      uniform (Chronocell), persistence (previous stage), L-mix (linear interpolation in real time of the
      neighbouring stages' fitted mixing), gauss (capture-time prior, VeloVAE-like), age (TACK-lite:
      constant influx from onset T0, net growth g, reservoir fraction, absorbing end), and an oracle q*
      fitted on half of the held-out cells and scored on the other half (swap);
 3. refit: decoder fitted with the age prior on the training stages; held-out LL with its own prediction.
Writes probe.json.

Usage: python chrono_prior_probe.py --prep ~/tack/out/repro/erythroid_prep.npz --out ~/tack/out/probe [--quick]
"""
import argparse
import json
import time

import chrono_common as cc  # sets BLAS threads before numpy
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

EPS = 1e-12


def grid_logl(traj, X, r):
    params = dict(traj.params)
    params["r"] = r
    return traj.get_logL(X, traj.theta, traj.t, traj.tau, traj.topo, params)[:, 0, :]


def cell_ll(L, q):
    return logsumexp(L + np.log(np.maximum(q, EPS))[None, :], axis=1)


def free_q(L, iters=400):
    """Mixing distribution maximising sum_i log sum_t q(t) exp(L_i(t)) (fixed-point EM)."""
    q = np.full(L.shape[1], 1.0 / L.shape[1])
    for _ in range(iters):
        lp = L + np.log(np.maximum(q, EPS))[None, :]
        post = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
        q = post.mean(0)
    return q


def _norm(q):
    q = np.maximum(q, EPS)
    return q / q.sum()


def gauss_prior(T_h, th, t_grid):
    mu0, slope, log_sd = th
    mu = mu0 + slope * T_h / 24.0
    return _norm(np.exp(-0.5 * ((t_grid - mu) / np.exp(log_sd)) ** 2))


def age_prior(T_h, th, t_grid):
    """Constant influx from onset T0 (h), net growth g (1/h), process unit = c hours, absorbing end,
    plus a reservoir point mass at t=0 with logistic weight in real time."""
    T0, log_c, g, w0, w1 = th
    c = np.exp(log_c)
    A = T_h - T0
    w = 1.0 / (1.0 + np.exp(-(w0 + w1 * T_h / 24.0)))
    q = np.zeros_like(t_grid)
    if A <= 0:
        q[0] = 1.0
        return _norm(q)
    d = t_grid[1] - t_grid[0]
    lo = np.concatenate([[0.0], (t_grid[1:] - d / 2)]) * c
    hi = np.concatenate([(t_grid[:-1] + d / 2), [np.inf]]) * c
    lo = np.minimum(lo, A)
    hi = np.minimum(hi, A)
    width = np.maximum(hi - lo, 0.0)
    if abs(g) < 1e-9:
        mass = width
    else:
        # integral of exp(g a) over [lo, hi], scaled by exp(-g * ref) for stability
        ref = A if g > 0 else 0.0
        mass = np.exp(g * (lo - ref)) * np.expm1(g * width) / g
    mass = np.maximum(mass, 0.0)
    if mass.sum() <= 0:
        mass[0] = 1.0
    entered = mass / mass.sum()
    q = (1 - w) * entered
    q[0] += w
    return _norm(q)


def fit_prior(kind, Ls, Ts, t_grid, starts, fixed_log_c=None):
    """Maximise the training log-likelihood sum_s sum_{i in s} log sum_t q(t | T_s) exp(L_i(t))."""
    n = sum(len(L) for L in Ls)

    def unpack(x):
        if kind == "age" and fixed_log_c is not None:
            return np.array([x[0], fixed_log_c, x[1], x[2], x[3]])
        return x

    def obj(x):
        th = unpack(x)
        f = age_prior if kind == "age" else gauss_prior
        tot = 0.0
        for L, T in zip(Ls, Ts):
            tot += cell_ll(L, f(T, th, t_grid)).sum()
        return -tot / n

    best = None
    for x0 in starts:
        if kind == "age" and fixed_log_c is not None:
            x0 = np.array([x0[0], x0[2], x0[3], x0[4]])
        r = minimize(obj, np.asarray(x0, float), method="Nelder-Mead",
                     options=dict(maxiter=1500, xatol=1e-3, fatol=1e-7))
        if best is None or r.fun < best.fun:
            best = r
    return unpack(best.x), -best.fun


def boot_ci(diff, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(1)
    return [float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(means.std())]


def fit_decoder(X, r, q0, procs, epoch, prior=None):
    from Chronocell.inference import Trajectory

    traj = Trajectory(cc.TOPO, cc.TAU, model="two_species_ss", verbose=0)
    return traj.fit(X, warm_start=True, fit_tau=True, params={"r": r}, Q=q0, prior=prior,
                    parallel=True, n_threads=procs, epoch=epoch)


def predictions(L_by_stage, train_stages, h, t_grid, age_starts, gauss_starts):
    """Mixing-distribution predictions for held-out stage h from training stages only."""
    Th = cc.STAGE_HOURS[h]
    q_stage = {s: free_q(L_by_stage[s]) for s in train_stages}
    hours = np.array([cc.STAGE_HOURS[s] for s in train_stages])
    before = [s for s in train_stages if cc.STAGE_HOURS[s] < Th]
    after = [s for s in train_stages if cc.STAGE_HOURS[s] > Th]
    prev = max(before, key=lambda s: cc.STAGE_HOURS[s]) if before else min(after, key=lambda s: cc.STAGE_HOURS[s])
    if before and after:
        nxt = min(after, key=lambda s: cc.STAGE_HOURS[s])
        lam = (Th - cc.STAGE_HOURS[prev]) / (cc.STAGE_HOURS[nxt] - cc.STAGE_HOURS[prev])
        lmix = (1 - lam) * q_stage[prev] + lam * q_stage[nxt]
    else:
        lmix = q_stage[prev]
    Ls = [L_by_stage[s] for s in train_stages]
    Ts = [cc.STAGE_HOURS[s] for s in train_stages]
    th_g, ll_g = fit_prior("gauss", Ls, Ts, t_grid, gauss_starts)
    th_a, ll_a = fit_prior("age", Ls, Ts, t_grid, age_starts)
    preds = dict(
        uniform=np.full(len(t_grid), 1.0 / len(t_grid)),
        persistence=q_stage[prev],
        lmix=lmix,
        gauss=gauss_prior(Th, th_g, t_grid),
        age=age_prior(Th, th_a, t_grid),
    )
    fitted = dict(gauss=dict(theta=th_g.tolist(), train_ll=ll_g), age=dict(theta=th_a.tolist(), train_ll=ll_a,
                  c_hours_per_unit=float(np.exp(th_a[1]))), prev=prev, train_hours=hours.tolist())
    return preds, fitted, q_stage


def c_profile(Ls, Ts, t_grid, th_best, grid):
    out = []
    for log_c in grid:
        start = [list(th_best[:1]) + [log_c] + list(th_best[2:])]
        _, ll = fit_prior("age", Ls, Ts, t_grid, start, fixed_log_c=log_c)
        out.append([float(np.exp(log_c)), float(ll)])
    return out


def evaluate(L_test, preds, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(L_test))
    A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
    ll_oracle = np.empty(len(L_test))
    ll_oracle[B] = cell_ll(L_test[B], free_q(L_test[A]))
    ll_oracle[A] = cell_ll(L_test[A], free_q(L_test[B]))
    ll = {k: cell_ll(L_test, q) for k, q in preds.items()}
    res = dict(n_test=int(len(L_test)), ll={k: float(v.mean()) for k, v in ll.items()},
               ll_oracle=float(ll_oracle.mean()),
               regret={k: boot_ci(ll_oracle - v) for k, v in ll.items()})
    pairs = [("age", "uniform"), ("age", "persistence"), ("age", "lmix"), ("age", "gauss"),
             ("gauss", "lmix"), ("gauss", "uniform"), ("lmix", "uniform")]
    res["paired"] = {f"{a}-{b}": boot_ci(ll[a] - ll[b]) for a, b in pairs}
    controls = ["uniform", "persistence", "lmix"]
    best_ctrl = min(controls, key=lambda k: res["regret"][k][0])
    reg, _, _, se = res["regret"][best_ctrl]
    res["K0"] = dict(best_control=best_ctrl, regret=reg, se=se, headroom_in_se=float(reg / se) if se > 0 else None)
    best_ref = res["regret"][best_ctrl][0]
    res["gap_closed"] = {k: float(1 - res["regret"][k][0] / best_ref) if best_ref > 0 else None
                         for k in ("age", "gauss")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    import pathlib

    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    X, depth_raw, genes, stage, celltype, batch = cc.load_prep(pathlib.Path(a.prep).expanduser())
    epoch = 5 if a.quick else 100
    folds = ["E7.75"] if a.quick else ["E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
    t_grid = np.linspace(cc.TAU[0], cc.TAU[-1], cc.M_GRID)
    age_starts = [[T0, np.log(c), 0.0, w0, 0.0] for T0 in (-24.0, 0.0) for c in (6.0, 24.0, 96.0) for w0 in (-2.0, 2.0)]
    gauss_starts = [[m0, s, np.log(sd)] for m0 in (0.0, 0.5) for s in (0.5, 1.0) for sd in (0.3, 0.8)]
    if a.quick:
        age_starts, gauss_starts = age_starts[:2], gauss_starts[:2]
    report = dict(quick=a.quick, epoch=epoch, n_genes=int(len(genes)), folds={},
                  note="adapter probe on Chronocell; NOT TACK; no clock claims on erythroid")
    T0 = time.time()
    for h in folds:
        tf = time.time()
        train = stage != h
        test = stage == h
        train_stages = [s for s in cc.STAGES if s != h]
        r_tr = depth_raw[train] / depth_raw[train].mean()
        r_te = depth_raw[test] / depth_raw[train].mean()
        q0 = cc.celltype_warm_q(celltype[train])

        # 1-2: uniform-prior decoder, frozen comparison
        trajU = fit_decoder(X[train], r_tr, q0, a.procs, epoch)
        L_tr = grid_logl(trajU, X[train], r_tr)
        L_te = grid_logl(trajU, X[test], r_te)
        st_tr = stage[train]
        L_by_stage = {s: L_tr[st_tr == s] for s in train_stages}
        preds, fitted, q_stage = predictions(L_by_stage, train_stages, h, t_grid, age_starts, gauss_starts)
        frozen = evaluate(L_te, preds, seed=1)
        Ls = [L_by_stage[s] for s in train_stages]
        Ts = [cc.STAGE_HOURS[s] for s in train_stages]
        prof = c_profile(Ls, Ts, t_grid, np.array(fitted["age"]["theta"]),
                         np.log(np.geomspace(2, 400, 6 if a.quick else 16)))

        # 3: refit the decoder with the age prior on training cells
        th_age = np.array(fitted["age"]["theta"])
        prior_tr = np.stack([age_prior(cc.STAGE_HOURS[s], th_age, t_grid) for s in st_tr])[:, None, :]
        trajA = fit_decoder(X[train], r_tr, q0, a.procs, epoch, prior=prior_tr)
        LA_tr = grid_logl(trajA, X[train], r_tr)
        LA_te = grid_logl(trajA, X[test], r_te)
        LA_by_stage = {s: LA_tr[st_tr == s] for s in train_stages}
        predsA, fittedA, _ = predictions(LA_by_stage, train_stages, h, t_grid, age_starts, gauss_starts)
        refit = evaluate(LA_te, predsA, seed=2)
        cross = dict(
            uniformDecoder_uniformPrior=float(cell_ll(L_te, preds["uniform"]).mean()),
            uniformDecoder_lmix=float(cell_ll(L_te, preds["lmix"]).mean()),
            ageDecoder_agePrior=float(cell_ll(LA_te, predsA["age"]).mean()),
            ageDecoder_lmix=float(cell_ll(LA_te, predsA["lmix"]).mean()),
            ageDecoder_agePrior_minus_uniformDecoder_lmix=boot_ci(cell_ll(LA_te, predsA["age"]) - cell_ll(L_te, preds["lmix"])),
            ageDecoder_agePrior_minus_uniformDecoder_uniform=boot_ci(cell_ll(LA_te, predsA["age"]) - cell_ll(L_te, preds["uniform"])),
        )
        report["folds"][h] = dict(
            terminal=(h == cc.STAGES[-1]), seconds=time.time() - tf,
            frozen=frozen, fitted=fitted, c_profile=prof, refit=refit, refit_fitted=fittedA, cross=cross,
            q=dict({k: v.tolist() for k, v in preds.items()}, oracle=free_q(L_te).tolist()),
        )
        (out / "probe.json").write_text(json.dumps(report, indent=1))
        print(h, json.dumps(dict(frozen_ll=frozen["ll"], oracle=frozen["ll_oracle"], K0=frozen["K0"],
                                 gap_closed=frozen["gap_closed"], c=fitted["age"]["c_hours_per_unit"],
                                 seconds=time.time() - tf)), flush=True)
    report["total_seconds"] = time.time() - T0
    (out / "probe.json").write_text(json.dumps(report, indent=1))
    print("DONE", report["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
