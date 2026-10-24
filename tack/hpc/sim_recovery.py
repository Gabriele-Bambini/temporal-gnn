"""Simulation recovery of the sci-FATE2 label-clock pipeline (required before any claim). Predictions and
amendment S-A1 are written in docs/superpowers/decisions.md before the full launch.

Synthetic data come from the frozen decoder of job 679647 (theta, tau, topology T3) on the SAME cells as the real
probe (same T, group, detection alpha; generation depth r, analysis depth r * lognormal(0, 0.1)), with the fitted
age prior (T0, g, w0, w1) and the scenario's true hour scale c. Times are generated on a 20x finer grid than the
analysis grid; ambiguous reads are generated with the real per-gene share and analysed exactly as in the probe.
Scenarios (states and counts use the same random stream in all scenarios):
  self   : c = c_A, beta as fitted. A second label set from the SAME cells and counts uses c = 2 c_A only for the
           labelling window ('double_label' control: must move c_label, not c_A).
  fast   : c = c_A, true unspliced half-life <= 15 min (beta_true = max(beta_fit, ln2/0.25 h * c)); unspliced capture
           Ub = beta_true / beta_fit keeps the observed U and S steady levels of the fitted decoder
  double : c = 2 c_A everywhere (states, kinetics in hours, labels)
The probe pipeline then runs unchanged (8 inits x 100 epochs, amendment A1, clocks, label profile, bands, frozen
decision rule), plus diagnostic arms: 'beta-oracle' (true beta in 1/h with the refitted decoder) and
'decoder-oracle' (true theta and its posteriors).

Usage: python sim_recovery.py --scenario self|fast|double [--quick] [--procs 8]
"""
import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import scifate2_clock_probe as probe  # noqa: E402
from label_model import new_fraction, phi  # noqa: E402

OUT_REAL = os.path.expanduser("~/tack/out/scifate2_clock")
BETA_H_FAST = np.log(2) / 0.25  # 1/h, unspliced half-life 15 min
FINE = 20
RSTEP = float(np.sqrt(probe.C_GRID[1] / probe.C_GRID[0]))  # half a grid step (ratio)


def species_fractions(Y, Y0, beta, gamma, h, beta_flux=None):
    if beta_flux is None:
        beta_flux = beta
    U, S, U0, S0 = Y[..., 0], Y[..., 1], Y0[..., 0], Y0[..., 1]
    old_u = U0 * np.exp(-beta * h)
    old_s = S0 * np.exp(-gamma * h) + beta_flux * U0 * phi(beta, gamma, h)
    f_u = np.where(U > 0, np.clip(U - old_u, 0, U) / np.maximum(U, 1e-300), 0.0)
    f_s = np.where(S > 0, np.clip(S - old_s, 0, S) / np.maximum(S, 1e-300), 0.0)
    return f_u, f_s


def true_theta(theta_fit, c_true, scenario):
    n_states = theta_fit.shape[1] - 2
    beta_fit, gamma = theta_fit[:, -2], theta_fit[:, -1]
    theta_true = theta_fit.copy()
    Ub = np.ones(len(beta_fit))
    n_nudged = 0
    if scenario == "fast":
        beta_true = np.maximum(beta_fit, BETA_H_FAST * c_true)
        close = np.abs(beta_true - gamma) < 1e-2 * beta_true
        beta_true[close] *= 1.02
        n_nudged = int(close.sum())
        Ub = beta_true / beta_fit
        theta_true[:, :n_states] = theta_fit[:, :n_states] / Ub[:, None]
        theta_true[:, -2] = beta_true
    return theta_true, Ub, n_nudged


def simulate(model, theta_true, Ub, tau, th_age, c_label_list, Tcell, r, alpha, amb_g, rng_state, rng_lab):
    """States and counts from rng_state (shared across scenarios); one label set per c in c_label_list."""
    tf = np.linspace(tau[0], tau[-1], FINE * probe.M)
    beta_true, gamma = theta_true[:, -2], theta_true[:, -1]
    Y = model.get_Y_hat(theta_true, tf, tau, probe.TOPO, {})  # true units
    th = np.array(th_age, float).copy()
    th[1] = np.log(c_label_list[0])  # the state clock is the first (scenario) c
    n, L = len(Tcell), probe.TOPO.shape[0]
    lin = rng_state.integers(0, L, size=n)
    tix = np.empty(n, int)
    for T in np.unique(Tcell):
        sel = np.where(Tcell == T)[0]
        q = np.maximum(probe.age_prior(T, th, tf), 0)
        tix[sel] = rng_state.choice(len(tf), size=len(sel), p=q / q.sum())
    Umean = Y[lin, tix, :, 0] * Ub[None, :]
    Smean = Y[lin, tix, :, 1]
    Amean = amb_g[None, :] / (1 - amb_g[None, :]) * (Umean + Smean)
    Uc = rng_state.poisson(r[:, None] * Umean)
    Sc = rng_state.poisson(r[:, None] * Smean)
    Ac = rng_state.poisson(r[:, None] * Amean)
    labels = []
    for c in c_label_list:
        h = probe.L_PULSE / c
        Y0 = model.get_Y_hat(theta_true, np.clip(tf - h, tau[0], None), tau, probe.TOPO, {})
        f_u, f_s = species_fractions(Y, Y0, beta_true, gamma, h)
        pu = np.clip(alpha[:, None] * f_u[lin, tix], 0, 1)
        ps = np.clip(alpha[:, None] * f_s[lin, tix], 0, 1)
        new = rng_lab.binomial(Uc, pu) + rng_lab.binomial(Sc, ps) + rng_lab.binomial(Ac, ps)
        labels.append(new.astype(float))
    return Uc.astype(float), Sc.astype(float), (Uc + Sc + Ac).astype(float), labels


def label_profile(model, theta, tau, q, new_s, tot_s, a_s, g_s, amb_g, beta_decay_per_h=None, beta_flux=None,
                  Ub=None):
    """Same score as the probe (new_fraction with ambiguous share).

    Oracle options: beta_decay_per_h = true splicing rate in 1/h (decay of old unspliced RNA, scaled with each c);
    beta_flux = rate converting the U units into splicing flux; Ub = capture factor applied to the unspliced means.
    """
    t = np.linspace(tau[0], tau[-1], probe.M)
    Y = model.get_Y_hat(theta, t, tau, probe.TOPO, {})
    if Ub is not None:
        Y[..., 0] *= Ub
    gamma = theta[:, -1]
    old_s = tot_s - new_s
    read_tot = tot_s.sum(1)
    ok = read_tot > 0
    ug = np.unique(g_s)
    rows = {sc: [] for sc in probe.ALPHA_SCALES}
    for c in probe.C_GRID:
        h = probe.L_PULSE / c
        beta = theta[:, -2] if beta_decay_per_h is None else beta_decay_per_h * c
        bflux = theta[:, -2] if beta_flux is None else beta_flux
        Y0 = model.get_Y_hat(theta, np.clip(t - h, tau[0], None), tau, probe.TOPO, {})
        if Ub is not None:
            Y0[..., 0] *= Ub
        pi = new_fraction(Y[..., 0], Y[..., 1], Y0[..., 0], Y0[..., 1], beta, gamma, h, amb_g, beta_flux=bflux)
        pi_cell = np.einsum("nlm,lmp->np", q, pi)
        for sc in probe.ALPHA_SCALES:
            P = np.clip(sc * a_s[:, None] * pi_cell, 1e-6, 1 - 1e-6)
            ll = (new_s * np.log(P) + old_s * np.log1p(-P)).sum(1) / np.maximum(read_tot, 1)
            rows[sc].append([ll[ok & (g_s == g)].mean() for g in ug])
    return {sc: np.array(v).T for sc, v in rows.items()}  # (G, nC)


def band_from(prof, n_boot, seed=7):
    rng = np.random.default_rng(seed)
    G = next(iter(prof.values())).shape[0]
    idx = rng.integers(0, G, size=(n_boot, G))
    bands, argmax = {}, {}
    for sc, M_ in prof.items():
        argmax[sc] = int(np.argmax(M_.mean(0)))
        bmax = np.argmax(M_[idx].mean(1), axis=1)
        bands[sc] = [float(probe.C_GRID[int(np.floor(np.percentile(bmax, 2.5)))]),
                     float(probe.C_GRID[int(np.ceil(np.percentile(bmax, 97.5)))])]
    lo = min(b[0] for b in bands.values())
    hi = max(b[1] for b in bands.values())
    return float(probe.C_GRID[argmax[1.0]]), lo, hi, argmax[1.0]


def posteriors(model, theta, tau, X, r):
    lf, t = probe.grid_loglik(model, X, theta, tau, r)
    return np.exp(lf - logsumexp(lf, axis=(1, 2), keepdims=True)), t


def dispersion_ratios(Uf, Sf, sp_):
    um, uv = Uf.mean(0), Uf.var(0)
    sm, sv = Sf.mean(0), Sf.var(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmedian(uv / (um + sp_ * um**2))), float(np.nanmedian(sv / (sm + sp_ * sm**2)))


def assess(c_true, c_label, lo, hi, argmax_idx):
    within = c_true / 1.25 <= c_label <= c_true * 1.25
    in_band = lo / RSTEP <= c_true <= hi * RSTEP
    edge = argmax_idx in (0, len(probe.C_GRID) - 1)
    return dict(c_label=c_label, band=[lo, hi], within_1_25=bool(within), band_contains_c_true=bool(in_band),
                R1=bool(within and in_band), peak_at_edge=bool(edge))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["self", "fast", "double"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    from multiprocessing import Pool

    from Chronocell.models import two_species_ss as model

    out = os.path.expanduser(f"~/tack/out/sim_recovery/{a.scenario}{'_quick' if a.quick else ''}")
    os.makedirs(out, exist_ok=True)
    T_start = time.time()
    real = json.load(open(os.path.join(OUT_REAL, "clock_probe.json")))
    dec = np.load(os.path.join(OUT_REAL, "decoder.npz"), allow_pickle=True)
    theta_fit, tau_fit = dec["theta"], dec["tau"]
    th_age = real["clocks"]["age"]["theta"]
    c_A = real["clocks"]["c_A"]
    sp_real = real["genes"]["sp"]
    c_true = {"self": c_A, "fast": c_A, "double": 2 * c_A}[a.scenario]
    label_cs = [c_true, 2 * c_A] if a.scenario == "self" else [c_true]

    obs, tot, new, U, S, genes, amb = probe.load(False)
    groups = (obs["timepoint"].astype(str) + "_" + obs["rep"].astype(str)).to_numpy()
    Tcell = obs["timepoint"].astype(str).map(probe.GROUP_T).to_numpy(float)
    keep, alpha, _ = probe.qc(obs, tot, genes)
    fit, score = probe.split(groups, keep, 1000)  # the real probe's split: same genes and depth as the real run
    r, gidx, _ = probe.depth_and_genes(U, S, fit, genes, amb, 150)
    assert list(genes[gidx].astype(str)) == list(dec["genes"].astype(str)), "gene set mismatch"
    if a.quick:  # smoke run only: fewer cells after the gene set is fixed
        fit, score = probe.split(groups, keep, 200)
        score &= np.random.default_rng(3).random(len(score)) < 0.25
    real_disp = dispersion_ratios(U[fit][:, gidx].toarray(), S[fit][:, gidx].toarray(), sp_real)
    del tot, new, U, S
    amb_g = np.nan_to_num(amb[gidx], nan=0.0)
    cells = fit | score
    fit_c, score_c = fit[cells], score[cells]
    T_c, g_c, a_c, r_gen = Tcell[cells], groups[cells], alpha[cells], r[cells]
    rng_state = np.random.default_rng(100)
    rng_lab = np.random.default_rng({"self": 201, "fast": 202, "double": 203}[a.scenario])
    r_obs = r_gen * np.exp(rng_state.normal(0, 0.1, size=len(r_gen)))
    theta_true, Ub, n_nudged = true_theta(theta_fit, c_true, a.scenario)
    Uc, Sc, totc, labels = simulate(model, theta_true, Ub, tau_fit, th_age, label_cs, T_c, r_gen, a_c, amb_g,
                                    rng_state, rng_lab)
    beta_true = theta_true[:, -2]
    hl_true = float(np.log(2) * c_true / np.median(beta_true))
    sim_disp = dispersion_ratios(Uc[fit_c], Sc[fit_c], sp_real)
    rep = dict(scenario=a.scenario, quick=a.quick, c_true=c_true, label_cs=label_cs, n_fit=int(fit_c.sum()),
               n_score=int(score_c.sum()), n_beta_nudged=n_nudged,
               truth=dict(unspliced_half_life_h=hl_true, gate2_true=bool(hl_true < 1.0),
                          beta_median_per_h=float(np.median(beta_true) / c_true),
                          gamma_median_per_h=float(np.median(theta_true[:, -1]) / c_true),
                          c_E_from_true_gamma=float(np.median(theta_true[:, -1]) * probe.HALF_LIFE_EXT / np.log(2))),
               dispersion_median_u_s=dict(real=real_disp, sim=sim_disp),
               observed_unspliced_share=float(Uc.sum() / totc.sum()))
    print("sim", json.dumps(rep), flush=True)

    def dump():
        with open(os.path.join(out, "sim.json"), "w") as fh:
            json.dump(rep, fh, indent=1)

    Xf = np.stack([Uc[fit_c], Sc[fit_c]], axis=-1)
    n_init, epoch = (2, 5) if a.quick else (8, 100)
    with Pool(min(a.procs, n_init)) as pool:
        fits = pool.map(probe.fit_one, [(Xf, r_obs[fit_c], s, epoch) for s in range(n_init)], chunksize=1)
    rhos = []
    for f in fits:
        post, tt = posteriors(model, f["theta"], f["tau"], Xf, r_obs[fit_c])
        rhos.append(float(spearmanr(post.sum(1) @ tt, T_c[fit_c]).correlation))
        del post
    forward = [f for f, rho in zip(fits, rhos) if rho > 0]
    rep["decoder"] = dict(elbos=[f["elbo"] for f in fits], spearman_t_vs_T=rhos)
    if a.quick and not forward:  # smoke run only: exercise the downstream code anyway
        forward = fits
        rep["decoder"]["smoke_override_A1"] = True
    if not forward:
        rep["verdict_frozen"] = "UNINFORMATIVE (A1: no forward initialisation)"
        rep["recovery"] = dict(R1=False, R2=False, R3=False)
        rep["gates"] = dict(gate1=False, gate2=False, unspliced_half_life_h_at_cA=None)
        dump()
        print("VERDICT", rep["verdict_frozen"], flush=True)
        return
    best = max(forward, key=lambda d: d["elbo"])
    theta, tau = best["theta"], best["tau"]
    beta_u, gamma_u = theta[:, -2], theta[:, -1]
    rep["decoder"].update(best_seed=best["seed"])

    # collection-time clocks (probe code)
    lf, t = probe.grid_loglik(model, Xf, theta, tau, r_obs[fit_c])
    Lbar = logsumexp(lf, axis=1) - np.log(probe.TOPO.shape[0])
    del lf
    Mi = Lbar.max(1)
    Ei = np.exp(Lbar - Mi[:, None])
    Ts = sorted(probe.GROUP_T.values())
    Tf = T_c[fit_c]
    Ms = [Mi[Tf == T] for T in Ts]
    Es = [Ei[Tf == T] for T in Ts]
    logit = lambda p: np.log(p / (1 - p))
    age_starts = [[logit((T0 - 48) / 36), np.log(c), 0.0, w0, 0.0]
                  for T0 in (60.0, 72.0) for c in (12.0, 24.0, 48.0) for w0 in (-2.0, 2.0)]
    gauss_starts = [[m0, s, np.log(sd)] for m0 in (0.0, 1.0) for s in (0.5, 1.0) for sd in (0.5, 1.5)]
    if a.quick:
        age_starts, gauss_starts = age_starts[:2], gauss_starts[:2]
    th_a, _ = probe.fit_prior("age", Ms, Es, Ts, t, age_starts)
    th_g, _ = probe.fit_prior("gauss", Ms, Es, Ts, t, gauss_starts)
    c_A_hat = float(np.exp(th_a[1]))
    c_G = float(24.0 / th_g[1]) if th_g[1] > 0 else None
    c_E = float(np.median(gamma_u) * probe.HALF_LIFE_EXT / np.log(2))
    rep["clocks"] = dict(c_A=c_A_hat, c_G=c_G, c_U=24.0, c_E=c_E, T0=float(48 + 36 / (1 + np.exp(-th_a[0]))))
    dump()

    # label clocks
    Xs = np.stack([Uc[score_c], Sc[score_c]], axis=-1)
    q, _ = posteriors(model, theta, tau, Xs, r_obs[score_c])
    a_s, g_s = np.clip(a_c[score_c], 1e-6, None), g_c[score_c]
    tot_s = totc[score_c]
    n_boot = 200 if a.quick else 2000
    hl_u = float(c_A_hat * np.log(2) / np.median(beta_u))
    gate2 = hl_u < 1.0
    rep["gates"] = dict(unspliced_half_life_h_at_cA=hl_u, gate2=bool(gate2))
    rep["R2"] = dict(c_A_hat=c_A_hat, within_1_25=bool(c_true / 1.25 <= c_A_hat <= c_true * 1.25))
    # decoder-oracle posteriors: true theta with the unspliced capture factor (Chronocell's 'Ub' parameter)
    t_or = np.linspace(tau_fit[0], tau_fit[-1], probe.M)
    lf_or = model.get_logL(Xs, theta_true, t_or, tau_fit, probe.TOPO, {"r": r_obs[score_c], "Ub": Ub})
    q_true = np.exp(lf_or - logsumexp(lf_or, axis=(1, 2), keepdims=True))
    del lf_or
    rep["labels"] = {}
    for c_lab, new_all in zip(label_cs, labels):
        new_s = new_all[score_c]
        key = "main" if c_lab == c_true else "double_label"
        prof = label_profile(model, theta, tau, q, new_s, tot_s, a_s, g_s, amb_g)
        c_label, lo, hi, am = band_from(prof, n_boot)
        res = assess(c_lab, c_label, lo, hi, am)
        res["S_profile"] = prof[1.0].mean(0).tolist()
        gate1 = not res["peak_at_edge"]
        inside = lambda c: (c is not None) and (lo / RSTEP <= c <= hi * RSTEP)
        others = [24.0, c_E, c_G]
        n_out = sum(1 for c in others if c is not None and not inside(c))

        def band_verdict():
            if inside(c_A_hat) and not inside(24.0) and not inside(c_E):
                return "POSITIVE"
            if inside(c_A_hat) and all(inside(c) for c in others if c is not None):
                return "UNINFORMATIVE (band contains all)"
            if (not inside(c_A_hat)) and n_out >= 2:
                return "NEGATIVE"
            return "INCONCLUSIVE"

        res["verdict_frozen"] = ("UNINFORMATIVE (gate 1)" if not gate1 else
                                 "UNINFORMATIVE (gate 2)" if not gate2 else band_verdict())
        res["verdict_without_gate2"] = "UNINFORMATIVE (gate 1)" if not gate1 else band_verdict()
        # oracle arms
        # beta-oracle: refitted decoder, but old unspliced RNA decays at the true splicing rate in 1/h (beta_true is
        # per process unit of the label clock c_lab)
        po = label_profile(model, theta, tau, q, new_s, tot_s, a_s, g_s, amb_g, beta_decay_per_h=beta_true / c_lab)
        res["beta_oracle"] = assess(c_lab, *band_from(po, n_boot))
        # decoder-oracle: true theta in process units, observed unspliced = Ub * true unspliced, true posteriors
        pd_ = label_profile(model, theta_true, tau_fit, q_true, new_s, tot_s, a_s, g_s, amb_g,
                            beta_flux=beta_true / Ub, Ub=Ub)
        res["decoder_oracle"] = assess(c_lab, *band_from(pd_, n_boot))
        rep["labels"][key] = res
        dump()
    main_res = rep["labels"]["main"]
    rep["recovery"] = dict(
        R1=main_res["R1"], R2=rep["R2"]["within_1_25"],
        R3=bool((not main_res["peak_at_edge"]) and (hl_true / 1.25 <= hl_u <= hl_true * 1.25)),
        hl_true=hl_true, hl_estimated=hl_u,
        fitted_beta_median_per_h_at_c_true=float(np.median(beta_u) / c_true),
        fitted_gamma_median_per_h_at_c_true=float(np.median(gamma_u) / c_true))
    rep["seconds"] = time.time() - T_start
    dump()
    print("RESULT", json.dumps(dict(scenario=a.scenario, c_true=c_true, clocks=rep["clocks"],
                                    recovery=rep["recovery"],
                                    labels={k: {kk: v[kk] for kk in ("c_label", "band", "R1", "verdict_frozen",
                                                                     "verdict_without_gate2")}
                                            | {"beta_oracle_c": v["beta_oracle"]["c_label"],
                                               "decoder_oracle_c": v["decoder_oracle"]["c_label"]}
                                            for k, v in rep["labels"].items()})), flush=True)


if __name__ == "__main__":
    main()
