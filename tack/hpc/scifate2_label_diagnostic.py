"""EXPLORATORY, post-hoc (declared after the UNINFORMATIVE verdict of job 679647): why do the labels ask for a
clock faster than 3 h per process unit?

Same cells and genes as the probe (same seeds). Model-free per-gene quantities on score cells:
  pi_obs_g = sum new / sum (alpha * total)            (detection-corrected new fraction)
  u_obs_g  = sum U / sum (U + S)                       (unspliced share)
  implied mRNA half-life: assuming steady state and that unspliced reads are all new,
  pi_obs = u + (1 - u) (1 - exp(-gamma L))  ->  gamma_lab = -ln(1 - (pi_obs - u)/(1 - u)) / L
Decoder side (frozen theta, posteriors with the uniform prior): predicted unspliced share and new fraction per
gene at c_A and at c = 3. Also the genome-wide detection-corrected new fraction.
"""
import json
import os

import numpy as np
from scipy.special import logsumexp

import scifate2_clock_probe as probe
from label_model import new_fraction


def main():
    from Chronocell.models import two_species_ss as model

    out = os.path.expanduser("~/tack/out/scifate2_clock")
    res = json.load(open(os.path.join(out, "clock_probe.json")))
    dec = np.load(os.path.join(out, "decoder.npz"), allow_pickle=True)
    theta, tau = dec["theta"], dec["tau"]
    obs, tot, new, U, S, genes, amb = probe.load(False)
    groups = (obs["timepoint"].astype(str) + "_" + obs["rep"].astype(str)).to_numpy()
    keep, alpha, _ = probe.qc(obs, tot, genes)
    fit, score = probe.split(groups, keep, 1000)
    r, gidx, ginfo = probe.depth_and_genes(U, S, fit, genes, amb, 150)
    assert list(genes[gidx].astype(str)) == list(dec["genes"].astype(str)), "gene set mismatch"
    a_s = alpha[score]
    new_s = new[score][:, gidx].toarray()
    tot_s = tot[score][:, gidx].toarray()
    U_s = U[score][:, gidx].toarray()
    S_s = S[score][:, gidx].toarray()
    pi_obs = new_s.sum(0) / (a_s[:, None] * tot_s).sum(0)
    u_obs = U_s.sum(0) / np.maximum((U_s + S_s).sum(0), 1)
    L = probe.L_PULSE
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (pi_obs - u_obs) / (1 - u_obs)
        gamma_lab = -np.log(1 - np.clip(x, 1e-6, 1 - 1e-6)) / L
    half_lab = np.log(2) / gamma_lab
    # genome-wide detection-corrected new fraction (all genes, score cells)
    allnew = np.asarray(new[score].sum(1)).ravel()
    alltot = np.asarray(tot[score].sum(1)).ravel()
    pi_global = float(allnew.sum() / (a_s * alltot).sum())

    # decoder predictions averaged over score cells
    Xs = probe.dense3(U, S, gidx, score)
    logL_s, t = probe.grid_loglik(model, Xs, theta, tau, r[score])
    q = np.exp(logL_s - logsumexp(logL_s, axis=(1, 2), keepdims=True))
    Y = model.get_Y_hat(theta, t, tau, probe.TOPO, {})
    w = q.sum(0) / q.sum()  # (L, m) average posterior mass
    U_pred = np.einsum("lm,lmp->p", w, Y[..., 0])
    S_pred = np.einsum("lm,lmp->p", w, Y[..., 1])
    u_pred = U_pred / (U_pred + S_pred)
    beta_u, gamma_u = theta[:, -2], theta[:, -1]
    preds = {}
    for name, c in (("c_A", res["clocks"]["c_A"]), ("c3", 3.0), ("c_E", res["clocks"]["c_E"])):
        h = L / c
        Y0 = model.get_Y_hat(theta, np.clip(t - h, tau[0], None), tau, probe.TOPO, {})
        pi = new_fraction(Y[..., 0], Y[..., 1], Y0[..., 0], Y0[..., 1], beta_u, gamma_u, h, amb[gidx])
        preds[name] = np.einsum("lm,lmp->p", w, pi)
    order = np.argsort(half_lab)
    g = genes[gidx].astype(str)
    rep = dict(
        note="EXPLORATORY post-hoc diagnostic, not confirmatory",
        pi_global_detection_corrected=pi_global,
        median_pi_obs=float(np.median(pi_obs)), median_u_obs=float(np.median(u_obs)),
        median_u_pred=float(np.median(u_pred)),
        median_half_life_lab_h=float(np.nanmedian(half_lab)),
        frac_genes_half_life_lab_lt_1h=float(np.nanmean(half_lab < 1)),
        median_pi_pred={k: float(np.median(v)) for k, v in preds.items()},
        corr_log_gamma_lab_vs_log_gamma_unit=float(np.corrcoef(np.log(gamma_lab[np.isfinite(gamma_lab)]),
                                                               np.log(gamma_u[np.isfinite(gamma_lab)]))[0, 1]),
        c_implied_by_median_gamma=float(np.median(gamma_u) / np.nanmedian(gamma_lab)),
        fastest_genes=[(g[i], round(float(half_lab[i]), 2), round(float(pi_obs[i]), 3), round(float(u_obs[i]), 3))
                       for i in order[:15]],
        slowest_genes=[(g[i], round(float(half_lab[i]), 2), round(float(pi_obs[i]), 3), round(float(u_obs[i]), 3))
                       for i in order[-15:]],
    )
    json.dump(rep, open(os.path.join(out, "label_diagnostic.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
