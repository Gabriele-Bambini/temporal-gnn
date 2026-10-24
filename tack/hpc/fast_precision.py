"""Precision in the SELF and FAST worlds with a perfect decoder, and joint identifiability of detection and clock.

For each world (SELF: slow splicing as fitted; FAST: unspliced half-life 15 min with capture Ub), data are
simulated on the real score cells as in sim_recovery.py and scored with the decoder-oracle (true theta, true Ub,
true posteriors). Outputs:
  1. 1-D: S(c) on a fine log grid (61 points in [c/3, 3c]) with exact alpha; continuous argmax (parabola on the
     three best points in log c); 95 % likelihood-ratio interval from the TOTAL log-likelihood (drop 1.92);
  2. alpha scan: argmax c for alpha x s, s in 0.6 ... 1.6, and the slope d log c / d log s;
  3. 2-D: total log-likelihood T(s, c) over s in [0.5, 2] (31 points) and the c grid; the joint 95 % region
     (drop 3.0, chi-square 2 df) and its extent in s and c; whether the joint maximum is closed (inside the box).
Overdispersion is ignored (binomial), so intervals are optimistic; the question is ridge versus closed peak.

Usage: python fast_precision.py [--quick]
"""
import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

import scifate2_clock_probe as probe  # noqa: E402
import sim_recovery as sr  # noqa: E402
from label_model import new_fraction  # noqa: E402


def pi_cells(model, theta, tau, q, amb_g, Ub, c_grid):
    """Predicted true-new read fraction per cell and gene for each c: array (nC, n, p)."""
    t = np.linspace(tau[0], tau[-1], probe.M)
    Y = model.get_Y_hat(theta, t, tau, probe.TOPO, {})
    Y[..., 0] *= Ub
    beta, gamma = theta[:, -2], theta[:, -1]
    out = []
    for c in c_grid:
        h = probe.L_PULSE / c
        Y0 = model.get_Y_hat(theta, np.clip(t - h, tau[0], None), tau, probe.TOPO, {})
        Y0[..., 0] *= Ub
        pi = new_fraction(Y[..., 0], Y[..., 1], Y0[..., 0], Y0[..., 1], beta, gamma, h, amb_g, beta_flux=beta / Ub)
        out.append(np.einsum("nlm,lmp->np", q, pi).astype(np.float32))
    return np.stack(out)


def total_ll(PI, new_s, old_s, a_s, s):
    """Total binomial log-likelihood for each c (vector over nC) at detection alpha x s."""
    P = np.clip(s * a_s[None, :, None] * PI, 1e-6, 1 - 1e-6)
    return (new_s[None] * np.log(P) + old_s[None] * np.log1p(-P)).sum(axis=(1, 2))


def cont_argmax(x_log, y):
    k = int(np.argmax(y))
    if k in (0, len(y) - 1):
        return float(np.exp(x_log[k])), True
    xs, ys = x_log[k - 1:k + 2], y[k - 1:k + 2]
    a, b, _ = np.polyfit(xs, ys, 2)
    return float(np.exp(-b / (2 * a))), False


def run_world(model, world, theta_fit, tau, th_age, c_true, cells, quick):
    Tc, gc, ac, rc, ambg = cells
    theta_true, Ub, _ = sr.true_theta(theta_fit, c_true, world)
    Uc, Sc, totc, labels = sr.simulate(model, theta_true, Ub, tau, th_age, [c_true], Tc, rc, ac, ambg,
                                       np.random.default_rng(100), np.random.default_rng(201))
    newc = labels[0]
    t_grid = np.linspace(tau[0], tau[-1], probe.M)
    lf = model.get_logL(np.stack([Uc, Sc], axis=-1), theta_true, t_grid, tau, probe.TOPO, {"r": rc, "Ub": Ub})
    q = np.exp(lf - logsumexp(lf, axis=(1, 2), keepdims=True))
    del lf
    c_grid = np.geomspace(c_true / 3, c_true * 3, 31 if quick else 61)
    PI = pi_cells(model, theta_true, tau, q, ambg, Ub, c_grid)
    old = totc - newc
    a_s = np.clip(ac, 1e-6, None)
    lc = np.log(c_grid)
    # 1-D, exact alpha
    T1 = total_ll(PI, newc, old, a_s, 1.0)
    c_hat, edge = cont_argmax(lc, T1)
    inside = c_grid[T1 >= T1.max() - 1.92]
    res = dict(world=world, c_true=c_true, unspliced_half_life_true_h=float(np.log(2) * c_true /
                                                                               np.median(theta_true[:, -2])),
               c_hat_exact_alpha=c_hat, at_edge=edge, lr95_exact_alpha=[float(inside.min()), float(inside.max())],
               ratio_c_hat_true=c_hat / c_true)
    # alpha scan
    scan = {}
    for s in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]:
        ch, e = cont_argmax(lc, total_ll(PI, newc, old, a_s, s))
        scan[str(s)] = dict(c_hat=ch, at_edge=e)
    pts = [(np.log(float(k)), np.log(v["c_hat"])) for k, v in scan.items() if not v["at_edge"]]
    res["alpha_scan"] = scan
    res["slope_dlogc_dlogalpha"] = float(np.polyfit(*np.array(pts).T, 1)[0]) if len(pts) > 2 else None
    # 2-D surface
    s_grid = np.geomspace(0.5, 2.0, 21 if quick else 31)
    T2 = np.stack([total_ll(PI, newc, old, a_s, s) for s in s_grid])  # (nS, nC)
    i, j = np.unravel_index(int(np.argmax(T2)), T2.shape)
    region = T2 >= T2.max() - 3.0
    si, cj = np.where(region)
    res["joint"] = dict(
        s_hat=float(s_grid[i]), c_hat=float(c_grid[j]),
        closed_peak=bool(0 < i < len(s_grid) - 1 and 0 < j < len(c_grid) - 1
                         and not region[0].any() and not region[-1].any()
                         and not region[:, 0].any() and not region[:, -1].any()),
        region95_s=[float(s_grid[si.min()]), float(s_grid[si.max()])],
        region95_c=[float(c_grid[cj.min()]), float(c_grid[cj.max()])],
        profile_c=[float(v) for v in T2.max(0) - T2.max()],
        profile_s=[float(v) for v in T2.max(1) - T2.max()],
        s_grid=s_grid.tolist(), c_grid=c_grid.tolist())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from Chronocell.models import two_species_ss as model

    real = json.load(open(os.path.join(sr.OUT_REAL, "clock_probe.json")))
    dec = np.load(os.path.join(sr.OUT_REAL, "decoder.npz"), allow_pickle=True)
    theta_fit, tau = dec["theta"], dec["tau"]
    th_age = real["clocks"]["age"]["theta"]
    c_A = real["clocks"]["c_A"]
    obs, tot, new, U, S, genes, amb = probe.load(False)
    groups = (obs["timepoint"].astype(str) + "_" + obs["rep"].astype(str)).to_numpy()
    Tcell = obs["timepoint"].astype(str).map(probe.GROUP_T).to_numpy(float)
    keep, alpha, _ = probe.qc(obs, tot, genes)
    fit, score = probe.split(groups, keep, 1000)
    r, gidx, _ = probe.depth_and_genes(U, S, fit, genes, amb, 150)
    del tot, new, U, S
    ambg = np.nan_to_num(amb[gidx], nan=0.0)
    Tc, gc, ac, rc = Tcell[score], groups[score], alpha[score], r[score]
    if a.quick:
        sub = np.random.default_rng(3).random(len(Tc)) < 0.2
        Tc, gc, ac, rc = Tc[sub], gc[sub], ac[sub], rc[sub]
    out = {}
    for world in ("self", "fast"):
        out[world] = run_world(model, world, theta_fit, tau, th_age, c_A, (Tc, gc, ac, rc, ambg), a.quick)
        print(world, json.dumps({k: v for k, v in out[world].items() if k not in ("alpha_scan",)}
                                | {"joint": {k: v for k, v in out[world]["joint"].items()
                                             if k not in ("profile_c", "profile_s", "s_grid", "c_grid")}}),
              flush=True)
    od = os.path.expanduser(f"~/tack/out/precision_study{'_quick' if a.quick else ''}")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "fast_precision.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
