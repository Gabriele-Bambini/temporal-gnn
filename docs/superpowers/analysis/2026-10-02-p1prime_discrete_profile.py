"""Numerical check of P1' with DISCRETE sampling times (the symbolic check covers continuous T only).

Population layer only: each cell's process time t = a / c is observed exactly (perfect decoder), a = age in
hours since entry, c = hours per process unit, absorbing end at t = 2. Snapshots at K = 5 real times.
We simulate three truths and profile the log-likelihood over c (other parameters re-optimised):
  edge   : constant influx from a known-shape onset T0, net growth g        -> c should be identified
  exp    : exponential influx (no onset in the window), growth g            -> c should NOT be identified
  pulse  : Gaussian pulse of influx + growth quadratic in age               -> c should NOT be identified
           (and the same truth fitted with growth linear in age only        -> weakly identified at most)
Output: profile curves (relative log-likelihood) and a verdict per case.
"""
import json

import numpy as np
from scipy.optimize import minimize

T_SAMPLES = np.array([12.0, 24.0, 36.0, 48.0, 60.0])  # hours
N_PER_T = 2000
GRID = np.linspace(0, 2, 201)
DA = 0.25  # age integration step (h)
C_TRUE = 24.0


def q_given_T(T, log_rho, log_G, c, a_max=2000.0, sub=8, n_abs=4000):
    """Discretised p(t | T) for t = a/c with absorbing end at t = 2.

    Interior bin j covers t in [edge_j, edge_{j+1}); its mass is integrated with `sub` midpoints in age
    (Jacobian c dt). The last bin collects all ages beyond c * (2 - dt/2) (absorbing), integrated to a_max.
    """
    from scipy.special import logsumexp

    d = GRID[1] - GRID[0]
    edges = np.concatenate([[0.0], GRID[:-1] + d / 2])
    lo, hi = edges[:-1], edges[1:]
    u = (np.arange(sub) + 0.5) / sub
    tt = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    a = c * tt
    with np.errstate(divide="ignore", invalid="ignore"):
        lw_int = log_rho(T - a) + log_G(a) + np.log(c * (hi - lo)[:, None] / sub)
    a_lo = c * edges[-1]
    aa = np.linspace(a_lo, max(a_max, a_lo + 1.0), n_abs + 1)
    am = 0.5 * (aa[:-1] + aa[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        lw_abs = log_rho(T - am) + log_G(am) + np.log(aa[1] - aa[0])
    lw_int = np.where(np.isfinite(lw_int), lw_int, -np.inf)
    lw_abs = np.where(np.isfinite(lw_abs), lw_abs, -np.inf)
    logq = np.concatenate([logsumexp(lw_int, axis=1), [logsumexp(lw_abs)]])
    if not np.isfinite(logq).any():  # nobody has entered yet: all mass at t = 0
        q = np.full(len(GRID), 1e-300)
        q[0] = 1.0
        return q / q.sum()
    q = np.exp(logq - logsumexp(logq[np.isfinite(logq)]))
    q = np.maximum(q, 1e-300)
    return q / q.sum()


def family(kind, th, c):
    if kind == "edge":
        T0, g = th
        log_rho = lambda E: np.where(E >= T0, 0.0, -np.inf)
        log_G = lambda a: g * a
    elif kind == "exp":
        lam, g = th
        log_rho = lambda E: lam * E
        log_G = lambda a: g * a
    elif kind == "pulse_quad":
        E0, log_s, g1, g2 = th
        log_rho = lambda E: -0.5 * ((E - E0) / np.exp(log_s)) ** 2
        log_G = lambda a: g1 * a + g2 * a**2
    elif kind == "pulse_lin":
        E0, log_s, g1 = th
        log_rho = lambda E: -0.5 * ((E - E0) / np.exp(log_s)) ** 2
        log_G = lambda a: g1 * a
    return log_rho, log_G


def loglik(kind, th, c, counts):
    log_rho, log_G = family(kind, th, c)
    return sum(float(counts[k] @ np.log(q_given_T(T, log_rho, log_G, c))) for k, T in enumerate(T_SAMPLES))


def simulate(kind, th, rng):
    log_rho, log_G = family(kind, th, C_TRUE)
    return np.stack([rng.multinomial(N_PER_T, q_given_T(T, log_rho, log_G, C_TRUE)) for T in T_SAMPLES])


def profile(kind, counts, starts, c_grid):
    out = []
    for c in c_grid:
        best = -np.inf
        for x0 in starts:
            r = minimize(lambda x: -loglik(kind, x, c, counts), np.asarray(x0, float), method="Nelder-Mead",
                         options=dict(maxiter=600, xatol=1e-4, fatol=1e-4))
            best = max(best, -r.fun)
        out.append(best)
    out = np.array(out)
    return (out - out.max()).tolist()


def main():
    rng = np.random.default_rng(0)
    c_grid = np.geomspace(6, 96, 13)
    cases = {
        "edge": dict(truth=("edge", [0.0, 0.0]), fit="edge", starts=[[-6.0, 0.0], [6.0, 0.01]]),
        "exp": dict(truth=("exp", [0.03, 0.0]), fit="exp", starts=[[0.0, 0.0], [0.05, 0.02], [-0.03, -0.02]]),
        "pulse_quad": dict(truth=("pulse_quad", [24.0, np.log(10.0), 0.0, 0.0]), fit="pulse_quad",
                           starts=[[24.0, np.log(10.0), 0.0, 0.0], [20.0, np.log(15.0), 0.01, 0.0]]),
        "pulse_lin": dict(truth=("pulse_quad", [24.0, np.log(10.0), 0.0, 0.0]), fit="pulse_lin",
                          starts=[[24.0, np.log(10.0), 0.0], [20.0, np.log(15.0), 0.01]]),
    }
    res = {}
    for name, cfg in cases.items():
        counts = simulate(*cfg["truth"], rng)
        prof = profile(cfg["fit"], counts, cfg["starts"], c_grid)
        p = np.array(prof)
        # likelihood-ratio 95% interval for c: profile within 1.92 of the maximum
        inside = c_grid[p >= -1.92]
        res[name] = dict(c_grid=c_grid.round(2).tolist(), rel_loglik=[round(v, 2) for v in prof],
                         c_hat=float(c_grid[int(np.argmax(p))]),
                         ci95=[float(inside.min()), float(inside.max())],
                         identified=bool(inside.min() > c_grid[0] and inside.max() < c_grid[-1]))
        print(name, json.dumps(res[name]), flush=True)
    return res


if __name__ == "__main__":
    out = main()
    import pathlib

    pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
