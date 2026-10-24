"""Batched log-domain Sinkhorn solvers with per-problem epsilon scaling (own code; numpy only).

solve(a, b, C, eps): balanced entropic OT, min <pi, C> + eps KL(pi | a x b), row sums a, column sums b (equal
totals), same iterations as tack/hpc/sinkhorn.py but with absolute potentials warm-started through a decreasing
schedule eps_k = max(eps, Cmax 2^-k) per problem.

solve_tv(a, b, C, eps, rho): unbalanced entropic OT with a SINK on the a side and a SOURCE on the b side, each
priced rho per unit of mass:
    min_pi  <pi, C> + eps KL(pi | R) + rho (|a| - |pi|) + rho (|b| - |pi|)   s.t.  pi 1 <= a,  pi^T 1 <= b,
a - pi 1 = mass destroyed at each a level, b - pi^T 1 = mass created at each b level (the one-sided total-variation
marginal penalty; in the eps -> 0 limit this is the dummy-node partial-transport problem: level -> sink cost rho,
source -> level cost rho, source -> sink 0). Reference R = m0 (a / |a|) x (b / |b|), m0 = min(|a|, |b|), so the
entropic term alone does not push mass into the sink / source (with R = a x b its total |a||b| << min(|a|, |b|)
for small nonzero fractions would). Dual updates (Sejourne et al. 2019, aprox of the one-sided TV = clipping):
    f = min(rho, eps [log(|a| / m0) - LSE_y(log b~_y + (g_y - C_xy) / eps)]),   g likewise,
    pi_xy = m0 a~_x b~_y exp((f_x + g_y - C_xy) / eps).
Stops per problem when the potentials move by < tol eps in a sweep (checked every check_every sweeps).
"""
import numpy as np


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def _schedule(eps, cmax):
    k = 0
    while True:
        e = np.maximum(eps, cmax * 0.5 ** k)
        yield e, bool(np.all(e <= eps))
        k += 1


# ------------------------------------------------------------------ balanced
def _sweep(f, g, la, lb, a_pos, b_pos, C, e):
    f = e[:, None] * (la - _lse((g[:, None, :] - C) / e[:, None, None], axis=2))
    f = np.where(a_pos, f, -np.inf)
    g = e[:, None] * (lb - _lse((f[:, :, None] - C) / e[:, None, None], axis=1))
    g = np.where(b_pos, g, -np.inf)
    return f, g


def _plan(f, g, C, e):
    return np.exp((f[:, :, None] + g[:, None, :] - C) / e[:, None, None])


def solve(a, b, C, eps, tol=1e-7, iters_stage=12, max_final=3000, check_every=10):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    G = a.shape[0]
    eps = np.broadcast_to(np.asarray(eps, float), (G,)).copy()
    with np.errstate(divide="ignore"):
        la, lb = np.log(a), np.log(b)
    a_pos, b_pos = a > 0, b > 0
    cmax = np.max(np.where(a_pos[:, :, None] & b_pos[:, None, :], C, 0.0), axis=(1, 2))
    f, g = np.zeros_like(a), np.zeros_like(b)
    for e, last in _schedule(eps, cmax):
        if last:
            break
        for _ in range(iters_stage):
            f, g = _sweep(f, g, la, lb, a_pos, b_pos, C, e)
    act = np.arange(G)
    n_it = 0
    for it in range(max_final):
        fa, ga = _sweep(f[act], g[act], la[act], lb[act], a_pos[act], b_pos[act], C[act], eps[act])
        f[act], g[act] = fa, ga
        n_it = it + 1
        if n_it % check_every == 0:
            err = np.abs(_plan(fa, ga, C[act], eps[act]).sum(2) - a[act]).max(1)
            act = act[err >= tol]
            if len(act) == 0:
                break
    pi = _plan(f, g, C, eps)
    assert np.all(np.isfinite(pi)), "non-finite plan"
    return pi, float(np.max(np.abs(pi.sum(2) - a))), n_it


# ------------------------------------------------------------------ unbalanced, one-sided TV (sink / source)
def _sweep_tv(f, g, lat, lbt, ca, cb, a_pos, b_pos, C, e, rho):
    ee = e[:, None]
    f = ee * (ca[:, None] - _lse(lbt[:, None, :] + (g[:, None, :] - C) / e[:, None, None], axis=2))
    f = np.where(a_pos, np.minimum(f, rho[:, None]), -np.inf)
    g = ee * (cb[:, None] - _lse(lat[:, :, None] + (f[:, :, None] - C) / e[:, None, None], axis=1))
    g = np.where(b_pos, np.minimum(g, rho[:, None]), -np.inf)
    # translation step (Sejourne et al. 2022): (f + c, g - c) leaves every unclipped plan entry unchanged; at the
    # optimum the side with more mass has a clipped (destroying / creating) level, i.e. max potential = rho there,
    # so jump straight to it instead of drifting eps log(|a| / |b|) per sweep. Fixed points are unchanged.
    c = np.where(ca > cb, rho - f.max(1), np.where(cb > ca, -(rho - g.max(1)), 0.0))
    return f + c[:, None], g - c[:, None]


def solve_tv(a, b, C, eps, rho, tol=1e-7, iters_stage=12, max_final=3000, check_every=10):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    G = a.shape[0]
    eps = np.broadcast_to(np.asarray(eps, float), (G,)).copy()
    rho = np.broadcast_to(np.asarray(rho, float), (G,)).copy()
    ta, tb = a.sum(1), b.sum(1)
    m0 = np.minimum(ta, tb)
    with np.errstate(divide="ignore"):
        lat, lbt = np.log(a / ta[:, None]), np.log(b / tb[:, None])
    ca, cb = np.log(ta / m0), np.log(tb / m0)
    a_pos, b_pos = a > 0, b > 0
    cmax = np.max(np.where(a_pos[:, :, None] & b_pos[:, None, :], C, 0.0), axis=(1, 2))
    cmax = np.maximum(cmax, rho)
    f, g = np.zeros_like(a), np.zeros_like(b)
    for e, last in _schedule(eps, cmax):
        if last:
            break
        for _ in range(iters_stage):
            f, g = _sweep_tv(f, g, lat, lbt, ca, cb, a_pos, b_pos, C, e, rho)
    act = np.arange(G)
    n_it, move = 0, np.zeros(G)
    for it in range(max_final):
        sl = (f[act], g[act], lat[act], lbt[act], ca[act], cb[act], a_pos[act], b_pos[act], C[act], eps[act], rho[act])
        fa, ga = _sweep_tv(*sl)
        n_it = it + 1
        if n_it % check_every == 0:
            with np.errstate(invalid="ignore"):
                df = np.where(a_pos[act], np.abs(fa - f[act]), 0.0).max(1)
                dg = np.where(b_pos[act], np.abs(ga - g[act]), 0.0).max(1)
            mv = np.maximum(df, dg) / eps[act]
            move[act] = mv
            f[act], g[act] = fa, ga
            act = act[mv >= tol]
            if len(act) == 0:
                break
        else:
            f[act], g[act] = fa, ga
    pi = m0[:, None, None] * np.exp(lat[:, :, None] + lbt[:, None, :]
                                    + (f[:, :, None] + g[:, None, :] - C) / eps[:, None, None])
    pi = np.where(a_pos[:, :, None] & b_pos[:, None, :], pi, 0.0)
    assert np.all(np.isfinite(pi)), "non-finite plan"
    # feasibility: rows <= a, columns <= b (exact up to the last sweep)
    over = max(float(np.max(pi.sum(2) - a)), float(np.max(pi.sum(1) - b)))
    return pi, float(move.max()), over, n_it
