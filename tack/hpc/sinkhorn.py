"""Batched entropic optimal transport between many pairs of small discrete measures (log-domain Sinkhorn).

solve(a, b, C, eps) returns the plans pi (G, n, n) minimising <pi, C> + eps KL(pi | a x b) with row sums a and
column sums b, for G independent problems at once: a, b (G, n) are nonnegative rows summing to 1 (zero entries
allowed), C (G, n, n) the costs, eps a scalar or (G,). Potentials are updated in the log domain, so tiny eps or
far-apart supports do not overflow; each problem stops iterating once its row marginal is within tol (active
set), so a few slow problems do not hold back the rest. Returns (pi, max row-marginal error).
"""
import numpy as np


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def _plan(f, g, Ce, eps):
    return np.exp(f[:, :, None] / eps[:, :, None] + g[:, None, :] / eps[:, :, None] - Ce)


def solve(a, b, C, eps, tol=1e-9, max_iter=5000, check_every=10):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    G = a.shape[0]
    eps = np.broadcast_to(np.asarray(eps, float), (G,)).reshape(-1, 1).copy()
    with np.errstate(divide="ignore"):
        la, lb = np.log(a), np.log(b)
    Ce = C / eps[:, :, None]
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    act = np.arange(G)
    for it in range(max_iter):
        e, c = eps[act], Ce[act]
        fa = e * (la[act] - _lse(g[act][:, None, :] / e[:, :, None] - c, axis=2))
        fa = np.where(a[act] > 0, fa, -np.inf)
        ga = e * (lb[act] - _lse(fa[:, :, None] / e[:, :, None] - c, axis=1))
        ga = np.where(b[act] > 0, ga, -np.inf)
        f[act], g[act] = fa, ga
        if (it + 1) % check_every == 0:
            err = np.abs(_plan(fa, ga, c, e).sum(2) - a[act]).max(1)
            act = act[err >= tol]
            if len(act) == 0:
                break
    pi = _plan(f, g, Ce, eps)
    assert np.all(np.isfinite(pi)), "non-finite transport plan"
    return pi, float(np.max(np.abs(pi.sum(2) - a)))
