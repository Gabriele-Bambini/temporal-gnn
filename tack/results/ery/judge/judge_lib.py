"""Judge: INDEPENDENT re-implementation of the metrics and the OT solver (nothing imported from the audited code).

Conventions kept on purpose (they are protocol, not method): PCA arrays from the npz files, 1500-cell subsamples
sub(n, seed) = sorted rng(seed).choice(n, min(n, 1500)), debiased Sinkhorn = transport cost <P, C> of the converged
entropic plan, eps = 0.05 * S, fixed S from the PAST cells only.
Solver: own Sinkhorn written from the fixed-point equations P = diag(u) K diag(v), K = exp(-C / eps),
u = (a / K v)^k, v = (b / K' u)^k, k = 1 (balanced) or tau / (tau + eps) (KL-relaxed). Convergence is judged on the
MARGINALS (balanced: max relative marginal error < 1e-9) or on the scalings (unbalanced: max |d log u| < 1e-10),
up to 20000 iterations, with a log-domain path (scipy logsumexp) when the kernel would underflow. Self terms use the
symmetric averaged update f <- (f + T(f)) / 2 (Feydy et al. 2019), a different iteration than the audited code.
"""
import numpy as np
from scipy.special import logsumexp

NSUB = 1500
STATS = {"calls": 0, "log_path": 0, "unconverged": 0, "max_marg_err": 0.0}


def sqdist(A, B):
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def sub(n, seed, k=NSUB):
    return np.sort(np.random.default_rng(seed).choice(n, min(n, k), replace=False))


def unif(n):
    return np.full(n, 1.0 / n)


def _log_sinkhorn(a, b, C, eps, k, maxit, tol):
    la, lb = np.log(a), np.log(b)
    f, g = np.zeros(len(a)), np.zeros(len(b))
    ok = False
    for _ in range(maxit):
        f_new = k * (eps * la - eps * logsumexp((g[None, :] - C) / eps, axis=1))
        g_new = k * (eps * lb - eps * logsumexp((f_new[:, None] - C) / eps, axis=0))
        dmax = max(np.abs(f_new - f).max(), np.abs(g_new - g).max()) / eps
        f, g = f_new, g_new
        if dmax < tol:
            ok = True
            break
    return np.exp((f[:, None] + g[None, :] - C) / eps), ok


def plan(a, b, C, eps, tau=None, maxit=20000, tol=1e-10):
    """Entropic (tau None) or KL-unbalanced entropic plan between weights a, b with cost C."""
    STATS["calls"] += 1
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    C = np.asarray(C, np.float64)
    k = 1.0 if tau is None else tau / (tau + eps)
    Ce = C / eps
    if Ce.min(1).max() > 500 or Ce.min(0).max() > 500:
        STATS["log_path"] += 1
        P, ok = _log_sinkhorn(a, b, C, eps, k, maxit, tol)
    else:
        K = np.exp(-Ce)
        u, v = np.ones(len(a)), np.ones(len(b))
        ok = False
        with np.errstate(all="ignore"):
            for it in range(maxit):
                u_new = (a / (K @ v)) ** k
                v = (b / (K.T @ u_new)) ** k
                if not (np.isfinite(u_new).all() and np.isfinite(v).all() and u_new.min() > 0 and v.min() > 0):
                    break
                if tau is None:
                    if it % 5 == 0:
                        r = u_new * (K @ v)
                        if np.abs(r / a - 1).max() < tol * 10:
                            u = u_new
                            ok = True
                            break
                else:
                    d = np.abs(np.log(u_new) - np.log(u)).max()
                    if d < tol:
                        u = u_new
                        ok = True
                        break
                u = u_new
        if not ok:
            STATS["log_path"] += 1
            P, ok = _log_sinkhorn(a, b, C, eps, k, maxit, tol)
        else:
            P = u[:, None] * K * v[None, :]
    if not ok:
        STATS["unconverged"] += 1
    if tau is None:
        err = max(np.abs(P.sum(1) / a - 1).max(), np.abs(P.sum(0) / b - 1).max())
        STATS["max_marg_err"] = max(STATS["max_marg_err"], float(err))
    return P


def self_plan(a, C, eps, maxit=20000, tol=1e-11):
    """Symmetric balanced problem by the averaged log update f <- (f + T(f)) / 2."""
    STATS["calls"] += 1
    la = np.log(a)
    f = np.zeros(len(a))
    ok = False
    for _ in range(maxit):
        Tf = eps * la - eps * logsumexp((f[None, :] - C) / eps, axis=1)
        f_new = 0.5 * (f + Tf)
        d = np.abs(f_new - f).max() / eps
        f = f_new
        if d < tol:
            ok = True
            break
    if not ok:
        STATS["unconverged"] += 1
    return np.exp((f[:, None] + f[None, :] - C) / eps)


def ot_cost(X, Y, wx, wy, eps):
    C = sqdist(X, Y)
    return float((plan(wx, wy, C, eps) * C).sum())


def ot_self(X, wx, eps):
    C = sqdist(X, X)
    return float((self_plan(wx, C, eps) * C).sum())


class Tgt:
    def __init__(self, Y, S):
        self.Y = np.asarray(Y, np.float64)
        self.w = unif(len(Y))
        self.S = S
        self.yy_ot = ot_self(self.Y, self.w, 0.05 * S)
        D = np.sqrt(sqdist(self.Y, self.Y))
        self.yyD = float(self.w @ D @ self.w)
        self.sw2 = float((self.w ** 2).sum())


def sd_fixed(X, wx, tgt):
    eps = 0.05 * tgt.S
    return (ot_cost(X, tgt.Y, wx, tgt.w, eps) - 0.5 * ot_self(X, wx, eps) - 0.5 * tgt.yy_ot) / tgt.S


def energies(X, wx, tgt):
    """(V-statistic energy as in the audited code, U-statistic energy with self pairs removed)."""
    X = np.asarray(X, np.float64)
    cross = float(wx @ np.sqrt(sqdist(X, tgt.Y)) @ tgt.w)
    xx = float(wx @ np.sqrt(sqdist(X, X)) @ wx)
    sx = float((wx ** 2).sum())
    ev = 2 * cross - xx - tgt.yyD
    eu = 2 * cross - xx / (1 - sx) - tgt.yyD / (1 - tgt.sw2)
    return ev, eu


def score(Z, w, tgt, seed=0):
    """Score a (weighted) predicted cloud: subsample seed 0 of the prediction (as the protocol), both energies and
    sd_fixed with the target's fixed S."""
    i = sub(len(Z), seed)
    wi = unif(len(i)) if w is None else np.asarray(w, np.float64)[i]
    keep = wi > 0
    i, wi = i[keep], wi[keep]
    wi = wi / wi.sum()
    X = np.asarray(Z, np.float64)[i]
    ev, eu = energies(X, wi, tgt)
    return {"energy": ev, "energy_U": eu, "sd_fixed": sd_fixed(X, wi, tgt), "ess": float(1 / (wi ** 2).sum()),
            "spread": float(np.sqrt(wi @ ((X - wi @ X) ** 2).sum(1)))}


def past_scale(Zp, seed=None, cap=3000):
    """median squared distance between distinct past cells; seed None = the first `cap` cells of a fixed permutation."""
    n = len(Zp)
    idx = np.arange(n) if n <= cap else np.sort(np.random.default_rng(123 if seed is None else seed).choice(n, cap, replace=False))
    C = sqdist(Zp[idx], Zp[idx])
    return float(np.median(C[np.triu_indices(len(idx), 1)]))


def nw_regress(src, vals, tgt, k=15):
    """Gaussian Nadaraya-Watson smoother; bandwidth = median distance to the k-th neighbour (first 800 source cells)
    - the bandwidth RULE of the audited method (part of the method's definition), re-coded."""
    m = min(len(src), 800)
    D = np.sqrt(sqdist(src[:m], src))
    D.sort(axis=1)
    h = float(np.median(D[:, min(k, D.shape[1] - 1)])) + 1e-9
    L = -sqdist(tgt, src) / (2 * h * h)
    L -= L.max(1, keepdims=True)
    W = np.exp(L)
    W /= W.sum(1, keepdims=True)
    return W @ vals
