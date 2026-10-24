"""Weighted Kullback-Leibler NMF (multiplicative updates) for bin-mean profiles.

X (B bins x G genes) holds CP10k bin means. The model X_b ~ h_b H says a bin is a mixture of K archetypal cell
states with CP10k profiles H_k (rows normalised to sum 1e4) and proportions h_b (rows sum to ~1). The loss is the
Poisson deviance of the bin's summed counts: row b is weighted by omega_b = n_b * lib_b / 1e4 (expected UMI in the
bin per unit CP10k), so a 7-cell bin weighs ~100x less than a 700-cell one. An optional mask (B x G, 0/1) hides
entries for cross-validation. Deterministic: NNDSVDa initialisation (or a warm start W0, H0).
"""
import numpy as np
from sklearn.decomposition._nmf import _initialize_nmf

TINY = 1e-10


def kl_loss(X, WH, Om):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(X > 0, X * np.log(np.maximum(X, TINY) / np.maximum(WH, TINY)), 0.0) - X + WH
    return float((Om * t).sum(dtype=np.float64))


def fit(X, K, omega, mask=None, n_iter=400, tol=1e-5, seed=0, W0=None, H0=None):
    X = np.ascontiguousarray(X, np.float32)
    omega = np.asarray(omega, np.float32)
    B, G = X.shape
    if W0 is None:
        W, H = _initialize_nmf(X if mask is None else X * mask, K, init="nndsvda", random_state=seed)
    else:
        W, H = W0.copy(), H0.copy()
    W = np.maximum(W, 1e-6).astype(np.float32)
    H = np.maximum(H, 1e-6).astype(np.float32)
    if mask is None:
        Xw = omega[:, None] * X
        Om = None
    else:
        Om = (omega[:, None] * mask).astype(np.float32)
        Xw = Om * X
    prev = None
    for it in range(n_iter):
        WH = np.maximum(W @ H, TINY)
        R = Xw / WH
        if Om is None:
            H *= (W.T @ R) / np.maximum((W * omega[:, None]).sum(0)[:, None], TINY)
        else:
            H *= (W.T @ R) / np.maximum(W.T @ Om, TINY)
        WH = np.maximum(W @ H, TINY)
        R = Xw / WH
        if Om is None:
            W *= (R @ H.T) / np.maximum(omega[:, None] * H.sum(1)[None, :], TINY)
        else:
            W *= (R @ H.T) / np.maximum(Om @ H.T, TINY)
        if it % 25 == 24:
            loss = kl_loss(X, W @ H, omega[:, None] if Om is None else Om)
            if prev is not None and abs(prev - loss) <= tol * max(abs(prev), 1.0):
                break
            prev = loss
    W, H = normalise(W.astype(np.float64), H.astype(np.float64))
    return W, H


def fit_subset(X, K, omega, n_top=4000, n_iter=1500, tol=1e-6, h_iter=300):
    """Fit W on the n_top genes with the most counts (all bins given), then H for every gene with W fixed.
    Returns W (B x K), H (K x G), both normalised (H rows sum to 1e4 over ALL genes)."""
    X = np.asarray(X, np.float64)
    omega = np.asarray(omega, np.float64)
    mc = (X * omega[:, None]).sum(0)
    top = np.argsort(-mc)[:min(n_top, int((mc > 0).sum()))]
    W, _ = fit(X[:, top], K, omega, n_iter=n_iter, tol=tol)
    Wf = W.astype(np.float32)
    Xw = (omega[:, None] * X).astype(np.float32)
    den = np.maximum((Wf * omega[:, None].astype(np.float32)).sum(0)[:, None], TINY)
    H = np.full((K, X.shape[1]), max(float(X.mean()), 1e-3) / K, np.float32)
    H[:, mc <= 0] = 0.0
    for _ in range(h_iter):
        H *= (Wf.T @ (Xw / np.maximum(Wf @ H, TINY))) / den
    return normalise(W, H.astype(np.float64))


def normalise(W, H):
    s = H.sum(1) / 1e4
    s = np.where(s > 0, s, 1.0)
    return W * s[None, :], H / s[:, None]


def project(X, H, n_iter=300):
    """Weights of new rows X (n x G) on fixed archetypes H (K x G): KL-optimal non-negative weights."""
    X = np.atleast_2d(X)
    n = X.shape[0]
    K = H.shape[0]
    W = np.full((n, K), 1.0 / K)
    Hs = H.sum(1)
    for _ in range(n_iter):
        WH = np.maximum(W @ H, TINY)
        W *= ((X / WH) @ H.T) / np.maximum(Hs[None, :], TINY)
    return W


def to_props(W):
    s = W.sum(1, keepdims=True)
    return W / np.where(s > 0, s, 1.0)
