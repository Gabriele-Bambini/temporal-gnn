"""Causal per-gene filters on the l1 scale L = log1p(CP10k) (G, B), with known noise R (G, B).

Target of transition t: y_t = L_{t+1} - L_t. Local-level forecast: yhat_t = x_{t|t} - L_t (E[L_{t+1} | bins <= t]).
Q is supplied as a function Qfun(b) -> (G,) used in the prediction step b-1 -> b; every Qfun below uses only the
changes y_k with k + 1 <= b (data of bins <= b), so x_{b|b} is a function of bins <= b only (strict).
"""
import numpy as np


def local_level(L, R, Qfun):
    G, B = L.shape
    X = np.zeros((G, B))
    Pf = np.zeros((G, B))
    Kg = np.zeros((G, B))
    x, P = L[:, 0].copy(), R[:, 0].copy()
    X[:, 0], Pf[:, 0], Kg[:, 0] = x, P, 1.0
    for b in range(1, B):
        Pm = P + Qfun(b)
        S = Pm + R[:, b]
        K = np.where(S > 0, Pm / np.where(S > 0, S, 1), 1.0)
        x = x + K * (L[:, b] - x)
        P = (1 - K) * Pm
        X[:, b], Pf[:, b], Kg[:, b] = x, P, K
    return X, Pf, Kg


def ll_forecast(L, X):
    """(T, G) forecasts of y_t = L_{t+1} - L_t, t = 0..B-2."""
    return np.stack([X[:, t] - L[:, t] for t in range(L.shape[1] - 1)])


# ---------------------------------------------------------------- per-gene Q from past changes (C6)
def q_pergene(L, R, mode, floor_frac=0.05):
    """mode 'base': Q_b = mean_{k<b} (y_k^2 - R_k - R_{k+1});
    'lag1': base + 2 mean_{k<b-1} (y_k y_{k+1} + R_{k+1})  (noise-corrected lag-1 autocovariance of the true
            change added: long-run variance of the true increments; = base while only one change is known);
    'ma1':  mean y_k^2 + 2 mean y_k y_{k+1}  (local-level MA(1) moment estimator, no known R; = base at b = 1).
    Floored at floor_frac * R_b. Returns Qfun and a dict of the fraction of genes at the floor per b."""
    Y = np.diff(L, axis=1)                                  # (G, B-1), y_k
    e = Y ** 2 - R[:, :-1] - R[:, 1:]
    c1 = Y[:, :-1] * Y[:, 1:] + R[:, 1:-1]                  # k = 0..B-3, needs bins <= k+2
    c1_raw = Y[:, :-1] * Y[:, 1:]
    ce, cc, ccr = np.cumsum(e, 1), np.cumsum(c1, 1), np.cumsum(c1_raw, 1)
    cy2 = np.cumsum(Y ** 2, 1)
    at_floor = {}

    def Qfun(b):
        nk = b                                              # changes k = 0..b-1
        base = ce[:, b - 1] / nk
        if mode == "base" or b < 2:
            q = base if mode != "ma1" else cy2[:, b - 1] / nk - (R[:, :b] + R[:, 1:b + 1]).mean(1)
        elif mode == "lag1":
            q = base + 2 * cc[:, b - 2] / (b - 1)
        elif mode == "ma1":
            q = cy2[:, b - 1] / nk + 2 * ccr[:, b - 2] / (b - 1)
        else:
            raise ValueError(mode)
        fl = floor_frac * R[:, b]
        at_floor[b] = float(np.mean(q < fl))
        return np.maximum(q, fl)
    return Qfun, at_floor


# ---------------------------------------------------------------- class Q (C3)
def classes_running(d, b, n_cls=5):
    """Class 0: running mean (cell-weighted, bins <= b) = 0; classes 1..4: quartiles of log running mean among the
    positive genes at bin b (boundaries from bins <= b)."""
    w = d.n[:b + 1]
    mbar = (d.m[:, :b + 1] * w[None, :]).sum(1) / w.sum()
    cls = np.zeros(d.G, int)
    pos = mbar > 0
    qs = np.quantile(np.log(mbar[pos]), np.linspace(0, 1, n_cls)[1:-1])
    cls[pos] = 1 + np.searchsorted(qs, np.log(mbar[pos]), side="right")
    return cls


def q_class(d, L, R, window, floor_frac=0.05, n_cls=5):
    """Class signal variance from past changes: Q_c,b = mean over genes of class c (classes at bin b) and changes
    k in the window (k + 1 <= b; window = 2 -> k = b-2, b-1; window = None -> all k < b) of y_k^2 - R_k - R_{k+1};
    each gene gets max(Q_c, floor_frac R_gb). Returns Qfun and the per-b class values."""
    Y = np.diff(L, axis=1)
    e = Y ** 2 - R[:, :-1] - R[:, 1:]
    vals = {}

    def Qfun(b):
        cls = classes_running(d, b, n_cls)
        lo = 0 if window is None else max(0, b - window)
        eb = e[:, lo:b].mean(1)
        qc = np.array([eb[cls == c].mean() if np.any(cls == c) else 0.0 for c in range(n_cls)])
        vals[b] = qc.tolist()
        return np.maximum(qc[cls], floor_frac * R[:, b])
    return Qfun, vals


# ---------------------------------------------------------------- local linear trend with per-gene grid ML (C6)
QL_GRID = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)
QS_GRID = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def llt_run(L, R, ql, qs, slope0_var=0.1):
    """Local-linear-trend filter (level, slope), fixed (ql, qs) for all genes. Returns filtered level, slope (G, B)
    and the innovation log-likelihood per gene and bin (0 at b = 0, 1)."""
    G, B = L.shape
    lv, sl = L[:, 0].copy(), np.zeros(G)
    P11, P12, P22 = R[:, 0].copy(), np.zeros(G), np.full(G, slope0_var)
    LV, SL, LL = np.zeros((G, B)), np.zeros((G, B)), np.zeros((G, B))
    LV[:, 0] = lv
    for b in range(1, B):
        # predict
        lv_p, sl_p = lv + sl, sl
        a11 = P11 + 2 * P12 + P22 + ql
        a12 = P12 + P22
        a22 = P22 + qs
        S = a11 + R[:, b]
        v = L[:, b] - lv_p
        S = np.maximum(S, 1e-12)
        k1, k2 = a11 / S, a12 / S
        lv, sl = lv_p + k1 * v, sl_p + k2 * v
        P11, P12, P22 = (1 - k1) * a11, (1 - k1) * a12, a22 - k2 * a12
        LV[:, b], SL[:, b] = lv, sl
        if b >= 2:
            LL[:, b] = -0.5 * (np.log(2 * np.pi * S) + v ** 2 / S)
    return LV, SL, LL


def llt_ml(L, R, ql_grid=QL_GRID, qs_grid=QS_GRID):
    """Per gene and per forecast time t: grid point maximising the innovation log-lik over bins <= t (strict).
    Returns forecasts (T, G) of y_t: 'slope' = slope_{t|t}; 'onestep' = level_{t|t} + slope_{t|t} - L_t;
    and the chosen grid index frequency per t."""
    G, B = L.shape
    runs = [(ql, qs) for ql in ql_grid for qs in qs_grid]
    LVs, SLs, CLL = [], [], []
    for ql, qs in runs:
        lv, sl, ll = llt_run(L, R, ql, qs)
        LVs.append(lv)
        SLs.append(sl)
        CLL.append(np.cumsum(ll, 1))
    LVs, SLs, CLL = np.stack(LVs), np.stack(SLs), np.stack(CLL)   # (M, G, B)
    T = B - 1
    slope = np.zeros((T, G))
    onestep = np.zeros((T, G))
    default = runs.index((1e-2, 1e-3)) if (1e-2, 1e-3) in runs else 0
    choice = {}
    for t in range(T):
        if t < 2:
            j = np.full(G, default)
        else:
            j = np.argmax(CLL[:, :, t], axis=0)
        g = np.arange(G)
        slope[t] = SLs[j, g, t]
        onestep[t] = LVs[j, g, t] + SLs[j, g, t] - L[:, t]
        choice[t] = np.bincount(j, minlength=len(runs)).tolist()
    return slope, onestep, runs, choice
