"""Population-level unbalanced OT with a sink / source between cell-state archetypes, Dileo & Sottoriva Task 2.

STRICT rolling origin: transition i maps snapshot i -> i+1 and reads snapshots 0..i only (gastrulation-human bins
2..31, pancreas-human bins 0..14; first N genes of the -human node files; raw UMI bin statistics).

Model at transition i, for every K in {4, 8, 16} (K_eff = min(K, i): fewer states than bins, so every state has past
mass to transport):
 1. archetypes: Poisson (KL) NMF of the pooled bin count matrix S_b = n_b m_b (bins 0..i) S ~ W H, H rows sum to 1
    (UMI distribution of a state), W_bk = UMI of bin b explained by state k. Deterministic farthest-point init.
 2. cells: per-state UMI per cell ell_k = 1 / c_k from the ridge fit sum_k W_bk c_k = n_b (weights n/(n+50), shrunk
    to the pooled 1/UMI-per-cell, clipped to [1/4, 4]x). Cell fractions pi_bk = W_bk c_k / sum_k'(.), absolute state
    masses w_bk = n_b pi_bk (cells; the total changes between bins). Per-cell mean profile of a state A_k = H_k / c_k,
    so m_b ~ pi_b A (mixture identity), and nonzero-fraction profile F_k from f_b ~ pi_b F (fixed-pi KL fit).
 3. cost: C_kl = mean_g (log1p(1e4 H_kg) - log1p(1e4 H_lg))^2 / mean off-diagonal value.
 4. couplings of the past pairs (j-1 -> j), j in the last R in {1, 3} pairs, on the masses of the fit at i:
      balanced    entropic OT between the proportions (no sink)                          [ablation]
      sink_abs    entropic OT on ABSOLUTE masses with one dummy node on each side: sources = K states + a reservoir
                  of mass N_dst, targets = K states + a sink of mass N_src; cost state->sink = reservoir->state =
                  lambda, reservoir->sink = 0. Plan block [:K,:K] = transport, [:K,K] = death (cells that leave),
                  [K,:K] = birth (cells that appear); deaths - births = N_src - N_dst exactly.
      sink_norm   same augmentation on the proportions (both totals 1): turnover by death + birth competes with
                  transport at equal totals (separates 'the sink' from 'absolute counts', which are sampling-driven)
      kluot_abs   KL-relaxed unbalanced entropic OT (Chizat) on absolute masses, marginal penalty tau (own
                  log-domain iterations); growth / decline per state = row-sum deficits of the plan
    eps in {0.05, 0.2} (normalised cost units), lambda in {0.1, 0.25, 0.5, 1} (turnover needs C_kl > 2 lambda), tau in {0.1, 1, 10}.
 5. rates: per pair the per-capita kernel Q_k. = P_k. / a_k (transport + death column) and births per source cell
    beta = P_birth / N_src; pooled over the last R pairs with weights omega_j a_k (omega = n/(n+50) of the smaller
    bin). Push one step: w_hat = w_i Q_transport + N_i beta, pi_hat = w_hat / sum (absolute total is not scored).
    Other composition forecasters in the same pipeline: persist (pi_i), lin1 (pi_i + (pi_i - pi_{i-1}), clipped),
    and the ORACLE (bin i+1 projected on the strict archetypes of the fit at i: NOT strict, the ceiling).
 6. map back (mixture shift, L in {1, 3}): m_hat = pi_hat A + rbar_L, rbar_L = omega-weighted mean of the residual
    m_b - pi_b A over the last L bins (L = 1: m_hat = m_i + (pi_hat - pi_i) A; L = 3: denoised level). f_hat the same
    with F. Targets: l1 = log1p(CP10k(m_hat)) - log1p(CP10k(m_i)); linear = m_hat - m_i; orig = finite-cell head:
    p0 = (1 - f_hat)^n_i, E = p0 clip(log(eps) - log(m_i + eps)) + (1 - p0) clip(log(m_hat/(1-p0) + eps) - log(m_i + eps)).
 7. selection: per transition the config with the best mean criterion J = (Pearson + tie-aware P@K up + down) / 3
    (all genes, K = 200 N / n_test, dist_bridge.criterion) over past transitions j in [max(1, i-8), i-1], within each
    reported group (family x K); ties within 1e-3 go to the earlier (simpler) config. i <= 1: trend fallback for
    l1 / linear, persist + head for orig.
Scores: fixed random 25 % of genes per transition, np.random.default_rng(1000 s + i), s in {0, 1, 2};
dileo_rescore.metrics (pearson_nan0, p_ta_up, p_ta_down, spearman); mean over transitions i >= 1 (and i >= 2), then
over the 3 mask seeds. Comparator recomputed here: trend (repeat last change, y_{i-1}).
Usage: python uot_sink.py <dataset>   ->   uot_sink_<dataset>.json / _scores.npz / .log
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
HPC = os.path.join(ROOT, "tack", "hpc")
sys.path.insert(0, HPC)
import dileo_protocol  # noqa: E402  (numpy only; torch is imported only inside patch())
import dileo_rescore  # noqa: E402  (metrics)
import dist_bridge  # noqa: E402  (criterion, n_cells)
import sinkhorn  # noqa: E402  (own batched log-domain balanced Sinkhorn)

SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
TARGETS = ("l1", "linear", "orig")
KS = (4, 8, 16)
LS = (1, 3)
EPSS = (0.05, 0.2)
RS = (1, 3)
LAMS = (0.1, 0.25, 0.5, 1.0)
TAUS = (0.1, 1.0, 10.0)
WINDOW = 8
TIE = 1e-3
FLOOR = 1e-6
EPS_T = 1e-8
NMF_ITERS = 400
FAMS = ("persist", "lin1", "balanced", "sink_abs", "sink_norm", "kluot_abs")
TINY = 1e-12


def omega(n):
    return np.asarray(n, float) / (np.asarray(n, float) + 50.0)


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, name):
        cache = os.path.join(HERE, f"_node_{name}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=False)
            X, genes = z["X"], z["genes"]
        else:
            n_genes = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
            st = ("s00_mean", "s01_median", "s02_std", "s03_frac_nonzero", "s04_sum")
            cols = [f"t{b:03d}_{s}" for b in SNAPS[name] for s in st]
            node = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=["gene"] + cols, nrows=n_genes)
            X = np.stack([np.stack([node[f"t{b:03d}_{s}"].to_numpy(np.float64) for s in st], 1) for b in SNAPS[name]])
            genes = node["gene"].astype(str).to_numpy().astype("U")
            np.savez(cache, X=X, genes=genes)
        self.name = name
        self.genes = genes
        self.B, self.G = X.shape[0], X.shape[1]
        self.T = self.B - 1
        self.m = np.maximum(X[:, :, 0].T, 0.0)                    # (G, B) raw per-cell mean
        self.f = np.clip(X[:, :, 3].T, 0.0, 1.0)                  # (G, B) nonzero fraction
        self.n = np.array([dist_bridge.n_cells(X[b]) for b in range(self.B)])
        self.S = self.m * self.n[None, :]                          # pooled counts per bin
        self.lib = self.m.sum(0)
        self.CP = 1e4 * self.m / self.lib[None, :]
        self.Lg = np.log1p(self.CP)
        self.n_test = int(round(0.25 * self.G))

    def Y(self, t):
        return np.stack([np.asarray(dileo_protocol.TARGET_NP[t](self.m[:, i], self.m[:, i + 1]), np.float64)
                         for i in range(self.T)])


def make_masks(G, T, n_test):
    mk = np.zeros((3, T, n_test), int)
    for s in range(3):
        for i in range(T):
            rng = np.random.default_rng(seed=1000 * s + i)
            mk[s, i] = np.sort(rng.permutation(G)[:n_test])
    return mk


# ------------------------------------------------------------------ archetypes
def kl_nmf(S, K, n_iter=NMF_ITERS):
    """S (B, G) >= 0 -> W (B, K), H (K, G) rows summing to 1, minimising KL(S | W H). Farthest-point init."""
    B, G = S.shape
    P = S / np.maximum(S.sum(1, keepdims=True), TINY)
    Lp = np.log1p(1e4 * P)
    idx = [int(np.argmax(S.sum(1)))]
    while len(idx) < K:
        d = np.min(((Lp[:, None, :] - Lp[None, idx, :]) ** 2).mean(-1), axis=1)
        d[idx] = -1
        idx.append(int(np.argmax(d)))
    H = 0.8 * P[idx] + 0.2 * P.mean(0)[None, :]
    H /= H.sum(1, keepdims=True)
    W = np.full((B, K), 1.0 / K) * S.sum(1, keepdims=True)
    for _ in range(n_iter):
        R = S / np.maximum(W @ H, TINY)
        H *= (W.T @ R) / np.maximum(W.sum(0)[:, None], TINY)
        s = H.sum(1)
        H /= np.maximum(s[:, None], TINY)
        W *= s[None, :]
        R = S / np.maximum(W @ H, TINY)
        W *= (R @ H.T)                                             # H rows sum to 1
    return W, H


def project_row(s, H, n_iter=300):
    """w >= 0 minimising KL(s | w H) with H fixed (rows sum to 1)."""
    w = np.full(H.shape[0], s.sum() / H.shape[0])
    for _ in range(n_iter):
        w *= H @ (s / np.maximum(w @ H, TINY))
    return w


def fixed_pi_kl(pi, X, w, A0, n_iter=120):
    """A >= 0 (K x G) minimising sum_b w_b KL(X_b | (pi A)_b), pi fixed."""
    A = np.maximum(A0, 1e-10)
    Xw = w[:, None] * X
    den = np.maximum((pi * w[:, None]).sum(0)[:, None], TINY)
    for _ in range(n_iter):
        A *= (pi.T @ (Xw / np.maximum(pi @ A, TINY))) / den
    return A


class Fit:
    """Strict state model for transition i (bins 0..i) at K_eff = min(K, i)."""

    def __init__(self, d, i, K):
        S = d.S[:, :i + 1].T
        keep = S.sum(0) > 0
        W, Hk = kl_nmf(S[:, keep], K)
        H = np.zeros((K, d.G))
        H[:, keep] = Hk
        n = d.n[:i + 1]
        X = W / n[:, None]                                         # X c ~ 1
        c0 = n.sum() / W.sum()
        om = omega(n)
        Gm = X.T @ (om[:, None] * X)
        lam = 0.1 * np.trace(Gm) / K
        c = np.linalg.solve(Gm + lam * np.eye(K), X.T @ om + lam * c0)
        c = np.clip(c, 0.25 * c0, 4.0 * c0)
        pi = W * c[None, :]
        self.cell_sum_ratio = pi.sum(1) / n                        # calibration check (should be ~1)
        pi /= pi.sum(1, keepdims=True)
        self.K, self.i, self.H, self.c, self.pi, self.n = K, i, H, c, pi, n
        self.A = H / c[:, None]                                    # per-cell raw mean of each state
        self.F = np.minimum(fixed_pi_kl(pi, d.f[:, :i + 1].T, n, 1 - np.exp(-self.A)), 1.0)
        Lh = np.log1p(1e4 * H)
        C = ((Lh[:, None, :] - Lh[None, :, :]) ** 2).mean(-1)
        off = C[~np.eye(K, dtype=bool)].mean() if K > 1 else 1.0
        self.C = C / max(off, TINY)
        self.cost_scale = float(off)
        self.Wumi = W
        self.d = d
        self._resid = {}
        self._plans = {}

    def resid(self, L):
        """omega-weighted residual of the mixture over the last L bins, for m and f."""
        if L not in self._resid:
            d, i = self.d, self.i
            lo = max(0, i - L + 1)
            w = omega(self.n[lo:])
            fm = self.pi[lo:] @ self.A                             # (L, G)
            ff = self.pi[lo:] @ self.F
            rm = ((d.m[:, lo:i + 1].T - fm) * w[:, None]).sum(0) / w.sum()
            rf = ((d.f[:, lo:i + 1].T - ff) * w[:, None]).sum(0) / w.sum()
            self._resid[L] = (rm, rf)
        return self._resid[L]

    def oracle(self):
        s = self.d.S[:, self.i + 1]
        w = project_row(s, self.H)
        x = w * self.c
        return x / x.sum()


# ------------------------------------------------------------------ OT solvers
def _floor(x):
    x = np.asarray(x, float)
    return x + FLOOR * x.sum(-1, keepdims=True)


def plans_balanced(A_, B_, C, eps):
    """A_, B_ (P, K) proportions; returns plans (P, K, K)."""
    a, b = _floor(A_), _floor(B_)
    a /= a.sum(1, keepdims=True)
    b /= b.sum(1, keepdims=True)
    P = len(a)
    pi, _ = sinkhorn.solve(a, b, np.broadcast_to(C, (P,) + C.shape).copy(), eps, max_iter=100000)
    return pi


def plans_sink(A_, B_, C, lam, eps):
    """A_, B_ (P, K) masses (absolute or proportions). Augmented balanced problem with a reservoir and a sink."""
    a, b = _floor(A_), _floor(B_)
    P, K = a.shape
    Na, Nb = a.sum(1), b.sum(1)
    aa = np.concatenate([a, Nb[:, None]], 1)
    bb = np.concatenate([b, Na[:, None]], 1)
    T = (Na + Nb)[:, None]
    lam = np.broadcast_to(np.asarray(lam, float), (P,))
    Ca = np.zeros((P, K + 1, K + 1))
    Ca[:, :K, :K] = C
    Ca[:, :K, K] = lam[:, None]
    Ca[:, K, :K] = lam[:, None]
    pi, err = sinkhorn.solve(aa / T, bb / T, Ca, eps, max_iter=100000)
    pi = pi * T[:, :, None]
    death, birth = pi[:, :K, K].sum(1), pi[:, K, :K].sum(1)
    chk = np.abs((death - birth) - (Na - Nb)) / (Na + Nb)
    assert chk.max() < 1e-5, ("sink balance violated", chk.max())
    return pi, a, b


def _lse(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def plans_kl(A_, B_, C, tau, eps, max_iter=20000, tol=1e-10):
    """KL-relaxed unbalanced entropic OT, reference a x b: min <P,C> + eps KL(P|ab) + tau KL(P1|a) + tau KL(P'1|b).
    P_kl = a_k b_l exp((f_k + g_l - C_kl) / eps), f = -(tau eps / (tau + eps)) log sum_l b_l exp((g_l - C_kl) / eps).
    Masses are divided by the mean of the two totals (ratio kept)."""
    a, b = _floor(A_), _floor(B_)
    s = 0.5 * (a.sum(1) + b.sum(1))[:, None]
    a, b = a / s, b / s
    P_, K = a.shape
    tau = np.broadcast_to(np.asarray(tau, float), (P_,))[:, None]
    eps = np.broadcast_to(np.asarray(eps, float), (P_,))[:, None]
    fi = tau * eps / (tau + eps)
    la, lb = np.log(a), np.log(b)
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    Ce = C[None, :, :] / eps[:, :, None]
    for it in range(max_iter):
        f_new = -fi * _lse(lb[:, None, :] + g[:, None, :] / eps[:, :, None] - Ce, axis=2)
        g_new = -fi * _lse(la[:, :, None] + f_new[:, :, None] / eps[:, :, None] - Ce, axis=1)
        delta = max(np.abs(f_new - f).max(), np.abs(g_new - g).max())
        f, g = f_new, g_new
        if delta < tol:
            break
    Pl = np.exp(la[:, :, None] + lb[:, None, :] + (f[:, :, None] + g[:, None, :]) / eps[:, :, None] - Ce)
    return Pl * s[:, :, None], a * s, b * s


# ------------------------------------------------------------------ composition forecasters
def pair_plans(fit, fam):
    """Per-pair couplings of the last max(RS) pairs (j-1 -> j) for every (eps, lam / tau) of a family, one batched
    solve; cached on the fit. Returns {hp_key: (js, Qt, death, beta, a, b, P)}."""
    if fam in fit._plans:
        return fit._plans[fam]
    i, K, pi, n = fit.i, fit.K, fit.pi, fit.n
    js = list(range(max(1, i - max(RS) + 1), i + 1))
    src = np.stack([pi[j - 1] for j in js])
    dst = np.stack([pi[j] for j in js])
    if fam.endswith("_abs"):
        src = src * n[[j - 1 for j in js]][:, None]
        dst = dst * n[js][:, None]
    if fam == "balanced":
        combos = [(eps, None) for eps in EPSS]
    elif fam in ("sink_abs", "sink_norm"):
        combos = [(eps, lam) for eps in EPSS for lam in LAMS]
    else:
        combos = [(eps, tau) for eps in EPSS for tau in TAUS]
    nJ = len(js)
    S_ = np.concatenate([src] * len(combos))
    D_ = np.concatenate([dst] * len(combos))
    E_ = np.repeat([c[0] for c in combos], nJ)
    if fam == "balanced":
        P = plans_balanced(S_, D_, fit.C, E_)
        a = _floor(S_)
        a /= a.sum(1, keepdims=True)
        b = None
        Qt = P / a[:, :, None]
        death = np.zeros_like(a)
        beta = np.zeros_like(a)
    elif fam in ("sink_abs", "sink_norm"):
        P, a, b = plans_sink(S_, D_, fit.C, np.repeat([c[1] for c in combos], nJ), E_)
        Qt = P[:, :K, :K] / a[:, :, None]
        death = P[:, :K, K] / a
        beta = P[:, K, :K] / a.sum(1, keepdims=True)
    else:
        P, a, b = plans_kl(S_, D_, fit.C, np.repeat([c[1] for c in combos], nJ), E_)
        Qt = P / a[:, :, None]
        death = np.zeros_like(a)
        beta = np.zeros_like(a)
    out = {}
    for c_, cb in enumerate(combos):
        sl = slice(c_ * nJ, (c_ + 1) * nJ)
        out[cb] = (js, Qt[sl], death[sl], beta[sl], a[sl], None if b is None else b[sl], P[sl])
    fit._plans[fam] = out
    return out


def pooled_forecast(fit, fam, hp):
    """pi_hat (K,) and coupling diagnostics from the last R pairs of the fit at i."""
    i, K, pi, n = fit.i, fit.K, fit.pi, fit.n
    if fam == "persist" or i < 1:
        return pi[i], None
    if fam == "lin1":
        h = np.maximum(pi[i] + (pi[i] - pi[i - 1]), 0)
        return h / h.sum(), None
    R, eps = hp["R"], hp["eps"]
    key = (eps, hp.get("lam", hp.get("tau")))
    js, Qt, death, beta, a, b, P = pair_plans(fit, fam)[key]
    sl = slice(max(0, len(js) - R), len(js))
    js, Qt, death, beta, a, P = js[sl], Qt[sl], death[sl], beta[sl], a[sl], P[sl]
    b = None if b is None else b[sl]
    ap = a / a.sum(1, keepdims=True)
    om = omega(np.minimum(n[[j - 1 for j in js]], n[js]))
    wj = om[:, None] * ap                                          # (R, K)
    Q = (wj[:, :, None] * Qt).sum(0) / np.maximum(wj.sum(0)[:, None], TINY)
    dth = (wj * death).sum(0) / np.maximum(wj.sum(0), TINY)
    bet = (om[:, None] * beta).sum(0) / om.sum()
    h = np.maximum(pi[i] @ Q + bet, 0)
    diag = None
    if fam in ("sink_abs", "sink_norm"):
        diag = {"pairs": js, "death_frac": dth, "birth_per_cell": bet,
                "death_total": P[:, :K, K].sum(1), "birth_total": P[:, K, :K].sum(1),
                "transport_total": P[:, :K, :K].sum((1, 2)), "src_total": a.sum(1), "dst_total": b.sum(1)}
    return h / h.sum(), diag


def build_configs():
    cfgs = []
    for K in KS:
        for L in LS:
            cfgs.append(("persist", K, L, {}))
            cfgs.append(("lin1", K, L, {}))
            for R in RS:
                for eps in EPSS:
                    cfgs.append(("balanced", K, L, {"R": R, "eps": eps}))
            for fam in ("sink_abs", "sink_norm"):
                for R in RS:
                    for eps in EPSS:
                        for lam in LAMS:
                            cfgs.append((fam, K, L, {"R": R, "eps": eps, "lam": lam}))
            for R in RS:
                for eps in EPSS:
                    for tau in TAUS:
                        cfgs.append(("kluot_abs", K, L, {"R": R, "eps": eps, "tau": tau}))
    return cfgs


def label(cfg):
    fam, K, L, hp = cfg
    return f"{fam}|K{K}|L{L}" + "".join(f"|{k}{v:g}" for k, v in hp.items())


def comp_key(cfg):
    fam, K, L, hp = cfg
    return (fam, K, tuple(sorted(hp.items())))


# ------------------------------------------------------------------ map back and targets
def map_back(fit, pih, L):
    rm, rf = fit.resid(L)
    mh = np.maximum(pih @ fit.A + rm, 0.0)
    fh = np.clip(pih @ fit.F + rf, 0.0, 1.0)
    fh = np.where(mh <= 0, 0.0, np.where(fh <= 1e-9, 1 - np.exp(-mh), fh))
    return mh, np.minimum(fh, mh)                                  # integer counts: f <= m


def target_pred(d, i, mh, fh, t):
    m0 = d.m[:, i]
    if t == "l1":
        return np.log1p(1e4 * mh / max(mh.sum(), TINY)) - d.Lg[:, i]
    if t == "linear":
        return mh - m0
    p0 = np.power(np.clip(1.0 - fh, 0.0, 1.0), d.n[i])
    vz = np.clip(np.log(EPS_T) - np.log(m0 + EPS_T), -5, 5)
    cond = mh / np.maximum(1.0 - p0, TINY)
    vn = np.clip(np.log(cond + EPS_T) - np.log(m0 + EPS_T), -5, 5)
    return np.where(mh > 0, p0 * vz + (1 - p0) * vn, vz)


# ------------------------------------------------------------------ scoring
MKEYS = ("pearson_nan0", "p_ta_up", "p_ta_down", "spearman", "p_at_200_up", "p_at_200_down")


def test_scores(y, p, mk_i):
    """(3 seeds, len(MKEYS))."""
    out = np.zeros((mk_i.shape[0], len(MKEYS)))
    for s in range(mk_i.shape[0]):
        r = dileo_rescore.metrics(y[mk_i[s]], p[mk_i[s]])
        out[s] = [r[k] for k in MKEYS]
    return out


def summ(sc, start):
    """sc (T, 3, M) -> mean over transitions >= start per seed, then over seeds."""
    x = np.nanmean(sc[start:], axis=0)                             # (3, M)
    return {k: round(float(x[:, j].mean()), 4) for j, k in enumerate(MKEYS[:3])} | \
        {"pcc_seed_sd": round(float(x[:, 0].std()), 4), "spearman": round(float(x[:, 3].mean()), 4)}


def triple(dct):
    return [dct["pearson_nan0"], dct["p_ta_up"], dct["p_ta_down"]]


def paired(a, b, start=2):
    """per-transition (mean over seeds) PCC and J differences, Wilcoxon over transitions >= start."""
    xa, xb = np.nanmean(a[start:], 1), np.nanmean(b[start:], 1)
    dp = xa[:, 0] - xb[:, 0]
    dj = xa[:, :3].mean(1) - xb[:, :3].mean(1)
    out = {}
    for k, v in (("dPCC", dp), ("dJ", dj)):
        nz = v[np.abs(v) > 1e-12]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
        out[k] = {"mean": round(float(v.mean()), 4), "wins": int((v > 1e-12).sum()), "losses": int((v < -1e-12).sum()),
                  "p_wilcoxon": round(p, 4)}
    return out


def main(name):
    t0 = time.time()
    d = Data(name)
    T = d.T
    mk = make_masks(d.G, T, d.n_test)
    q = 200.0 / d.n_test
    Y = {t: d.Y(t) for t in TARGETS}
    cfgs = build_configs()
    C = len(cfgs)
    print(name, "G", d.G, "T", T, "configs", C, "cells", np.round(d.n).astype(int).tolist(), flush=True)

    # groups: (family set, K set)
    groups = {}
    for fset_name, fset in (("persist", ("persist",)), ("lin1", ("lin1",)), ("balanced", ("balanced",)),
                            ("sink_abs", ("sink_abs",)), ("sink_norm", ("sink_norm",)), ("kluot_abs", ("kluot_abs",)),
                            ("sink_any", ("sink_abs", "sink_norm")), ("noOT", ("persist", "lin1")),
                            ("full", FAMS)):
        for kname, kset in (("Kany", KS),) + tuple((f"K{K}", (K,)) for K in KS):
            groups[f"{fset_name}|{kname}"] = [r for r, c in enumerate(cfgs) if c[0] in fset and c[1] in kset]
    # fixed-L persist sub-groups (level baselines) and fixed single configs for the ablation table
    groups["persist|L1|Kany"] = [r for r, c in enumerate(cfgs) if c[0] == "persist" and c[2] == 1]
    groups["persist|L3|Kany"] = [r for r, c in enumerate(cfgs) if c[0] == "persist" and c[2] == 3]
    gnames = list(groups)

    J = {t: np.full((C, T), np.nan) for t in TARGETS}
    TV = np.full((C, T), np.nan)
    Jor = {t: np.full((len(KS), len(LS), T), np.nan) for t in TARGETS}
    Sg = {t: {g: np.full((T, 3, len(MKEYS)), np.nan) for g in gnames} for t in TARGETS}
    Sel = {t: {g: [None] * T for g in gnames} for t in TARGETS}
    Sor = {t: {f"oracle|K{K}|L{L}": np.full((T, 3, len(MKEYS)), np.nan) for K in KS for L in LS} for t in TARGETS}
    Sor_sel = {t: {f"oracle|K{K}|Lsel": np.full((T, 3, len(MKEYS)), np.nan) for K in KS} for t in TARGETS}
    Sor_sel_any = {t: np.full((T, 3, len(MKEYS)), np.nan) for t in TARGETS}
    Str = {t: np.full((T, 3, len(MKEYS)), np.nan) for t in TARGETS}
    fit_info, sink_diag = {}, {}
    fallback_orig = cfgs.index(("persist", KS[0], 1, {}))

    for i in range(1, T):
        ti = time.time()
        fits = {K: Fit(d, i, min(K, i)) for K in KS}
        tfit = time.time() - ti
        fit_info[i] = {f"K{K}": {"K_eff": fits[K].K, "cell_sum_ratio_range": [round(float(fits[K].cell_sum_ratio.min()), 3),
                                                                              round(float(fits[K].cell_sum_ratio.max()), 3)],
                                 "cost_scale": round(fits[K].cost_scale, 4)} for K in KS}
        orc = {K: fits[K].oracle() for K in KS}
        comp = {}
        preds = {t: np.zeros((C, d.G), np.float32) for t in TARGETS}
        for r, cfg in enumerate(cfgs):
            fam, K, L, hp = cfg
            fit = fits[K]
            ck = comp_key(cfg)
            if ck not in comp:
                comp[ck] = pooled_forecast(fit, fam, hp)
            pih, diag = comp[ck]
            TV[r, i] = 0.5 * np.abs(pih - orc[K]).sum()
            if diag is not None and i == T - 1:
                sink_diag[label(cfg)] = diag
            mh, fh = map_back(fit, pih, L)
            for t in TARGETS:
                p = target_pred(d, i, mh, fh, t)
                preds[t][r] = p
                J[t][r, i] = dist_bridge.criterion(Y[t][i], p, q)
        # oracle composition (NOT strict)
        orp = {}
        for a_, K in enumerate(KS):
            for b_, L in enumerate(LS):
                mh, fh = map_back(fits[K], orc[K], L)
                for t in TARGETS:
                    p = target_pred(d, i, mh, fh, t)
                    orp[(t, K, L)] = p
                    Jor[t][a_, b_, i] = dist_bridge.criterion(Y[t][i], p, q)
                    Sor[t][f"oracle|K{K}|L{L}"][i] = test_scores(Y[t][i], p, mk[:, i])
        # selection (strict: J of transitions < i) and test scoring
        lo = max(1, i - WINDOW)
        for t in TARGETS:
            Str[t][i] = test_scores(Y[t][i], Y[t][i - 1], mk[:, i])
            for g in gnames:
                rows = groups[g]
                if i - lo < 1:
                    if t == "orig":                                # level + finite-cell head (same for all groups)
                        p = preds[t][fallback_orig]
                        Sel[t][g][i] = "fallback:" + label(cfgs[fallback_orig])
                    else:
                        p = Y[t][i - 1]
                        Sel[t][g][i] = "fallback:trend"
                else:
                    sc = np.array([np.nanmean(J[t][r, lo:i]) for r in rows])
                    best = np.nanmax(sc)
                    r = [rr for rr, s in zip(rows, sc) if s >= best - TIE][0]
                    p = preds[t][r]
                    Sel[t][g][i] = label(cfgs[r])
                Sg[t][g][i] = test_scores(Y[t][i], p, mk[:, i])
            # oracle with L chosen on past oracle J (composition is the oracle anyway)
            for a_, K in enumerate(KS):
                if i - lo < 1:
                    b_ = 0
                else:
                    b_ = int(np.argmax([np.nanmean(Jor[t][a_, bb, lo:i]) for bb in range(len(LS))]))
                Sor_sel[t][f"oracle|K{K}|Lsel"][i] = test_scores(Y[t][i], orp[(t, K, LS[b_])], mk[:, i])
            if i - lo < 1:
                a_, b_ = 0, 0
            else:
                flat = [(np.nanmean(Jor[t][aa, bb, lo:i]), aa, bb) for aa in range(len(KS)) for bb in range(len(LS))]
                _, a_, b_ = max(flat, key=lambda x: x[0])
            Sor_sel_any[t][i] = test_scores(Y[t][i], orp[(t, KS[a_], LS[b_])], mk[:, i])
        print(name, "transition", i, "fit", round(tfit, 1), "s total", round(time.time() - t0, 1), "s", flush=True)

    # ------------------------------------------------------------------ assemble
    out = {"dataset": name, "n_genes": d.G, "n_transitions": T, "n_test": d.n_test,
           "cells": [int(round(x)) for x in d.n], "protocol": __doc__, "n_configs": C, "targets": {}}
    for t in TARGETS:
        res = {"trend": {"i>=1": summ(Str[t], 1), "i>=2": summ(Str[t], 2)}, "groups": {}, "oracle": {}, "paired": {}}
        for g in gnames:
            sel = Sel[t][g]
            fam_share = {}
            for s in sel[2:]:
                if s and not s.startswith("fallback"):
                    fam_share[s.split("|")[0]] = fam_share.get(s.split("|")[0], 0) + 1
            kshare = {}
            for s in sel[2:]:
                if s and not s.startswith("fallback"):
                    kshare[s.split("|")[1]] = kshare.get(s.split("|")[1], 0) + 1
            res["groups"][g] = {"i>=1": summ(Sg[t][g], 1), "i>=2": summ(Sg[t][g], 2), "family_share_i>=2": fam_share,
                                "K_share_i>=2": kshare, "selected": sel,
                                "per_transition_pcc": [None if not np.isfinite(v) else round(float(v), 4)
                                                       for v in np.nanmean(Sg[t][g][:, :, 0], 1)]}
        for k, v in Sor[t].items():
            res["oracle"][k] = {"i>=1": summ(v, 1), "i>=2": summ(v, 2)}
        for k, v in Sor_sel[t].items():
            res["oracle"][k] = {"i>=1": summ(v, 1), "i>=2": summ(v, 2)}
        res["oracle"]["oracle|Ksel|Lsel"] = {"i>=1": summ(Sor_sel_any[t], 1), "i>=2": summ(Sor_sel_any[t], 2)}
        pairs = [("sink_abs|Kany", "balanced|Kany"), ("sink_norm|Kany", "balanced|Kany"),
                 ("sink_any|Kany", "balanced|Kany"), ("kluot_abs|Kany", "balanced|Kany"),
                 ("sink_abs|Kany", "lin1|Kany"), ("sink_norm|Kany", "lin1|Kany"), ("sink_any|Kany", "persist|Kany"),
                 ("balanced|Kany", "lin1|Kany"), ("full|Kany", "noOT|Kany")]
        for K in KS:
            pairs += [(f"sink_abs|K{K}", f"balanced|K{K}"), (f"sink_norm|K{K}", f"balanced|K{K}")]
        for a, b in pairs:
            res["paired"][f"{a} - {b}"] = paired(Sg[t][a], Sg[t][b])
        for a in ("sink_abs|Kany", "sink_norm|Kany", "sink_any|Kany", "balanced|Kany", "lin1|Kany", "persist|Kany",
                  "full|Kany"):
            res["paired"][f"{a} - trend"] = paired(Sg[t][a], Str[t])
        # hindsight-best fixed config per family (NOT strict)
        hind = {}
        for fam in FAMS:
            rows = [r for r, c in enumerate(cfgs) if c[0] == fam]
            best = max(rows, key=lambda r: np.nanmean(J[t][r, 1:]))
            hind[fam] = {"config": label(cfgs[best]), "J_mean_all_genes_i>=1": round(float(np.nanmean(J[t][best, 1:])), 4)}
        res["hindsight_best_fixed_config_NOT_strict"] = hind
        out["targets"][t] = res

    # composition forecast error vs the oracle projection (TV, i >= 2): per family, best-mean config and l1-selected
    tv = {}
    for fam in FAMS:
        for K in KS:
            rows = [r for r, c in enumerate(cfgs) if c[0] == fam and c[1] == K and c[2] == 1]
            means = [float(np.nanmean(TV[r, 2:])) for r in rows]
            j = int(np.argmin(means))
            tv[f"{fam}|K{K}"] = {"best_config_hindsight": label(cfgs[rows[j]]), "tv_best": round(means[j], 4),
                                 "tv_median_over_hp": round(float(np.median(means)), 4)}
    out["composition_tv_vs_oracle_i>=2"] = tv

    # sink interpretation at the last transition (fit on all bins <= T-1)
    fitL = {K: Fit(d, T - 1, min(K, T - 1)) for K in KS}
    interp = {}
    for K in (4, 8):
        fit = fitL[K]
        Hn = fit.H / np.maximum(fit.H.mean(0, keepdims=True), TINY)
        expr = fit.H.max(0) > 5e-5
        markers = []
        for k in range(fit.K):
            sc = np.where(expr, Hn[k], -1)
            markers.append([str(g) for g in d.genes[np.argsort(-sc)[:8]]])
        js = list(range(1, T))
        src = np.stack([fit.pi[j - 1] for j in js])
        dst = np.stack([fit.pi[j] for j in js])
        ent = {"markers": markers, "pi_over_bins": np.round(fit.pi, 3).tolist(),
               "ell_umi_per_cell": np.round(1 / fit.c, 1).tolist()}
        for fam in ("sink_abs", "sink_norm"):
            for lam in (0.5, 1.0):
                if fam == "sink_abs":
                    s_, d_ = src * fit.n[:-1, None], dst * fit.n[1:, None]
                else:
                    s_, d_ = src, dst
                P, a, b = plans_sink(s_, d_, fit.C, lam, 0.05)
                Kf = fit.K
                death = P[:, :Kf, Kf]
                birth = P[:, Kf, :Kf]
                tr = P[:, :Kf, :Kf].sum((1, 2))
                ent[f"{fam}|lam{lam:g}|eps0.05"] = {
                    "per_pair": [{"pair": f"{d.n[j - 1]:.0f}->{d.n[j]:.0f} cells (snap {j - 1}->{j})",
                                  "death_total": round(float(death[q_].sum()), 2), "birth_total": round(float(birth[q_].sum()), 2),
                                  "transported": round(float(tr[q_]), 2),
                                  "death_frac_by_state": np.round(death[q_] / a[q_], 3).tolist(),
                                  "birth_by_state": np.round(birth[q_], 2).tolist()} for q_, j in enumerate(js)],
                    "mean_death_frac_by_state": np.round((death / a).mean(0), 3).tolist(),
                    "birth_share_by_state": np.round(birth.sum(0) / max(birth.sum(), TINY), 3).tolist(),
                    "turnover_share_of_source_mass": round(float(death.sum() / a.sum()), 4),
                    "corr_net_change_vs_cell_count_change": round(float(np.corrcoef(birth.sum(1) - death.sum(1),
                                                                                  b.sum(1) - a.sum(1))[0, 1]), 4)
                    if fam == "sink_abs" else None,
                    "corr_gross_turnover_vs_abs_log_cellcount_ratio": round(float(stats.spearmanr(
                        death.sum(1) / a.sum(1), np.abs(np.log(fit.n[1:] / fit.n[:-1]))).correlation), 4)}
        interp[f"K{K}"] = ent
    out["sink_interpretation_last_fit"] = interp
    out["fit_info"] = fit_info
    out["sink_diag_last_transition_sample"] = {k: {kk: (np.round(vv, 4).tolist() if isinstance(vv, np.ndarray) else vv)
                                                   for kk, vv in v.items()}
                                               for k, v in list(sink_diag.items())[:6]}
    out["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(os.path.join(HERE, f"uot_sink_{name}_scores.npz"), labels=np.array([label(c) for c in cfgs]),
                        **{f"J_{t}": J[t] for t in TARGETS}, TV=TV,
                        **{f"S_{t}_{g.replace('|', '_')}": Sg[t][g] for t in TARGETS for g in gnames},
                        **{f"S_trend_{t}": Str[t] for t in TARGETS})
    with open(os.path.join(HERE, f"uot_sink_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print_table(out)
    print(name, "done", out["seconds"], "s", flush=True)


def print_table(out):
    name = out["dataset"]
    for t in TARGETS:
        r = out["targets"][t]
        print(f"== {name} target={t}  [PCC, P@200 up, P@200 down] tie-aware, i>=1 | i>=2")
        print(f"  {'trend':28s} {triple(r['trend']['i>=1'])} | {triple(r['trend']['i>=2'])}")
        for g in ("persist|Kany", "persist|L1|Kany", "persist|L3|Kany", "lin1|Kany", "balanced|Kany", "sink_abs|Kany",
                  "sink_norm|Kany", "sink_any|Kany", "kluot_abs|Kany", "noOT|Kany", "full|Kany"):
            e = r["groups"][g]
            print(f"  {g:28s} {triple(e['i>=1'])} | {triple(e['i>=2'])}  fam {e['family_share_i>=2']} K {e['K_share_i>=2']}")
        for K in KS:
            for fam in ("lin1", "balanced", "sink_abs", "sink_norm"):
                e = r["groups"][f"{fam}|K{K}"]
                print(f"  {fam + '|K' + str(K):28s} {triple(e['i>=1'])} | {triple(e['i>=2'])}")
            e = r["oracle"][f"oracle|K{K}|Lsel"]
            print(f"  {'ORACLE|K' + str(K) + '|Lsel':28s} {triple(e['i>=1'])} | {triple(e['i>=2'])}")
            for L in LS:
                e = r["oracle"][f"oracle|K{K}|L{L}"]
                print(f"  {'ORACLE|K' + str(K) + '|L' + str(L):28s} {triple(e['i>=1'])} | {triple(e['i>=2'])}")
        e = r["oracle"]["oracle|Ksel|Lsel"]
        print(f"  {'ORACLE|Ksel|Lsel':28s} {triple(e['i>=1'])} | {triple(e['i>=2'])}")
        for k, v in r["paired"].items():
            print(f"  paired {k:40s} dPCC {v['dPCC']['mean']:+.4f} ({v['dPCC']['wins']}/{v['dPCC']['losses']}, "
                  f"p={v['dPCC']['p_wilcoxon']}) dJ {v['dJ']['mean']:+.4f} ({v['dJ']['wins']}/{v['dJ']['losses']}, "
                  f"p={v['dJ']['p_wilcoxon']})")


if __name__ == "__main__":
    main(sys.argv[1])
