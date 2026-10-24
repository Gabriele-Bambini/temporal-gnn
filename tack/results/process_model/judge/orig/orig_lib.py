"""orig pipeline of the process-model v2 build, importing the copied verify_3/c12_binomial_head.py (never its main()).

Detection (c12, exact grid binomial filter): eta = logit p on 101 points in [-15, 5], random walk N(0, tau^2),
K_b = rint(f_b n_b) ~ Bin(n_b, p_b); (tau, mu0) chosen at every t by the prequential log-likelihood of K_0..K_t.
Heads (c12 formulas, same RNG keys [key, t, pool, 0/1]) for every pool of dist_bridge.POOLS, delta = 0:
  base = pooled f (DistBridge no-SB, the level + observation-head baseline); A = p_hat everywhere;
  B = A with P0 = E_post[(1-p)^n] for K_t = 0 genes; C (primary) = posterior P0 for every gene;
  D = base with posterior P0 for K_t = 0 genes.
  y = 5 (1 - P0) if m_t = 0, else -5 P0 + (1 - P0) E[clip log ratio | K >= 1].
Rolling pool selection per variant as DistBridge.predict (t <= 2 default kalman 0.2; else best mean J over the past
window of 8, ties within 1e-3 -> earlier pool).
NASS orig full (build copy of nass.py) and O2 = the NASS orig stacker with F['p0'] := C's posterior P0 and
F['obs_box4'] := C's rolling head output (same feature count). The fallback (i = 0, 1) is computed from the
ORIGINAL F before the swap, so every NASS-family orig model shares it (NASS's convention).
"""
import os
import sys

import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
if B not in sys.path:
    sys.path.insert(0, B)
import pm_lib as L  # noqa: E402  (sets GRAPH_CACHE, imports build/dsdata.py and build/nass.py first)

D, NS, DB = L.D, L.NS, L.DB
_saved = list(sys.path)
sys.path.insert(0, HERE)
import common as C3  # noqa: E402  (build/orig/common.py: the verify_3 copy)
import c12_binomial_head as c12  # noqa: E402
sys.path[:] = _saved
assert os.path.dirname(os.path.abspath(C3.__file__)) == HERE, C3.__file__
assert os.path.dirname(os.path.abspath(c12.__file__)) == HERE, c12.__file__
assert C3.D is D, "orig/common.py picked a dsdata other than build/dsdata.py"

VARS = ("base", "A_phat", "B_phat+post0", "C_post_all", "D_base+post0")


def detection(d):
    T, G = d.T, d.G
    K = np.rint(d.f * d.n[None, :])
    n = d.n
    settings = [(tau, mu0) for tau in c12.TAUS for mu0 in c12.MU0S]
    cum = []
    for tau, mu0 in settings:
        pll, _ = c12.run_filter(K, n, tau, mu0)
        cum.append(np.cumsum(pll.sum(0)))
    cum = np.array(cum)
    chosen = [int(np.argmax(cum[:, t])) for t in range(T)]
    p_hat = np.zeros((T, G))
    logP0_plug = np.zeros((T, G))
    logP0_post = np.zeros((T, G))
    for si in sorted(set(chosen)):
        ts = [t for t in range(T) if chosen[t] == si]
        _, kept = c12.run_filter(K, n, settings[si][0], settings[si][1], keep=set(ts))
        for t in ts:
            pr = kept[t]
            lpr = np.log(np.maximum(pr, 1e-300))
            p_hat[t] = pr @ c12.PG
            nt = max(int(round(n[t])), 1)
            logP0_plug[t] = nt * np.log1p(-p_hat[t])
            logP0_post[t] = logsumexp(lpr + nt * c12.LQ[None, :], axis=1)
    return {"K": K, "p_hat": p_hat, "logP0_plug": logP0_plug, "logP0_post": logP0_post,
            "setting": [f"tau={settings[c][0]},mu0={settings[c][1]}" for c in chosen]}


def heads(d, det, Y):
    T, G, Bn = d.T, d.G, d.B
    n = d.n
    K, p_hat = det["K"], det["p_hat"]
    trip = [DB.triplet(d.X[b]) for b in range(Bn)]
    pools = {p: (DB.box_pool if p[0] == "box" else DB.kalman_pool)(trip, list(n), p[1]) for p in DB.POOLS}
    preds = {v: {p: [None] * T for p in DB.POOLS} for v in VARS}
    key = 1 + (d.name == "pancreas-human")
    for pi, p in enumerate(DB.POOLS):
        for t in range(T):
            H = pools[p][t] if t > 0 else trip[0]
            fP, mu, var = DB.positive_part(H)
            m_now = d.m[:, t]
            nt = max(int(round(n[t])), 1)
            act = np.where((m_now > 0) & (fP > 0))[0]
            c_pool = c12.cond_part(fP, mu, var, m_now, n[t], np.random.default_rng([key, t, pi, 0]), act)
            c_phat = c12.cond_part(p_hat[t], mu, var, m_now, n[t], np.random.default_rng([key, t, pi, 1]), act)
            P0_pool = (1 - fP) ** nt
            P0_plug, P0_post = np.exp(det["logP0_plug"][t]), np.exp(det["logP0_post"][t])
            zero = K[:, t] == 0
            preds["base"][p][t] = c12.combine(P0_pool, c_pool, m_now, act)
            preds["A_phat"][p][t] = c12.combine(P0_plug, c_phat, m_now, act)
            preds["B_phat+post0"][p][t] = c12.combine(np.where(zero, P0_post, P0_plug), c_phat, m_now, act)
            preds["C_post_all"][p][t] = c12.combine(P0_post, c_phat, m_now, act)
            preds["D_base+post0"][p][t] = c12.combine(np.where(zero, P0_post, P0_pool), c_pool, m_now, act)
    q = 200.0 / round(0.25 * G)
    sel, ch = {}, {}
    for v in VARS:
        J = {p: [DB.criterion(Y[j], preds[v][p][j], q) for j in range(T)] for p in DB.POOLS}
        s_, c_ = [], []
        for t in range(T):
            if t <= 2:
                c = DB.DEFAULT[:2]
            else:
                W = range(max(1, t - DB.WINDOW), t)
                sc = [float(np.mean([J[p][j] for j in W])) for p in DB.POOLS]
                best = max(sc)
                c = [p for p, s in zip(DB.POOLS, sc) if s >= best - 1e-3][0]
            s_.append(preds[v][c][t])
            c_.append(f"{c[0]}{c[1]}")
        sel[v], ch[v] = s_, c_
    return sel, ch


def nass_orig(d, det, C_sel):
    tg = "orig"
    Y = d.Y(tg)
    F, cls = NS.build_features(d, tg, Y, None, None)
    fb = NS.fallback(d, tg, Y, F)                       # from the ORIGINAL F (shared fallback)
    G_ = NS.GROUPS
    full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"] + G_["obs"]
    st = NS.Stacker(F, cls, Y, full)
    P_full, ch_full = NS.run_learned(st, [full], Y, d, fb)
    del st
    F2 = dict(F)
    F2["p0"] = np.exp(det["logP0_post"])
    F2["obs_box4"] = np.stack([np.asarray(c, np.float64) for c in C_sel])
    st2 = NS.Stacker(F2, cls, Y, full)
    P_O2, ch_O2 = NS.run_learned(st2, [full], Y, d, fb)
    del st2
    trend = [fb[0]] + [Y[i - 1] for i in range(1, d.T)]
    return Y, {"NASS_full_orig": P_full, "O2_nass+C_head": P_O2, "trend": trend}, {"NASS_full_orig": ch_full,
                                                                                   "O2_nass+C_head": ch_O2}


def orig_all(d):
    """All orig models (strict). Returns Y, predictions dict, chosen dict, detection settings."""
    Y = d.Y("orig")
    det = detection(d)
    sel, ch = heads(d, det, Y)
    Y2, pn, chn = nass_orig(d, det, sel["C_post_all"])
    assert np.array_equal(Y, Y2)
    preds = {**{f"{v}[rolling]": sel[v] for v in VARS}, **pn}
    chosen = {**{f"{v}[rolling]": ch[v] for v in VARS}, **chn}
    return Y, preds, chosen, det["setting"]
