"""C12: DistBridge's finite-cell observation head on orig with detection from a per-gene logit-random-walk binomial
filter instead of the pooled plug-in f, strict rolling origin; and the n_{t+1}-invariance of the E[(1-p)^n] ranking.

Filter (exact, on a grid): eta_b = logit p_b on 101 points in [-15, 5]; eta_b = eta_{b-1} + N(0, tau^2);
K_b ~ Bin(n_b, p_b), K_b = f_b n_b (integer to 6e-5). Prior at the first snapshot N(mu0, 3^2). (tau, mu0) on a grid,
chosen at every forecast time t by the prequential log-likelihood of K_0..K_t (bins <= t only).
Predictive of p_{t+1} given bins <= t: post_t convolved with the tau kernel. p_hat = its mean.
Head (same formulas as dist_bridge.expected_target, re-implemented so P0 and f can be set separately): bin t+1 =
n_t cells from the law with detection f and the pool's positive part (mu, var);
  y = -5 P0 (m_t > 0) or 5 (1 - P0) (m_t = 0), plus (1 - P0) E[clip target | K >= 1] for m_t > 0 (MC, 256 draws).
Variants: base = pooled f (DistBridge, delta = 0); A = p_hat everywhere (P0 = (1 - p_hat)^n);
  B = A but P0 = E_post[(1-p)^n] for genes with K_t = 0 (the literal wording); C = posterior P0 for every gene;
  D = base for K_t > 0, posterior P0 for K_t = 0 (isolates the zero genes).
Pools: dist_bridge.POOLS (box 2/4/6/8, kalman 0.1/0.2/0.35), delta = 0; rolling selection as DistBridge.predict
(t <= 2 default kalman 0.2; else best mean J over past window 8, ties within 1e-3 -> earlier pool), per variant.
Invariance (among genes with K_t > 0, fixed p_hat): log E_n[(1 - p_hat)^n] under n = n_t, n_t exp(+-sigma_m) (50/50,
sigma_m = median past |dlog n|), lognormal (sd = past sd of dlog n, 21-point Gauss-Hermite), oracle n_{t+1}
(reference only, not strict); Spearman vs n = n_t; the same with the posterior of p; and float64-naive values (ties).
Scores: common.score on orig, 1000 s + i masks, i >= 1 (and i >= 2).
"""
import json
import os
import time

import numpy as np
from scipy import stats
from scipy.special import logsumexp

import common as C

DB = C.D.DB
EG = np.linspace(-15, 5, 101)
PG = 1 / (1 + np.exp(-EG))
LP, LQ = np.log(PG), np.log1p(-PG)
TAUS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5)
MU0S = (-8.0, -6.0, -4.0)
S0 = 3.0
DRAWS = 256
CLIP = 5.0


def kernel(tau):
    M = np.exp(-0.5 * ((EG[None, :] - EG[:, None]) / tau) ** 2)
    return M / M.sum(1, keepdims=True)


def run_filter(K, n, tau, mu0, keep=None):
    """Returns prequential log-lik per gene and bin (G, B) and, for b in keep, the predictive of p_{b+1} (G, 101)."""
    G, B = K.shape
    M = kernel(tau)
    prior = np.exp(-0.5 * ((EG - mu0) / S0) ** 2)
    prior /= prior.sum()
    pri = np.broadcast_to(prior, (G, len(EG)))
    pll = np.zeros((G, B))
    kept = {}
    for b in range(B):
        if b > 0:
            pri = post @ M
        ll = K[:, b:b + 1] * LP[None, :] + (n[b] - K[:, b:b + 1]) * LQ[None, :]
        mx = ll.max(1, keepdims=True)
        w = pri * np.exp(ll - mx)
        z = w.sum(1)
        pll[:, b] = np.log(np.maximum(z, 1e-300)) + mx[:, 0]
        post = w / np.maximum(z, 1e-300)[:, None]
        if keep is not None and b in keep:
            kept[b] = post @ M
    return pll, kept


def cond_part(f, mu, var, m_now, n, rng, act):
    """E[clip(log(S/n + eps) - log(m_now + eps)) | K >= 1] for genes act (dist_bridge.expected_target's sampler)."""
    n = max(int(round(n)), 1)
    fa, mua, va = f[act], mu[act], var[act]
    P0s = (1 - fa) ** n
    u = rng.random((len(act), DRAWS))
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.ceil(np.log1p(-u * (1 - P0s[:, None])) / np.log1p(-fa[:, None]))
    J = np.clip(np.nan_to_num(J, nan=1.0, posinf=n, neginf=1.0), 1, n).astype(np.int64)
    Kc = 1 + rng.binomial(n - J, fa[:, None])
    lam = (mua - 1)[:, None] * Kc
    vv = va[:, None] * Kc
    kind = np.broadcast_to(DB._law_kind(mua, va)[:, None], lam.shape)
    extra = np.zeros_like(lam)
    nb, po = kind == 2, kind == 1
    if nb.any():
        ln, vn = lam[nb], vv[nb]
        extra[nb] = rng.negative_binomial(ln ** 2 / (vn - ln), ln / vn)
    if po.any():
        extra[po] = rng.poisson(lam[po])
    return DB.log_delta(np.repeat(m_now[act, None], DRAWS, 1), (Kc + extra) / n).mean(1)


def combine(P0, cond, m_now, act):
    y = np.where(m_now > 0, -CLIP * P0, CLIP * (1 - P0)).astype(np.float64)
    y[act] = y[act] + (1 - P0[act]) * cond
    return y


def gh(k=21):
    x, w = np.polynomial.hermite_e.hermegauss(k)
    return x, w / w.sum()


def main():
    out = {}
    for name in C.NAMES:
        t0 = time.time()
        d = C.load(name)
        G, B, T = d.G, d.B, d.T
        K = np.rint(d.f * d.n[None, :])
        n = d.n
        Y = d.Y("orig")
        # ---- filter hyperparameters by prequential log-lik (strict)
        settings = [(tau, mu0) for tau in TAUS for mu0 in MU0S]
        cum = []
        for tau, mu0 in settings:
            pll, _ = run_filter(K, n, tau, mu0)
            cum.append(np.cumsum(pll.sum(0)))
        cum = np.array(cum)                                         # (S, B): log-lik of K_0..K_b
        chosen = [int(np.argmax(cum[:, t])) for t in range(T)]
        o = {"filter_setting_per_t": [f"tau={settings[c][0]},mu0={settings[c][1]}" for c in chosen]}
        # ---- predictives at every t with the chosen setting
        p_hat = np.zeros((T, G))
        logP0_plug = np.zeros((T, G))
        logP0_post = np.zeros((T, G))
        inv = {}
        dln = np.diff(np.log(n))
        xg, wg = gh()
        for si in sorted(set(chosen)):
            ts = [t for t in range(T) if chosen[t] == si]
            _, kept = run_filter(K, n, settings[si][0], settings[si][1], keep=set(ts))
            for t in ts:
                pr = kept[t]
                lpr = np.log(np.maximum(pr, 1e-300))
                p_hat[t] = pr @ PG
                nt = max(int(round(n[t])), 1)
                logP0_plug[t] = nt * np.log1p(-p_hat[t])
                logP0_post[t] = logsumexp(lpr + nt * LQ[None, :], axis=1)
                # invariance: n distributions (strict sigmas from past dlog n, bins <= t)
                past = dln[:t]
                sm = float(np.median(np.abs(past))) if len(past) else 0.5
                ss = float(np.sqrt(np.mean(past ** 2))) if len(past) else 0.5
                ndist = {"n_t": ([n[t]], [1.0]),
                         "n_t*exp(+-sigma_med)": ([n[t] * np.exp(-sm), n[t] * np.exp(sm)], [0.5, 0.5]),
                         "lognormal(sd_past)": (list(n[t] * np.exp(ss * xg)), list(wg))}
                if t + 1 < B:
                    ndist["oracle n_{t+1}"] = ([n[t + 1]], [1.0])
                pos = K[:, t] > 0
                res = {"sigma_med": round(sm, 3), "sigma_sd": round(ss, 3), "n_genes_K>0": int(pos.sum())}
                base_plug = base_post = None
                for k, (ns, ws) in ndist.items():
                    ns, lw = np.array(ns), np.log(np.array(ws))
                    plug = logsumexp(ns[None, :] * np.log1p(-p_hat[t][pos])[:, None] + lw[None, :], axis=1)
                    post = logsumexp(logsumexp(lpr[pos][:, :, None] + ns[None, None, :] * LQ[None, :, None], axis=1)
                                     + lw[None, :], axis=1)
                    naive = np.exp(plug)
                    if base_plug is None:
                        base_plug, base_post, base_naive = plug, post, naive
                        res[f"{k}: frac P0 == 0 in float64 (plug)"] = round(float(np.mean(naive == 0)), 4)
                        continue
                    res[f"spearman plug-in p_hat, {k} vs n_t (log space)"] = round(float(stats.spearmanr(base_plug, plug).correlation), 6)
                    res[f"spearman plug-in p_hat, {k} vs n_t (float64 naive, ties)"] = round(float(stats.spearmanr(base_naive, naive).correlation), 6)
                    res[f"spearman posterior p, {k} vs n_t"] = round(float(stats.spearmanr(base_post, post).correlation), 6)
                inv[t] = res
        o["invariance_per_t"] = inv
        keys = [k for k in inv[1] if k.startswith("spearman")]
        o["invariance_summary_t>=1"] = {k: {"min": round(min(inv[t][k] for t in range(1, T) if k in inv[t]), 6),
                                            "mean": round(float(np.mean([inv[t][k] for t in range(1, T) if k in inv[t]])), 6)}
                                        for k in keys}
        # ---- heads per pool
        trip = [DB.triplet(d.X[b]) for b in range(B)]
        pools = {p: (DB.box_pool if p[0] == "box" else DB.kalman_pool)(trip, list(n), p[1]) for p in DB.POOLS}
        VARS = ("base", "A_phat", "B_phat+post0", "C_post_all", "D_base+post0")
        preds = {v: {p: [None] * T for p in DB.POOLS} for v in VARS}
        chk = None
        key = 1 + (name == "pancreas-human")
        for pi, p in enumerate(DB.POOLS):
            for t in range(T):
                H = pools[p][t] if t > 0 else trip[0]
                fP, mu, var = DB.positive_part(H)
                m_now = d.m[:, t]
                nt = max(int(round(n[t])), 1)
                act = np.where((m_now > 0) & (fP > 0))[0]
                c_pool = cond_part(fP, mu, var, m_now, n[t], np.random.default_rng([key, t, pi, 0]), act)
                c_phat = cond_part(p_hat[t], mu, var, m_now, n[t], np.random.default_rng([key, t, pi, 1]), act)
                P0_pool = (1 - fP) ** nt
                P0_plug, P0_post = np.exp(logP0_plug[t]), np.exp(logP0_post[t])
                zero = K[:, t] == 0
                preds["base"][p][t] = combine(P0_pool, c_pool, m_now, act)
                preds["A_phat"][p][t] = combine(P0_plug, c_phat, m_now, act)
                preds["B_phat+post0"][p][t] = combine(np.where(zero, P0_post, P0_plug), c_phat, m_now, act)
                preds["C_post_all"][p][t] = combine(P0_post, c_phat, m_now, act)
                preds["D_base+post0"][p][t] = combine(np.where(zero, P0_post, P0_pool), c_pool, m_now, act)
                if chk is None and t == 5:
                    ref = DB.expected_target(H, m_now, n[t], np.random.default_rng([key, t, pi, 0]))
                    chk = float(np.max(np.abs(ref - preds["base"][p][t])))
        o["check_base_equals_dist_bridge.expected_target (max abs diff, same rng)"] = chk
        # ---- rolling selection (as DistBridge.predict, delta = 0)
        q = 200.0 / round(0.25 * G)
        mk = C.masks(G, T)
        o["scores"] = {}
        sel_log = {}
        for v in VARS:
            J = {p: [DB.criterion(Y[j], preds[v][p][j], q) for j in range(T)] for p in DB.POOLS}
            sel, ch = [], []
            for t in range(T):
                if t <= 2:
                    c = DB.DEFAULT[:2]
                else:
                    W = range(max(1, t - DB.WINDOW), t)
                    sc = [float(np.mean([J[p][j] for j in W])) for p in DB.POOLS]
                    best = max(sc)
                    c = [p for p, s_ in zip(DB.POOLS, sc) if s_ >= best - 1e-3][0]
                sel.append(preds[v][c][t])
                ch.append(f"{c[0]}{c[1]}")
            sel_log[v] = ch
            for lab, P in (("rolling", sel), ("box4", preds[v][("box", 4)]), ("kalman0.2", preds[v][("kalman", 0.2)])):
                rows = C.score(Y, P, mk)
                o["scores"][f"{v} [{lab}]"] = {"i>=1": C.summ(rows, range(1, T)), "i>=2": C.summ(rows, range(2, T))}
                if lab == "rolling":
                    np.save(os.path.join(C.HERE, f"c12_rows_{v}_{name}.npy"), rows)
        o["selected_pools"] = sel_log
        # detection diagnostics
        o["p_hat_vs_f_t"] = {"median_abs_logit_diff_K>0": round(float(np.median(np.abs(
            (np.log(p_hat[1:] / (1 - p_hat[1:])) - np.log(np.clip(d.f[:, 1:T].T, 1e-9, 1) / np.clip(1 - d.f[:, 1:T].T, 1e-9, 1)))[K[:, 1:T].T > 0]))), 3),
            "mean P0_post among K_t=0 genes (t>=1)": round(float(np.mean([np.exp(logP0_post[t][K[:, t] == 0]).mean() for t in range(1, T)])), 4)}
        o["seconds"] = round(time.time() - t0)
        out[name] = o
        print("==", name, "filter", o["filter_setting_per_t"])
        print("  check", chk)
        for k, v in o["scores"].items():
            print(f"  {k:34s} i>=1 {C.fmt(v['i>=1'])} | i>=2 {C.fmt(v['i>=2'])}")
        print("  invariance", json.dumps(o["invariance_summary_t>=1"]))
        print("  diag", o["p_hat_vs_f_t"], "sec", o["seconds"], flush=True)
        json.dump(out, open(os.path.join(C.HERE, "c12_binomial_head.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
