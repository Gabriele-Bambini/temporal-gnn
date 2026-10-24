"""State-space follow-up: damped-trend Kalman filters and gene-adaptive noise hyperparameters (all strict).

Why: predictors.py shows (i) the innovations of a single pooled-theta Kalman are under-dispersed for low-expressed
genes and over-dispersed / positively autocorrelated for high-expressed ones (one signal variance does not fit all
genes), (ii) the noise-corrected autocorrelation of the true curve changes is ~ +0.5..0.6 at lag 1 and ~ 0 at lag 2+
(noise_acf.py), i.e. the slope is short-lived: a damped trend (slope AR(1), phi < 1) rather than a random-walk slope.
Models (raw mean or log1p(CP10k) observation space, R and Q as in predictors.kalman_inputs):
  level   mu_{t+1} = mu_t + phi b_t + w,  b_{t+1} = phi b_t + v  (phi = 1: local linear trend; theta_b = 0: local level)
  kd_RO        configuration chosen by rolling-origin criterion J (as in predictors.py) over the damped grid
  ml_strat     per expression decile (strata from bins <= t), the configuration maximising the Gaussian
               log-likelihood of that decile's past innovations (bins <= t): empirical-Bayes, strict
  ml_gene      the same per gene
  obs_ml_strat (orig) the ml_strat raw forecast pushed through the finite-cell observation model
Output: kalman_ml.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, SNAPS, TRAIN_SPLIT, TARGETS, tgt  # noqa: E402
import predictors as pr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def kalman_phi(X, R, Qs, T, th_l, th_b, phi):
    """Returns forecasts (T, G) of bin t+1 made at t (t = 0..T-1) and innovation log-lik (T, G): row t-1 = bin t."""
    G = X.shape[0]
    lvl, b = X[:, 0].copy(), np.zeros(G)
    P11, P12 = R[:, 0].copy(), np.zeros(G)
    P22 = (th_b ** 2) * Qs[:, 0] * 10.0
    fc = np.zeros((T, G))
    ll = np.zeros((T, G))
    fc[0] = lvl
    for t in range(1, T + 1):
        ql, qb = th_l ** 2 * Qs[:, t], th_b ** 2 * Qs[:, t]
        lp, bp = lvl + phi * b, phi * b
        P11p = P11 + 2 * phi * P12 + phi ** 2 * P22 + ql
        P12p = phi * P12 + phi ** 2 * P22
        P22p = phi ** 2 * P22 + qb
        S = P11p + R[:, t]
        v = X[:, t] - lp
        ll[t - 1] = -0.5 * (np.log(S) + v * v / S)
        K1, K2 = P11p / S, P12p / S
        lvl, b = lp + K1 * v, bp + K2 * v
        P11, P12, P22 = (1 - K1) * P11p, (1 - K1) * P12p, P22p - K2 * P12p
        if t < T:
            fc[t] = lvl + phi * b
    return fc, ll


def grid(space):
    # theta_b up to 10 x theta-scale: a fast-adapting slope (sanity check: R -> 0, theta_b >> theta_l reproduces
    # raw-scale linear extrapolation); phi in {0.5, 0.75, 1} spans the measured lag-1 persistence of true changes
    tl = (0.001, 0.05, 0.2, 0.8) if space == "raw" else (0.001, 0.02, 0.1, 0.4)
    tb = (0.1, 0.4, 1.0, 3.0, 10.0) if space == "raw" else (0.01, 0.03, 0.1, 0.3, 1.0)
    g = [(a, 0.0, 1.0) for a in tl]
    g += [(a, b, p) for a in tl for b in tb for p in (0.5, 0.75, 1.0)]
    return g


def main():
    t0 = time.time()
    out = {}
    for name in SNAPS:
        d = load(name)
        m, s, f, n = d["m"], d["s"], d["f"], d["n"]
        T = TRAIN_SPLIT[name]
        G = m.shape[0]
        Y = {t: np.stack([tgt(t, m[:, i], m[:, i + 1]) for i in range(T)]) for t in TARGETS}
        truths = {t: pr.Truth(Y[t]) for t in TARGETS}
        rng = np.random.default_rng(0)
        n_test = int(round(0.25 * G))
        masks = [[np.sort(rng.permutation(G)[:n_test]) for _ in range(T)] for _ in range(pr.N_DRAWS)]
        strata = pr.strata_seq(m, n, T)
        res = {}
        for space in ("raw", "l1"):
            X, R, Q = pr.kalman_inputs(m, s, n, space)
            gr = grid(space)
            FC, LL = [], []
            for c in gr:
                fc, ll = kalman_phi(X, R, Q, T, *c)
                FC.append(fc.astype(np.float32))
                LL.append(ll.astype(np.float32))
            FC, LL = np.stack(FC), np.stack(LL)                # (C, T, G)
            cumLL = np.cumsum(LL, 1, out=LL)                    # cumLL[c, i] = sum over bins 1..i+1 -> use i-1 for bins <= i
            conv = pr.raw_to_targets if space == "raw" else pr.l1_to_targets
            # ---- J-based rolling-origin selection over the damped grid
            fam = pr.run_family(f"kd_{space}", gr, lambda c: conv(m, FC[gr.index(c)]), truths, T, gr.index(gr[2]))
            # ---- empirical-Bayes selection per stratum / per gene from past innovations
            seq_s = np.zeros((T, G))
            seq_g = np.zeros((T, G))
            chosen_s = []
            default = gr.index(gr[2])
            for i in range(T):
                if i == 0:
                    seq_s[i] = FC[default, i]
                    seq_g[i] = FC[default, i]
                    chosen_s.append({})
                    continue
                past = cumLL[:, i - 1]                          # log-lik of bins 1..i (all <= i)
                lab = strata[i]
                ch = {}
                for q in np.unique(lab):
                    g = lab == q
                    c = int(np.argmax(past[:, g].sum(1)))
                    seq_s[i, g] = FC[c, i, g]
                    ch[int(q)] = str(gr[c])
                chosen_s.append(ch)
                cg = np.argmax(past, 0)
                seq_g[i] = FC[cg, i, np.arange(G)]
            Ps, Pg = conv(m, seq_s), conv(m, seq_g)
            r = {}
            for t in TARGETS:
                F = fam[t]
                sc = pr.mask_score(Y[t], F["seq"], masks)
                hb = pr.mask_score(Y[t], F["best_seq"], masks)
                r[t] = {"kd_RO": sc, "kd_hindsight_best_NOT_STRICT": {"config": str(gr[F["best"]]), **{k: v for k, v in hb.items() if k != "per_transition_from_i0"}},
                        "kd_RO_selected_configs": [str(gr[c]) for c in F["sel"]],
                        "ml_strat": pr.mask_score(Y[t], Ps[t], masks), "ml_gene": pr.mask_score(Y[t], Pg[t], masks)}
            if space == "raw":
                Po = pr.fc_obs(m, s, f, n, T, seq_s)
                r["orig"]["obs_ml_strat"] = pr.mask_score(Y["orig"], Po, masks)
                Po2 = pr.fc_obs(m, s, f, n, T, F_best_damped(fam, gr, FC, truths, T))
                r["orig"]["obs_kd_bestJ_raw_RO"] = pr.mask_score(Y["orig"], Po2, masks)
            # how often does the ML choice include a slope, and is it damped? (last transition, per decile)
            last = chosen_s[-1]
            r["ml_strat_choice_last_transition_by_decile"] = last
            phis = [eval(v)[2] if eval(v)[1] > 0 else 0.0 for ch in chosen_s[1:] for v in ch.values()]
            r["ml_strat_frac_with_slope"] = float(np.mean([p > 0 for p in phis]))
            r["ml_strat_frac_damped_among_slope"] = float(np.mean([p < 1 for p in phis if p > 0])) if any(p > 0 for p in phis) else None
            sel_j = [eval(str(gr[c])) for c in fam["l1"]["sel"][2:]]
            r["kd_RO_l1_frac_slope"] = float(np.mean([c[1] > 0 for c in sel_j]))
            r["kd_RO_l1_phi_counts"] = {str(p): int(sum(1 for c in sel_j if c[1] > 0 and c[2] == p)) for p in (0.5, 0.75, 1.0)}
            sel_j = [eval(str(gr[c])) for c in fam["linear"]["sel"][2:]]
            r["kd_RO_linear_phi_counts"] = {str(p): int(sum(1 for c in sel_j if c[1] > 0 and c[2] == p)) for p in (0.5, 0.75, 1.0)}
            res[space] = r
            print(f"== {name} kalman space={space} ({time.time() - t0:.0f}s)  slope chosen {r['ml_strat_frac_with_slope']:.2f}, damped|slope {r['ml_strat_frac_damped_among_slope']}")
            print(f"   ML choice last transition by decile: {last}")
            print(f"   RO phi counts l1 {r['kd_RO_l1_phi_counts']} linear {r['kd_RO_linear_phi_counts']}")
            for t in TARGETS:
                for k in ("kd_RO", "ml_strat", "ml_gene", "obs_ml_strat", "obs_kd_bestJ_raw_RO"):
                    if k in r[t]:
                        x = r[t][k]
                        print(f"   {t:6s} {k:20s} PCC {x['pcc']:+.3f}+-{x['pcc_sd']:.3f} up {x['p200_up']:.3f} dn {x['p200_down']:.3f}")
                hb = r[t]["kd_hindsight_best_NOT_STRICT"]
                print(f"   {t:6s} hindsight {hb['config']:18s} PCC {hb['pcc']:+.3f} up {hb['p200_up']:.3f} dn {hb['p200_down']:.3f}")
        out[name] = res
    with open(os.path.join(HERE, "kalman_ml.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


def F_best_damped(fam, gr, FC, truths, T):
    """Raw forecast sequence of the J-selected (rolling-origin, orig criterion) damped config, for the obs model."""
    sel = fam["orig"]["sel"]
    return np.stack([FC[sel[i], i] for i in range(T)])


if __name__ == "__main__":
    main()
