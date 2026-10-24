"""Strict per-gene heteroscedastic Kalman filter (local level + damped local trend) with the KNOWN sampling noise
of every bin mean as observation variance: the principled form of 'per-gene history + noise model'.

Space g: l1 -> log1p(CP10k_b); linear -> m_b. Observation variance of g_b per gene and bin:
  linear: R = s_b^2 / (n_b - 1), floored at (1 / n_b)^2 (a zero bin mean still has an uncertainty of ~1 cell);
  l1    : delta method, R = (1e4 / L_b)^2 s_b^2 / (n_b - 1) / (1 + CP10k_b)^2, floored at (1e4 / (L_b n_b))^2.
State (level, slope), x_b = [[1, 1], [0, phi]] x_{b-1} + w, Q = diag(q_l, q_s) * scale; scale = 'rel': the gene's
median R over the bins seen so far, 'abs': the median over genes of that quantity (one number per transition).
Prediction for transition a -> b: (level_{a|a} + phi * slope_{a|a}) - g_a (the target subtracts the OBSERVED g_a).
Grid of (q_l, q_s, phi, scale); per transition the configuration is chosen by rolling origin on past transitions
only (mean all-gene Pearson over transitions 1..i-1, i >= 2); the best fixed configuration in hindsight is also
reported (non-causal choice, labelled). Output: results_kalman.json.
"""
import itertools
import json
import os

import numpy as np

import variance_decomp as V

OUT = V.OUT
QL = (0.03, 0.3, 3.0, 30.0, 300.0)
QS = (0.0, 0.03, 0.3, 3.0, 30.0, 300.0)
PHI = (1.0, 0.7)
SCALE = ("rel", "abs")
CONFIGS = list(itertools.product(QL, QS, PHI, SCALE))


def space(tg, m, s, n):
    L = m.sum(0, keepdims=True)
    if tg == "l1":
        cp = 1e4 * m / L
        g = np.log1p(cp)
        R = (1e4 / L) ** 2 * s ** 2 / np.maximum(n - 1, 1)[None, :] / (1 + cp) ** 2
        R = np.maximum(R, (1e4 / (L * n[None, :])) ** 2)
    else:
        g = m.copy()
        R = np.maximum(s ** 2 / np.maximum(n - 1, 1)[None, :], (1.0 / n[None, :]) ** 2)
    return g, R


def run_filter(g, R, cfg):
    """Returns pred[:, a] = one-step forecast of g_{a+1} made with bins <= a."""
    ql, qs, phi, sc = cfg
    G, B = g.shape
    lev = g[:, 0].copy()
    slo = np.zeros(G)
    P11, P12, P22 = R[:, 0].copy(), np.zeros(G), R[:, 0].copy()   # level starts at g_0; slope prior ~ 1 obs sd
    pred = np.zeros((G, B))
    for b in range(B):
        if b > 0:
            scale = np.median(R[:, :b + 1], 1) if sc == "rel" else np.full(G, np.median(np.median(R[:, :b + 1], 1)))
            # predict
            lev, slo = lev + slo, phi * slo
            p11 = P11 + 2 * P12 + P22 + ql * scale
            p12 = phi * (P12 + P22)
            p22 = phi ** 2 * P22 + qs * scale
            # update with g_b
            S = p11 + R[:, b]
            k1, k2 = p11 / S, p12 / S
            inn = g[:, b] - lev
            lev, slo = lev + k1 * inn, slo + k2 * inn
            P11, P12, P22 = (1 - k1) * p11, (1 - k1) * p12, p22 - k2 * p12
        pred[:, b] = lev + phi * slo
    return pred


def main():
    out = {}
    for name in V.SNAPS:
        m, s, f, n = V.load(name)
        G, nbin = m.shape
        n_tr = nbin - 1
        rng_mask = np.random.default_rng(99)
        n_test = int(round(0.25 * G))
        masks = [[np.sort(rng_mask.permutation(G)[:n_test]) for _ in range(V.N_MASK)] for _ in range(n_tr)]
        res = {}
        for tg in ("l1", "linear"):
            Y = np.stack([V.T(tg, m[:, i], m[:, i + 1]) for i in range(n_tr)], 1)
            g, R = space(tg, m, s, n)
            preds = []
            for cfg in CONFIGS:
                pr = run_filter(g, R, cfg)
                preds.append(np.stack([pr[:, i] - g[:, i] for i in range(n_tr)], 1))
            C = np.array([[V.corr(Y[:, i], P[:, i]) for i in range(1, n_tr)] for P in preds])
            best = int(np.argmax(C.mean(1)))
            sel = {i: int(np.argmax(C[:, :i - 1].mean(1))) for i in range(2, n_tr)}
            rows_sel = [V.score(Y[:, i], preds[sel[i]][:, i], masks[i]) for i in range(2, n_tr)]
            rows_best = [V.score(Y[:, i], preds[best][:, i], masks[i]) for i in range(2, n_tr)]
            res[tg] = {
                "strict_rolling_selected_i>=2": {k: float(np.mean([r[k] for r in rows_sel])) for k in ("pearson_nan0", "p_at_200_up", "p_at_200_down")},
                "best_fixed_in_hindsight_i>=2": {"config": list(map(str, CONFIGS[best])),
                                                  **{k: float(np.mean([r[k] for r in rows_best])) for k in ("pearson_nan0", "p_at_200_up", "p_at_200_down")}},
                "per_transition_selected_pearson": [round(r["pearson_nan0"], 3) for r in rows_sel],
                "selected_configs": {int(i): list(map(str, CONFIGS[c])) for i, c in sel.items()},
            }
            print(name, tg, json.dumps({k: v for k, v in res[tg].items() if k not in ("selected_configs",)}), flush=True)
        out[name] = res
    with open(os.path.join(OUT, "results_kalman.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
