"""Strict per-gene local-polynomial forecasts in each target's own space (does a long-window smoother reach the
noise ceiling? tests the 'small but persistent true change' reading of posthoc.py).

l1     : g_b = log1p(CP10k_b) (each bin normalised by its own library), prediction g_hat(b) - g(a);
linear : g_b = m_b (raw mean), prediction g_hat(b) - m_a;
orig   : local fit on m_b, then the finite-cell expected orig target of a law whose mean is the forecast (the level
         law of the window rescaled to the forecast mean, n = n_a cells).
g_hat(b) = weighted least-squares fit over the last W snapshot bins <= a (time = bin index), degree 0 (level) or 1
(level + phi * slope, phi in {0.5, 1}), weights n_b (precision of a bin mean) or equal. All bins used are <= a.
Selection: (W, degree, phi, weights) chosen per transition by rolling origin on past transitions only (mean
all-gene Pearson on transitions < i, needs i >= 2); also reported: the best fixed configuration in hindsight
(non-causal choice of one configuration for the whole dataset, labelled as such).
Output: results_trendfit.json.
"""
import itertools
import json
import os

import numpy as np

import variance_decomp as V

OUT = V.OUT
WS = (2, 3, 4, 6, 8, 12, 40)
CONFIGS = [(W, 0, 0.0, wt) for W in WS for wt in ("n", "eq")] + \
          [(W, 1, phi, wt) for W in WS if W >= 3 for phi in (0.5, 1.0) for wt in ("n", "eq")]


def g_space(tg, m):
    if tg == "l1":
        return np.log1p(1e4 * m / m.sum(0, keepdims=True))
    return m


def forecast(gs, n, a, cfg):
    W, deg, phi, wt = cfg
    ks = np.arange(max(0, a - W + 1), a + 1)
    w = (n[ks] if wt == "n" else np.ones(len(ks))).astype(float)
    t = ks.astype(float)
    G = gs[:, ks]
    tb = (w * t).sum() / w.sum()
    gb = (G * w).sum(1) / w.sum()
    if deg == 0 or len(ks) < 2:
        return gb
    stt = (w * (t - tb) ** 2).sum()
    slope = ((G - gb[:, None]) * (w * (t - tb))).sum(1) / stt
    return gb + slope * ((a - tb) + phi * 1.0)


def main():
    out = {}
    for name in V.SNAPS:
        m, s, f, n = V.load(name)
        G, nbin = m.shape
        n_tr = nbin - 1
        Hs = [V.H_of(m[:, j], s[:, j] * np.sqrt(n[j] / max(n[j] - 1, 1)), f[:, j]) for j in range(nbin)]
        rng_mask = np.random.default_rng(99)
        n_test = int(round(0.25 * G))
        masks = [[np.sort(rng_mask.permutation(G)[:n_test]) for _ in range(V.N_MASK)] for _ in range(n_tr)]
        res = {}
        for tg in ("l1", "linear", "orig"):
            Y = np.stack([V.T(tg, m[:, i], m[:, i + 1]) for i in range(n_tr)], 1)
            gs = g_space("l1" if tg == "l1" else "linear", m)
            P = {}
            for ci, cfg in enumerate(CONFIGS):
                for i in range(1, n_tr):
                    gh = forecast(gs, n, i, cfg)
                    if tg == "orig":
                        W = cfg[0]
                        ks = list(range(max(0, i - W + 1), i + 1))
                        lev = V.mix([Hs[k] for k in ks], [n[k] for k in ks])
                        sc = np.where(lev[:, 0] > 0, np.maximum(gh, 0) / np.where(lev[:, 0] > 0, lev[:, 0], 1), 0.0)
                        H = lev * np.column_stack([sc, np.ones(G), sc ** 2])
                        rng = np.random.default_rng(1000 * ci + i)
                        P[(ci, i)] = V.T(tg, m[:, i][:, None], V.sample_means(H, n[i], 48, rng)).mean(1)
                    else:
                        P[(ci, i)] = gh - gs[:, i]
            if tg == "orig":
                pass
            R = np.array([[V.corr(Y[:, i], P[(ci, i)]) for i in range(1, n_tr)] for ci in range(len(CONFIGS))])  # (C, n_tr-1)
            best_fixed = int(np.argmax(R.mean(1)))
            sel = {}
            for i in range(2, n_tr):
                sel[i] = int(np.argmax(R[:, :i - 1].mean(1)))   # columns 0..i-2 = transitions 1..i-1
            rows_sel = [V.score(Y[:, i], P[(sel[i], i)], masks[i]) for i in range(2, n_tr)]
            rows_best = [V.score(Y[:, i], P[(best_fixed, i)], masks[i]) for i in range(2, n_tr)]
            top = np.argsort(-R.mean(1))[:5]
            res[tg] = {
                "strict_rolling_selected_i>=2": {k: float(np.mean([r[k] for r in rows_sel])) for k in ("pearson_nan0", "p_at_200_up", "p_at_200_down")},
                "selected_configs": {int(i): list(map(str, CONFIGS[c])) for i, c in sel.items()},
                "best_fixed_in_hindsight_i>=2": {"config": list(map(str, CONFIGS[best_fixed])),
                                                  **{k: float(np.mean([r[k] for r in rows_best])) for k in ("pearson_nan0", "p_at_200_up", "p_at_200_down")}},
                "top5_configs_mean_allgene_pearson_i>=1": [[list(map(str, CONFIGS[c])), float(R[c].mean())] for c in top],
                "level4_n_mean_allgene_pearson_i>=1": float(R[CONFIGS.index((4, 0, 0.0, "n"))].mean()),
            }
            print(name, tg, json.dumps({k: v for k, v in res[tg].items() if k != "selected_configs"}), flush=True)
        out[name] = res
    with open(os.path.join(OUT, "results_trendfit.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
