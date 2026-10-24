"""Is the gap between oracle-coefficient and strict per-gene history fits (posthoc.py) a regime effect?

Same per-gene features as posthoc.py (lags y_{i-1..i-3}, deviation of g_a from the cell-weighted level of the last 4
and 8 bins), coefficients varying by 5 expression-level classes of m_a, fitted STRICTLY on past transitions with
exponential recency weights w_j = 0.5 ** ((i - 1 - j) / h), half-life h in transitions (h = inf: equal pooling as
in posthoc.py). h is also chosen per transition by rolling origin (mean all-gene Pearson on transitions < i).
Targets l1 and linear (orig is saturated by the level + finite-cell observation model). Output: results_regime.json.
"""
import json
import os

import numpy as np

import variance_decomp as V
from posthoc import interact, level_class

OUT = V.OUT
HS = (0.5, 1.0, 2.0, 4.0, np.inf)


def wls(X, y, w):
    X1 = np.column_stack([np.ones(len(X)), X])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X1 * sw[:, None], y * sw, rcond=None)
    return coef


def main():
    out = {}
    for name in V.SNAPS:
        m, s, f, n = V.load(name)
        G, nbin = m.shape
        n_tr = nbin - 1
        Hs = [V.H_of(m[:, j], s[:, j], f[:, j]) for j in range(nbin)]
        rng_mask = np.random.default_rng(99)
        n_test = int(round(0.25 * G))
        masks = [[np.sort(rng_mask.permutation(G)[:n_test]) for _ in range(V.N_MASK)] for _ in range(n_tr)]
        res = {}
        for tg in ("l1", "linear"):
            Y = np.stack([V.T(tg, m[:, i], m[:, i + 1]) for i in range(n_tr)], 1)
            feats = []
            for i in range(n_tr):
                cols = [Y[:, i - k] if i - k >= 0 else np.zeros(G) for k in (1, 2, 3)]
                for K in (4, 8):
                    ks = list(range(max(0, i - K + 1), i + 1))
                    lev = V.mix([Hs[k] for k in ks], [n[k] for k in ks])
                    cols.append(V.T(tg, m[:, i], lev[:, 0]))
                feats.append(interact(np.column_stack(cols), level_class(m[:, i])))
            P = {h: {} for h in HS}
            for i in range(2, n_tr):
                Xp = np.vstack([feats[j] for j in range(1, i)])
                yp = np.concatenate([Y[:, j] for j in range(1, i)])
                for h in HS:
                    w = np.concatenate([np.full(G, 1.0 if not np.isfinite(h) else 0.5 ** ((i - 1 - j) / h)) for j in range(1, i)])
                    c = wls(Xp, yp, w)
                    P[h][i] = c[0] + feats[i] @ c[1:]
            C = {h: {i: V.corr(Y[:, i], P[h][i]) for i in P[h]} for h in HS}
            sel = {}
            for i in range(3, n_tr):
                sel[i] = max(HS, key=lambda h: np.mean([C[h][j] for j in range(2, i)]))
            r = {}
            for h in HS:
                rows = [V.score(Y[:, i], P[h][i], masks[i]) for i in range(2, n_tr)]
                r[f"halflife_{h}_i>=2"] = {k: float(np.mean([rw[k] for rw in rows])) for k in ("pearson_nan0", "p_at_200_up", "p_at_200_down")}
            rows = [V.score(Y[:, i], P[sel[i]][i], masks[i]) for i in range(3, n_tr)]
            r["rolling_selected_halflife_i>=3"] = {k: float(np.mean([rw[k] for rw in rows])) for k in ("pearson_nan0", "p_at_200_up", "p_at_200_down")}
            r["selected_halflives"] = {int(i): str(h) for i, h in sel.items()}
            res[tg] = r
            print(name, tg, json.dumps({k: v for k, v in r.items() if k != "selected_halflives"}), flush=True)
        out[name] = res
    with open(os.path.join(OUT, "results_regime.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
