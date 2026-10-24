"""(1b) Robustness of the GRN lag-1 signal found under l1 (grn_signal.json): stronger nulls and a lean strict test.

Per transition a -> b, genes with >= 1 regulator in bin a, own controls as in grn_signal (o_last, o_rev, o_lvl,
o_f, o_zero):
  grn        : importance-weighted mean of the regulators' last change (bin-a edges);
  grn_prev   : the same with the bin a-1 edges (graph one bin older);
  lvlnull    : every edge's regulator replaced by a random regulator of the same expression decile at bin a
               (5 draws, mean of the partial correlations): keeps in-degree, weights and the regulators' level;
  out        : importance-weighted mean of the targets' last change (for regulators only, reversed direction);
  binary     : unweighted mean over all regulators (edge presence only).
Partial correlations (OLS residualisation on own), sign counts over transitions, and a lean strict rolling-origin
ridge: own vs own + grn (+ its reversion analogue) vs own + lvlnull analogue (same feature form).
Output: grn_check.json.
"""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge

import common
from grn_signal import graph_mats, partial_corr, row_norm, zscore

NULL_DRAWS = 5


def level_matched(u, lvl, rng):
    regs = np.unique(u)
    dec = np.searchsorted(np.quantile(lvl[regs], np.linspace(0.1, 0.9, 9)), lvl[regs], side="right")
    newid = {}
    mapping = np.arange(len(lvl))
    for k in np.unique(dec):
        grp = regs[dec == k]
        mapping[grp] = rng.permutation(grp)
    return mapping[u]


def main():
    out = {}
    rng = np.random.default_rng(13)
    for name in common.SNAPS:
        d = common.load_node(name)
        N = d["m"].shape[0]
        snaps = common.SNAPS[name]
        trans = common.transitions(name)
        u_all, i_all, ts_all, w_all = common.load_edges(name)
        Y = {T: [common.target(T, d["m"][:, a], d["m"][:, b]) for a, b in trans] for T in common.TARGETS}
        rows = {T: [] for T in common.TARGETS}
        feats = {T: {} for T in common.TARGETS}
        for ti in range(2, len(trans)):
            a = trans[ti][0]
            hb = [s for s in snaps if s <= a]
            p = hb[-2]
            sel = ts_all == a
            u, i, w = u_all[sel], i_all[sel], w_all[sel].astype(float)
            selp = ts_all == p
            W = row_norm(graph_mats(u, i, w, N))
            Wp = row_norm(graph_mats(u_all[selp], i_all[selp], w_all[selp].astype(float), N))
            Wb = row_norm(graph_mats(u, i, np.ones_like(w), N))
            Wout = row_norm(graph_mats(u, i, w, N).T.tocsr())
            m = d["m"][:, a]
            lvl = np.log1p(1e4 * m / m.sum())
            Wn = [row_norm(graph_mats(level_matched(u, lvl, rng), i, w, N)) for _ in range(NULL_DRAWS)]
            has = np.asarray((graph_mats(u, i, w, N) > 0).sum(1)).ravel() > 0
            isreg = np.isin(np.arange(N), np.unique(u))
            lo = max(0, len(hb) - 4)
            wt = d["n"][hb[lo:]]
            pool4 = (d["m"][:, hb[lo:]] * wt).sum(1) / wt.sum()
            for T in common.TARGETS:
                yprev = Y[T][ti - 1]
                orev = common.target(T, m, pool4)
                own = np.column_stack([yprev, orev, lvl, d["f"][:, a], (m == 0).astype(float)])
                y = Y[T][ti]
                r = {"transition": f"{trans[ti][0]}->{trans[ti][1]}"}
                for tag, Wx in (("grn", W), ("grn_prev", Wp), ("binary", Wb)):
                    r[tag] = partial_corr(y[has], (Wx @ yprev)[has], own[has])
                r["lvlnull"] = float(np.mean([partial_corr(y[has], (Wx @ yprev)[has], own[has]) for Wx in Wn]))
                r["out"] = partial_corr(y[isreg], (Wout @ yprev)[isreg], own[isreg])
                rows[T].append(r)
                feats[T][ti] = {"own": own, "grn": np.column_stack([W @ yprev, W @ orev, has]),
                                "lvlnull": np.column_stack([Wn[0] @ yprev, Wn[0] @ orev, has])}
        res = {}
        for T in common.TARGETS:
            R = rows[T]
            summ = {k: {"mean": float(np.mean([r[k] for r in R])), "n_positive": int(sum(r[k] > 0 for r in R)),
                        "n": len(R)} for k in ("grn", "grn_prev", "binary", "lvlnull", "out")}
            valid = sorted(feats[T])
            evals = valid[1:]
            mk = common.masks(N, len(evals), 5, seed=11)
            ridge = {}
            for sname in ("own", "own+grn", "own+lvlnull"):
                def X(t):
                    parts = [feats[T][t]["own"]] + ([] if sname == "own" else [feats[T][t][sname.split("+")[1]]])
                    M = np.column_stack(parts)
                    return np.column_stack([zscore(M[:, j]) for j in range(M.shape[1])])
                Xs = {t: X(t) for t in valid}
                ys = {t: zscore(Y[T][t]) for t in valid}
                P = []
                for t in evals:
                    tr = [s for s in valid if s < t]
                    mdl = Ridge(alpha=10.0).fit(np.vstack([Xs[s] for s in tr]), np.concatenate([ys[s] for s in tr]))
                    P.append(mdl.predict(Xs[t]))
                sc = common.score_masked([Y[T][t] for t in evals], P, mk)
                ridge[sname] = {k: v for k, v in sc.items() if k != "per_transition"}
            res[T] = {"partial_summary": summ, "strict_ridge": ridge,
                      "per_transition": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in R]}
            print(name, T, {k: (round(v["mean"], 3), f"{v['n_positive']}/{v['n']}") for k, v in summ.items()})
            print("   ridge", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in ridge.items()}, flush=True)
        out[name] = res
    with open(os.path.join(common.OUT, "grn_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
