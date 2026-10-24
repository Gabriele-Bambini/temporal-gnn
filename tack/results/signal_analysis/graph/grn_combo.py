"""(1c) Is the small GRN gain under l1 additive to noise-aware weighting of the own history?

Strict rolling-origin ridge (fitted on transitions < i, z-scored per transition), features from bins <= a:
  own+noise          : o_last, o_rev, o_lvl, o_f, o_zero, se, o_last x lvl, o_rev x lvl, o_last x se, o_rev x se
                       (as followups.py F2);
  own+noise+grn      : + importance-weighted regulators' last change and reversion move (bin-a edges), has_reg;
  own+noise+lvlnull  : + the same on an expression-level-matched random-regulator graph (grn_check.level_matched).
Output: grn_combo.json.
"""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge

import common
from grn_check import level_matched
from grn_signal import graph_mats, row_norm, zscore


def main():
    rng = np.random.default_rng(17)
    out = {}
    for name in common.SNAPS:
        d = common.load_node(name)
        N = d["m"].shape[0]
        snaps = common.SNAPS[name]
        trans = common.transitions(name)
        n = d["n"]
        u_all, i_all, ts_all, w_all = common.load_edges(name)
        Y = {T: [common.target(T, d["m"][:, a], d["m"][:, b]) for a, b in trans] for T in common.TARGETS}
        feats = {T: {} for T in common.TARGETS}
        for ti in range(1, len(trans)):
            a = trans[ti][0]
            hb = [s for s in snaps if s <= a]
            sel = ts_all == a
            u, i, w = u_all[sel], i_all[sel], w_all[sel].astype(float)
            m, s = d["m"][:, a], d["s"][:, a]
            lvl = np.log1p(1e4 * m / m.sum())
            se = np.log(np.sqrt(s ** 2 / n[a]) / (m + 1.0 / n[a]) + 1e-6)
            W = row_norm(graph_mats(u, i, w, N))
            Wn = row_norm(graph_mats(level_matched(u, lvl, rng), i, w, N))
            has = (np.asarray((graph_mats(u, i, w, N) > 0).sum(1)).ravel() > 0).astype(float)
            lo = max(0, len(hb) - 4)
            wt = n[hb[lo:]]
            pool4 = (d["m"][:, hb[lo:]] * wt).sum(1) / wt.sum()
            for T in common.TARGETS:
                yprev, orev = Y[T][ti - 1], common.target(T, m, pool4)
                zl, zr, zv, zs = zscore(yprev), zscore(orev), zscore(lvl), zscore(se)
                base = np.column_stack([yprev, orev, lvl, d["f"][:, a], (m == 0).astype(float), se,
                                        zl * zv, zr * zv, zl * zs, zr * zs])
                feats[T][ti] = {"own+noise": base,
                                "own+noise+grn": np.column_stack([base, W @ yprev, W @ orev, has]),
                                "own+noise+lvlnull": np.column_stack([base, Wn @ yprev, Wn @ orev, has])}
        res = {}
        for T in common.TARGETS:
            valid = sorted(feats[T])
            evals = valid[1:]
            mk = common.masks(N, len(evals), 5, seed=11)
            res[T] = {}
            for sname in ("own+noise", "own+noise+grn", "own+noise+lvlnull"):
                Xs = {t: np.column_stack([zscore(c) for c in feats[T][t][sname].T]) for t in valid}
                ys = {t: zscore(Y[T][t]) for t in valid}
                P = []
                for t in evals:
                    tr = [x for x in valid if x < t]
                    mdl = Ridge(alpha=10.0).fit(np.vstack([Xs[x] for x in tr]), np.concatenate([ys[x] for x in tr]))
                    P.append(mdl.predict(Xs[t]))
                sc = common.score_masked([Y[T][t] for t in evals], P, mk)
                res[T][sname] = {k: v for k, v in sc.items() if k != "per_transition"}
            print(name, T, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res[T].items()}, flush=True)
        out[name] = res
    with open(os.path.join(common.OUT, "grn_combo.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
