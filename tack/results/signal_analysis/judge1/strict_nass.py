"""Judge 1 strictness test for NASS (design_best-signal), run on my copy in rerun_nass_p.

For a test transition k: truncate the data to bins 0..k+1 (nothing after k+1 exists) AND replace bin k+1 by a
gene-permuted copy of itself (so the label of transition k is garbage). Rebuild every feature and rerun the learner
(full feature set, full 20-config rolling-origin grid, and the fixed h = 0.5 variant). If NASS is strict, the
prediction for transition k must be identical to the one from the untouched full data.
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rerun_nass_p"))
import dsdata as D  # noqa: E402
import nass  # noqa: E402


def truncated(d, k, permute_next=True, seed=5):
    e = copy.copy(d)
    X = d.X[:k + 2].copy()
    if permute_next:
        X[k + 1] = X[k + 1][np.random.default_rng(seed).permutation(d.G)]
    e.X = X
    e.bins = d.bins[:k + 2]
    e.B, e.G = X.shape[0], X.shape[1]
    e.T = e.B - 1
    e.m = np.maximum(X[:, :, 0].T, 0)
    e.s = X[:, :, 2].T
    e.f = X[:, :, 3].T
    e.n = np.array([D.DB.n_cells(X[b]) for b in range(e.B)])
    e.lib = e.m.sum(0)
    e.CP = 1e4 * e.m / e.lib[None, :]
    e.L = np.log1p(e.CP)
    return e


def run(d, tg, k):
    Y = d.Y(tg)
    F, cls = nass.build_features(d, tg, Y, None, None)
    fb = nass.fallback(d, tg, Y, F)
    G_ = nass.GROUPS
    full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"] + (G_["obs"] if tg == "orig" else [])
    st = nass.Stacker(F, cls, Y, full)
    p_ro, ch = nass.run_learned(st, [full], Y, d, fb)
    fx = [nass.CONFIGS.index((0.5, 1e-3, 0))]
    p_fx, _ = nass.run_learned(st, [full], Y, d, fb, cfg_idx=fx)
    feats = {n: F[n][k] for n in full}
    return p_ro[k], p_fx[k], feats, ch[k]


def main(name, ks):
    d = D.Data(name)
    out = {}
    for k in ks:
        for tg in D.TARGETS:
            a_ro, a_fx, a_f, a_ch = run(d, tg, k)
            e = truncated(d, k)
            b_ro, b_fx, b_f, b_ch = run(e, tg, k)
            fd = {n: float(np.max(np.abs(a_f[n] - b_f[n]))) for n in a_f}
            res = {"max_abs_diff_pred_RO": float(np.max(np.abs(a_ro - b_ro))),
                   "max_abs_diff_pred_h0.5": float(np.max(np.abs(a_fx - b_fx))),
                   "max_abs_diff_any_feature": max(fd.values()),
                   "worst_feature": max(fd, key=fd.get), "chosen_full": a_ch, "chosen_trunc": b_ch,
                   "pred_scale_sd": float(np.std(a_ro))}
            out[f"{name}|{tg}|k={k}"] = res
            print(name, tg, k, json.dumps(res), flush=True)
    fo = os.path.join(HERE, "strict_nass.json")
    prev = json.load(open(fo)) if os.path.exists(fo) else {}
    prev.update(out)
    json.dump(prev, open(fo, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
