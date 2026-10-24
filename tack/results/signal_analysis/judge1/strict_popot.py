"""Judge 1 strictness test for PopOT (design_population-ot), run on my copy in rerun_popot.

Transition k: truncate to bins 0..k+1 and gene-permute bin k+1; compare the forecast law and the three target
predictions of EVERY config (120) at transition k with the full-data values.
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "crossgene"))
sys.path.insert(0, os.path.join(HERE, "rerun_popot"))
import common  # noqa: E402
import popot  # noqa: E402


def truncated(d, k, seed=5):
    e = copy.copy(d)
    sl = slice(0, k + 2)
    p = np.random.default_rng(seed).permutation(d.N)
    e.M, e.SD, e.F = d.M[:, sl].copy(), d.SD[:, sl].copy(), d.F[:, sl].copy()
    for A in (e.M, e.SD, e.F):
        A[:, k + 1] = A[p, k + 1]
    e.nc = d.nc[sl].copy()          # per-bin statistic; invariant to the permutation
    e.bins = d.bins[sl]
    e.lib = e.M.sum(0)
    e.CP = 1e4 * e.M / e.lib
    e.L = np.log1p(e.CP)
    e.N, e.S = e.M.shape
    e.T = e.S - 1
    return e


def preds(d, k):
    mdl = popot.PopOT(d)
    cfgs = popot.build_configs()
    out = {t: [] for t in popot.TARGETS}
    for cfg in cfgs:
        Th = mdl.law(k, cfg)
        for t in popot.TARGETS:
            out[t].append(mdl.target_pred(k, Th, t, popot.label(cfg)))
    return {t: np.array(v) for t, v in out.items()}, cfgs


def main(name, ks):
    d = common.Data(name)
    res = {}
    for k in ks:
        a, cfgs = preds(d, k)
        b, _ = preds(truncated(d, k), k)
        ot = np.array([c[2] in popot.OT for c in cfgs])
        r = {t: {"maxdiff_all": float(np.abs(a[t] - b[t]).max()), "maxdiff_OT_cfgs": float(np.abs(a[t] - b[t])[ot].max()),
                 "pred_sd": float(a[t].std())} for t in popot.TARGETS}
        res[f"{name}|k={k}"] = r
        print(name, k, json.dumps(r), flush=True)
    fo = os.path.join(HERE, "strict_popot.json")
    prev = json.load(open(fo)) if os.path.exists(fo) else {}
    prev.update(res)
    json.dump(prev, open(fo, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
