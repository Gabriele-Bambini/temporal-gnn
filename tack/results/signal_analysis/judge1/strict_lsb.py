"""Judge 1 strictness test for LatentSB (design_latent-sb), run on my copy in rerun_lsb.

Transition k: truncate to bins 0..k+1 and gene-permute bin k+1. Compare, for EVERY latent config, the l1 and linear
heads at origin k, and the orig head for every ORIG_CFG, with the full-data values. Any difference = a bin > k
entered. Also measures the effect of self.nmed (median cell count over ALL bins, used as a weight cap in factor()).
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rerun_lsb"))
import lsb_common as C  # noqa: E402
import run_lsb as R  # noqa: E402
from lsb_model import LatentSB  # noqa: E402


def truncated(d, k, seed=5):
    e = copy.copy(d)
    sl = slice(0, k + 2)
    e.M, e.SD, e.F, e.SUM = d.M[:, sl].copy(), d.SD[:, sl].copy(), d.F[:, sl].copy(), d.SUM[:, sl].copy()
    p = np.random.default_rng(seed).permutation(d.N)
    for A in (e.M, e.SD, e.F, e.SUM):
        A[:, k + 1] = A[p, k + 1]
    e.bins = d.bins[sl]
    ok = e.M > 0
    e.nc = np.array([np.round(np.median(e.SUM[ok[:, s], s] / e.M[ok[:, s], s])) for s in range(e.M.shape[1])])
    e.N, e.S = e.M.shape
    e.T = e.S - 1
    e.lib = e.M.sum(0)
    e.CP = 1e4 * e.M / e.lib
    e.X = np.log1p(e.CP)
    e.trip = [np.stack([e.M[:, s], e.F[:, s], e.SD[:, s] ** 2 + e.M[:, s] ** 2], 1) for s in range(e.S)]
    return e


def heads(d, k, nmed=None):
    mod = LatentSB(d)
    if nmed is not None:
        mod.nmed = nmed
    l1, lin = [], []
    for cfg in R.CFGS:
        x = mod.levels(k, cfg)
        l1.append(mod.head_l1(k, x[1]))
        lin.append(mod.head_linear(k, x[1], 0.5))
    cl = R.CFGS.index((4, "cur", "ar", None, 4.0, "mom"))
    xn, xx, _ = mod.levels(k, R.CFGS[cl])
    org = [mod.head_orig(k, xx if s == "next" else xn, kp, a, [9, k, oi]) for oi, (kp, a, s) in enumerate(R.ORIG_CFGS)]
    return np.array(l1), np.array(lin), np.array(org), mod.nmed


def main(name, ks):
    d = C.Data(name)
    out = {}
    for k in ks:
        a1, al, ao, nmed_full = heads(d, k)
        e = truncated(d, k)
        b1, bl, bo, nmed_tr = heads(e, k)
        # same truncated data, but nmed forced to the full-data value: isolates the nmed channel
        c1, cl_, co, _ = heads(e, k, nmed=nmed_full)
        kpos = np.array([c[0] > 0 for c in R.CFGS])
        res = {"nmed_full": nmed_full, "nmed_trunc": nmed_tr,
               "l1_maxdiff_k0": float(np.abs(a1 - b1)[~kpos].max()), "l1_maxdiff_kpos": float(np.abs(a1 - b1)[kpos].max()),
               "linear_maxdiff_k0": float(np.abs(al - bl)[~kpos].max()),
               "linear_maxdiff_kpos": float(np.abs(al - bl)[kpos].max()),
               "orig_maxdiff": float(np.abs(ao - bo).max()),
               "l1_maxdiff_kpos_with_full_nmed": float(np.abs(a1 - c1)[kpos].max()),
               "orig_maxdiff_with_full_nmed": float(np.abs(ao - co).max()),
               "l1_kpos_pred_corr_full_vs_trunc_min": float(min(np.corrcoef(a1[c], b1[c])[0, 1]
                                                                for c in np.where(kpos)[0])),
               "l1_pred_sd": float(np.std(a1))}
        out[f"{name}|k={k}"] = res
        print(name, k, json.dumps(res), flush=True)
    fo = os.path.join(HERE, "strict_lsb.json")
    prev = json.load(open(fo)) if os.path.exists(fo) else {}
    prev.update(out)
    json.dump(prev, open(fo, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
