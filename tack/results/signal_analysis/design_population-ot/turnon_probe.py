"""Side probe for the orig observation head (strict): does a long-memory law for genes at zero in bin i help?

popot.py found that the turn-on AUC (genes with m_i = 0, label m_{i+1} > 0) of the 4-bin pooled law's nonzero
fraction is 0.79 / 0.72 (gastr / pancr), below the 0.86 / 0.81 of the plain 'fraction of past bins expressed'.
Here the orig prediction is dist_bridge.expected_target on the box4 law (= level + observation model, the popot
baseline), except that genes at zero in bin i take the law pooled over a longer past: box8, all past bins, or a
recency-weighted pool (weights n_b * 0.5^((i - b) / h), h in {2, 4}). Everything reads snapshots <= i only.
Scores as popot.py (crossgene common.masks, Pearson and P@200 with np.argsort). Output turnon_probe.json.
"""
import json
import os
import sys
import zlib

import numpy as np

import popot
from popot import common, dist_bridge


def pooled(T3, n, lo, i, h=None):
    b = np.arange(lo, i + 1)
    w = n[lo:i + 1] * (1.0 if h is None else 0.5 ** ((i - b) / h))
    return (T3[:, :, lo:i + 1] * w).sum(-1) / w.sum()


def main():
    out = {"protocol": __doc__}
    for name in ("gastrulation-human", "pancreas-human"):
        d = common.Data(name)
        mk = common.masks(d)
        Y = common.all_targets(d, "orig")
        T3 = np.stack([d.M, d.F, d.SD ** 2 + d.M ** 2])
        variants = {"box4 (baseline)": None, "zeros<-box8": ("box", 8), "zeros<-all_past": ("box", 10 ** 6),
                    "zeros<-recency_h2": ("rec", 2), "zeros<-recency_h4": ("rec", 4)}
        res = {}
        for vn, v in variants.items():
            P = np.zeros((d.T, d.N))
            aucs = []
            for i in range(d.T):
                base = pooled(T3, d.nc, max(0, i - 3), i)
                law = base.copy()
                z = d.M[:, i] <= 0
                if v is not None:
                    alt = pooled(T3, d.nc, max(0, i - v[1] + 1), i) if v[0] == "box" else pooled(T3, d.nc, 0, i, v[1])
                    law[:, z] = alt[:, z]
                law = popot.fixup(law)
                rng = np.random.default_rng([zlib.crc32(name.encode()), i, zlib.crc32(vn.encode())])
                P[i] = dist_bridge.expected_target(law.T.copy(), d.M[:, i], d.nc[i], rng)
                if i >= 1:
                    aucs.append(popot.auc(law[1][z], d.M[z, i + 1] > 0))
            sc = common.test_scores(Y, P, mk)
            res[vn] = {**popot.summ(sc), "turn_on_auc_i>=1": round(float(np.nanmean(aucs)), 4)}
            print(name, vn, res[vn]["i>=1"], res[vn]["all"], res[vn]["turn_on_auc_i>=1"], flush=True)
        out[name] = res
    json.dump(out, open(os.path.join(popot.HERE, "turnon_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
