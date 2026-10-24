"""C3 sensitivity: analog bandwidth (0.5, 2 sd) and recency half-life h (0.5, 2) of the 6-feature ridge (i >= 2)."""
import json
import os

import numpy as np

import common as C
import kf
import c3_share as S

out = {}
for name in C.NAMES:
    d = C.load(name)
    Y, L, T = d.Y("l1"), d.L, d.T
    Rn, Rf = C.R_l1(d, False), C.R_l1(d, True)
    sh, Sv, _ = S.shares(Y, Rn)
    qc, _ = kf.q_class(d, L, Rf, None)
    Xc, _, _ = kf.local_level(L, Rf, qc)
    mk = C.masks(d.G, T)
    o = {}
    for h in (0.5, 1.0, 2.0):
        for bw in (0.5, 1.0, 2.0):
            S.H_REC, S.BW = h, bw
            for lab, z, an in (("recency", sh, False), ("analog(share)", sh, True), ("analog(log|S|)", np.log(np.abs(Sv)), True)):
                if lab == "recency" and bw != 1.0:
                    continue
                P = S.ridge_forecasts(d, L, Y, Xc, Rf, z, an)
                o[f"h={h} bw={bw} {lab}"] = C.summ(C.score(Y, P, mk), range(2, T))
    out[name] = o
    for k, v in o.items():
        print(C.SHORT[name], f"{k:32s}", C.fmt(v), flush=True)
json.dump(out, open(os.path.join(C.HERE, "c3_bw_sens.json"), "w"), indent=1)
