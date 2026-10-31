"""DESCRIPTIVE metric check (uses the true target stage; not a forecast, never used for selection).

Question: does a metric reward a forecast that is merely CONTRACTED toward its centroid? A proper distribution
distance compared between two random halves A, B of the SAME true stage must be smallest at a = 0 (no contraction).
For each origin 2-5 (origin-s PCA space, target = true stage s+1, halves exactly as the diag_metric floor: permutation
seed 4, first 1500 cells of each half) we contract half A by a in {0, .1, .2, .3, .4, .5} and score it against B:
  sd_fixed  diag_metric's fixed-scale debiased Sinkhorn quantity: [<P_xy,C> - <P_xx,C>/2 - <P_yy,C>/2] / S_s
            (transport COST of the entropic plans only, eps = 0.05 S_s)
  sd_full   the textbook Sinkhorn divergence (Feydy et al. 2019): the same with the full entropic value
            OT_eps = <P,C> + eps KL(P | a x b) in all three terms, same plans, same eps, same / S_s
  energy, mmd as in diag_metric.
Also, the same check with half A first shifted by the true mean difference to the previous stage (a displaced
forecast), to see whether contraction pays off when the location is wrong.
Output: metric_check.json, printed table.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attempt_forecast as A  # noqa: E402

F = A.F


def ot_full(A_, B_, wa, wb, eps, sym=False):
    C = F.sqdist(A_, B_)
    P = F.sinkhorn_sym(wa, C, eps) if sym else F.sinkhorn(wa, wb, C, eps)
    ab = wa[:, None] * wb[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(P > 0, P * np.log(P / ab), 0.0).sum() - P.sum() + ab.sum()
    return float((P * C).sum()), float((P * C).sum() + eps * kl)


def sd_pair(X, Y, S):
    wx, wy = F.unif(len(X)), F.unif(len(Y))
    eps = 0.05 * S
    xy = ot_full(X, Y, wx, wy, eps)
    xx = ot_full(X, X, wx, wx, eps, sym=True)
    yy = ot_full(Y, Y, wy, wy, eps, sym=True)
    return ((xy[0] - xx[0] / 2 - yy[0] / 2) / S, (xy[1] - xx[1] / 2 - yy[1] / 2) / S)


def main():
    out = {}
    for s in (2, 3, 4, 5):
        d = F.load(A.DATA, s)
        S = F.past_scale(d["Z_past"])
        Zt = d["Z_fut"].astype(np.float64)
        half = np.random.default_rng(4).permutation(len(Zt))
        ZA, ZB = Zt[half[: len(Zt) // 2]][: F.NSUB], Zt[half[len(Zt) // 2:]][: F.NSUB]
        st = d["stage_past"]
        shift = Zt.mean(0) - d["Z_past"][st == s].astype(np.float64).mean(0)
        rows = []
        for disp in (0.0, 1.0):
            for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
                X = ZA - disp * shift
                c = X.mean(0)
                X = c + (1 - a) * (X - c)
                sdf, sdu = sd_pair(X, ZB, S)
                wb = F.unif(len(ZB))
                yy_mmd = float(wb @ F.rbf(ZB, ZB, S) @ wb)
                rows.append({"displaced": disp, "a": a, "sd_fixed": sdf, "sd_full": sdu, "energy": F.energy(X, ZB),
                             "mmd": F.mmd2(X, ZB, F.unif(len(X)), wb, S, yy_mmd)})
                print(s, rows[-1], flush=True)
        out[str(s)] = {"S": S, "n_A": len(ZA), "n_B": len(ZB), "shift_norm": float(np.linalg.norm(shift)),
                       "rows": rows}
    with open(os.path.join(HERE, "metric_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
