"""Judge 1: is the per-gene OT/SB increment (design_best-signal/_ot_<ds>.npz, made by ot_feature.py) spanned by
quantities any per-gene filter already has?  Per transition i >= 1, OLS over genes of inc_tg[i] on
  (a) the pooled-law mean trend over L = min(3, i) bins, in the target formula: tfun(box4 mean at i-L -> at i) / L
  (b) (a) + the nonzero-fraction trend pushed through the finite-cell zero probability (orig: dP0 from f alone)
  (c) (b) + last observed change + box-4 reversion move.
Reports median / mean R^2 over transitions (all genes and genes with a nonzero increment).
Also re-derives the OT increment for one transition on a gene subset with my own call of the dist_bridge primitives
(different Monte Carlo stream) and reports its correlation with the saved array.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rerun_nass_p"))
import dsdata as D  # noqa: E402
from dsdata import DB  # noqa: E402

SRC = os.path.join(os.path.dirname(HERE), "design_best-signal")


def r2(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (res ** 2).sum() / tot) if tot > 0 else np.nan


def main():
    out = {}
    for name in ("pancreas-human", "gastrulation-human"):
        d = D.Data(name)
        ot = np.load(os.path.join(SRC, f"_ot_{name}.npz"))
        trip = [DB.triplet(d.X[b]) for b in range(d.B)]
        box = DB.box_pool(trip, list(d.n), 4)
        res = {}
        for tg in ("orig", "l1"):
            Y = d.Y(tg)
            Z = d.CP if tg == "l1" else d.m
            rows = []
            for i in range(1, d.T):
                inc = ot[f"inc_{tg}"][i].astype(np.float64)
                L = min(3, i)
                mA, mB = box[i - L][:, 0], box[i][:, 0]
                if tg == "l1":
                    trend = D.tfun("l1", 1e4 * mA / mA.sum(), 1e4 * mB / mB.sum()) / L
                else:
                    trend = D.tfun("orig", mA, mB) / L
                fA, fB = np.clip(box[i - L][:, 1], 0, 1), np.clip(box[i][:, 1], 0, 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fP = np.where((fA > 0) & (fB > 0), np.minimum(1, fB * (fB / np.where(fA > 0, fA, 1)) ** (1 / L)), fB)
                nn = max(int(round(d.n[i])), 1)
                dp0 = (1 - fP) ** nn - (1 - fB) ** nn
                rev4 = D.tfun(tg, Z[:, i], (Z[:, max(0, i - 3):i + 1] * d.n[max(0, i - 3):i + 1]).sum(1) / d.n[max(0, i - 3):i + 1].sum())
                a = np.column_stack([trend])
                b_ = np.column_stack([trend, dp0])
                c = np.column_stack([trend, dp0, Y[i - 1], rev4])
                nz = np.abs(inc) > 1e-9
                rows.append([r2(inc, a), r2(inc, b_), r2(inc, c), r2(inc[nz], c[nz]) if nz.sum() > 10 else np.nan,
                             float(nz.mean()), float(np.std(inc)), float(np.std(Y[i]))])
            A = np.array(rows)
            res[tg] = {"median_R2_trend": round(float(np.nanmedian(A[:, 0])), 3),
                       "median_R2_trend+dP0": round(float(np.nanmedian(A[:, 1])), 3),
                       "median_R2_trend+dP0+lag1+rev4": round(float(np.nanmedian(A[:, 2])), 3),
                       "mean_R2_trend+dP0+lag1+rev4": round(float(np.nanmean(A[:, 2])), 3),
                       "median_R2_all_on_nonzero_inc_genes": round(float(np.nanmedian(A[:, 3])), 3),
                       "share_genes_nonzero_inc": round(float(np.mean(A[:, 4])), 3),
                       "median_sd_inc_over_sd_target": round(float(np.median(A[:, 5] / A[:, 6])), 4)}
            print(name, tg, res[tg], flush=True)
        # spot re-derivation of one transition on a gene subset
        i = 5
        L = min(3, i)
        sub = np.arange(3000)
        H, A0 = box[i][sub], box[i - L][sub]
        cp = DB.coupling(A0, H, L)
        Hot = DB.sb_step(H, cp, L, 1.0)
        m_now = d.m[sub, i]
        e0 = DB.expected_target(H, m_now, d.n[i], np.random.default_rng([555, 0]), draws=1024)
        e1 = DB.expected_target(Hot, m_now, d.n[i], np.random.default_rng([555, 1]), draws=1024)
        mine = (e1 - e0).astype(np.float64)
        saved = ot["inc_orig"][i][sub].astype(np.float64)
        ok = (np.abs(mine) + np.abs(saved)) > 0
        res["spot_rederive_orig_i5_first3000"] = {"corr": round(float(np.corrcoef(mine[ok], saved[ok])[0, 1]), 4),
                                                  "max_abs_diff": round(float(np.abs(mine - saved).max()), 4),
                                                  "sinkhorn_err": float(cp["err"]), "n_coupled": int(len(cp["idx"]))}
        print(name, res["spot_rederive_orig_i5_first3000"], flush=True)
        out[name] = res
    json.dump(out, open(os.path.join(HERE, "ot_span.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
