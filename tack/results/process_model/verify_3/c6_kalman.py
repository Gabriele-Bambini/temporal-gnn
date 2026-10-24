"""C6: standalone per-gene local-level Kalman forecaster on l1 with known R (strict), vs NASS and a per-gene ML
local-linear-trend filter, on the 1000 s + i masks, i >= 2, all transitions and the quiet ones (share < 0.2).

LL: R = common.R_l1 floored at its m = 0 value (1/n^2 on the raw scale); per-gene Q_b from past changes k < b
(kf.q_pergene) floored at 0.05 R_b; forecast y_t = x_{t|t} - L_t. 'Lag-1 noise correction' is ambiguous, so three
Q estimators are run: base (known-R moment: mean(y^2 - R_k - R_{k+1})), lag1 (base + 2 x noise-corrected lag-1
autocovariance), ma1 (mean y^2 + 2 mean y_k y_{k+1}, no R). Side runs: R without floor.
LLT: kf.llt_ml (grid of level / slope variances, per gene max innovation log-lik over bins <= t); forecasts
slope_{t|t} and level + slope - L_t.
NASS: predictions from nass_rerun.py (nass_l1_rows_<name>.npy, same masks).
Quiet transitions: share_i < 0.2 with the CONTEXT-calibrated share (R without floor, c3_share.py); the floored-R
share set is reported too.
"""
import json
import os

import numpy as np
from scipy import stats

import common as C
import kf


def shares(Y, R):
    v = np.array([y.var() for y in Y])
    return 1 - np.array([(R[:, i] + R[:, i + 1]).mean() for i in range(len(Y))]) / v


def main():
    out = {}
    for name in C.NAMES:
        d = C.load(name)
        Y, L, T = d.Y("l1"), d.L, d.T
        mk = C.masks(d.G, T)
        Rf, Rn = C.R_l1(d, True), C.R_l1(d, False)
        sh_n, sh_f = shares(Y, Rn), shares(Y, Rf)
        preds, floor_frac = {}, {}
        for rk, R in (("Rfloor", Rf), ("Rnofloor", Rn)):
            for mode in ("base", "lag1", "ma1"):
                if rk == "Rnofloor" and mode != "base":
                    continue
                q, atf = kf.q_pergene(L, R, mode)
                X, _, Kg = kf.local_level(L, R, q)
                P = list(kf.ll_forecast(L, X))
                P[0] = P[1] = None
                key = f"LL pergeneQ[{mode}] {rk}"
                preds[key] = P
                floor_frac[key] = {"mean_b>=2": round(float(np.mean([atf[b] for b in range(2, d.B)])), 3),
                                   "b=2": round(atf[2], 3), "b=last": round(atf[d.B - 1], 3)}
        slope, onestep, runs, choice = kf.llt_ml(L, Rf)
        preds["LLT ML slope_t|t"] = [None, None] + list(slope[2:])
        preds["LLT ML level+slope-L_t"] = [None, None] + list(onestep[2:])
        # sensitivity: the lower edge 1e-4 of the level grid binds for most genes -> extended grid down to 0
        slope_x, onestep_x, runs_x, choice_x = kf.llt_ml(L, Rf, ql_grid=(0.0, 1e-6, 1e-5) + kf.QL_GRID)
        preds["LLT ML(ext grid) slope_t|t"] = [None, None] + list(slope_x[2:])
        preds["LLT ML(ext grid) level+slope-L_t"] = [None, None] + list(onestep_x[2:])
        preds["trend"] = [None, None] + [Y[t - 1] for t in range(2, T)]
        rows = {k: C.score(Y, p, mk) for k, p in preds.items()}
        rows["NASS full"] = np.load(os.path.join(C.HERE, f"nass_l1_rows_{name}.npy"))
        idx = list(range(2, T))
        quiet = [i for i in idx if sh_n[i] < 0.2]
        quiet_f = [i for i in idx if sh_f[i] < 0.2]
        loud = [i for i in idx if sh_n[i] >= 0.2]
        o = {"quiet_transitions(share<0.2, R_nofloor, i>=2)": quiet, "quiet_transitions(share<0.2, R_floor, i>=2)": quiet_f,
             "Q_at_floor_fraction": floor_frac,
             "LLT_grid_choice_last_t": dict(zip([f"{a:g},{b:g}" for a, b in runs], choice[T - 1])),
             "LLT_ext_grid_choice_last_t": dict(zip([f"{a:g},{b:g}" for a, b in runs_x], choice_x[T - 1]))}
        o["scores"] = {}
        for k, r in rows.items():
            o["scores"][k] = {"i>=2": C.summ(r, idx), "quiet": C.summ(r, quiet), "quiet_Rfloor": C.summ(r, quiet_f),
                              "share>=0.2": C.summ(r, loud)}
        per = {k: np.nanmean(r[:, :, 0], 0) for k, r in rows.items()}
        o["per_transition_pcc"] = {k: [None if not np.isfinite(x) else round(float(x), 3) for x in v] for k, v in per.items()}
        pr = {}
        for a in ("LL pergeneQ[base] Rfloor", "LL pergeneQ[lag1] Rfloor", "LLT ML level+slope-L_t"):
            for b in ("NASS full", "LLT ML level+slope-L_t", "LL pergeneQ[base] Rfloor"):
                if a == b:
                    continue
                dl = per[a][idx] - per[b][idx]
                pr[f"{a} - {b}"] = {"mean": round(float(dl.mean()), 4), "better": int((dl > 1e-9).sum()),
                                    "worse": int((dl < -1e-9).sum()), "wilcoxon_p": round(float(stats.wilcoxon(dl).pvalue), 4)}
        o["paired_pcc_i>=2"] = pr
        out[name] = o
        print("==", name, "quiet", quiet, "quiet(Rfloor)", quiet_f)
        print("  Q at floor", floor_frac)
        for k, v in o["scores"].items():
            print(f"  {k:30s} i>=2 {C.fmt(v['i>=2'])} | quiet {C.fmt(v['quiet'])} | quiet(Rfloor) {C.fmt(v['quiet_Rfloor'])}"
                  f" | share>=0.2 {C.fmt(v['share>=0.2'])}")
        for k, v in pr.items():
            print("  paired", k, v)
    json.dump(out, open(os.path.join(C.HERE, "c6_kalman.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
