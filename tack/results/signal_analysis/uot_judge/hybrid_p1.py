"""Judge: is the selected 'sink OT' of the population prototype just a closed-form hybrid trend?

At lambda = 0.1 with normalised cost (mean off-diagonal C = 1) turnover costs 2 lambda = 0.2, below almost every
pairwise cost, so the eps -> 0 sink_norm plan of one pair a -> b keeps min(a_k, b_k) in place, kills a_k - b_k and
gives birth to b_k - a_k. With R = 1 the push-forward is then the closed form
    HYB:  pi_hat_k  propto  pi_i,k min(1, pi_i,k / pi_{i-1},k) + max(pi_i,k - pi_{i-1},k, 0)
(multiplicative decline of shrinking states, additive growth of growing ones; no transport at all).
This script, on the judge copy p1/uot_sink.py (strict fits at every transition):
 1. TV(pi_hat of sink_norm lam=0.1 R=1, HYB) per K and eps, and TV of the configuration that the strict rolling-origin
    selection of the group sink_norm|Kany actually picked (read from the judge rerun JSON) against HYB at the same K.
 2. Scores HYB as a family (K in {4, 8, 16} x L in {1, 3}, the same map back, the same strict rolling-origin selection,
    the same masks) and pairs it with sink_norm|Kany, balanced|Kany and persist|L3|Kany from the judge rerun.
Usage: python hybrid_p1.py <dataset>   ->  hybrid_p1_<dataset>.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "p1")
sys.path.insert(0, P1)
import uot_sink as U  # noqa: E402
import dist_bridge  # noqa: E402


def hyb(pi, i):
    a, b = pi[i - 1], pi[i]
    h = b * np.minimum(1.0, b / np.maximum(a, 1e-300)) + np.maximum(b - a, 0.0)
    return h / h.sum()


def parse(lbl):
    parts = lbl.split("|")
    fam, K, L = parts[0], int(parts[1][1:]), int(parts[2][1:])
    hp = {}
    for p in parts[3:]:
        for key in ("R", "eps", "lam", "tau"):
            if p.startswith(key):
                v = float(p[len(key):])
                hp[key] = int(v) if key == "R" else v
                break
    return fam, K, L, hp


def main():
    name = sys.argv[1]
    d = U.Data(name)
    T = d.T
    mk = U.make_masks(d.G, T, d.n_test)
    q = 200.0 / d.n_test
    tg_list = ("l1", "linear")
    Y = {t: d.Y(t) for t in tg_list}
    rerun = json.load(open(os.path.join(P1, f"uot_sink_{name}.json")))
    sc = np.load(os.path.join(P1, f"uot_sink_{name}_scores.npz"))
    cfgs = [(K, L) for K in U.KS for L in U.LS]
    J = {t: np.full((len(cfgs), T), np.nan) for t in tg_list}
    S = {t: np.full((T, 3, len(U.MKEYS)), np.nan) for t in tg_list}
    tv_rows, tv_sel = [], {t: [] for t in tg_list}
    lam_sel = {t: {} for t in tg_list}
    for i in range(1, T):
        fits = {K: U.Fit(d, i, min(K, i)) for K in U.KS}
        preds = {t: [] for t in tg_list}
        for r, (K, L) in enumerate(cfgs):
            fit = fits[K]
            ph = hyb(fit.pi, i)
            mh, fh = U.map_back(fit, ph, L)
            for t in tg_list:
                p = U.target_pred(d, i, mh, fh, t)
                preds[t].append(p)
                J[t][r, i] = dist_bridge.criterion(Y[t][i], p, q)
        for K in U.KS:
            fit = fits[K]
            ph = hyb(fit.pi, i)
            for eps in U.EPSS:
                ps, _ = U.pooled_forecast(fit, "sink_norm", {"R": 1, "eps": eps, "lam": 0.1})
                tv_rows.append({"i": i, "K": K, "K_eff": fit.K, "eps": eps, "tv": float(0.5 * np.abs(ps - ph).sum())})
        lo = max(1, i - U.WINDOW)
        for t in tg_list:
            if i - lo < 1:
                p = Y[t][i - 1]
            else:
                s_ = np.array([np.nanmean(J[t][r, lo:i]) for r in range(len(cfgs))])
                r = int(np.where(s_ >= s_.max() - U.TIE)[0][0])
                p = preds[t][r]
            S[t][i] = U.test_scores(Y[t][i], p, mk[:, i])
            lbl = rerun["targets"][t]["groups"]["sink_norm|Kany"]["selected"][i]
            if lbl and not lbl.startswith("fallback"):
                fam, K, L, hp = parse(lbl)
                lam_sel[t][str(hp.get("lam"))] = lam_sel[t].get(str(hp.get("lam")), 0) + 1
                fit = fits[K]
                ps, _ = U.pooled_forecast(fit, fam, hp)
                ph = hyb(fit.pi, i)
                mh_s, fh_s = U.map_back(fit, ps, L)
                mh_h, fh_h = U.map_back(fit, ph, L)
                ys, yh = U.target_pred(d, i, mh_s, fh_s, t), U.target_pred(d, i, mh_h, fh_h, t)
                tv_sel[t].append({"i": i, "label": lbl, "tv_vs_hyb_same_K": round(float(0.5 * np.abs(ps - ph).sum()), 4),
                                  "pred_corr_vs_hyb_same_K_L": round(float(np.corrcoef(ys, yh)[0, 1]), 4)})
        print(name, "transition", i, "done", flush=True)
    out = {"dataset": name, "tv_sinknorm_lam0.1_R1_vs_hyb": {}, "selected_sink_norm_vs_hyb": tv_sel,
           "selected_sink_norm_lambda_counts": lam_sel, "scores": {}, "paired": {}}
    for K in U.KS:
        for eps in U.EPSS:
            v = [r["tv"] for r in tv_rows if r["K"] == K and r["eps"] == eps and r["i"] >= 2]
            out["tv_sinknorm_lam0.1_R1_vs_hyb"][f"K{K}_eps{eps:g}"] = {"median": round(float(np.median(v)), 4),
                                                                      "max": round(float(np.max(v)), 4)}
    for t in tg_list:
        out["scores"][t] = {"HYB_closed_form|Kany(strict sel)": {"i>=1": U.summ(S[t], 1), "i>=2": U.summ(S[t], 2)}}
        for g in ("sink_norm_Kany", "balanced_Kany", "persist_L3_Kany", "lin1_Kany", "sink_abs_Kany"):
            out["scores"][t][g] = {"i>=1": U.summ(sc[f"S_{t}_{g}"], 1)}
            out["paired"][f"{t}: HYB - {g}"] = U.paired(S[t], sc[f"S_{t}_{g}"])
        print(name, t, json.dumps(out["scores"][t]), flush=True)
        for k, v in out["paired"].items():
            if k.startswith(t):
                print("  ", k, json.dumps(v), flush=True)
    np.savez_compressed(os.path.join(HERE, f"hybrid_p1_{name}_scores.npz"), **{f"S_{t}": S[t] for t in tg_list})
    with open(os.path.join(HERE, f"hybrid_p1_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
