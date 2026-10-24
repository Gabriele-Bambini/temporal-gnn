"""C10: absolute noise-corrected l1 signal variance per transition vs cell counts.

S_i = var_g(y_i) - mean_g(R_i + R_{i+1}), y = l1 change over all G genes. C2 is not visible here, so R is defined
here: delta-method noise variance of L_b = log1p(CP10k_b) of a bin mean, R_b = kap_b^2 s^2 / (n_b - 1) / (1 + CP_b)^2
(population sd s, kap = 1e4 / library), 'plain'; and NASS's floored version, s^2 -> max(s^2, running cell-weighted
mean count) + 1/n, 'floor' (gives zero-count genes a nonzero noise). Correlations over transitions: Pearson and
Spearman of S_i with 0.5 (log n_i + log n_{i+1}) and with (log n_{i+1} - log n_i)^2; all transitions, excluding
transitions touching a bin with n < 10, and dropping the single largest (dlog n)^2 point.
"""
import json
import os

import numpy as np
from scipy import stats

import common as C


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3:
        return None, None
    return round(float(stats.pearsonr(a, b)[0]), 4), round(float(stats.spearmanr(a, b).correlation), 4)


def main():
    res = {}
    for name in C.NAMES:
        d = C.Data(name)
        Y = d.Y("l1")
        T = d.T
        ln = np.log(d.n)
        gm = 0.5 * (ln[:-1] + ln[1:])
        dl2 = (ln[1:] - ln[:-1]) ** 2
        small = np.array([min(d.n[i], d.n[i + 1]) < 10 for i in range(T)])
        out = {"n": d.n.round().tolist(), "excluded_transitions": np.where(small)[0].tolist(),
               "geo_logn": gm.round(4).tolist(), "dlogn2": dl2.round(4).tolist()}
        for kind in ("plain", "floor"):
            R = d.R_L(floor=kind == "floor")
            vy = Y.var(1)
            r0, r1 = R[:, :-1].mean(0), R[:, 1:].mean(0)
            S = vy - r0 - r1
            keep = ~small
            top = np.argmax(dl2)
            keep_top = np.ones(T, bool)
            keep_top[top] = False
            o = {"S": S.round(6).tolist(), "var_y": vy.round(6).tolist(), "noise_t": r0.round(6).tolist(),
                 "noise_t1": r1.round(6).tolist(),
                 "budget_mean_share(noise_t, noise_t1, signal)": [round(float(np.mean(r0 / vy)), 4),
                                                                   round(float(np.mean(r1 / vy)), 4),
                                                                   round(float(np.mean(S / vy)), 4)],
                 "budget_pooled_share": [round(float(r0.sum() / vy.sum()), 4), round(float(r1.sum() / vy.sum()), 4),
                                         round(float(S.sum() / vy.sum()), 4)],
                 "corr_geo_all": corr(S, gm), "corr_dl2_all": corr(S, dl2),
                 "corr_geo_excl": corr(S[keep], gm[keep]), "corr_dl2_excl": corr(S[keep], dl2[keep]),
                 "corr_dl2_drop_top": corr(S[keep_top], dl2[keep_top]), "corr_geo_drop_top": corr(S[keep_top], gm[keep_top]),
                 "top_dl2_transition": int(top),
                 "corr_geo_i>=1": corr(S[1:], gm[1:]), "corr_dl2_i>=1": corr(S[1:], dl2[1:]),
                 "S_excluded": {int(i): round(float(S[i]), 6) for i in np.where(small)[0]},
                 "S_mean_all": round(float(S.mean()), 6), "S_mean_excl": round(float(S[keep].mean()), 6),
                 "S_median_all": round(float(np.median(S)), 6), "n_negative_S": int((S < 0).sum())}
            out[kind] = o
            print(f"== {name} R={kind}: excluded {out['excluded_transitions']}")
            print("  S_i:", np.round(S, 5).tolist())
            print(f"  budget mean shares {o['budget_mean_share(noise_t, noise_t1, signal)']} pooled {o['budget_pooled_share']}")
            print(f"  corr(S, geo log n) all {o['corr_geo_all']} excl {o['corr_geo_excl']} drop-top {o['corr_geo_drop_top']} i>=1 {o['corr_geo_i>=1']}")
            print(f"  corr(S, dlogn^2)  all {o['corr_dl2_all']} excl {o['corr_dl2_excl']} drop-top(t={top}) {o['corr_dl2_drop_top']} i>=1 {o['corr_dl2_i>=1']}")
            print(f"  S excluded {o['S_excluded']}  mean all {o['S_mean_all']} excl {o['S_mean_excl']} neg {o['n_negative_S']}")
        res[name] = out
    json.dump(res, open(os.path.join(C.OUT, "c10_signal_vs_n.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
