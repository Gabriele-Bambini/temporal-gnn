"""Judge: comparisons of the two sink prototypes with the external numbers (trend, DistBridge, NASS, their corrected
best), made robust to the mask convention.
The external numbers were computed on OTHER gene masks (DistBridge / trend / their models: their 12 x 5 saved test-gene
sets, tack/results/dileo_task2/rescore; NASS: 20 masks seed 20261022, design_best-signal/results.json), and the
pancreas l1 trend PCC alone moves from 0.177 (spec masks) to 0.159 (their masks) to 0.141 (NASS masks). So besides the
raw (unpaired) numbers, each model is reported as a gain over the trend scored on the SAME masks (difference in
differences), and against NASS transition by transition (i >= 2): d_t = (ours_t - trend_t) - (NASS_t - NASStrend_t).
Ours: judge reruns (p1 *_scores.npz; p2 sel_preds rescored with the judge's own metrics on the spec masks).
Usage: python external_compare.py  ->  external_compare.json
"""
import collections
import json
import os

import numpy as np
from scipy import stats

import indep_check as IC

HERE = os.path.dirname(os.path.abspath(__file__))
SA = os.path.dirname(HERE)
RES = os.path.join(os.path.dirname(SA), "dileo_task2", "rescore")


def per_trans(Y, P, mk):
    """(T, 3): mean over the 3 spec masks of PCC (NaN -> 0), tie-aware P@200 up, down; row 0 NaN."""
    T = len(Y)
    out = np.full((T, 3), np.nan)
    for i in range(1, T):
        v = []
        for s in range(3):
            r = IC.my_metrics(Y[i][mk[s][i]], P[i][mk[s][i]])
            v.append([0.0 if not np.isfinite(r[0]) else r[0], r[1], r[2]])
        out[i] = np.mean(v, 0)
    return out


def wil(d):
    nz = d[np.abs(d) > 1e-12]
    return round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None


def main():
    nass = json.load(open(os.path.join(SA, "design_best-signal", "results.json")))
    out = {}
    for name in ("pancreas-human", "gastrulation-human"):
        X, genes = IC.load_csv(name)
        m = np.maximum(X[:, :, 0], 0.0)
        cp = 1e4 * m / m.sum(1, keepdims=True)
        Yall = {"l1": np.log1p(cp[1:]) - np.log1p(cp[:-1]), "linear": m[1:] - m[:-1]}
        T = len(Yall["l1"])
        mk = IC.masks(X.shape[1], T)
        z1 = np.load(os.path.join(HERE, "p1", f"uot_sink_{name}_scores.npz"))
        z2 = np.load(os.path.join(HERE, "p2", f"sel_preds_{name}.npz"))
        o = {}
        for tg in ("l1", "linear"):
            Y = Yall[tg]
            tr = np.full_like(Y, np.nan)
            tr[1:] = Y[:-1]
            rows = {"trend": per_trans(Y, tr, mk)}
            for g in ("sink_norm_Kany", "sink_abs_Kany", "balanced_Kany", "persist_L3_Kany", "lin1_Kany", "kluot_abs_Kany"):
                S = z1[f"S_{tg}_{g}"]                                   # (T, 3, M): pearson_nan0, p_ta_up, p_ta_down
                rows[f"pop::{g}"] = np.nanmean(S[:, :, :3], 1)
            tr1 = np.nanmean(z1[f"S_trend_{tg}"][:, :, :3], 1)
            trend_check = float(np.nanmax(np.abs(tr1[1:] - rows["trend"][1:])))   # p1 (float32 metrics) vs judge (float64)
            print(f"   [check] p1 trend vs judge trend, max abs diff per transition: {trend_check:.2e}")
            for fam in ("uot", "uot_forced", "bal", "persist"):
                rows[f"gene::{fam}"] = per_trans(Y, z2[f"{fam}__{tg}"].astype(np.float64), mk)
            tab = {}
            for k, v in rows.items():
                a1, a2 = np.nanmean(v[1:], 0), np.nanmean(v[2:], 0)
                tab[k] = {"i>=1": np.round(a1, 4).tolist(), "i>=2": np.round(a2, 4).tolist(),
                          "gain_over_trend_same_masks_i>=1": np.round(a1 - np.nanmean(rows["trend"][1:], 0), 4).tolist()}
            # external, on their masks
            r = json.load(open(os.path.join(RES, f"rescore_{name}_{tg}.json")))
            acc = collections.defaultdict(list)
            for k, v in r["ours_on_their_masks"].items():
                for s, vv in v.items():
                    e = vv["i>=1"]
                    acc[k.split("@")[0]].append([e["pearson_nan0"], e["p_ta_up"], e["p_ta_down"]])
            ext = {}
            trt = np.mean(acc[f"trend_{tg}"], 0)
            db = np.mean(acc[f"distbridge_strict_{tg}"], 0)
            ext["trend@their_masks"] = np.round(trt, 4).tolist()
            ext["DistBridge@their_masks"] = np.round(db, 4).tolist()
            ext["DistBridge_gain_over_trend_their_masks"] = np.round(db - trt, 4).tolist()
            pre = {}
            for k, v in r["theirs"].items():
                if "/pre/" in k:
                    pre[k] = np.mean([[vv["i>=1"]["pearson_nan0"], vv["i>=1"]["p_ta_up"], vv["i>=1"]["p_ta_down"]]
                                      for vv in v.values()], 0)
            best = np.max(np.array(list(pre.values())), 0)
            ext["their_corrected_best_per_metric_max_over_strict_pre_models"] = np.round(best, 4).tolist()
            ext["their_best_gain_over_trend_their_masks"] = np.round(best - trt, 4).tolist()
            bestm = max(pre, key=lambda k: pre[k][0])
            ext["their_best_single_model_by_pcc"] = [bestm, np.round(pre[bestm], 4).tolist()]
            # NASS, transition-paired
            pt = nass[name]["targets"][tg]["per_transition_pcc_up_down"]
            NF, NT = np.array(pt["full"], float), np.array(pt["trend"], float)
            ext["NASS_full_i>=2@nass_masks"] = np.round(np.nanmean(NF[2:], 0), 4).tolist()
            ext["NASS_trend_i>=2@nass_masks"] = np.round(np.nanmean(NT[2:], 0), 4).tolist()
            ext["NASS_gain_over_trend_i>=2"] = np.round(np.nanmean(NF[2:] - NT[2:], 0), 4).tolist()
            pair = {}
            for k, v in rows.items():
                if k == "trend":
                    continue
                dd = (v[2:, 0] - rows["trend"][2:, 0]) - (NF[2:, 0] - NT[2:, 0])
                raw = v[2:, 0] - NF[2:, 0]
                pair[k] = {"DiD_pcc_mean": round(float(np.mean(dd)), 4), "wins": int((dd > 0).sum()),
                           "losses": int((dd < 0).sum()), "p": wil(dd),
                           "raw_pcc_minus_NASS_mean": round(float(np.mean(raw)), 4), "raw_p": wil(raw)}
            o[tg] = {"ours_spec_masks": tab, "external": ext, "vs_NASS_transition_paired_i>=2": pair,
                     "p1_trend_vs_judge_trend_max_abs_diff": trend_check}
            print("==", name, tg)
            for k, v in tab.items():
                print(f"   {k:24s} {v['i>=1']}  gain/trend {v['gain_over_trend_same_masks_i>=1']}")
            for k, v in ext.items():
                print(f"   {k:60s} {v}")
            for k, v in pair.items():
                print(f"   vsNASS {k:24s} {json.dumps(v)}")
        out[name] = o
    with open(os.path.join(HERE, "external_compare.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
