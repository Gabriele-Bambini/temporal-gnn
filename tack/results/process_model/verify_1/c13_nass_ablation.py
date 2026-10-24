"""C13 part C: NASS (l1) ablations, run from NASS's own functions without its main() (nothing in design_best-signal is
written). Variants: full (reference), full-cg (SVD level features rev_svd3 / rev_svd8 removed), full + svdkf forecast
(factor-space LLT forecast of c13_factor.py as one extra feature; rows strict: transition i uses bins <= i), several
svdkf versions, and full-cg + svdkf (SVD levels replaced by the factor filter). GRN / OT off (graph = None,
ot = None), as in NASS's 'full' row. Same rolling-origin learner and grid as NASS.
Scoring: NASS's 20 masks (seed 20261022, comparable with the NASS rows of nass.log) and CONTEXT masks
(default_rng(1000 s + i), 3 seeds); mean PCC / P@200 at i >= 2 (and i >= 1 for CONTEXT masks); paired per-transition
deltas vs full (nass.paired: Wilcoxon over transitions i >= 2, NASS masks).
"""
import json
import os
import sys

import numpy as np

import common as C

sys.path.insert(0, C.DBS)
import dsdata as D  # noqa: E402
import nass as NS  # noqa: E402

EXTRA = ("svdkf_auto", "svdkf_lvl_3", "svdkf_chg_3", "svdkf_lvl_8", "pergene_LLT")


def score_ctx(Y, P, mk, lo):
    T = len(Y)
    rows = np.full((3, T, 3), np.nan)
    for s in range(3):
        for i in range(lo, T):
            t = mk[s, i]
            m = C.dileo_rescore.metrics(Y[i][t], P[i][t])
            rows[s, i] = (m["pearson"], m["p_ta_up"], m["p_ta_down"])
    return rows


def main():
    res = {}
    for name in C.NAMES:
        d = D.Data(name)
        Y = d.Y("l1")
        dc = C.Data(name)
        assert np.max(np.abs(Y - dc.Y("l1"))) < 1e-9, "target mismatch between loaders"
        F, cls = NS.build_features(d, "l1", Y, None, None)
        z = np.load(os.path.join(C.OUT, f"c13_forecasts_{name}.npz"))
        for k in EXTRA:
            F[k] = z[k].astype(np.float64)
            F[k][0] = 0.0
        fb = NS.fallback(d, "l1", Y, F)
        G_ = NS.GROUPS
        full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"]
        nocg = [k for k in full if k not in G_["cg"]]
        st = NS.Stacker(F, cls, Y, full + list(EXTRA))
        variants = {"full": [full], "full-cg": [nocg]}
        for k in EXTRA:
            variants[f"full+{k}"] = [full + [k]]
        variants["full-cg+svdkf_auto"] = [nocg + ["svdkf_auto"]]
        variants["full-cg+svdkf_lvl_3"] = [nocg + ["svdkf_lvl_3"]]
        preds = {v: NS.run_learned(st, s, Y, d, fb)[0] for v, s in variants.items()}
        nmk = D.masks(d.G, d.T)
        cmk = dc.masks()
        S = {v: D.score(Y, p, nmk) for v, p in preds.items()}
        out = {}
        for v, p in preds.items():
            t2 = D.summarize(S[v], list(range(2, d.T)))
            rc = score_ctx(Y, p, cmk, 1)
            c1 = np.nanmean(rc[:, 1:], 1).mean(0)
            c2 = np.nanmean(rc[:, 2:], 1).mean(0)
            out[v] = {"nass_masks_i>=2": [t2["pcc"], t2["p200_up"], t2["p200_down"], t2["pta_up"], t2["pta_down"]],
                      "ctx_i>=2": [round(float(x), 4) for x in c2], "ctx_i>=1": [round(float(x), 4) for x in c1]}
            if v != "full":
                out[v]["paired_vs_full"] = NS.paired(S[v], S["full"], list(range(2, d.T)))
            pv = out[v].get("paired_vs_full", {}).get("pcc", {})
            print(f"{name} {v:24s} NASS-masks i>=2 {out[v]['nass_masks_i>=2'][:3]} | ctx i>=2 {out[v]['ctx_i>=2']} "
                  f"i>=1 {out[v]['ctx_i>=1']} | dPCC vs full {pv.get('mean_delta')} ({pv.get('better')}/{pv.get('worse')}, p={pv.get('wilcoxon_p')})",
                  flush=True)
        res[name] = out
        json.dump(res, open(os.path.join(C.OUT, "c13_nass_ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
