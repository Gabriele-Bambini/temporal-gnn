"""C4: per-expression-class PCC of the l1 change y_i with the unweighted 4-bin reversion mean(L_{i-3..i}) - L_i and
with the lag-1 change y_{i-1}; pooled PCC of the 8-bin reversion; class shares of sum(y^2).

Transitions are 0-based snapshot transition indices (i: snapshot i -> i+1). Gastrulation 9..19, 27, 28 (raw bins
11->12 .. 21->22, 29->30, 30->31); pancreas 3..13. Classes from CP10k at bin i (current bin only, no future):
zero (m_i = 0), (0, 0.1], (0.1, 1], (1, 10], (10, 100], and the leftover > 100 (not in the question's list).
Primary pooling: all genes x listed transitions, raw values. Secondary: mean of per-transition PCCs, and the
3 CONTEXT test masks (genes of the mask of each transition), averaged over seeds.
"""
import json
import os

import numpy as np

import common as C

TR = {"gastrulation-human": list(range(9, 20)) + [27, 28], "pancreas-human": list(range(3, 14))}
CLS = ("zero", "le0.1", "0.1-1", "1-10", "10-100", "gt100")


def classes(m_i, cp_i):
    c = np.full(len(m_i), 5)
    c[cp_i <= 100] = 4
    c[cp_i <= 10] = 3
    c[cp_i <= 1] = 2
    c[cp_i <= 0.1] = 1
    c[m_i <= 0] = 0
    return c


def box_rev(L, i, k):
    lo = max(0, i - k + 1)
    return L[:, lo:i + 1].mean(1) - L[:, i], i - lo + 1


def main():
    res = {}
    for name in C.NAMES:
        d = C.Data(name)
        Y = d.Y("l1")
        mk = d.masks()
        rows = []
        for i in TR[name]:
            r4, w4 = box_rev(d.L, i, 4)
            r8, w8 = box_rev(d.L, i, 8)
            rows.append(dict(i=i, y=Y[i], lag=Y[i - 1], r4=r4, r8=r8, w8=w8, c=classes(d.m[:, i], d.CP[:, i])))
        y = np.concatenate([r["y"] for r in rows])
        lag = np.concatenate([r["lag"] for r in rows])
        r4 = np.concatenate([r["r4"] for r in rows])
        r8 = np.concatenate([r["r8"] for r in rows])
        c = np.concatenate([r["c"] for r in rows])
        ss = (y ** 2).sum()
        out = {"transitions": TR[name], "box8_window_len": [r["w8"] for r in rows], "classes": {}}
        for k, cn in enumerate(CLS):
            sel = c == k
            per_t_rev = [C.pcc(r["y"][r["c"] == k], r["r4"][r["c"] == k]) for r in rows if (r["c"] == k).sum() > 2]
            per_t_lag = [C.pcc(r["y"][r["c"] == k], r["lag"][r["c"] == k]) for r in rows if (r["c"] == k).sum() > 2]
            # masks: per seed, pool mask genes over listed transitions
            mrev, mlag = [], []
            for s in range(3):
                idx = [(r, mk[s, r["i"]]) for r in rows]
                yy = np.concatenate([r["y"][t][r["c"][t] == k] for r, t in idx])
                rr = np.concatenate([r["r4"][t][r["c"][t] == k] for r, t in idx])
                ll = np.concatenate([r["lag"][t][r["c"][t] == k] for r, t in idx])
                mrev.append(C.pcc(yy, rr))
                mlag.append(C.pcc(yy, ll))
            out["classes"][cn] = {
                "n_gene_transitions": int(sel.sum()),
                "pcc_rev4_pooled": C.pcc(y[sel], r4[sel]), "pcc_lag1_pooled": C.pcc(y[sel], lag[sel]),
                "pcc_rev4_mean_per_t": float(np.nanmean(per_t_rev)), "pcc_lag1_mean_per_t": float(np.nanmean(per_t_lag)),
                "pcc_rev4_masks": float(np.mean(mrev)), "pcc_lag1_masks": float(np.mean(mlag)),
                "ss_share": float((y[sel] ** 2).sum() / ss), "gene_share": float(sel.mean())}
        full8 = np.concatenate([np.full(len(r["y"]), r["w8"] == 8) for r in rows])
        out["pcc_rev8_pooled_all"] = C.pcc(y, r8)
        out["pcc_rev8_pooled_full_window_only"] = C.pcc(y[full8], r8[full8]) if full8.any() else None
        out["pcc_rev8_mean_per_t"] = float(np.nanmean([C.pcc(r["y"], r["r8"]) for r in rows]))
        out["pcc_rev4_pooled_all"] = C.pcc(y, r4)
        out["pcc_lag1_pooled_all"] = C.pcc(y, lag)
        res[name] = out
        print(f"== {name} transitions {TR[name]}")
        for cn, v in out["classes"].items():
            print(f"  {cn:7s} n={v['n_gene_transitions']:7d} rev4 {v['pcc_rev4_pooled']:+.3f} (per-t {v['pcc_rev4_mean_per_t']:+.3f},"
                  f" masks {v['pcc_rev4_masks']:+.3f})  lag1 {v['pcc_lag1_pooled']:+.3f} (per-t {v['pcc_lag1_mean_per_t']:+.3f},"
                  f" masks {v['pcc_lag1_masks']:+.3f})  SS share {v['ss_share']:.4f} gene share {v['gene_share']:.4f}")
        print(f"  rev8 pooled all {out['pcc_rev8_pooled_all']:+.3f} | full-window only {out['pcc_rev8_pooled_full_window_only']}"
              f" | mean per-t {out['pcc_rev8_mean_per_t']:+.3f} ; rev4 all {out['pcc_rev4_pooled_all']:+.3f} lag1 all {out['pcc_lag1_pooled_all']:+.3f}")
    json.dump(res, open(os.path.join(C.OUT, "c4_classes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
