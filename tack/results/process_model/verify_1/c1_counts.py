"""C1: integer structure of the node-file summaries (raw UMI counts?). Over every raw bin and over snapshot bins."""
import json
import os

import numpy as np

import common as C

LISTED = {"gastrulation-human": [2, 3, 9, 182, 792, 929, 982, 616, 340, 217, 225, 248, 290, 304, 431, 440, 333, 247,
                                 246, 230, 275, 316, 355, 350, 264, 233, 228, 306, 231, 138, 46, 7],
          "pancreas-human": [111, 612, 361, 116, 50, 53, 68, 79, 191, 218, 132, 286, 698, 616, 105]}


def dist_int(x):
    return float(np.max(np.abs(x - np.round(x)))) if x.size else float("nan")


def block(X, bins):
    K_all, S_all, R_all, gap_all, nb = [], [], [], [], []
    per_bin = []
    for b in bins:
        xb = X[:, b]
        ok = xb[:, 0] > 0
        ratio = xb[ok, 4] / xb[ok, 0]
        n_b = float(np.median(ratio))
        K = xb[ok, 3] * n_b
        K_all.append(K)
        S_all.append(xb[ok, 4])
        R_all.append(ratio)
        gap_all.append(np.abs(ratio - n_b))
        nb.append(n_b)
        per_bin.append({"bin": b, "n_b": n_b, "n_expr": int(ok.sum()), "max_gap_ratio_nb": float(np.max(np.abs(ratio - n_b))),
                        "max_dist_K": dist_int(K), "max_dist_sum": dist_int(xb[ok, 4]), "max_dist_ratio": dist_int(ratio)})
    K, S, Rt, gap = map(np.concatenate, (K_all, S_all, R_all, gap_all))
    worst = max(per_bin, key=lambda r: r["max_dist_K"])
    return {"n_gene_bins_expressed": int(K.size), "max_dist_K": dist_int(K), "max_dist_sum": dist_int(S),
            "max_dist_ratio": dist_int(Rt), "max_gap_ratio_nb": float(np.max(gap)), "median_K": float(np.median(K)),
            "share_K_le_3": float(np.mean(K <= 3 + 1e-9)), "share_K_le_3_rounded": float(np.mean(np.round(K) <= 3)),
            "share_K_eq_1": float(np.mean(np.round(K) == 1)), "n_b": nb, "worst_K_bin": worst, "per_bin": per_bin}


def main():
    res = {}
    for name in C.NAMES:
        X = C.load_all_bins(name)
        G, B = X.shape[0], X.shape[1]
        # extra: medians half-integer? zero-mean genes with nonzero stats?
        med = X[:, :, 1]
        r = {"G": G, "B_all": B,
             "all_bins": block(X, list(range(B))),
             "snapshot_bins": block(X, C.SNAPS[name]),
             "max_dist_2xmedian": dist_int(2 * med),
             "n_zero_mean_with_nonzero_frac": int(((X[:, :, 0] <= 0) & (X[:, :, 3] > 0)).sum())}
        nb = np.array(r["all_bins"]["n_b"])
        r["n_b_minus_listed"] = (nb - np.array(LISTED[name])).tolist()
        r["n_b_match_listed"] = bool(np.allclose(nb, LISTED[name], atol=1e-6))
        res[name] = r
        for k in ("all_bins", "snapshot_bins"):
            q = r[k]
            print(f"{name} {k}: N={q['n_gene_bins_expressed']} maxdist K={q['max_dist_K']:.3g} sum={q['max_dist_sum']:.3g} "
                  f"sum/mean={q['max_dist_ratio']:.3g} max|sum/mean-n_b|={q['max_gap_ratio_nb']:.3g} "
                  f"median K={q['median_K']} share K<=3={q['share_K_le_3']:.4f} worstK bin={q['worst_K_bin']['bin']}")
        print(f"  n_b={np.round(nb, 4).tolist()} match listed={r['n_b_match_listed']} 2*median maxdist={r['max_dist_2xmedian']:.3g}")
    json.dump(res, open(os.path.join(C.OUT, "c1_counts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
