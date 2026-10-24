"""(3) Gene-level heterogeneity: which gene groups carry the target variance, the true top / bottom lists and
the Pearson / P@200 of strict reference forecasters, under each target (orig, l1, linear).

Partitions of the genes at transition a -> b (all strict, from bin a, except 'crossing', which is descriptive):
  expr     : zero_a (m_a = 0), Q1..Q5 = quintiles of log1p(CP10k_a) among expressed genes;
  top1     : top 1 % by m_a vs rest (the very highly expressed genes);
  frac     : f_a in {0}, (0, .02], (.02, .1], (.1, .3], (.3, 1];
  outdeg   : out-degree in the bin-a GRN: nonreg (0), reg_low (<= median of regulators), reg_high, hub50 (top 50);
  indeg    : in-degree in the bin-a GRN: 0, tertiles among targets;
  crossing : zz (0 -> 0), on (0 -> +), off (+ -> 0), pp (+ -> +)  [uses bin b: description only].
Per partition and group, averaged over transitions (i >= 1): share of genes; share of sum (y - mean)^2 (where the
target's variance lives); share of the true top-k up / down lists (all genes, k = 800 = 200 / 0.25, tie-aware
inclusion weights); for each reference forecaster, the group's share of the Pearson numerator
sum (y - ybar)(p - pbar) / den (the group shares sum to the all-gene Pearson) and of the tie-aware top-k hits;
and the Pearson of the forecaster with the group removed.
Reference forecasters (strict): orig -> ObsLevel(4) (the core of DistBridge's win), trend; l1 / linear -> trend,
reversion to the cell-weighted mean of the last 4 bins. Hub / expression association: Spearman of out-degree and
in-degree with log1p(CP10k).
Output: heterogeneity.json in this folder.
"""
import json
import os

import numpy as np
from scipy import stats

import common
import task2_forecasters as tf

KALL = 800


def hist_arrays(d, bins):
    return [np.stack([d["m"][:, b], d["med"][:, b], d["s"][:, b], d["f"][:, b], d["S"][:, b]], 1) for b in bins]


def cp10k(m):
    return 1e4 * m / max(m.sum(), 1e-12)


def partitions(d, a, b, indeg, outdeg):
    m, f = d["m"][:, a], d["f"][:, a]
    N = len(m)
    lvl = np.log1p(cp10k(m))
    P = {}
    g = np.full(N, "zero_a", dtype=object)
    ex = m > 0
    q = np.quantile(lvl[ex], [0.2, 0.4, 0.6, 0.8])
    g[ex] = np.array([f"Q{k + 1}" for k in np.searchsorted(q, lvl[ex], side="right")], dtype=object)
    P["expr"] = g
    t1 = np.full(N, "rest", dtype=object); t1[m >= np.quantile(m, 0.99)] = "top1pct"
    P["top1"] = t1
    fg = np.full(N, "f=0", dtype=object)
    for lo, hi, lab in ((0, .02, "f<=.02"), (.02, .1, "f<=.1"), (.1, .3, "f<=.3"), (.3, 1.01, "f>.3")):
        fg[(f > lo) & (f <= hi)] = lab
    P["frac"] = fg
    og = np.full(N, "nonreg", dtype=object)
    reg = outdeg > 0
    med = np.median(outdeg[reg])
    og[reg & (outdeg <= med)] = "reg_low"; og[reg & (outdeg > med)] = "reg_high"
    og[np.argsort(-outdeg)[:50]] = "hub50"
    P["outdeg"] = og
    ig = np.full(N, "indeg0", dtype=object)
    tg = indeg > 0
    qq = np.quantile(indeg[tg], [1 / 3, 2 / 3])
    ig[tg] = np.array([f"in_T{k + 1}" for k in np.searchsorted(qq, indeg[tg], side="right")], dtype=object)
    P["indeg"] = ig
    za, zb = m == 0, d["m"][:, b] == 0
    cg = np.full(N, "pp", dtype=object)
    cg[za & zb] = "zz"; cg[za & ~zb] = "on"; cg[~za & zb] = "off"
    P["crossing"] = cg
    return P


def _incl(v, k):
    thr = np.sort(v)[::-1][k - 1]
    above, eq = v > thr, v == thr
    q = above.astype(float)
    q[eq] = (k - above.sum()) / eq.sum()
    return q


def group_stats(y, preds, P):
    yc = y - y.mean()
    tot2 = (yc ** 2).sum()
    tu, td = _incl(y, KALL), _incl(-y, KALL)
    out = {}
    for pname, lab in P.items():
        res = {}
        for gname in np.unique(lab):
            s = lab == gname
            r = {"share_genes": float(s.mean()), "share_var": float((yc[s] ** 2).sum() / tot2),
                 "share_true_top_up": float(tu[s].sum() / KALL), "share_true_top_down": float(td[s].sum() / KALL),
                 "mean_y": float(y[s].mean())}
            for fname, p in preds.items():
                pc = p - p.mean()
                den = np.sqrt(tot2 * (pc ** 2).sum())
                r[f"{fname}:pcc_contrib"] = float((yc[s] * pc[s]).sum() / den) if den > 0 else 0.0
                pu, pd_ = _incl(p, KALL), _incl(-p, KALL)
                r[f"{fname}:hits_up"] = float((tu[s] * pu[s]).sum() / KALL)
                r[f"{fname}:hits_down"] = float((td[s] * pd_[s]).sum() / KALL)
                r[f"{fname}:pcc_without_group"] = common.pearson(y[~s], p[~s]) if (~s).sum() > 10 else float("nan")
            res[gname] = r
        out[pname] = res
    for fname, p in preds.items():
        out.setdefault("_all", {})[fname] = {"pcc": common.pearson(y, p),
                                             "hits_up": float((tu * _incl(p, KALL)).sum() / KALL),
                                             "hits_down": float((td * _incl(-p, KALL)).sum() / KALL)}
    return out


def average(list_of):
    """Average nested dicts of floats over transitions."""
    if isinstance(list_of[0], dict):
        keys = set().union(*[x.keys() for x in list_of])
        return {k: average([x[k] for x in list_of if k in x]) for k in sorted(keys)}
    return float(np.nanmean(list_of))


def main():
    out = {}
    for name in common.SNAPS:
        d = common.load_node(name)
        N = d["m"].shape[0]
        snaps = common.SNAPS[name]
        trans = common.transitions(name)
        u_all, i_all, ts_all, _ = common.load_edges(name)
        per = {T: [] for T in common.TARGETS}
        hub_expr = []
        for ti, (a, b) in enumerate(trans):
            if ti == 0:
                continue
            sel = ts_all == a
            u, i = u_all[sel], i_all[sel]
            u, i = u[(u < N) & (i < N)], i[(u < N) & (i < N)]
            outdeg = np.bincount(u, minlength=N).astype(float)
            indeg = np.bincount(i, minlength=N).astype(float)
            lvl = np.log1p(cp10k(d["m"][:, a]))
            reg, tg = outdeg > 0, indeg > 0
            hub_expr.append({"spearman_outdeg_lvl_regs": float(stats.spearmanr(outdeg[reg], lvl[reg]).correlation),
                             "spearman_indeg_lvl_targets": float(stats.spearmanr(indeg[tg], lvl[tg]).correlation),
                             "spearman_indeg_lvl_all": float(stats.spearmanr(indeg, lvl).correlation),
                             "frac_regs_among_expressed": float(reg.sum() / max((d["m"][:, a] > 0).sum(), 1))})
            P = partitions(d, a, b, indeg, outdeg)
            hist_bins = [s for s in snaps if s <= a]
            p = hist_bins[-2]
            lo = max(0, len(hist_bins) - 4)
            w = d["n"][hist_bins[lo:]]
            pool4 = (d["m"][:, hist_bins[lo:]] * w).sum(1) / w.sum()
            for T in common.TARGETS:
                y = common.target(T, d["m"][:, a], d["m"][:, b])
                preds = {"trend": common.target(T, d["m"][:, p], d["m"][:, a]),
                         "rev4": common.target(T, d["m"][:, a], pool4)}
                if T == "orig":
                    preds["obs4"] = tf.ObsLevel(4).predict(hist_arrays(d, hist_bins), hist_bins, None).astype(np.float64)
                per[T].append(group_stats(y, preds, P))
        out[name] = {"hub_expression": average(hub_expr),
                     **{T: average(per[T]) for T in common.TARGETS}}
        print("==", name, out[name]["hub_expression"])
        for T in common.TARGETS:
            print(" ", T, "all:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out[name][T]["_all"].items()})
            for part in ("crossing", "expr", "top1", "outdeg"):
                for g, r in out[name][T][part].items():
                    ref = "obs4" if T == "orig" else "trend"
                    print(f"    {part:8s} {g:8s} genes={r['share_genes']:.3f} var={r['share_var']:.3f} "
                          f"top_up={r['share_true_top_up']:.3f} top_dn={r['share_true_top_down']:.3f} "
                          f"{ref}: pcc_c={r[ref + ':pcc_contrib']:.3f} hits_up={r[ref + ':hits_up']:.3f} "
                          f"hits_dn={r[ref + ':hits_down']:.3f} pcc_wo={r[ref + ':pcc_without_group']:.3f}", flush=True)
    with open(os.path.join(common.OUT, "heterogeneity.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
