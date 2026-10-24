"""C7: does the GRN add information about the l1 change beyond each gene's own history? Partial correlations per
transition.

Graph: per-bin GRNBoost2 edges of the same build (gastrulation-human.csv / pancreas-human.csv, cached as u, i, ts, w
in the session scratchpad edge_cache; u = regulator, i = target, ts = raw bin, w = importance, as NASS's load_graph).
W_t = bin-t graph, rows (targets) normalised to sum 1. Wbar = mean of the row-normalised graphs of snapshot bins
t, t-1, t-2 (fewer at t < 2), rows renormalised to sum 1. Strict: only bins <= t.
Aggregates: (a) W_t y_{t-1}; (b) Wbar y_{t-1}; (c) Wbar y_{t-1} given W_t y_{t-1}; (d) W_t y_{t-1} given Wbar y_{t-1};
(e) W_t rev_box4 (NASS's rev_box4: cell-weighted 4-bin box of CP10k vs today, on the l1 scale).
Controls: set 'listed' = intercept + lag1..3 + rev_box2/4/8 + level L_t + SNR terms (logse, frac, lag1_x_se,
rev4_x_se) + has-regulator flag(s) (NASS features); set 'nass_cls' = NASS's full feature set (lags, box, Kalman,
SNR, SVD) interacted with its 5 expression classes plus class dummies plus has-regulator flags (the stacker design).
Partial correlation = Pearson of the residuals of y_t and of the aggregate after least squares on the controls,
on the genes of each CONTEXT test mask (default_rng(1000 s + i), 3 seeds, averaged) and on all genes.
Null: level-matched relabelling of regulators (NASS level_null logic): regulators of the graphs used are permuted
within deciles of L_t among regulators; aggregate = W x[perm]; 10 draws per transition, same controls.
Transitions i >= 1 (lag-1 needed); also i >= 3 (Wbar uses 3 graphs, lag3 defined).
"""
import json
import os
import sys
import time

import numpy as np
from scipy import sparse

import common as C

sys.path.insert(0, C.DBS)
import dsdata as D  # noqa: E402
import nass as NS  # noqa: E402

N_NULL = 10
LISTED = ["lag1", "lag2", "lag3", "rev_box2", "rev_box4", "rev_box8", "logse", "frac", "lag1_x_se", "rev4_x_se"]


def rownorm(A):
    s = np.asarray(A.sum(1)).ravel()
    return sparse.diags(np.where(s > 0, 1.0 / np.where(s > 0, s, 1), 0.0)) @ A, s > 0


def graphs(d):
    z = np.load(os.path.join(C.EDGE_CACHE, f"{d.name}_edges.npz"))
    u, tg, ts, w = z["u"], z["i"], z["ts"], z["w"]
    info = {"n_edges": int(len(u)), "max_u": int(u.max()), "max_i": int(tg.max()), "ts_values": np.unique(ts).tolist()}
    assert max(info["max_u"], info["max_i"]) < d.G
    order = np.argsort(ts, kind="stable")
    ts_s = ts[order]
    out = {}
    for j, b in enumerate(d.bins):
        lo, hi = np.searchsorted(ts_s, b), np.searchsorted(ts_s, b, side="right")
        sel = order[lo:hi]
        A = sparse.csr_matrix((w[sel].astype(np.float64), (tg[sel], u[sel])), shape=(d.G, d.G))
        W, has = rownorm(A)
        out[j] = (W.tocsr(), has, np.unique(u[sel]))
    info["edges_per_bin"] = {int(b): int(out[j][0].nnz) for j, b in enumerate(d.bins)}
    return out, info


def perm_within_deciles(regs, lvl, rng, G):
    mp = np.arange(G)
    dec = np.searchsorted(np.quantile(lvl[regs], np.linspace(0.1, 0.9, 9)), lvl[regs], side="right")
    for k in np.unique(dec):
        grp = regs[dec == k]
        mp[grp] = rng.permutation(grp)
    return mp


def resid_basis(Z):
    Q, Rr = np.linalg.qr(Z)
    keep = np.abs(np.diag(Rr)) > 1e-10 * np.abs(np.diag(Rr)).max()
    return Q[:, keep]


def res_(Q, v):
    return v - Q @ (Q.T @ v)


def off(v, u):
    uu = (u * u).sum()
    return v - u * ((u * v).sum() / uu) if uu > 0 else v


def cor(a, b):
    """Correlation of two residual vectors (both already orthogonal to the intercept in Q)."""
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def zsc(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


def main():
    res = {}
    for name in C.NAMES:
        t0 = time.time()
        d = D.Data(name)
        dc = C.Data(name)
        Y = d.Y("l1")
        F, cls = NS.build_features(d, "l1", Y, None, None)
        G_ = NS.GROUPS
        fullset = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"]
        g, ginfo = graphs(d)
        print(f"{name}: graphs loaded {time.time() - t0:.0f}s, {ginfo['n_edges']} edges", flush=True)
        mk = dc.masks()
        T = d.T
        keys = ("a", "b", "c", "d", "e")
        real = {cs: {gs: np.full((T, 5), np.nan) for gs in ("mask", "all")} for cs in ("listed", "nass_cls")}
        null = {cs: {gs: np.full((T, N_NULL, 5), np.nan) for gs in ("mask", "all")} for cs in ("listed", "nass_cls")}
        frac_has = []
        for i in range(1, T):
            W, has, regs = g[i]
            idx = [j for j in (i, i - 1, i - 2) if j >= 0]
            Wb = sum(g[j][0] for j in idx) / len(idx)
            Wb, hasb = rownorm(Wb)
            regs_b = np.unique(np.concatenate([g[j][2] for j in idx]))
            frac_has.append(float(has.mean()))
            y = Y[i]
            lag, rev = F["lag1"][i], F["rev_box4"][i]
            Lt = d.L[:, i]
            base_listed = np.column_stack([np.ones(d.G), Lt] + [F[k][i] for k in LISTED] +
                                          [has.astype(float), hasb.astype(float)])
            Fi = np.column_stack([zsc(F[k][i]) for k in fullset])
            base_cls = np.column_stack([Fi * (cls[i] == c)[:, None] for c in range(NS.N_CLS)] +
                                       [(cls[i] == c)[:, None].astype(float) for c in range(NS.N_CLS)] +
                                       [has.astype(float), hasb.astype(float), Lt])
            aggs = [(W @ lag, Wb @ lag, W @ rev)]
            rng = np.random.default_rng([7, i, len(name)])
            for _ in range(N_NULL):
                mp = perm_within_deciles(regs, Lt, rng, d.G)
                mpb = perm_within_deciles(regs_b, Lt, rng, d.G)
                aggs.append((W @ lag[mp], Wb @ lag[mpb], W @ rev[mp]))
            for cs, base in (("listed", base_listed), ("nass_cls", base_cls)):
                for gs in ("mask", "all"):
                    sets = [mk[s, i] for s in range(3)] if gs == "mask" else [np.arange(d.G)]
                    vals = np.zeros((len(aggs), len(sets), 5))
                    for si, t in enumerate(sets):
                        Q = resid_basis(base[t])
                        ry = res_(Q, y[t])
                        for ai, (wl, wbl, wr) in enumerate(aggs):
                            ra, rb, re = res_(Q, wl[t]), res_(Q, wbl[t]), res_(Q, wr[t])
                            # Frisch-Waugh: partial given base + other aggregate = residualise on the other's residual
                            vals[ai, si] = (cor(ry, ra), cor(ry, rb), cor(off(ry, ra), off(rb, ra)),
                                            cor(off(ry, rb), off(ra, rb)), cor(ry, re))
                    v = vals.mean(1)
                    real[cs][gs][i] = v[0]
                    null[cs][gs][i] = v[1:]
            print(f"  i={i} ({time.time() - t0:.0f}s) listed/mask a..e {np.round(real['listed']['mask'][i], 4).tolist()} "
                  f"null mean {np.round(null['listed']['mask'][i].mean(0), 4).tolist()}", flush=True)
        out = {"graph_info": ginfo, "frac_targets_with_regulator": frac_has, "summary": {},
               "per_transition": {cs: {gs: np.round(real[cs][gs], 5).tolist() for gs in real[cs]} for cs in real}}
        for cs in real:
            for gs in ("mask", "all"):
                for rname, lo in (("i>=1", 1), ("i>=3", 3)):
                    r = real[cs][gs][lo:]
                    nl = null[cs][gs][lo:]
                    nm = nl.mean(1)                                # (n_t, 5) null mean per transition
                    draw_means = nl.mean(0)                        # (N_NULL, 5) mean over transitions per draw
                    sm = {}
                    for k, key in enumerate(keys):
                        sm[key] = {"mean": round(float(np.mean(r[:, k])), 5), "n_pos": int((r[:, k] > 0).sum()),
                                   "n": int(len(r)), "null_mean": round(float(np.mean(nm[:, k])), 5),
                                   "null_sd_of_draw_means": round(float(np.std(draw_means[:, k])), 5),
                                   "null_n_pos_mean": round(float(np.mean((nl[:, :, k] > 0).sum(0))), 2),
                                   "n_t_real_gt_null_mean": int((r[:, k] > nm[:, k]).sum()),
                                   "real_minus_null_mean": round(float(np.mean(r[:, k] - nm[:, k])), 5)}
                    out["summary"][f"{cs}/{gs}/{rname}"] = sm
                    print(f"  {cs:8s} {gs:4s} {rname}: " + "  ".join(
                        f"{key}={sm[key]['mean']:+.4f} ({sm[key]['n_pos']}/{sm[key]['n']}) null {sm[key]['null_mean']:+.4f}"
                        f"+-{sm[key]['null_sd_of_draw_means']:.4f} ({sm[key]['null_n_pos_mean']:.1f})" for key in keys))
        res[name] = out
        json.dump(res, open(os.path.join(C.OUT, "c7_grn_partial.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
