"""Quick look 8: the bin GRN (GRNBoost2 on the cells of ONE bin = the only within-bin cross-gene information) as a
within-bin covariance operator. Partial correlation of the l1 change y_i with graph-propagated quantities, given the
full own-history control set (lags 1-3, box 2/4/8 reversions, logse, frac, SNR interactions, class x feature
interactions); in-sample per transition (diagnostic). W_b[target, regulator] row-normalised importances of bin b.
 G_cur   = W_i  y_{i-1}        (NASS grn_lag1; bin-i graph)
 G_prev  = W_{i-1} y_{i-1}     (graph of the previous bin: does the CURRENT bin covariance matter?)
 G_mom2  = W_i (L_i - L_{i-2})/2
 G_sym   = Wsym_i y_{i-1}      (W + W^T, row-normalised)
 G_2hop  = W_i W_i y_{i-1}
 G_rev   = W_i rev_box4        (neighbours' deviation from their recent mean: co-noise would give a positive sign)
 G_null  = level-matched regulator permutation of W_i, times y_{i-1}"""
import os, sys
import numpy as np
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicklook import load, corr, pcorr

EC = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\edge_cache"
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}

def rown(A):
    s = np.asarray(A.sum(1)).ravel()
    return sparse.diags(np.where(s > 0, 1.0 / np.where(s > 0, s, 1), 0.0)) @ A

def zs(v):
    sd = v.std(); return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)

for name in sys.argv[1:] or ["pancreas-human", "gastrulation-human"]:
    m, med, s, f, tot, n = load(name)
    G, B = m.shape; T = B - 1
    lib = m.sum(0); kap = 1e4 / lib; CP = m * kap[None, :]; L = np.log1p(CP)
    Y = np.diff(L, axis=1).T
    RL = kap[None, :] ** 2 * s ** 2 / np.maximum(n - 1, 1)[None, :] / (1 + CP) ** 2
    z = np.load(os.path.join(EC, f"{name}_edges.npz"))
    u, it, ts, w = z["u"], z["i"], z["ts"], z["w"]
    order = np.argsort(ts, kind="stable"); ts_s = ts[order]
    rng = np.random.default_rng(13)
    Ws, Wsym, Wnull = {}, {}, {}
    for j, b in enumerate(SNAPS[name]):
        lo, hi = np.searchsorted(ts_s, b), np.searchsorted(ts_s, b, side="right")
        sel = order[lo:hi]
        A = sparse.csr_matrix((w[sel].astype(np.float64), (it[sel], u[sel])), shape=(G, G))
        Ws[j] = rown(A); Wsym[j] = rown(A + A.T)
        regs = np.unique(u[sel]); lv = L[:, j]
        dec = np.searchsorted(np.quantile(lv[regs], np.linspace(0.1, 0.9, 9)), lv[regs], side="right")
        mp = np.arange(G)
        for k in np.unique(dec):
            grp = regs[dec == k]; mp[grp] = rng.permutation(grp)
        Wnull[j] = rown(sparse.csr_matrix((w[sel].astype(np.float64), (it[sel], mp[u[sel]])), shape=(G, G)))
    res = {}
    for i in range(3, T):
        mi, fi = m[:, i], f[:, i]
        boxes = []
        for k in (2, 4, 8):
            lo = max(0, i - k + 1); ww = n[lo:i + 1]
            boxes.append(np.log1p((CP[:, lo:i + 1] * ww).sum(1) / ww.sum()) - L[:, i])
        se = np.log(np.sqrt(np.maximum(RL[:, i], 1e-12)) + 1e-6)
        ctrl = [Y[i - 1], Y[i - 2], Y[i - 3]] + boxes + [se, fi, zs(Y[i - 1]) * zs(se), zs(boxes[1]) * zs(se)]
        cl = np.digitize(np.where(mi > 0, np.log10(np.where(mi > 0, mi, 1)), -9), [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])
        for c in range(5):
            ind = (cl == c).astype(float)
            ctrl += [ind] + [v * ind for v in (Y[i - 1], boxes[0], boxes[1], boxes[2])]
        mom2 = (L[:, i] - L[:, i - 2]) / 2
        Wa3 = (Ws[i] + Ws[i - 1] + Ws[i - 2]) / 3.0
        cand = {"G_avg3": Wa3 @ Y[i - 1], "G_avg3_2hop": Wa3 @ (Wa3 @ Y[i - 1]), "G_cur": Ws[i] @ Y[i - 1], "G_prev": Ws[i - 1] @ Y[i - 1], "G_mom2": Ws[i] @ mom2,
                "G_sym": Wsym[i] @ Y[i - 1], "G_2hop": Ws[i] @ (Ws[i] @ Y[i - 1]), "G_rev": Ws[i] @ boxes[1],
                "G_null": Wnull[i] @ Y[i - 1]}
        for k, v in cand.items():
            res.setdefault(k, []).append((pcorr(Y[i], v, ctrl), corr(Y[i], v)))
        # head-to-head: G_cur given G_prev and vice versa
        res.setdefault("G_avg3|G_cur", []).append((pcorr(Y[i], cand["G_avg3"], ctrl + [cand["G_cur"]]), np.nan))
        res.setdefault("G_cur|G_avg3", []).append((pcorr(Y[i], cand["G_cur"], ctrl + [cand["G_avg3"]]), np.nan))
        res.setdefault("G_cur|G_prev", []).append((pcorr(Y[i], cand["G_cur"], ctrl + [cand["G_prev"]]), np.nan))
        res.setdefault("G_prev|G_cur", []).append((pcorr(Y[i], cand["G_prev"], ctrl + [cand["G_cur"]]), np.nan))
        res.setdefault("G_mom2|G_cur", []).append((pcorr(Y[i], cand["G_mom2"], ctrl + [cand["G_cur"]]), np.nan))
        res.setdefault("G_2hop|G_cur", []).append((pcorr(Y[i], cand["G_2hop"], ctrl + [cand["G_cur"]]), np.nan))
        res.setdefault("G_sym|G_cur", []).append((pcorr(Y[i], cand["G_sym"], ctrl + [cand["G_cur"]]), np.nan))
    print(f"===== {name}  (transitions i>=3, k={T-3})")
    for k, v in res.items():
        a = np.array(v)
        print(f"  {k:13s} partial {np.nanmean(a[:, 0]):+.3f} (pos {int((a[:, 0] > 0).sum())}/{len(a)})   raw corr {np.nanmean(a[:, 1]):+.3f}")
