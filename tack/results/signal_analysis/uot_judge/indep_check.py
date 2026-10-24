"""Judge: independent data load + targets + masks + metrics (no prototype code imported).

1. Reads the raw '-human' node CSVs (first N rows, N = rows of <name>_y_outdeg.csv), builds X (B, G, 5) and compares it
   to the npz caches the prototypes read (design_best-signal/_node_*.npz for uot_gene_sink, uot_pop_sink/_node_*.npz and
   the judge's own rebuilt p1 cache).
2. Cells per bin (median over expressed genes of s04_sum / s00_mean) vs the CONTEXT list.
3. Own targets (l1, linear), own masks (default_rng(1000 s + i).permutation(G)[:n_test]), own metrics (Pearson,
   tie-aware P@200 up / down) -> trend scores, i >= 1 and i >= 2.
4. Rescoring of the gene-sink prototype's SAVED selected predictions (sel_preds_<ds>.npz) with these own metrics.
Usage: python indep_check.py   ->  indep_check.json
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
GRN = os.path.join(ROOT, "tgl-grn-1CCD", "GRN")
SA = os.path.join(ROOT, "tack", "results", "signal_analysis")
HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
CONTEXT_CELLS = {"gastrulation-human": [9, 182, 792, 929, 982, 616, 340, 217, 225, 248, 290, 304, 431, 440, 333, 247, 246,
                                        230, 275, 316, 355, 350, 264, 233, 228, 306, 231, 138, 46, 7],
                 "pancreas-human": [111, 612, 361, 116, 50, 53, 68, 79, 191, 218, 132, 286, 698, 616, 105]}
STATS = ("s00_mean", "s01_median", "s02_std", "s03_frac_nonzero", "s04_sum")


def load_csv(name):
    N = len(pd.read_csv(os.path.join(GRN, f"{name}_y_outdeg.csv")))
    cols = [f"t{b:03d}_{s}" for b in SNAPS[name] for s in STATS]
    df = pd.read_csv(os.path.join(GRN, f"{name}_node.csv"), usecols=["gene"] + cols, nrows=N)
    assert len(df) == N
    X = np.stack([np.stack([df[f"t{b:03d}_{s}"].to_numpy(np.float64) for s in STATS], 1) for b in SNAPS[name]])
    return X, df["gene"].astype(str).to_numpy()


def cells(X):
    out = []
    for b in range(X.shape[0]):
        ok = X[b, :, 0] > 0
        out.append(float(np.median(X[b, ok, 4] / X[b, ok, 0])))
    return np.array(out)


def incl(v, k):
    order = np.sort(v)[::-1]
    thr = order[k - 1]
    above = v > thr
    eq = v == thr
    p = above.astype(float)
    p[eq] = (k - above.sum()) / eq.sum()
    return p


def my_metrics(y, p, k=200):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    if np.all(p == p[0]) or np.all(y == y[0]):
        r = np.nan
    else:
        yc, pc = y - y.mean(), p - p.mean()
        r = float((yc * pc).sum() / np.sqrt((yc ** 2).sum() * (pc ** 2).sum()))
    up = float((incl(y, k) * incl(p, k)).sum() / k)
    dn = float((incl(-y, k) * incl(-p, k)).sum() / k)
    return r, up, dn


def masks(G, T):
    nt = int(round(0.25 * G))
    return [[np.sort(np.random.default_rng(1000 * s + i).permutation(G)[:nt]) for i in range(T)] for s in range(3)]


def score(Y, P, mk, start, nan0=False):
    """Y, P (T, G); mean over transitions >= start per seed, then over seeds."""
    per_seed = []
    for s in range(3):
        rows = []
        for i in range(start, len(Y)):
            if not np.all(np.isfinite(P[i])):
                continue
            r = my_metrics(Y[i][mk[s][i]], P[i][mk[s][i]])
            if nan0 and not np.isfinite(r[0]):
                r = (0.0, r[1], r[2])
            rows.append(r)
        per_seed.append(np.nanmean(np.array(rows), 0))
    return [round(float(x), 4) for x in np.mean(per_seed, 0)]


def main():
    out = {}
    for name in ("pancreas-human", "gastrulation-human"):
        X, genes = load_csv(name)
        B, G = X.shape[0], X.shape[1]
        T = B - 1
        o = {"G": G, "B": B}
        caches = {"design_best-signal (read by uot_gene_sink)": os.path.join(SA, "design_best-signal", f"_node_{name}.npz"),
                  "uot_pop_sink own cache": os.path.join(SA, "uot_pop_sink", f"_node_{name}.npz"),
                  "judge p1 rebuilt cache": os.path.join(HERE, "p1", f"_node_{name}.npz")}
        for k, f in caches.items():
            if os.path.exists(f):
                z = np.load(f, allow_pickle=False)
                Xc = z["X"]
                same_shape = Xc.shape == X.shape
                o[f"cache_vs_csv::{k}"] = {"shape": list(Xc.shape), "same_shape": bool(same_shape),
                                           "max_abs_diff": float(np.abs(Xc - X).max()) if same_shape else None,
                                           "genes_equal": bool(np.array_equal(z["genes"].astype(str), genes))
                                           if "genes" in z.files else "no genes array"}
            else:
                o[f"cache_vs_csv::{k}"] = "missing"
        n = cells(X)
        o["cells_rounded"] = [int(round(x)) for x in n]
        o["cells_match_context"] = o["cells_rounded"] == CONTEXT_CELLS[name]
        m = np.maximum(X[:, :, 0], 0.0)                       # (B, G)
        cp = 1e4 * m / m.sum(1, keepdims=True)
        Y = {"l1": np.log1p(cp[1:]) - np.log1p(cp[:-1]), "linear": m[1:] - m[:-1]}
        mk = masks(G, T)
        tr = {}
        for tg, y in Y.items():
            P = np.full_like(y, np.nan)
            P[1:] = y[:-1]
            tr[tg] = {"i>=1": score(y, P, mk, 1), "i>=2": score(y, P, mk, 2),
                      "allgene_pcc_i>=1": round(float(np.mean([np.corrcoef(y[i], y[i - 1])[0, 1] for i in range(1, T)])), 4)}
        o["trend_own_code"] = tr
        # rescoring of the gene-sink prototype's saved selected predictions
        f = os.path.join(SA, "uot_gene_sink", f"sel_preds_{name}.npz")
        if os.path.exists(f):
            z = np.load(f)
            rs = {}
            for key in z.files:
                fam, tg = key.split("__")
                rs[key] = {"i>=1": score(Y[tg], z[key].astype(np.float64), mk, 1),
                           "i>=2": score(Y[tg], z[key].astype(np.float64), mk, 2)}
            o["gene_sink_saved_preds_rescored_own_metrics"] = rs
        f = os.path.join(HERE, "p2", f"sel_preds_{name}.npz")
        if os.path.exists(f):
            z = np.load(f)
            z0 = np.load(os.path.join(SA, "uot_gene_sink", f"sel_preds_{name}.npz"))
            o["gene_sink_rerun_vs_saved_preds_max_abs_diff"] = {
                k: float(np.nanmax(np.abs(z[k].astype(np.float64) - z0[k].astype(np.float64)))) for k in z.files}
        out[name] = o
        print(name, json.dumps(o, indent=1), flush=True)
    with open(os.path.join(HERE, "indep_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
