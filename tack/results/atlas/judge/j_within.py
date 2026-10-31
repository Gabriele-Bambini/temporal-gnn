"""JUDGE, EXPLORATORY: ot_pairs vs union WITHIN single cell types (independent re-implementation of the checker's
within-lineage test), on the atlas (k = 1..7) and on the erythroid parity products (k = 2..5).

Unit = (k, cell type T) with >= 100 cells of T at k-1, at k and at k+1 (types are atlas annotations, used only to cut
units, never as model input). Inside a unit everything is the frozen recipe restricted to T: fit subsamples
IB.sub(n, seed = 1000 + r / 2000 + r), target IB.sub(n_held_T, seed = 1), n_pred = min(n_held_T, 1500),
IB.methods(..., which = prev/next/union/indep/ot_pairs/sb_half) with rng default_rng(1000 k + r), IB.subsample_pred,
ED_U in the interp<k> PCA space, mean over 3 reps. Removes between-type composition change from the comparison.
Usage: python j_within.py -> j_within.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
SETS = {"atlas": (os.path.join(SCR, "atlas"), range(1, 8)), "ery": (os.path.join(SCR, "ery"), range(2, 6))}
WHICH = ("prev", "next", "union", "indep", "ot_pairs", "sb_half")
MIN_N = 100


def main():
    res = {}
    for name, (data, ks) in SETS.items():
        units = {}
        t0 = time.time()
        for k in ks:
            d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
            st = d["stage_fit"]
            Zf, Xf, Zh = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32), d["Z_held"].astype(np.float64)
            comps = d["pca_comps"].astype(np.float64)
            ct, ch = d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
            S = IB.med_scale(Zf)
            for T in sorted(set(ch)):
                A = np.where((st == k - 1) & (ct == T))[0]
                B = np.where((st == k + 1) & (ct == T))[0]
                H = np.where(ch == T)[0]
                if min(len(A), len(B), len(H)) < MIN_N:
                    continue
                n_pred = min(len(H), IB.NSUB)
                tgt = IB.Target(Zh[H[IB.sub(len(H), seed=1)]], None, S, with_sd=False)
                reps = []
                for r in range(3):
                    rng = np.random.default_rng(1000 * k + r)
                    iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
                    ZA, XA = Zf[A[iA]], Xf[A[iA]].astype(np.float64)
                    ZB, XB = Zf[B[iB]], Xf[B[iB]].astype(np.float64)
                    M = IB.methods(ZA, XA, ZB, XB, np.zeros(len(iA)), comps, n_pred, rng, which=WHICH)
                    M.pop("_plan", None)
                    rec = {}
                    for m, (Z, X, w, gm) in M.items():
                        Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
                        rec[m] = IB.ed_uv(Zs, ws, tgt)[0]
                    reps.append(rec)
                mean = {m: float(np.mean([rp[m] for rp in reps])) for m in WHICH}
                units[f"{k}|{T}"] = {"k": k, "type": T, "n": [int(len(A)), int(len(H)), int(len(B))], "mean": mean,
                                     "rep_gap_ot_minus_union": [rp["ot_pairs"] - rp["union"] for rp in reps]}
                print(name, k, T, units[f"{k}|{T}"]["n"], {m: round(v, 3) for m, v in mean.items()}, flush=True)
        U = list(units.values())
        ratio = [u["mean"]["ot_pairs"] / u["mean"]["union"] for u in U]
        summ = {"n_units": len(U),
                "ot_pairs<union": sum(u["mean"]["ot_pairs"] < u["mean"]["union"] for u in U),
                "ot_pairs_beats_prev_next_union": sum(all(u["mean"]["ot_pairs"] < u["mean"][b] for b in ("prev", "next", "union")) for u in U),
                "median_ot_pairs_over_union": float(np.median(ratio)),
                "sb_half<union": sum(u["mean"]["sb_half"] < u["mean"]["union"] for u in U),
                "median_sb_half_over_union": float(np.median([u["mean"]["sb_half"] / u["mean"]["union"] for u in U])),
                "ot_pairs<indep": sum(u["mean"]["ot_pairs"] < u["mean"]["indep"] for u in U),
                "units_ot_wins": [f"k{u['k']} {u['type']}" for u in U if u["mean"]["ot_pairs"] < u["mean"]["union"]],
                "seconds": round(time.time() - t0, 1)}
        res[name] = {"units": units, "summary": summ}
        print(name, json.dumps(summ), flush=True)
    json.dump(res, open(os.path.join(HERE, "j_within.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
