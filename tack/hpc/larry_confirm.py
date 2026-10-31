"""Pre-registered LARRY test (tack/prereg/2026-10-29-larry-prereg.md): does OT win where cells really move and the true
couplings are known (Weinreb et al. 2020, in vitro, days 2/4/6, clonal barcodes)?

H1  interpolation of day 4 from days 2 and 6 with the frozen erythroid / atlas methods (atlas_confirm.run_interp on
    ~/tack/out/larry/<pop>/interp1.npz): units = the 4 day-4 culture wells (2 populations x 2 wells), per-well ED_U.
H3  coupling accuracy against the clonal truth, units = 2 populations x 2 transitions (2->4, 4->6), clonally labelled
    cells only (couple_<ab>.npz): clonal enrichment E(P) = [sum_ij P_ij 1(c_i = c_j) / sum P] / [sum_c a_c b_c]
    for P = entropic OT plan (interp_bench_frozen.plan, eps_rel 0.05), kNN coupling (each source cell uniform over its
    10 nearest target cells in PCA) and cell-type coupling (each source cell uniform over target cells with the same
    annotated type). Output ~/tack/out/larry/larry_confirm.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_confirm as AC  # noqa: E402
import interp_bench_frozen as IB  # noqa: E402

OUT = os.path.expanduser("~/tack/out/larry")
POPS = ("sca1pos", "sca1neg")
K_NN = 10


def enrichment(P, ca, cb):
    same = ca[:, None] == cb[None, :]
    m = float((P * same).sum() / P.sum())
    a = {c: v / len(ca) for c, v in zip(*np.unique(ca, return_counts=True))}
    b = {c: v / len(cb) for c, v in zip(*np.unique(cb, return_counts=True))}
    base = sum(a[c] * b.get(c, 0.0) for c in a)
    return m / base, m, base


def couplings(Za, Zb, ta, tb):
    P_ot, _ = IB.plan(Za, Zb, 0.05)
    _, ind = NearestNeighbors(n_neighbors=K_NN).fit(Zb).kneighbors(Za)
    P_knn = np.zeros((len(Za), len(Zb)))
    np.put_along_axis(P_knn, ind, 1.0 / K_NN, axis=1)
    P_ct = (ta[:, None] == tb[None, :]).astype(float)
    rows = P_ct.sum(1, keepdims=True)
    P_ct = np.where(rows > 0, P_ct / np.maximum(rows, 1), 1.0 / len(Zb))
    return {"ot": P_ot, "knn": P_knn, "celltype": P_ct}


def wins(units, a, b):
    n = len(units)
    need = int(np.ceil(2 * n / 3))
    w = sum(u[a] < u[b] for u in units)
    mean_ok = bool(np.mean([u[a] for u in units]) < np.mean([u[b] for u in units]))
    return {"wins": int(w), "of": n, "need": need, "mean_ok": mean_ok, "pass": bool(w >= need and mean_ok)}


def main():
    res = {"interp": {}, "couple": {}}
    wells = []
    for pop in POPS:
        d = os.path.join(OUT, pop)
        r = AC.run_interp(d, 1)
        z = np.load(os.path.join(d, "interp1.npz"), allow_pickle=True)
        st, Zf, Zh = z["stage_fit"], z["Z_fit"].astype(np.float64), z["Z_held"].astype(np.float64)
        rad = np.sqrt(((Zh - Zh.mean(0)) ** 2).sum(1).mean())
        r["separation"] = float(np.linalg.norm(Zf[st == 2].mean(0) - Zf[st == 0].mean(0)) / rad)
        res["interp"][pop] = r
        for w in r["pools"]:
            wells.append({m: r["pool_mean"][m][w] for m in AC.INTERP_METHODS})
        print(pop, "interp", {m: round(v["ED_U"], 4) for m, v in r["mean"].items()}, "sep", round(r["separation"], 2), flush=True)
        for ab in ("24", "46"):
            c = np.load(os.path.join(d, f"couple_{ab}.npz"), allow_pickle=True)
            Ps = couplings(c["Z_a"].astype(np.float64), c["Z_b"].astype(np.float64),
                           c["celltype_a"].astype(str), c["celltype_b"].astype(str))
            e = {k: enrichment(P, c["clone_a"], c["clone_b"]) for k, P in Ps.items()}
            res["couple"][f"{pop}_{ab}"] = {k: {"enrichment": v[0], "same_clone_mass": v[1], "indep_mass": v[2]} for k, v in e.items()}
            print(pop, ab, {k: round(v[0], 2) for k, v in e.items()}, flush=True)
    cu = [{k: -v["enrichment"] for k, v in u.items()} for u in res["couple"].values()]   # lower = better for wins()
    res["verdicts"] = {
        "H1_ot_pairs": {b: wins(wells, "ot_pairs", b) for b in ("prev", "next", "union")},
        "H1b_uot": {b: wins(wells, "uot", b) for b in ("prev", "next", "union")},
        "H1c_sb_half": {b: wins(wells, "sb_half", b) for b in ("prev", "next", "union")},
        "H3_ot_vs_knn": wins(cu, "ot", "knn"),
        "H3b_ot_vs_celltype": wins(cu, "ot", "celltype"),
    }
    for h in ("H1_ot_pairs", "H1b_uot", "H1c_sb_half"):
        res["verdicts"][h]["pass"] = all(v["pass"] for v in res["verdicts"][h].values())
    print("VERDICTS", json.dumps(res["verdicts"]), flush=True)
    with open(os.path.join(OUT, "larry_confirm.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
