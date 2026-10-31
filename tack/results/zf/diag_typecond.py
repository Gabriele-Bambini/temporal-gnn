"""EXPLORATORY (not pre-registered): does conditioning on cell type rescue interpolation on the zebrafish GAP14 products?
Held timepoint k from k-1 and k+1 (interp<k>.npz, PCA space), ED_U against the held subsample (<= 1500, seed 1):
  union     50/50 of the two neighbours (cells uniform within side)
  ot_glob   frozen McCann OT pairs on all cells (interp_bench_frozen.methods)
  tc_shift  per cell type present on both sides (>= 10 cells): prev cells + delta_c / 2, next cells - delta_c / 2,
            delta_c = mean(next_c) - mean(prev_c); other cells unshifted; masses as union
  tc_ot     per shared type: McCann OT pairs inside the type (eps_rel 0.05), type mass = union mass of the type;
            unshared cells kept with their union mass
  floor     held embryos split in two random halves: ED_U(half A, half B) (a real replicate of the same timepoint)
Cell types (authors' annotation) are read only for the neighbours, never for the held cells.
Writes diag_typecond.json next to this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = os.environ.get("ZF_DATA", r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\zf")
MIN_T, REPS, N = 10, 3, 1500


def ed(P, w, tgt):
    return IB.ed_uv(P, w / w.sum(), tgt)[0]


def unit(k):
    d = np.load(os.path.join(SCR, f"interp{k}.npz"), allow_pickle=True)
    st, Zf, Zh = d["stage_fit"], d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
    ct, emb = d["celltype_fit"].astype(str), d["sample_held"].astype(str)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    S = IB.med_scale(Zf)
    tgt = IB.Target(Zh[IB.sub(len(Zh), seed=1)], None, S, with_sd=False)
    wA, wB = np.full(len(A), 0.5 / len(A)), np.full(len(B), 0.5 / len(B))
    P_union, w_union = np.vstack([Zf[A], Zf[B]]), np.r_[wA, wB]
    shared = [c for c in sorted(set(ct[A]) & set(ct[B])) if (ct[A] == c).sum() >= MIN_T and (ct[B] == c).sum() >= MIN_T]
    ZA, ZB = Zf[A].copy(), Zf[B].copy()
    for c in shared:
        ia, ib = ct[A] == c, ct[B] == c
        dl = Zf[B][ib].mean(0) - Zf[A][ia].mean(0)
        ZA[ia] += dl / 2
        ZB[ib] -= dl / 2
    out = {"k": k, "n_shared_types": len(shared),
           "shared_mass": float(sum(wA[ct[A] == c].sum() + wB[ct[B] == c].sum() for c in shared)),
           "union": ed(P_union, w_union, tgt), "tc_shift": ed(np.vstack([ZA, ZB]), w_union, tgt)}
    og, ot = [], []
    for r in range(REPS):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        M = IB.methods(Zf[A[iA]], Zf[A[iA]], Zf[B[iB]], Zf[B[iB]], np.zeros(len(iA)), np.eye(Zf.shape[1]), N, rng,
                       which=("ot_pairs",))
        og.append(ed(M["ot_pairs"][0], np.ones(N), tgt))
        pts, ws = [], []
        for c in shared:
            a, b = A[ct[A] == c], B[ct[B] == c]
            mass = wA[ct[A] == c].sum() + wB[ct[B] == c].sum()
            n_c = max(int(round(N * mass)), 5)
            Pp, _ = IB.plan(Zf[a], Zf[b], 0.05)
            i, j = IB.draw_pairs(Pp, n_c, rng)
            pts.append(0.5 * Zf[a][i] + 0.5 * Zf[b][j])
            ws.append(np.full(n_c, mass / n_c))
        for side, w in ((A, wA), (B, wB)):
            un = ~np.isin(ct[side], shared)
            if un.any():
                pts.append(Zf[side][un])
                ws.append(w[un])
        ot.append(ed(np.vstack(pts), np.concatenate(ws), tgt))
    out["ot_glob"], out["tc_ot"] = float(np.mean(og)), float(np.mean(ot))
    ue = np.unique(emb)
    rng = np.random.default_rng(5)
    half = set(rng.permutation(ue)[: len(ue) // 2])
    h1, h2 = np.isin(emb, list(half)), ~np.isin(emb, list(half))
    out["floor"] = ed(Zh[h1], np.ones(h1.sum()), IB.Target(Zh[h2], None, S, with_sd=False))
    return out


def main():
    res = [unit(k) for k in range(1, 10)]
    for r in res:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    keys = ("union", "ot_glob", "tc_shift", "tc_ot", "floor")
    summ = {m: float(np.mean([r[m] for r in res])) for m in keys}
    summ.update({f"{m}_beats_union": int(sum(r[m] < r["union"] for r in res)) for m in ("ot_glob", "tc_shift", "tc_ot")})
    print("MEAN", summ)
    with open(os.path.join(HERE, "diag_typecond.json"), "w") as fh:
        json.dump({"units": res, "summary": summ}, fh, indent=1)


if __name__ == "__main__":
    main()
