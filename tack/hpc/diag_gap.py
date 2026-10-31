"""EXPLORATORY diagnosis (not pre-registered): does OT interpolation start to beat the 50/50 mix when the gap is wider?

For gap g = 1..4 and held stage k (g <= k <= 8 - g) of the Pijuan-Sala atlas without blood (cells of atlas_prep.load):
fit = stages k-g and k+g (fit-only preprocessing, atlas_prep.transform), held = stage k; the frozen erythroid methods
(interp_bench_frozen.methods, 3 reps, seeds and call order as atlas_confirm.run_interp, so g = 1 reproduces
tack/results/atlas/confirm.json). Extra numbers per unit (separate random streams, the method draws are untouched):
separation |mu_B - mu_A| / RMS radius(held), ED_U(prev, next), ORACLE alpha-mixture of prev/next, biological-pool floor
of the held stage (one meta 'pool' vs the other pools) with union and ot_pairs on the same pool targets, and per cell type
separation (types with >= 50 cells at k-g, k, k+g). Output ~/tack/out/atlas/diag_gap.json.
"""
import importlib
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_prep as AP  # noqa: E402
import interp_bench_frozen as IB  # noqa: E402

METHODS = ("prev", "next", "union", "indep", "ot_pairs", "sb_half", "uot")
OUT = os.path.expanduser("~/tack/out/atlas/diag_gap.json")
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
MIN_POOL, MIN_TYPE = 100, 50


def ed(P, w, tgt):
    return IB.ed_uv(P, w, tgt)[0]


def unit(X, stage, pool, ct, genes, cc_genes, k, g, reps=3):
    t0 = time.time()
    fit = np.where((stage == k - g) | (stage == k + g))[0]
    held = np.where(stage == k)[0]
    r = AP.transform(X, fit, held, genes, cc_genes)
    # same dtypes as the saved products read by atlas_confirm.run_interp
    Zf = r["Z_fit"].astype(np.float32).astype(np.float64)
    Xf16 = r["X_fit"].astype(np.float16)
    Xf = Xf16.astype(np.float32)
    Zh = r["Z_other"].astype(np.float32).astype(np.float64)
    Xh = r["X_other"].astype(np.float16).astype(np.float32)
    comps = r["comps"].astype(np.float32).astype(np.float64)
    st = stage[fit]
    gl = [x.capitalize() for x in r["genes"]]
    ccm = np.array([x in set(IB.S_GENES + IB.G2M_GENES) for x in gl])
    Xcc = Xf16[:, ccm].astype(np.float64)
    cc_fit = (Xcc - Xcc.mean(0)).mean(1)
    A, B = np.where(st == k - g)[0], np.where(st == k + g)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], Xh[it], S, with_sd=False)
    hpool, hct = pool[held], ct[held]
    pools = [p for p in sorted(set(hpool)) if (hpool == p).sum() >= MIN_POOL]
    ptg = {}
    for p in pools:
        ip = np.where(hpool == p)[0]
        ip = ip[IB.sub(len(ip), seed=3)]
        ptg[p] = IB.Target(Zh[ip], None, S, with_sd=False)
    out = {"k": k, "g": g, "n_prev": int(len(A)), "n_next": int(len(B)), "n_held": int(len(Zh)), "pools": pools, "reps": []}
    keep0 = {}
    for rep in range(reps):
        rng = np.random.default_rng(1000 * k + rep)
        iA, iB = IB.sub(len(A), seed=1000 + rep), IB.sub(len(B), seed=2000 + rep)
        ZA, XA = Zf[A[iA]], Xf[A[iA]].astype(np.float64)
        ZB, XB = Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        M.pop("_plan", None)
        M.pop("_uot", None)
        rec = {}
        for name, (Z, Xm, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, Xm, w, n_pred, rng)
            if name not in METHODS:
                continue
            rec[name] = {"ED_U": ed(Zs, ws, tgt), "ED_U_gene": IB.ed_uv(Xs, ws, tgt, gene=True)[0],
                         "pool_ED_U": {p: ed(Zs, ws, ptg[p]) for p in pools}}
            if rep == 0 and name in ("union", "ot_pairs"):
                keep0[name] = (Zs, ws)
        out["reps"].append(rec)
    out["mean"] = {m: float(np.mean([rp[m]["ED_U"] for rp in out["reps"]])) for m in METHODS}
    out["mean_gene"] = {m: float(np.mean([rp[m]["ED_U_gene"] for rp in out["reps"]])) for m in METHODS}
    # ---- extras (own streams)
    mA, mB, mH = Zf[A].mean(0), Zf[B].mean(0), Zh.mean(0)
    rad = float(np.sqrt(((Zh - mH) ** 2).sum(1).mean()))
    out["separation"] = float(np.linalg.norm(mB - mA) / rad)
    # where the held mean sits on the prev->next chord (0 = prev, 1 = next) and how far off the chord
    d = mB - mA
    tpos = float((mH - mA) @ d / (d @ d))
    out["mean_t"] = tpos
    out["off_chord"] = float(np.linalg.norm(mH - mA - tpos * d) / rad)
    jA, jB = A[IB.sub(len(A), seed=5)], B[IB.sub(len(B), seed=6)]
    out["ED_prev_next"] = ed(Zf[jA], IB.unif(len(jA)), IB.Target(Zf[jB], None, S, with_sd=False))
    P = np.vstack([Zf[jA], Zf[jB]])
    mix = {}
    for a in ALPHAS:
        w = np.r_[np.full(len(jA), a / len(jA)), np.full(len(jB), (1 - a) / len(jB))]
        keep = w > 0
        mix[float(a)] = ed(P[keep], w[keep] / w[keep].sum(), tgt)
    ab = min(mix, key=mix.get)
    out["oracle_mix"] = {"alpha_prev": ab, "ED_U": mix[ab], "alpha_0.5": mix[0.5]}
    # biological-pool floor: one pool vs the other pools of the held stage
    fl = {}
    for p in pools:
        io = np.where(hpool != p)[0]
        if len(io) < MIN_POOL:
            continue
        io = io[IB.sub(len(io), seed=4)]
        fl[p] = {"floor": ed(Zh[io], IB.unif(len(io)), ptg[p]),
                 "union": ed(*keep0["union"], ptg[p]), "ot_pairs": ed(*keep0["ot_pairs"], ptg[p]),
                 "n": int((hpool == p).sum())}
    out["pool_floor"] = fl
    # per cell type separation
    ty = {}
    sA, sB = ct[fit][A], ct[fit][B]
    for c in sorted(set(sA) & set(sB) & set(hct)):
        a_, b_, h_ = Zf[A][sA == c], Zf[B][sB == c], Zh[hct == c]
        if min(len(a_), len(b_), len(h_)) < MIN_TYPE:
            continue
        rr = float(np.sqrt(((h_ - h_.mean(0)) ** 2).sum(1).mean()))
        ty[c] = {"sep": float(np.linalg.norm(b_.mean(0) - a_.mean(0)) / rr), "n": [len(a_), len(h_), len(b_)]}
    out["type_sep"] = ty
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    X, obs, genes = AP.load()
    meta = pd.read_csv(AP.META, sep="\t", usecols=["cell", "pool", "sequencing.batch"])
    obs = obs.merge(meta, on="cell", how="left")
    stage = np.array([AP.STAGES.index(s) for s in obs["stage"]])
    pool = obs["pool"].astype(str).to_numpy()
    ct = obs["celltype"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc_genes = set(str(x).capitalize() for x in list(s_g) + list(g2m))
    res = {"pool_by_stage": {AP.STAGES[i]: sorted(set(pool[stage == i])) for i in range(9)},
           "batch_by_pool": {str(p): sorted(set(obs["sequencing.batch"].astype(str)[pool == p])) for p in sorted(set(pool))},
           "units": []}
    for g in (1, 2, 3, 4):
        for k in range(g, 9 - g):
            u = unit(X, stage, pool, ct, genes, cc_genes, k, g)
            res["units"].append(u)
            print("g", g, "k", k, "sep", round(u["separation"], 2), {m: round(v, 3) for m, v in u["mean"].items()},
                  "oracle_mix", u["oracle_mix"], f"[{u['sec']}s]", flush=True)
            with open(OUT, "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
