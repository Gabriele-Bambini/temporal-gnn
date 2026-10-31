"""AUDIT 4 (uot growth prior, touches only H1b): the proliferation (cc) genes in the interp products.
(a) gene matching: capitalize() vs case-insensitive matching of the Tirosh S/G2M lists against the HVG symbols
    (does mouse-symbol capitalisation lose genes?), number of cc genes among the 2000 HVGs per k, atlas vs erythroid;
(b) quality of the HVG-only score used by atlas_confirm.run_interp: correlation with the 96-gene score that
    atlas_prep stored in interp<k>.npz['cc_fit'] (same fit cells, all kept genes, centred on fit cells);
(c) EXPLORATORY impact: run_interp re-implemented line by line (asserted to reproduce confirm.json bit for bit with the
    frozen HVG score), then uot recomputed with the stored 96-gene score and with no growth prior (cc = 0, i.e. beta 0).
    The other methods are untouched by construction (uot draws come last and uot is never subsampled). H1b recomputed.
Output: audit/cc.json.
"""
import json
import os

import numpy as np

from _common import AC, ATLAS, AUDIT, ERY, IB, confirm, dump, load

CC = set(IB.S_GENES + IB.G2M_GENES)
CC_LOWER = {g.lower() for g in CC}


def hvg_cc(d):
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in CC for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    return (Xcc - Xcc.mean(0)).mean(1), ccm


def run_interp_cc(k, cc_mode):
    """line-by-line copy of atlas_confirm.run_interp; cc_mode in {'frozen', 'stored96', 'none'}; returns uot reps."""
    d = load(ATLAS, f"interp{k}.npz")
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
    comps = d["pca_comps"].astype(np.float64)
    cc_fit, _ = hvg_cc(d)
    if cc_mode == "stored96":
        cc_fit = d["cc_fit"].astype(np.float64)
    elif cc_mode == "none":
        cc_fit = np.zeros(len(st))
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], Xh[it], S, with_sd=False)
    reps = []
    for r in range(3):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        info = M.pop("_uot")
        M.pop("_plan", None)
        rec = {}
        for name, (Z, X, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
            if name in ("uot", "ot_pairs", "union"):
                rec[name] = IB.ed_uv(Zs, ws, tgt)[0]
        rec["growth_cv"] = info["growth_cv"]
        reps.append(rec)
    return reps


def main():
    cj = confirm()
    with open(os.path.join(AUDIT, "confirm_local.json")) as fh:
        cl = json.load(fh)
    out = {"atlas": {}, "ery": {}}
    for k in range(1, 8):
        d = load(ATLAS, f"interp{k}.npz")
        g = d["genes"].astype(str)
        cc_h, ccm = hvg_cc(d)
        ci = np.array([x.lower() in CC_LOWER for x in g])
        r = {"n_cc_hvg_capitalize": int(ccm.sum()), "n_cc_hvg_case_insensitive": int(ci.sum()),
             "cc_hvg_genes": sorted(g[ccm].tolist()), "confirm_cc_genes": cj["interp"][str(k)]["cc_genes"],
             "corr_hvg_vs_stored96": float(np.corrcoef(cc_h, d["cc_fit"].astype(np.float64))[0, 1]),
             "sd_hvg_score": float(cc_h.std()), "sd_stored96": float(d["cc_fit"].std())}
        out["atlas"][k] = r
        print("atlas", k, r, flush=True)
    for k in range(2, 6):
        d = load(ERY, f"interp{k}.npz")
        _, ccm = hvg_cc(d)
        out["ery"][k] = {"n_cc_hvg_capitalize": int(ccm.sum())}
    # exploratory impact on uot
    imp = {}
    for k in range(1, 8):
        row = {}
        for mode in ("frozen", "stored96", "none"):
            reps = run_interp_cc(k, mode)
            row[mode] = {"uot_ED_U": float(np.mean([x["uot"] for x in reps])),
                         "growth_cv": float(np.mean([x["growth_cv"] for x in reps]))}
            if mode == "frozen":  # faithfulness of the copy: bitwise vs the local frozen run, 1e-9 vs the HPC run
                for r_, x in enumerate(reps):
                    for m in ("uot", "ot_pairs", "union"):
                        assert x[m] == cl["interp"][str(k)]["reps"][r_][m]["ED_U"], (k, r_, m)
                        assert abs(x[m] - cj["interp"][str(k)]["reps"][r_][m]["ED_U"]) < 1e-9, (k, r_, m)
                row["copy_reproduces_frozen_run"] = True
        imp[k] = row
        print("uot impact", k, row, flush=True)
    out["uot_impact_exploratory"] = imp
    base = {int(k): {m: v["ED_U"] for m, v in r["mean"].items()} for k, r in cj["interp"].items()}
    out["H1b_exploratory"] = {}
    for mode in ("frozen", "stored96", "none"):
        tab = {k: {**base[k], "uot": imp[k][mode]["uot_ED_U"]} for k in base}
        out["H1b_exploratory"][mode] = AC.beats(tab, "uot", ["prev", "next", "union"], 5)
    print(out["H1b_exploratory"])
    dump("cc.json", out)


if __name__ == "__main__":
    main()
