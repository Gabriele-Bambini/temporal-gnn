"""EXPLORATORY (not pre-registered): does OT interpolation beat the 50/50 union WITHIN lineages of the atlas?

Units, per held stage k = 1..7 (interp<k>.npz, the product's own PCA space, nothing refitted):
  type   one atlas cell type with >= MIN_N cells at each of k-1, k, k+1 (fit AND target restricted to that type)
  group  a coarse group of common.GROUPS (defined from the type names) with >= MIN_N cells at each of the 3 stages
  shared all cells of the types present (>= MIN_PRESENT cells) at all three stages, i.e. the stage minus the
         one-sided (appearing / disappearing) types
For every unit the frozen IB.methods + IB scoring run through common.run_interp_subset (mirror of
atlas_confirm.run_interp: rng 1000k+r, subsample seeds 1000+r / 2000+r, target seed 1, n_pred = min(n_held, 1500),
proliferation score centred on ALL fit cells as frozen). ED_U is compared only within a unit (the scale differs).
Also per unit: ED_U(prev, next) on the rep-0 subsamples (the union's convexity bonus is 1/4 of it).
Usage: python within_lineage.py [--reps 3] [--min-n 100]   -> within_lineage.json
"""
import argparse
import os
import time

import numpy as np

import common as C

IB = C.IB
MIN_PRESENT = 10


def summary(units, fam):
    us = [u for u in units if u["family"] == fam]
    if not us:
        return None
    g = lambda u, m: u["mean"][m]["ED_U"]  # noqa: E731
    out = {"n_units": len(us)}
    for a in ("ot_pairs", "uot", "sb_half"):
        out[f"{a}<union"] = int(sum(g(u, a) < g(u, "union") for u in us))
        out[f"{a}_beats_prev_next_union"] = int(sum(all(g(u, a) < g(u, b) for b in ("prev", "next", "union")) for u in us))
        out[f"median_{a}/union"] = float(np.median([g(u, a) / g(u, "union") for u in us]))
    out["ot_pairs<indep"] = int(sum(g(u, "ot_pairs") < g(u, "indep") for u in us))
    out["best_method_counts"] = {}
    for u in us:
        b = min(C.METHODS, key=lambda m: g(u, m))
        out["best_method_counts"][b] = out["best_method_counts"].get(b, 0) + 1
    out["units_ot_pairs<union"] = [f"k{u['k']}:{u['name']}" for u in us if g(u, "ot_pairs") < g(u, "union")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--ks", type=int, nargs="+", default=list(range(1, 8)))
    args = ap.parse_args()
    t0 = time.time()
    units = []
    for k in args.ks:
        d = C.load_interp(k)
        st = d["stage_fit"]
        ctf, cth = d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
        Zf = d["Z_fit"].astype(np.float64)
        S = IB.med_scale(Zf)
        types = sorted(set(ctf) | set(cth))
        cnt = {t: (int(((st == k - 1) & (ctf == t)).sum()), int((cth == t).sum()), int(((st == k + 1) & (ctf == t)).sum()))
               for t in types}
        todo = [("type", t, [t]) for t in types if min(cnt[t]) >= args.min_n]
        for gname, mem in C.GROUPS.items():
            c3 = [sum(cnt.get(t, (0, 0, 0))[i] for t in mem) for i in range(3)]
            if min(c3) >= args.min_n:
                todo.append(("group", gname, mem))
        shared = [t for t in types if min(cnt[t]) >= MIN_PRESENT]
        todo.append(("shared", "shared_types", shared))
        for fam, name, mem in todo:
            fk, hk = np.isin(ctf, mem), np.isin(cth, mem)
            o = C.run_interp_subset(d, k, fit_keep=fk, held_keep=hk, reps=args.reps, pools=False)
            A, B = np.where((st == k - 1) & fk)[0], np.where((st == k + 1) & fk)[0]
            ZA, ZB = Zf[A[IB.sub(len(A), seed=1000)]], Zf[B[IB.sub(len(B), seed=2000)]]
            eab = IB.ed_uv(ZA, IB.unif(len(ZA)), IB.Target(ZB, None, S, with_sd=False))[0]
            u = {"family": fam, "k": k, "held": C.STAGES[k], "name": name, "n_prev": o["n_prev"], "n_held": o["n_held"],
                 "n_next": o["n_next"], "mean": o["mean"], "sd_ED_U": o["sd"], "ED_U_prev_next_rep0": eab,
                 "quarter_ED_prev_next": 0.25 * eab}
            if fam == "shared":
                u["types"] = mem
                u["held_frac_kept"] = float(hk.mean())
                u["fit_frac_kept_prev"] = float(fk[st == k - 1].mean())
                u["fit_frac_kept_next"] = float(fk[st == k + 1].mean())
            units.append(u)
            print(f"k{k} {fam:6s} {name[:28]:28s} n={o['n_prev']}/{o['n_held']}/{o['n_next']} " +
                  " ".join(f"{m}={o['mean'][m]['ED_U']:.3f}" for m in ("prev", "next", "union", "ot_pairs", "uot", "indep")) +
                  f" [{time.time() - t0:.0f}s]", flush=True)
    out = {"script": "within_lineage.py", "exploratory": True, "reps": args.reps, "min_n": args.min_n,
           "note": "EXPLORATORY, not pre-registered; does not change the pre-registered verdicts (H1/H1b FAIL).",
           "summary": {fam: summary(units, fam) for fam in ("type", "group", "shared")}, "units": units,
           "seconds": round(time.time() - t0, 1)}
    C.dump(os.path.join(C.HERE, "within_lineage.json"), out)
    print(out["summary"])


if __name__ == "__main__":
    main()
