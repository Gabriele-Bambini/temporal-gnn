"""EXPLORATORY (not pre-registered): are the OT midpoint clouds under-dispersed (entropic contraction), and does a
sharper plan (smaller eps_rel) change the ot_pairs vs union comparison? ORACLE-labelled: eps_rel is scanned against the
held stage, so no setting here is a valid method.

Units: each full held stage k = 1..7 (3 reps, frozen subsample seeds 1000+r / 2000+r, target seed 1) and each atlas
cell type with >= 100 cells at k-1, k, k+1 (rep 0 only). Per unit:
  spread_ratio  RMS distance to the (weighted) mean of a prediction / that of the held target (IB convention):
                prev, next, union, indep, ot_pairs, uot (frozen methods, rep rng 1000k+r) and the held cells outside
                the target subsample (reference).
  eps scan      ot_pairs with eps_rel in EPS (own rng; 0.05 is the frozen value): ED_U, spread ratio and the plan's
                median effective number of partners per source cell exp(H(P_i. / r_i)).
Usage: python spread_eps.py   -> spread_eps.json
"""
import os
import time

import numpy as np

import common as C

IB = C.IB
EPS = [0.01, 0.02, 0.05, 0.1, 0.2]
MIN_N = 100


def spread(Z, w, tgt):
    mu = w @ Z
    return float(np.sqrt(w @ ((Z - mu) ** 2).sum(1)) / np.sqrt(((tgt.Z - tgt.Z.mean(0)) ** 2).sum(1).mean()))


def unit(d, k, fk, hk, reps):
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh = d["Z_held"].astype(np.float64)[hk]
    comps = d["pca_comps"].astype(np.float64)
    cc_fit, _ = C.cc_score(d)
    A, B = np.where((st == k - 1) & fk)[0], np.where((st == k + 1) & fk)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], None, S, with_sd=False)
    rest = np.setdiff1d(np.arange(len(Zh)), it)
    recs = []
    for r in range(reps):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        M.pop("_plan", None)
        M.pop("_uot", None)
        rec = {"spread": {}, "ED_U": {}, "eps": {}}
        for name, (Z, X, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
            if name in ("prev", "next", "union", "indep", "ot_pairs", "uot", "sb_half"):
                rec["spread"][name] = spread(Zs, ws, tgt)
                rec["ED_U"][name] = IB.ed_uv(Zs, ws, tgt)[0]
        if len(rest) > 1:
            rec["spread"]["held_rest"] = spread(Zh[rest], IB.unif(len(rest)), tgt)
        rg = np.random.default_rng(70_000 + 100 * k + r)
        for e in EPS:
            P, _ = IB.plan(ZA, ZB, e)
            rowp = P / P.sum(1, keepdims=True)
            H = -(rowp * np.log(np.maximum(rowp, 1e-300))).sum(1)
            i, j = IB.draw_pairs(P, n_pred, rg)
            Zp = 0.5 * ZA[i] + 0.5 * ZB[j]
            rec["eps"][str(e)] = {"ED_U": IB.ed_uv(Zp, IB.unif(n_pred), tgt)[0], "spread": spread(Zp, IB.unif(n_pred), tgt),
                                  "median_eff_partners": float(np.median(np.exp(H)))}
        recs.append(rec)
    avg = lambda f: float(np.mean([f(x) for x in recs]))  # noqa: E731
    out = {"n": [int(len(A)), int(len(Zh)), int(len(B))], "reps": reps,
           "spread": {m: avg(lambda x: x["spread"][m]) for m in recs[0]["spread"]},
           "ED_U": {m: avg(lambda x: x["ED_U"][m]) for m in recs[0]["ED_U"]},
           "eps": {e: {q: avg(lambda x: x["eps"][e][q]) for q in ("ED_U", "spread", "median_eff_partners")} for e in recs[0]["eps"]}}
    be = min(out["eps"], key=lambda e: out["eps"][e]["ED_U"])
    out["ORACLE_best_eps"] = be
    out["ORACLE_best_eps_ED_U"] = out["eps"][be]["ED_U"]
    out["ORACLE_best_eps_beats_union"] = bool(out["eps"][be]["ED_U"] < out["ED_U"]["union"])
    return out


def main():
    t0 = time.time()
    res = {"script": "spread_eps.py", "exploratory": True,
           "note": "EXPLORATORY; eps scan is ORACLE-labelled; nothing here changes the pre-registered verdicts.",
           "full": {}, "types": []}
    for k in range(1, 8):
        d = C.load_interp(k)
        st, ctf, cth = d["stage_fit"], d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
        res["full"][k] = unit(d, k, np.ones(len(st), bool), np.ones(len(cth), bool), 3)
        f = res["full"][k]
        print(f"k{k} full spread " + " ".join(f"{m}={v:.3f}" for m, v in f["spread"].items()) +
              " | eps " + " ".join(f"{e}:{v['ED_U']:.3f}/{v['spread']:.2f}/{v['median_eff_partners']:.0f}" for e, v in f["eps"].items()) +
              f" | union {f['ED_U']['union']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        for t in sorted(set(ctf) | set(cth)):
            if min(((st == k - 1) & (ctf == t)).sum(), (cth == t).sum(), ((st == k + 1) & (ctf == t)).sum()) < MIN_N:
                continue
            u = unit(d, k, ctf == t, cth == t, 1)
            u.update({"k": k, "type": t})
            res["types"].append(u)
            print(f"   k{k} {t[:24]:24s} spread ot {u['spread']['ot_pairs']:.3f} union {u['spread']['union']:.3f} "
                  f"held_rest {u['spread'].get('held_rest', float('nan')):.3f} | best eps {u['ORACLE_best_eps']} "
                  f"{u['ORACLE_best_eps_ED_U']:.3f} vs union {u['ED_U']['union']:.3f}", flush=True)
        C.dump(os.path.join(C.HERE, "spread_eps.json"), res)
    fl = res["full"]
    res["summary"] = {
        "full_n_k_ot_pairs_spread_below_union": int(sum(fl[k]["spread"]["ot_pairs"] < fl[k]["spread"]["union"] for k in fl)),
        "full_mean_spread": {m: float(np.mean([fl[k]["spread"][m] for k in fl])) for m in fl[1]["spread"]},
        "full_n_k_ORACLE_best_eps_beats_union": int(sum(fl[k]["ORACLE_best_eps_beats_union"] for k in fl)),
        "full_ORACLE_best_eps": {k: fl[k]["ORACLE_best_eps"] for k in fl},
        "types_n": len(res["types"]),
        "types_n_ot_pairs_spread_below_union": int(sum(u["spread"]["ot_pairs"] < u["spread"]["union"] for u in res["types"])),
        "types_n_ORACLE_best_eps_beats_union": int(sum(u["ORACLE_best_eps_beats_union"] for u in res["types"])),
        "types_median_spread": {m: float(np.median([u["spread"][m] for u in res["types"]])) for m in ("prev", "next", "union", "ot_pairs", "uot", "indep")},
    }
    res["seconds"] = round(time.time() - t0, 1)
    C.dump(os.path.join(C.HERE, "spread_eps.json"), res)
    print(res["summary"])


if __name__ == "__main__":
    main()
