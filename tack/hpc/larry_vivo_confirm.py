"""Pre-registered LARRY in vivo coupling test (tack/prereg/2026-10-30-larry-invivo-prereg.md): an independent check of
the in vitro finding (larry_robust, exploratory) that, at the SAME spread, the OT coupling puts more mass on true
sister pairs than kNN and than the same kernel without mass constraints.

Units: ~/tack/out/larry_vivo/<cohort>_<mouse>.npz (larry_vivo_prep: day-2 cohort sample -> recipient mouse of the same
cohort, >= 50 shared clones). Couplings, clonal enrichment and spread exactly as in larry_robust:
  ot    balanced entropic OT plan (interp_bench_frozen.plan), eps_rel grid EPS_OT, Sinkhorn marginal error recorded
  soft  row-normalised Gaussian kernel (OT without mass constraints), eps_rel grid EPS_SOFT
  knn   uniform over the K nearest targets, K grid KS
  spread = exp(mean row entropy) = effective number of partners per source cell
Hypotheses; rule "OT beats X" = ratio > 1 in >= ceil(2n/3) of the n units AND geometric mean of the ratios > 1:
  H1 (primary)   OT at eps_rel 0.05 vs kNN interpolated (log-log) at OT's spread
  H2 (primary)   OT at eps_rel 0.05 vs soft interpolated at OT's spread
  H3 (secondary) best converged OT on the grid (marginal error <= TOL) vs best kNN on the grid (both tuned on the
                 truth); a unit without a converged OT plan counts as a loss and is left out of the geometric mean
Descriptive: OT eps 0.05 vs kNN K = 10 (the in vitro H3 comparison); the same H1 / H2 ratios on REPS random 80 %
subsamples; whether OT's spread lies inside the kNN / soft grids. Output ~/tack/out/larry_vivo/larry_vivo_confirm.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_bench_frozen as IB  # noqa: E402
import larry_robust as LR  # noqa: E402

OUT = os.path.expanduser("~/tack/out/larry_vivo")
EPS_OT = (0.005, 0.01, 0.02, 0.05, 0.1)
EPS_SOFT = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
KS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
EPS0 = 0.05
TOL = 1e-3
REPS = 3
FRAC = 0.8


def ot_curve(Za, Zb, ca, cb, grid):
    out = []
    for e in grid:
        IB.STATS["max_marg_err"] = 0.0
        P = IB.plan(Za, Zb, e)[0]
        out.append({"param": e, "spread": LR.spread(P), "enrichment": LR.enrichment(P, ca, cb)[0],
                    "marg_err": IB.STATS["max_marg_err"]})
    return out


def unit(Za, Zb, ca, cb, eps_ot=EPS_OT):
    cur = {"ot": ot_curve(Za, Zb, ca, cb, eps_ot),
           "soft": LR.curve("soft", Za, Zb, ca, cb, EPS_SOFT),
           "knn": LR.curve("knn", Za, Zb, ca, cb, [k for k in KS if k < len(Zb)])}
    ot0 = next(c for c in cur["ot"] if c["param"] == EPS0)
    matched = {}
    for fam in ("knn", "soft"):
        e, inside = LR.at_spread(cur[fam], ot0["spread"])
        matched[fam] = {"at_ot_spread": e, "ratio": ot0["enrichment"] / e, "inside_grid": inside}
    conv = [c for c in cur["ot"] if c["marg_err"] <= TOL]
    best_ot = max(conv, key=lambda c: c["enrichment"]) if conv else None
    best_knn = max(cur["knn"], key=lambda c: c["enrichment"])
    k10 = next(c for c in cur["knn"] if c["param"] == 10)
    return {"curves": cur, "ot0": ot0, "matched": matched, "best_ot": best_ot, "best_knn": best_knn,
            "ratio_best": best_ot["enrichment"] / best_knn["enrichment"] if best_ot else float("nan"),
            "ratio_fixed_knn10": ot0["enrichment"] / k10["enrichment"]}


def rule(ratios):
    r = np.asarray(ratios, float)
    ok = np.isfinite(r)
    need = int(np.ceil(2 * len(r) / 3))
    wins = int((r[ok] > 1).sum())
    g = float(np.exp(np.mean(np.log(r[ok])))) if ok.any() else float("nan")
    return {"wins": wins, "of": int(len(r)), "need": need, "gmean": g, "pass": bool(wins >= need and g > 1)}


def main():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    res = {"units": {}}
    for name in sorted(meta["units"]):
        c = np.load(os.path.join(OUT, f"{name}.npz"), allow_pickle=True)
        Za, Zb = c["Z_a"].astype(np.float64), c["Z_b"].astype(np.float64)
        ca, cb = c["clone_a"], c["clone_b"]
        u = unit(Za, Zb, ca, cb)
        rng = np.random.default_rng(0)
        u["reps"] = []
        for _ in range(REPS):
            ia = np.sort(rng.choice(len(Za), int(FRAC * len(Za)), replace=False))
            ib = np.sort(rng.choice(len(Zb), int(FRAC * len(Zb)), replace=False))
            v = unit(Za[ia], Zb[ib], ca[ia], cb[ib], eps_ot=(EPS0,))
            u["reps"].append({f: v["matched"][f]["ratio"] for f in ("knn", "soft")})
        res["units"][name] = u
        print(name, "ot0", (round(u["ot0"]["spread"], 1), round(u["ot0"]["enrichment"], 2), f"{u['ot0']['marg_err']:.1e}"),
              "H1", round(u["matched"]["knn"]["ratio"], 2), "H2", round(u["matched"]["soft"]["ratio"], 2),
              "H3", round(u["ratio_best"], 2), "fixedK10", round(u["ratio_fixed_knn10"], 2),
              "reps", [(round(r["knn"], 2), round(r["soft"], 2)) for r in u["reps"]], flush=True)
        for f, cur in u["curves"].items():
            print("  ", f, [(x["param"], round(x["spread"], 1), round(x["enrichment"], 2)) for x in cur], flush=True)
    U = list(res["units"].values())
    res["verdicts"] = {
        "H1_ot_vs_knn_matched": rule([u["matched"]["knn"]["ratio"] for u in U]),
        "H2_ot_vs_soft_matched": rule([u["matched"]["soft"]["ratio"] for u in U]),
        "H3_best_ot_vs_best_knn": rule([u["ratio_best"] for u in U]),
        "desc_ot_vs_knn10": rule([u["ratio_fixed_knn10"] for u in U]),
        "desc_all_inside_grid": all(m["inside_grid"] for u in U for m in u["matched"].values()),
        "desc_ot0_max_marg_err": max(u["ot0"]["marg_err"] for u in U),
    }
    print("VERDICTS", json.dumps(res["verdicts"]), flush=True)
    with open(os.path.join(OUT, "larry_vivo_confirm.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
