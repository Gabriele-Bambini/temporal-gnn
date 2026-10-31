"""EXPLORATORY robustness check of the pre-registered LARRY coupling result (H3: OT > kNN, 3/4 units, job 726981).
Not a pre-registration: the H3 verdict stands as frozen. Question: does the OT win depend on the chosen settings
(eps_rel 0.05, K = 10), or on how spread the coupling is (betting on fewer partners), rather than on WHICH partners
OT picks?

Same units and cells as larry_confirm (couple_<ab>.npz, 2 populations x 2 transitions), same clonal enrichment.
Coupling families, each on a grid:
  ot    balanced entropic OT plan (interp_bench_frozen.plan), eps_rel in EPS
  soft  row-normalised Gaussian kernel exp(-C / (eps_rel median C)): OT without the mass constraints (= the first
        Sinkhorn half-step), i.e. a soft kNN; same EPS grid
  knn   uniform over the K nearest targets, K in KS
Spread of a coupling = exp(mean row entropy) = effective number of partners per source cell (exactly K for knn).
Readouts per unit:
  (a) curves enrichment vs spread for the three families
  (b) matched spread: OT at the frozen eps_rel 0.05 vs knn / soft interpolated (log-log) at OT's spread
  (c) best of grid per family (tuned on the clonal truth itself: symmetric, optimistic for every family)
  (d) stability: (b) on REPS random 80 % subsamples of source and target cells
Reading fixed before running:
  OT vs knn  ROBUST           (b) OT / knn > 1 in >= 3 of 4 units and on the geometric mean, AND (c) best ot > best
                              knn in >= 3 of 4 units
             SPREAD ARTEFACT  (b) OT / knn <= 1 in >= 2 of 4 units (the frozen win came from the spread, not the picks)
             MIXED            otherwise
  mass constraint helps       (b) OT / soft > 1 in >= 3 of 4 units (the transport part, not local similarity, helps)
Output ~/tack/out/larry/larry_robust.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_bench_frozen as IB  # noqa: E402
from larry_confirm import OUT, POPS, enrichment  # noqa: E402,F401

EPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
EPS_FROZEN = 0.05
KS = (1, 2, 3, 5, 10, 20, 50, 100, 200)
REPS = 3
FRAC = 0.8


def spread(P):
    R = P / P.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(R > 0, R * np.log(R), 0.0).sum(1)
    return float(np.exp(H.mean()))


def knn_plan(Za, Zb, k):
    _, ind = NearestNeighbors(n_neighbors=k).fit(Zb).kneighbors(Za)
    P = np.zeros((len(Za), len(Zb)))
    np.put_along_axis(P, ind, 1.0 / k, axis=1)
    return P


def soft_plan(Za, Zb, eps_rel):
    C = IB.sqdist(Za, Zb)
    L = -C / (eps_rel * np.median(C))
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def make_plan(fam, Za, Zb, g):
    if fam == "ot":
        return IB.plan(Za, Zb, g)[0]
    return soft_plan(Za, Zb, g) if fam == "soft" else knn_plan(Za, Zb, g)


def curve(fam, Za, Zb, ca, cb, grid):
    out = []
    for g in grid:
        P = make_plan(fam, Za, Zb, g)
        out.append({"param": g, "spread": spread(P), "enrichment": enrichment(P, ca, cb)[0]})
    return out


def at_spread(cur, s):
    """Log-log interpolation of the enrichment at spread s along a grid curve; flag False if s is off the grid
    (then the value is clamped to the nearest end)."""
    xs = np.log([c["spread"] for c in cur])
    ys = np.log([max(c["enrichment"], 1e-12) for c in cur])
    o = np.argsort(xs)
    x = np.log(s)
    return float(np.exp(np.interp(x, xs[o], ys[o]))), bool(xs.min() <= x <= xs.max())


def run_unit(Za, Zb, ca, cb, ot_grid=EPS):
    cur = {"ot": curve("ot", Za, Zb, ca, cb, ot_grid), "soft": curve("soft", Za, Zb, ca, cb, EPS),
           "knn": curve("knn", Za, Zb, ca, cb, KS)}
    ot0 = next(c for c in cur["ot"] if c["param"] == EPS_FROZEN)
    matched = {}
    for fam in ("knn", "soft"):
        e, inside = at_spread(cur[fam], ot0["spread"])
        matched[fam] = {"enrichment_at_ot_spread": e, "ratio_ot": ot0["enrichment"] / e, "inside_grid": inside}
    best = {f: max(cur[f], key=lambda c: c["enrichment"]) for f in cur}
    return {"curves": cur, "ot_frozen": ot0, "matched": matched, "best": best}


def gmean(r):
    return float(np.exp(np.mean(np.log(r))))


def main():
    res = {"units": {}}
    for pop in POPS:
        for ab in ("24", "46"):
            c = np.load(os.path.join(OUT, pop, f"couple_{ab}.npz"), allow_pickle=True)
            Za, Zb = c["Z_a"].astype(np.float64), c["Z_b"].astype(np.float64)
            ca, cb = c["clone_a"], c["clone_b"]
            u = run_unit(Za, Zb, ca, cb)
            k10 = next(x for x in u["curves"]["knn"] if x["param"] == 10)
            u["check_frozen"] = {"ot_eps0.05": u["ot_frozen"]["enrichment"], "knn_K10": k10["enrichment"]}
            rng = np.random.default_rng(0)
            u["reps"] = []
            for _ in range(REPS):
                ia = np.sort(rng.choice(len(Za), int(FRAC * len(Za)), replace=False))
                ib = np.sort(rng.choice(len(Zb), int(FRAC * len(Zb)), replace=False))
                v = run_unit(Za[ia], Zb[ib], ca[ia], cb[ib], ot_grid=(EPS_FROZEN,))
                u["reps"].append({"ot_spread": v["ot_frozen"]["spread"],
                                  **{f"ratio_ot_{f}": v["matched"][f]["ratio_ot"] for f in ("knn", "soft")}})
            name = f"{pop}_{ab}"
            res["units"][name] = u
            print(name, "check", {k: round(v, 2) for k, v in u["check_frozen"].items()},
                  "ot_spread", round(u["ot_frozen"]["spread"], 1),
                  "ratio_ot_knn", round(u["matched"]["knn"]["ratio_ot"], 2),
                  "ratio_ot_soft", round(u["matched"]["soft"]["ratio_ot"], 2),
                  "best", {f: (b["param"], round(b["enrichment"], 2)) for f, b in u["best"].items()},
                  "reps_knn", [round(r["ratio_ot_knn"], 2) for r in u["reps"]], flush=True)
            for f, cur in u["curves"].items():
                print("  ", f, [(x["param"], round(x["spread"], 1), round(x["enrichment"], 2)) for x in cur], flush=True)
    U = list(res["units"].values())
    rk = [u["matched"]["knn"]["ratio_ot"] for u in U]
    rs = [u["matched"]["soft"]["ratio_ot"] for u in U]
    bk = [bool(u["best"]["ot"]["enrichment"] > u["best"]["knn"]["enrichment"]) for u in U]
    robust = sum(r > 1 for r in rk) >= 3 and gmean(rk) > 1 and sum(bk) >= 3
    artefact = sum(r <= 1 for r in rk) >= 2
    res["reading"] = {"ratio_ot_knn": rk, "gmean_ot_knn": gmean(rk), "best_ot_gt_best_knn": bk,
                      "ot_vs_knn": "ROBUST" if robust else ("SPREAD ARTEFACT" if artefact else "MIXED"),
                      "ratio_ot_soft": rs, "gmean_ot_soft": gmean(rs),
                      "mass_constraint_helps": bool(sum(r > 1 for r in rs) >= 3),
                      "all_inside_grid": all(m["inside_grid"] for u in U for m in u["matched"].values())}
    print("READING", json.dumps(res["reading"]), flush=True)
    with open(os.path.join(OUT, "larry_robust.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
