"""Pre-registered LARRY cytokine-perturbation coupling test (tack/prereg/2026-10-30-larry-cytokine-prereg.md): the same
frozen matched-spread test as larry_vivo_confirm (unit() and rule() imported unchanged, same grids), on the 9 units of
larry_cyto_prep (~/tack/out/larry_cyto/<condition>_<a><b>.npz: day 2 -> day 4 per condition, day 4 -> day 6 within a
condition).
  H1 (primary)   OT at eps_rel 0.05 vs kNN interpolated (log-log) at OT's spread
  H2 (primary)   OT at eps_rel 0.05 vs soft interpolated at OT's spread
  H3 (secondary) best converged OT on the grid vs best kNN on the grid
Rule: ratio > 1 in >= ceil(2n/3) units AND geometric mean > 1. Descriptive as in vivo, plus every rule split by
transition type (day 2 -> 4 units share one source sample; day 4 -> 6 units do not). Output
~/tack/out/larry_cyto/larry_cyto_confirm.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import larry_vivo_confirm as VC  # noqa: E402

OUT = os.path.expanduser("~/tack/out/larry_cyto")


def verdicts(units):
    U = list(units.values())
    return {
        "H1_ot_vs_knn_matched": VC.rule([u["matched"]["knn"]["ratio"] for u in U]),
        "H2_ot_vs_soft_matched": VC.rule([u["matched"]["soft"]["ratio"] for u in U]),
        "H3_best_ot_vs_best_knn": VC.rule([u["ratio_best"] for u in U]),
        "desc_ot_vs_knn10": VC.rule([u["ratio_fixed_knn10"] for u in U]),
        "desc_all_inside_grid": all(m["inside_grid"] for u in U for m in u["matched"].values()),
        "desc_ot0_max_marg_err": max(u["ot0"]["marg_err"] for u in U),
    }


def main():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    res = {"units": {}}
    for name in sorted(meta["units"]):
        c = np.load(os.path.join(OUT, f"{name}.npz"), allow_pickle=True)
        Za, Zb = c["Z_a"].astype(np.float64), c["Z_b"].astype(np.float64)
        ca, cb = c["clone_a"], c["clone_b"]
        u = VC.unit(Za, Zb, ca, cb)
        rng = np.random.default_rng(0)
        u["reps"] = []
        for _ in range(VC.REPS):
            ia = np.sort(rng.choice(len(Za), int(VC.FRAC * len(Za)), replace=False))
            ib = np.sort(rng.choice(len(Zb), int(VC.FRAC * len(Zb)), replace=False))
            v = VC.unit(Za[ia], Zb[ib], ca[ia], cb[ib], eps_ot=(VC.EPS0,))
            u["reps"].append({f: v["matched"][f]["ratio"] for f in ("knn", "soft")})
        res["units"][name] = u
        print(name, "ot0", (round(u["ot0"]["spread"], 1), round(u["ot0"]["enrichment"], 2), f"{u['ot0']['marg_err']:.1e}"),
              "H1", round(u["matched"]["knn"]["ratio"], 2), "H2", round(u["matched"]["soft"]["ratio"], 2),
              "H3", round(u["ratio_best"], 2), "fixedK10", round(u["ratio_fixed_knn10"], 2),
              "reps", [(round(r["knn"], 2), round(r["soft"], 2)) for r in u["reps"]], flush=True)
        for f, cur in u["curves"].items():
            print("  ", f, [(x["param"], round(x["spread"], 1), round(x["enrichment"], 2)) for x in cur], flush=True)
    res["verdicts"] = verdicts(res["units"])
    res["by_transition"] = {t: verdicts({n: u for n, u in res["units"].items() if n.endswith(t)})
                            for t in ("_24", "_46")}
    print("VERDICTS", json.dumps(res["verdicts"]), flush=True)
    print("BY_TRANSITION", json.dumps(res["by_transition"]), flush=True)
    with open(os.path.join(OUT, "larry_cyto_confirm.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
