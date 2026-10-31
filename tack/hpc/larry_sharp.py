"""POST HOC extension of larry_robust (written after its fixed reading came out MIXED, job 728038): the OT grid stopped
at eps_rel 0.01 (4-22 effective partners per cell) while kNN reached 1 partner, so 'best of grid' compared a sharp kNN
with a blunt OT. Here: OT at eps_rel 0.005 / 0.002 / 0.001 on the same 4 units with the same clonal enrichment, Sinkhorn
marginal error reported (small eps may not converge in 2000 steps), kNN at matched spread (K 1-200). Descriptive only:
the larry_robust reading (MIXED) stands. Output ~/tack/out/larry/larry_sharp.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_bench_frozen as IB  # noqa: E402
import larry_robust as LR  # noqa: E402

EPS_SHARP = (0.005, 0.002, 0.001)


def main():
    res = {}
    for pop in LR.POPS:
        for ab in ("24", "46"):
            c = np.load(os.path.join(LR.OUT, pop, f"couple_{ab}.npz"), allow_pickle=True)
            Za, Zb = c["Z_a"].astype(np.float64), c["Z_b"].astype(np.float64)
            ca, cb = c["clone_a"], c["clone_b"]
            rows = []
            for e in EPS_SHARP:
                IB.STATS["max_marg_err"], IB.STATS["log_fallbacks"] = 0.0, 0
                P = IB.plan(Za, Zb, e)[0]
                rows.append({"eps_rel": e, "spread": LR.spread(P), "enrichment": LR.enrichment(P, ca, cb)[0],
                             "marg_err": IB.STATS["max_marg_err"], "log_fallback": IB.STATS["log_fallbacks"] > 0})
            knn = LR.curve("knn", Za, Zb, ca, cb, LR.KS)
            for r in rows:
                e, inside = LR.at_spread(knn, r["spread"])
                r.update({"knn_at_spread": e, "ratio_ot_knn": r["enrichment"] / e, "inside_grid": inside})
            name = f"{pop}_{ab}"
            res[name] = {"ot": rows, "knn": knn,
                         "best_ot_sharp": max(r["enrichment"] for r in rows),
                         "best_knn": max(k["enrichment"] for k in knn)}
            print(name, [(r["eps_rel"], round(r["spread"], 1), round(r["enrichment"], 2), round(r["ratio_ot_knn"], 2),
                          f"{r['marg_err']:.1e}", r["log_fallback"]) for r in rows],
                  "best_ot_sharp", round(res[name]["best_ot_sharp"], 2), "best_knn", round(res[name]["best_knn"], 2),
                  flush=True)
    with open(os.path.join(LR.OUT, "larry_sharp.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
