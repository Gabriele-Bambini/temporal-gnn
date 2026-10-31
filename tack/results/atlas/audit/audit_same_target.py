"""AUDIT 6 (scoring symmetry): did every method see the same target subsample, the same pools and the same
prediction size? IB.ed_uv is wrapped (pass-through, result unchanged) while the FROZEN atlas_confirm.run_interp and
run_forecast run; every call records (target object, gene flag, n points, weight sum, max weight). Checks per unit:
one held-stage target object for every method's ED_U and ED_U_gene, the target = the seed-1 subsample of the held /
future stage, one target object per pool, every prediction has n_pred points (interp) or the same seed-0 subsample
of the stage-s cells (forecast), weights sum to 1. Output: audit/same_target.json.
"""
import numpy as np

from _common import AC, ATLAS, IB, confirm, dump, load

CALLS = []
_orig = IB.ed_uv


def wrapped(P, wp, tgt, gene=False):
    CALLS.append({"tgt": id(tgt), "gene": gene, "n": int(len(P)), "wsum": float(np.sum(wp)), "wmax": float(np.max(wp)),
                  "n_tgt": int(len(tgt.Z)), "P0": float(np.asarray(P, np.float64)[0, 0])})
    return _orig(P, wp, tgt, gene)


IB.ed_uv = wrapped


def interp(k, cj):
    CALLS.clear()
    r = AC.run_interp(ATLAS, k)
    d = load(ATLAS, f"interp{k}.npz")
    n_held = len(d["Z_held"])
    n_pred = min(n_held, IB.NSUB)
    pools = r["pools"]
    per = 2 + len(pools)  # ED_U, ED_U_gene, one per pool, for each of the 7 scored methods
    # the frozen loop calls ed_uv only for the 7 INTERP_METHODS, in M order, 3 reps
    assert len(CALLS) == 3 * 7 * per, (k, len(CALLS))
    main_t = {c["tgt"] for i, c in enumerate(CALLS) if i % per in (0, 1)}
    pool_t = [{CALLS[j * per + 2 + q]["tgt"] for j in range(len(CALLS) // per)} for q in range(len(pools))]
    out = {"n_calls": len(CALLS), "one_main_target": len(main_t) == 1,
           "main_target_size": CALLS[0]["n_tgt"], "expected_target_size": min(n_held, IB.NSUB),
           "one_target_per_pool": all(len(s) == 1 for s in pool_t),
           "pool_target_sizes": [CALLS[2 + q]["n_tgt"] for q in range(len(pools))],
           "all_pred_n": sorted({c["n"] for c in CALLS}), "expected_n_pred": n_pred,
           "max_wsum_dev": max(abs(c["wsum"] - 1) for c in CALLS),
           "matches_confirm": all(abs(r["mean"][m]["ED_U"] - cj["interp"][str(k)]["mean"][m]["ED_U"]) < 1e-9
                                  for m in AC.INTERP_METHODS)}
    smp = d["sample_held"].astype(str)
    out["pool_sizes_ok"] = out["pool_target_sizes"] == [int((smp == p).sum()) for p in pools]
    out["ok"] = (out["one_main_target"] and out["main_target_size"] == out["expected_target_size"]
                 and out["one_target_per_pool"] and out["all_pred_n"] == [n_pred] and out["max_wsum_dev"] < 1e-9
                 and out["matches_confirm"] and out["pool_sizes_ok"])
    return out


def forecast(s, cj):
    """frozen run_forecast call order: per family (persist 1, meanshift 1, uot_rw 13, tilt_rw 6 grid points):
    back.score per grid point (ED_U, ED_U_gene), then final.score (ED_U, ED_U_gene, one per pool)."""
    CALLS.clear()
    r = AC.run_forecast(ATLAS, s)
    d = load(ATLAS, f"origin{s}.npz")
    n_pools = len([p for p in set(d["sample_fut"].astype(str)) if (d["sample_fut"].astype(str) == p).sum() >= 20])
    sizes = (1, 1, len(AC.UOT_GRID), len(AC.TILT_GRID))
    assert len(CALLS) == 2 * sum(sizes) + 4 * (2 + n_pools), (s, len(CALLS))
    back, final, pools, i = [], [], [], 0
    for fam, g in zip(("persist", "meanshift", "uot_rw", "tilt_rw"), sizes):
        back += CALLS[i:i + 2 * g]
        i += 2 * g
        final.append((fam, CALLS[i:i + 2]))
        pools.append(CALLS[i + 2:i + 2 + n_pools])
        i += 2 + n_pools
    n_now = int((d["stage_past"] == s).sum())
    n_prev = int((d["stage_past"] == s - 1).sum())
    f_ids = {c["tgt"] for _, cs in final for c in cs}
    out = {"n_calls": len(CALLS), "back_one_target": len({c["tgt"] for c in back}) == 1,
           "final_one_target": len(f_ids) == 1, "back_ne_final": not ({c["tgt"] for c in back} & f_ids),
           "pools_one_target_each": all(len({pl[q]["tgt"] for pl in pools}) == 1 for q in range(n_pools)),
           "back_pred_n": sorted({c["n"] for c in back}), "final_pred_n": sorted({c["n"] for _, cs in final for c in cs}),
           "expected_back_n": min(n_prev, IB.NSUB), "expected_final_n": min(n_now, IB.NSUB),
           "final_target_n": final[0][1][0]["n_tgt"], "expected_target_n": min(len(d["Z_fut"]), IB.NSUB),
           "persist_uotrw_tilt_same_first_point": len({cs[0]["P0"] for fam, cs in final if fam != "meanshift"}) == 1,
           "max_wsum_dev": max(abs(c["wsum"] - 1) for c in CALLS),
           "matches_confirm": all(abs(r[f]["ED_U"] - cj["forecast"][str(s)][f]["ED_U"]) < 1e-9 for f in r)}
    out["ok"] = (out["back_one_target"] and out["final_one_target"] and out["back_ne_final"]
                 and out["pools_one_target_each"] and out["back_pred_n"] == [out["expected_back_n"]]
                 and out["final_pred_n"] == [out["expected_final_n"]] and out["final_target_n"] == out["expected_target_n"]
                 and out["persist_uotrw_tilt_same_first_point"] and out["max_wsum_dev"] < 1e-9 and out["matches_confirm"])
    return out


def main():
    cj = confirm()
    res = {"interp": {}, "forecast": {}}
    for k in range(1, 8):
        res["interp"][k] = interp(k, cj)
        print("interp", k, res["interp"][k], flush=True)
    for s in range(2, 8):
        res["forecast"][s] = forecast(s, cj)
        print("forecast", s, res["forecast"][s], flush=True)
    res["ALL_OK_interp"] = all(v["ok"] for v in res["interp"].values())
    res["ALL_OK_forecast"] = all(v["ok"] for v in res["forecast"].values())
    print("ALL_OK_interp", res["ALL_OK_interp"], "ALL_OK_forecast", res["ALL_OK_forecast"])
    dump("same_target.json", res)


if __name__ == "__main__":
    main()
