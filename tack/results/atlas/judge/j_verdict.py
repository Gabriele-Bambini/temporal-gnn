"""JUDGE (atlas confirmation): independent recomputation of the five pre-registered verdicts from confirm.json.

Not exploratory: (1) LF-normalised sha256 of the four frozen files vs the pre-registration; (2) per-unit ED_U = mean of
the 3 reps; (3) the prereg rule ("strictly lower in >= ceil(2n/3) units AND lower mean, against every listed B"), read
both per baseline and jointly (a unit counts only if A beats all B in it, as the frozen beats()); (4) stored verdicts.
Secondary (prereg-listed, not used for the verdict): gene-space ED_U, per-pool ED_U.
EXPLORATORY facts recorded for the report: H1c triviality (indep vs union), sb_half vs union, uot_rw == persist ties,
rep-level ot_pairs - union signs.
Usage: python j_verdict.py   (reads ../confirm.json, writes j_verdict.json next to this file)
"""
import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PREREG = os.path.join(ROOT, "tack", "prereg", "2026-10-25-atlas-confirmation-prereg.md")
FROZEN = ["atlas_prep.py", "atlas_confirm.py", "interp_bench_frozen.py", "ery_forecast.py"]
C = json.load(open(os.path.join(HERE, "..", "confirm.json")))


def sha_lf(path):
    b = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def rule(vals, a, bs):
    """vals[unit][method] -> per-baseline and joint reading of the prereg rule."""
    units = sorted(vals, key=int)
    n = len(units)
    need = math.ceil(2 * n / 3)
    per_b = {}
    for b in bs:
        w = [u for u in units if vals[u][a] < vals[u][b]]
        t = [u for u in units if vals[u][a] == vals[u][b]]
        ma, mb = float(np.mean([vals[u][a] for u in units])), float(np.mean([vals[u][b] for u in units]))
        per_b[b] = {"wins": len(w), "ties": len(t), "losses": n - len(w) - len(t), "win_units": w, "tie_units": t,
                    "mean_A": ma, "mean_B": mb, "pass": len(w) >= need and ma < mb}
    joint_units = [u for u in units if all(vals[u][a] < vals[u][b] for b in bs)]
    mean_ok = all(per_b[b]["mean_A"] < per_b[b]["mean_B"] for b in bs)
    return {"n_units": n, "need": need, "per_baseline": per_b, "joint_wins": len(joint_units),
            "joint_win_units": joint_units, "mean_ok": mean_ok,
            "pass_joint": len(joint_units) >= need and mean_ok,
            "pass_per_baseline": all(per_b[b]["pass"] for b in bs)}


def main():
    out = {}
    # (1) sha256
    txt = open(PREREG, encoding="utf-8").read()
    out["sha256_LF"] = {}
    for f in FROZEN:
        h = sha_lf(os.path.join(ROOT, "tack", "hpc", f))
        out["sha256_LF"][f] = {"sha256": h, "in_prereg": h in txt}
    # (2) per-unit means from reps
    I, F = C["interp"], C["forecast"]
    maxdiff = {"ED_U": 0.0, "ED_U_gene": 0.0, "pool": 0.0}
    for k, r in I.items():
        assert len(r["reps"]) == 3
        for m in r["mean"]:
            for key in ("ED_U", "ED_U_gene"):
                mm = float(np.mean([rp[m][key] for rp in r["reps"]]))
                maxdiff[key] = max(maxdiff[key], abs(mm - r["mean"][m][key]))
            for p in r["pools"]:
                mm = float(np.mean([rp[m]["pool_ED_U"][p] for rp in r["reps"]]))
                maxdiff["pool"] = max(maxdiff["pool"], abs(mm - r["pool_mean"][m][p]))
    out["rep_mean_maxdiff"] = maxdiff
    ki = {k: {m: v["ED_U"] for m, v in r["mean"].items()} for k, r in I.items()}
    kg = {k: {m: v["ED_U_gene"] for m, v in r["mean"].items()} for k, r in I.items()}
    fo = {s: {f: v["ED_U"] for f, v in r.items()} for s, r in F.items()}
    fg = {s: {f: v["ED_U_gene"] for f, v in r.items()} for s, r in F.items()}
    tests = {"H1_ot_pairs": ("i", "ot_pairs", ["prev", "next", "union"]),
             "H1b_uot": ("i", "uot", ["prev", "next", "union"]),
             "H1c_pairing": ("i", "ot_pairs", ["indep"]),
             "H2_uot_rw": ("f", "uot_rw", ["persist", "meanshift"]),
             "H2b_ot_needed": ("f", "uot_rw", ["tilt_rw"])}
    out["verdicts"], out["gene_space_secondary"] = {}, {}
    for h, (kind, a, bs) in tests.items():
        r = rule(ki if kind == "i" else fo, a, bs)
        st = C["verdicts"][h]
        r["stored"] = st
        r["matches_stored"] = (st["wins"] == r["joint_wins"] and st["need"] == r["need"]
                               and st["mean_ok"] == r["mean_ok"] and st["pass"] == r["pass_joint"])
        r["readings_agree"] = r["pass_joint"] == r["pass_per_baseline"]
        out["verdicts"][h] = r
        g = rule(kg if kind == "i" else fg, a, bs)
        out["gene_space_secondary"][h] = {"joint_wins": g["joint_wins"], "pass": g["pass_joint"],
                                          "per_baseline": {b: (v["wins"], v["ties"], v["losses"], round(v["mean_A"], 4),
                                                               round(v["mean_B"], 4)) for b, v in g["per_baseline"].items()}}
    # per-pool secondary (3-rep pool means): interp ot_pairs/uot vs union, forecast uot_rw vs persist
    pool = {}
    for a in ("ot_pairs", "uot", "sb_half", "indep"):
        cells = [(k, p) for k, r in I.items() for p in r["pools"]]
        w = sum(I[k]["pool_mean"][a][p] < I[k]["pool_mean"]["union"][p] for k, p in cells)
        pool[f"{a}<union"] = f"{w}/{len(cells)}"
    cells = [(s, p) for s, r in F.items() for p in r["persist"]["pool_ED_U"]]
    for a in ("uot_rw", "tilt_rw"):
        w = sum(F[s][a]["pool_ED_U"][p] < F[s]["persist"]["pool_ED_U"][p] for s, p in cells)
        t = sum(F[s][a]["pool_ED_U"][p] == F[s]["persist"]["pool_ED_U"][p] for s, p in cells)
        pool[f"{a}_vs_persist_W/T/L"] = f"{w}/{t}/{len(cells) - w - t}"
    w = sum(F[s]["uot_rw"]["pool_ED_U"][p] < F[s]["tilt_rw"]["pool_ED_U"][p] for s, p in cells)
    t = sum(F[s]["uot_rw"]["pool_ED_U"][p] == F[s]["tilt_rw"]["pool_ED_U"][p] for s, p in cells)
    pool["uot_rw_vs_tilt_rw_W/T/L"] = f"{w}/{t}/{len(cells) - w - t}"
    out["pool_secondary"] = pool
    # EXPLORATORY facts
    ex = {}
    ex["indep_vs_union"] = {k: (round(v["indep"], 3), round(v["union"], 3)) for k, v in ki.items()}
    ex["n_k_union_lt_indep"] = sum(v["union"] < v["indep"] for v in ki.values())
    ex["n_k_order_union_lt_otpairs_lt_indep"] = sum(v["union"] < v["ot_pairs"] < v["indep"] for v in ki.values())
    ex["sb_half_vs_union"] = {k: (round(v["sb_half"], 3), round(v["union"], 3)) for k, v in ki.items()}
    ex["n_k_sb_half_lt_union"] = sum(v["sb_half"] < v["union"] for v in ki.values())
    ex["mean_ED_U"] = {m: round(float(np.mean([v[m] for v in ki.values()])), 4) for m in ki["1"]}
    ex["gap_ot_pairs_minus_union"] = {k: round(v["ot_pairs"] - v["union"], 4) for k, v in ki.items()}
    reps = {}
    for k, r in I.items():
        reps[k] = [round(rp["ot_pairs"]["ED_U"] - rp["union"]["ED_U"], 4) for rp in r["reps"]]
    ex["rep_level_ot_pairs_minus_union"] = reps
    ex["n_rep_pairs_ot_pairs_worse"] = sum(x > 0 for v in reps.values() for x in v)
    ties = {}
    for s, r in F.items():
        ties[s] = {"hp_uot_rw": r["uot_rw"]["hp"], "hp_tilt_rw": r["tilt_rw"]["hp"],
                   "uot_rw_eq_persist": r["uot_rw"]["ED_U"] == r["persist"]["ED_U"],
                   "tilt_rw_eq_persist": r["tilt_rw"]["ED_U"] == r["persist"]["ED_U"],
                   "uot_rw_minus_persist": round(r["uot_rw"]["ED_U"] - r["persist"]["ED_U"], 4),
                   "backtest_uot_rw": round(r["uot_rw"]["backtest_ED_U"], 4),
                   "backtest_persist": round(r["persist"]["backtest_ED_U"], 4),
                   "ess_uot_rw": round(r["uot_rw"]["ess"], 1)}
    ex["forecast_ties"] = ties
    # if the 3 ties had counted as wins (NOT the prereg rule): wins and mean
    w_lenient = sum(fo[s]["uot_rw"] <= fo[s]["persist"] for s in fo)
    ex["H2_if_ties_counted_as_wins"] = {"wins": w_lenient, "mean_uot_rw": float(np.mean([fo[s]["uot_rw"] for s in fo])),
                                        "mean_persist": float(np.mean([fo[s]["persist"] for s in fo]))}
    # effect size of the H1 gap vs union (units are NOT independent: neighbouring k share stages; descriptive only)
    from scipy import stats
    g = np.array([ki[k]["ot_pairs"] - ki[k]["union"] for k in sorted(ki, key=int)])
    rel = np.array([(ki[k]["ot_pairs"] - ki[k]["union"]) / ki[k]["union"] for k in sorted(ki, key=int)])
    tq = stats.t.ppf(0.975, len(g) - 1)
    ex["gap_t_interval_95_nominal"] = [float(g.mean() - tq * g.std(ddof=1) / np.sqrt(len(g))),
                                       float(g.mean() + tq * g.std(ddof=1) / np.sqrt(len(g)))]
    ex["gap_mean_sd"] = [float(g.mean()), float(g.std(ddof=1))]
    ex["rel_gap_ot_pairs_vs_union"] = [round(float(x), 3) for x in rel]
    ex["coin_null_P_ge5_of7"] = float(sum(math.comb(7, i) for i in range(5, 8)) / 2 ** 7)
    ex["coin_null_P_0_of7"] = 1 / 2 ** 7
    ex["coin_null_P_ge3_of4"] = float(sum(math.comb(4, i) for i in range(3, 5)) / 2 ** 4)
    out["exploratory"] = ex
    json.dump(out, open(os.path.join(HERE, "j_verdict.json"), "w"), indent=1, default=float)
    for h, r in out["verdicts"].items():
        print(h, "joint", r["joint_wins"], "/", r["n_units"], "need", r["need"], "mean_ok", r["mean_ok"],
              "PASS" if r["pass_joint"] else "FAIL", "| matches stored", r["matches_stored"], "| readings agree",
              r["readings_agree"], "|", {b: (v["wins"], v["ties"], v["losses"], round(v["mean_A"], 4), round(v["mean_B"], 4))
                                         for b, v in r["per_baseline"].items()})
    print("sha", {f: v["in_prereg"] for f, v in out["sha256_LF"].items()})
    print("rep_mean_maxdiff", maxdiff)
    print("gene", {h: (v["joint_wins"], v["pass"]) for h, v in out["gene_space_secondary"].items()})
    print("pool", pool)
    print(json.dumps(ex, indent=0, default=float))


if __name__ == "__main__":
    main()
