"""JUDGE, EXPLORATORY (oracle-labelled): forecasting grid in hindsight, to separate SELECTION from MODEL CLASS for H2.

For every origin s (atlas 1..7, erythroid parity products 1..5) the frozen AC.Problem / forecast / score are used
unchanged to score persist, meanshift, all 13 uot_rw grid points (AC.UOT_GRID) and all 6 tilt_rw points against the
TRUE stage s+1 (this uses the target: ORACLE, never a verdict). Then:
  - the rolling-origin choice at s is recomputed as argmin of the origin s-1 scores and checked against the stored run;
  - rank of that choice in the final grid at s, number of lam > 0 points below persist, best lam > 0 point;
  - Spearman over grid points between the backtest (origin s-1) and final (origin s) scores;
  - the H2 count against persist if hp had been chosen in hindsight (ORACLE; not the prereg rule).
Usage: python j_forecast.py -> j_forecast.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import atlas_confirm as AC  # noqa: E402

SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
SETS = {"atlas": (os.path.join(SCR, "atlas"), range(1, 8), os.path.join(HERE, "..", "confirm.json")),
        "ery": (os.path.join(SCR, "ery"), range(1, 6), os.path.join(HERE, "..", "parity_erythroid.json"))}


def hpkey(fam, hp):
    return fam + ("" if not hp else "|" + ",".join(f"{k}={v}" for k, v in sorted(hp.items())))


def grid_scores(data, s):
    P = AC.origin_problem(data, s)
    out = {"persist": P.score(*P.forecast("persist", {}))["ED_U"],
           "meanshift": P.score(*P.forecast("meanshift", {}))["ED_U"]}
    for fam, grid in (("uot_rw", AC.UOT_GRID), ("tilt_rw", AC.TILT_GRID)):
        for hp in grid:
            sc = P.score(*P.forecast(fam, hp))
            out[hpkey(fam, hp)] = sc["ED_U"]
            out[hpkey(fam, hp) + "#ess"] = sc["ess"]
    return out


def main():
    res = {}
    for name, (data, origins, stored_path) in SETS.items():
        t0 = time.time()
        stored = json.load(open(stored_path))["forecast"]
        G = {s: grid_scores(data, s) for s in origins}
        per = {}
        for s in list(origins)[1:]:
            row = {"persist": G[s]["persist"]}
            for fam, grid in (("uot_rw", AC.UOT_GRID), ("tilt_rw", AC.TILT_GRID)):
                keys = [hpkey(fam, hp) for hp in grid]
                back = np.array([G[s - 1][k] for k in keys])
                fin = np.array([G[s][k] for k in keys])
                ch = int(np.argmin(back))
                pos = [i for i, hp in enumerate(grid) if hp["lam"] > 0]
                best_pos = pos[int(np.argmin(fin[pos]))]
                row[fam] = {
                    "rolling_choice": grid[ch], "rolling_matches_stored": grid[ch] == stored[str(s)][fam]["hp"],
                    "rolling_final_ED_U": float(fin[ch]),
                    "rolling_matches_stored_ED_U": abs(fin[ch] - stored[str(s)][fam]["ED_U"]) < 1e-9,
                    "rank_of_choice_in_final_grid": int(1 + np.sum(fin < fin[ch])), "grid_size": len(grid),
                    "oracle_best": grid[int(np.argmin(fin))], "oracle_best_ED_U": float(fin.min()),
                    "best_lam_pos": grid[best_pos], "best_lam_pos_ED_U": float(fin[best_pos]),
                    "n_lam_pos_below_persist": int(np.sum(fin[pos] < G[s]["persist"])), "n_lam_pos": len(pos),
                    "best_lam_pos_rel_gain_vs_persist": float((G[s]["persist"] - fin[best_pos]) / G[s]["persist"]),
                    "spearman_backtest_vs_final": float(spearmanr(back, fin)[0]),
                    "backtest_gain_of_choice_vs_persist": float(G[s - 1]["persist"] - back[ch]),
                }
            per[str(s)] = row
        ss = [str(s) for s in list(origins)[1:]]
        summ = {}
        for fam in ("uot_rw", "tilt_rw"):
            ora = [per[s][fam]["oracle_best_ED_U"] for s in ss]
            orp = [per[s][fam]["best_lam_pos_ED_U"] for s in ss]
            rol = [per[s][fam]["rolling_final_ED_U"] for s in ss]
            pe = [per[s]["persist"] for s in ss]
            summ[fam] = {
                "all_rolling_choices_match_stored": all(per[s][fam]["rolling_matches_stored"] for s in ss),
                "ORACLE_strict_wins_vs_persist_best_any": int(sum(o < p for o, p in zip(ora, pe))),
                "ORACLE_strict_wins_vs_persist_best_lam_pos": int(sum(o < p for o, p in zip(orp, pe))),
                "ORACLE_mean_best_any": float(np.mean(ora)), "ORACLE_mean_best_lam_pos": float(np.mean(orp)),
                "rolling_mean": float(np.mean(rol)), "persist_mean": float(np.mean(pe)),
                "ORACLE_rel_gain_best_lam_pos": [round(per[s][fam]["best_lam_pos_rel_gain_vs_persist"], 4) for s in ss],
                "ranks_of_rolling_choice": [per[s][fam]["rank_of_choice_in_final_grid"] for s in ss],
                "spearman_backtest_vs_final": [round(per[s][fam]["spearman_backtest_vs_final"], 3) for s in ss],
                "median_spearman": float(np.median([per[s][fam]["spearman_backtest_vs_final"] for s in ss])),
            }
        res[name] = {"grid": {str(s): G[s] for s in origins}, "per_origin": per, "summary": summ,
                     "seconds": round(time.time() - t0, 1)}
        print(name, json.dumps(summ, indent=0), flush=True)
        for s in ss:
            u = per[s]["uot_rw"]
            print(name, "s", s, "persist", round(per[s]["persist"], 4), "| rolling", u["rolling_choice"],
                  round(u["rolling_final_ED_U"], 4), "rank", u["rank_of_choice_in_final_grid"], "| best lam>0",
                  u["best_lam_pos"], round(u["best_lam_pos_ED_U"], 4), "n_below", u["n_lam_pos_below_persist"],
                  "| rho", round(u["spearman_backtest_vs_final"], 2), "| backtest gain", round(u["backtest_gain_of_choice_vs_persist"], 4),
                  flush=True)
    # per-pool view of the only real atlas win (s = 6) and of s = 4, from the stored run
    C = json.load(open(SETS["atlas"][2]))["forecast"]
    res["atlas_pools_stored"] = {s: {p: (round(C[s]["uot_rw"]["pool_ED_U"][p], 3), round(C[s]["persist"]["pool_ED_U"][p], 3))
                                     for p in C[s]["persist"]["pool_ED_U"]} for s in ("4", "6", "7")}
    print("atlas pools (uot_rw, persist)", res["atlas_pools_stored"])
    json.dump(res, open(os.path.join(HERE, "j_forecast.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
