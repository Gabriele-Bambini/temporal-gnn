"""EXPLORATORY support: check that the local machine + the mechanism helpers reproduce confirm.json before any analysis.
(a) common.run_interp_subset with no masks vs confirm.json interp[k].mean / reps, k = 1..7;
(b) frozen atlas_confirm.run_forecast(s) vs confirm.json forecast[s], s = 2..7 (hp, ED_U, backtest).
Writes check_repro.json. Usage: python check_repro.py
"""
import os
import time

import numpy as np

import common as C

def main():
    ref = C.confirm()
    out = {"script": "check_repro.py", "exploratory": True, "interp": {}, "forecast": {}}
    t0 = time.time()
    for k in range(1, 8):
        o = C.run_interp_subset(C.load_interp(k), k)
        diff = max(abs(o["reps"][r][m]["ED_U"] - ref["interp"][str(k)]["reps"][r][m]["ED_U"])
                   for r in range(3) for m in C.METHODS)
        diffg = max(abs(o["reps"][r][m]["ED_U_gene"] - ref["interp"][str(k)]["reps"][r][m]["ED_U_gene"])
                    for r in range(3) for m in C.METHODS)
        out["interp"][k] = {"max_abs_diff_ED_U": diff, "max_abs_diff_ED_U_gene": diffg}
        print("interp", k, diff, diffg, f"[{time.time() - t0:.0f}s]", flush=True)
    for s in range(2, 8):
        o = C.AC.run_forecast(C.DATA, s)
        rs = ref["forecast"][str(s)]
        out["forecast"][s] = {f: {"hp_same": o[f]["hp"] == rs[f]["hp"],
                                  "abs_diff_ED_U": abs(o[f]["ED_U"] - rs[f]["ED_U"]),
                                  "abs_diff_backtest": abs(o[f]["backtest_ED_U"] - rs[f]["backtest_ED_U"])}
                              for f in o}
        print("forecast", s, out["forecast"][s], f"[{time.time() - t0:.0f}s]", flush=True)
    out["all_ok"] = bool(all(v["max_abs_diff_ED_U"] < 1e-9 for v in out["interp"].values()) and
                         all(x["hp_same"] and x["abs_diff_ED_U"] < 1e-9 and x["abs_diff_backtest"] < 1e-9
                             for v in out["forecast"].values() for x in v.values()))
    out["seconds"] = round(time.time() - t0, 1)
    print("ALL_OK", out["all_ok"])
    C.dump(os.path.join(C.HERE, "check_repro.json"), out)


if __name__ == "__main__":
    main()
