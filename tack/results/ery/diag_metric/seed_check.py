"""Robustness extra (NOT pre-registered): are the corrected verdicts stable to the 1500-cell subsampling?

For origins 2-5, the Mode-B forecasts (hp selected on the backtest by energy or by sd_fixed, read from rescore.json)
are rebuilt (noise, if any, from a seeded generator) and re-scored against the true next stage with 5 other
subsample seeds (prediction and target). For every seed the pre-registered rule is re-applied. Output seed_check.json.
"""
import json
import os

import numpy as np

import ery_forecast_fixed as ef
import verdicts as vd

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(5)


def main():
    res = json.load(open(os.path.join(HERE, "rescore.json")))
    tabs = {m: {k: {} for k in SEEDS} for m in ("sd_fixed", "energy")}
    for s in (2, 3, 4, 5):
        d = ef.load(ef.DATA, s)
        Zt = d["Z_fut"]
        S = res["origins"][str(s)]["S_fixed"]
        tg = {k: ef.Target(Zt, S, seed=200 + k) for k in SEEDS}
        for m in ("sd_fixed", "energy"):
            for fi, f in enumerate(vd.FAMS):
                hp = res["origins"][str(s)]["families"][f]["sel"][m]
                Zf, _, w = ef.forecast(f, hp, d, s, (s, s - 1), rng=np.random.default_rng([s, fi, 7]))
                for k in SEEDS:
                    i = ef.sub(len(Zf), seed=100 + k)
                    X = Zf[i]
                    wx = ef.unif(len(X)) if w is None else w[i] / w[i].sum()
                    t = tg[k]
                    if m == "energy":
                        val = ef.energy(X, t.Y, None if w is None else w[i])
                    else:
                        val = float(ef.sinkhorn_div_fixed(X, t.Y, wx, t.wy, S, t.yy_ot))
                    tabs[m][k].setdefault(str(s), {})[f] = val
            print(s, m, "done", flush=True)
        ef._DRIFT.clear()
    out = {"note": "extra robustness check, not pre-registered; Mode-B hp, 5 subsample seeds", "metrics": {}}
    for m, per in tabs.items():
        rows = []
        for k, v in per.items():
            r = vd.verdicts(v)
            rows.append({"seed": k, "OT_helps": r["OT_helps"], "velocity_helps": r["velocity_helps"],
                         "velocity_vs_ot_and_baselines": r["extra_velocity_vs_ot_and_baselines"],
                         "sink_helps": r["sink_helps"],
                         "vel_beats_persist_origins": {f: [o for o in vd.ORIG if v[o][f] < v[o]["persist"]]
                                                       for f in vd.VELF},
                         "table": v})
        out["metrics"][m] = rows
        print(m, [(x["seed"], x["OT_helps"], x["velocity_helps"], x["velocity_vs_ot_and_baselines"], x["sink_helps"])
                  for x in rows])
        for x in rows:
            print("   seed", x["seed"], "vel beats persist at", x["vel_beats_persist_origins"])
    with open(os.path.join(HERE, "seed_check.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
