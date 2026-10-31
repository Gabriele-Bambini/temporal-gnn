"""Judge extras (independent code, judge_lib only):
(1) EVALUATION ONLY: uot_rw (at the audited 'roll' hp, by sd_fixed and by energy) vs persistence scored against each
    embryo pool (>= 100 cells) of the true target stage, energy V and U - does the post-hoc pass hold pool by pool?
(2) DESCRIPTIVE metric probe in the forecasting spaces: half A of the true stage s+1 contracted toward its mean by a,
    scored against half B (sd_fixed with the fixed S_s, energy V / U). Which metric prefers a > 0?
Output: judge/indep/extra.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_lib as J  # noqa: E402
from judge_forecast import Origin, forecast_new  # noqa: E402

STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    ERY = os.path.dirname(HERE)
    obs = pd.read_csv(os.path.join(a.data, "obs.csv"))
    codes = np.array([STAGES.index(x) for x in obs["stage"].astype(str)])
    out = {"pools": {}, "probe": {}}
    for s in (2, 3, 4, 5):
        d = dict(np.load(os.path.join(a.data, f"origin{s}.npz"), allow_pickle=True))
        Zfut = d.pop("Z_fut").astype(np.float64)
        smp_fut = obs["sample"].astype(str).to_numpy()[codes == s + 1]
        O = Origin({k: v for k, v in d.items() if not k.endswith("_fut")})
        st = O.st
        S = float(np.median(J.sqdist(O.Z[J.sub(len(O.Z), 7)], O.Z[J.sub(len(O.Z), 7)])[
            np.triu_indices(min(len(O.Z), 1500), 1)]))
        prev = json.load(open(os.path.join(ERY, "attempt_forecast", f"scores_o{s - 1}.json")))["final"]["uot_rw"]
        hps = {m: json.loads(min(prev, key=lambda k: prev[k][m])) for m in ("sd_fixed", "energy")}
        Zn = O.Z[st == s]
        preds = {"persist": (Zn, None)}
        for m, hp in hps.items():
            preds[f"uot_rw[{m} hp]"] = forecast_new(O, "uot_rw", hp, s, None)
        res = {"hp": hps, "pools": {}}
        for p in sorted(set(smp_fut), key=int):
            idx = np.where(smp_fut == p)[0]
            if len(idx) < 100:
                continue
            tg = J.Tgt(Zfut[idx[J.sub(len(idx), 1)]], S)
            res["pools"][p] = {}
            for name, (Z, w) in preds.items():
                i = J.sub(len(Z), 0)
                wi = J.unif(len(i)) if w is None else w[i] / w[i].sum()
                ev, eu = J.energies(Z[i], wi, tg)
                res["pools"][p][name] = {"energy": ev, "energy_U": eu}
        out["pools"][str(s)] = res
        # probe
        perm = np.random.default_rng(4).permutation(len(Zfut))
        hA, hB = perm[: len(Zfut) // 2][:1500], perm[len(Zfut) // 2:][:1500]
        tB = J.Tgt(Zfut[hB], S)
        rows = []
        for aa in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            Z = Zfut[hA]
            Z = Z.mean(0) + (1 - aa) * (Z - Z.mean(0))
            ev, eu = J.energies(Z, J.unif(len(Z)), tB)
            rows.append({"a": aa, "sd_fixed": J.sd_fixed(Z, J.unif(len(Z)), tB), "energy": ev, "energy_U": eu})
        out["probe"][str(s)] = rows
        print(s, json.dumps(res, default=float)[:900], flush=True)
        print(s, "probe", [(r["a"], round(r["sd_fixed"], 4), round(r["energy"], 4), round(r["energy_U"], 4)) for r in rows], flush=True)
    with open(os.path.join(HERE, "indep", "extra.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
