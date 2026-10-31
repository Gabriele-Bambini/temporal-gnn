"""Subsampling-seed check of the headline comparisons (scoring only; hp fixed to the PRIMARY 'roll' selection).

For origins 2-5, the forecasts of the key new families (hp chosen by the rolling-origin backtest under the metric being
judged) and persistence are re-scored with 3 other subsample seeds (prediction subsample seed 100+k, target subsample
seed 200+k). Reports the paired difference family - persist under sd_fixed and energy. Seed variation does not
capture embryo-pool variation (see the per-pool breakdown in pools_o<s>.json).
Output: seed_check.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attempt_forecast as A  # noqa: E402

F = A.F
FAMS = ["uot_rw", "mk_km", "tilt_rw", "ms_smooth", "vel_ss_nm", "desample", "contract", "c_uot"]
SEEDS = (1, 2, 3)


def sc(Zf, w, Zt, S, k, m):
    i = F.sub(len(Zf), seed=100 + k)
    if w is not None:
        i = i[w[i] > 0]
    X = Zf[i]
    wx = F.unif(len(X)) if w is None else w[i] / w[i].sum()
    t = F.Target(Zt, S, seed=200 + k)
    if m == "energy":
        return F.energy(X, t.Y, wx)
    return float(F.sinkhorn_div_fixed(X, t.Y, wx, t.wy, t.S, t.yy_ot))


def main():
    obs, codes = A.load_obs(A.DATA)
    out = {}
    for s in (2, 3, 4, 5):
        cur, prev = A.load_scores(s), A.load_scores(s - 1)
        F._DRIFT.clear()
        d, smp_past, smp_fut, km, S, P = A.build_final(A.DATA, s, obs, codes)
        out[str(s)] = {}
        for m in ("sd_fixed", "energy"):
            base = [sc(P.Zn, None, d["Z_fut"], S, k, m) for k in SEEDS]
            out[str(s)][m] = {"persist": base}
            for fam in FAMS:
                hp = A.selected_hps(fam, s, m, cur, prev)["roll"]
                Zf, Xf, w = A.FAMILIES[fam][1](P, hp)
                v = [sc(Zf, w, d["Z_fut"], S, k, m) for k in SEEDS]
                diff = [a - b for a, b in zip(v, base)]
                out[str(s)][m][fam] = {"hp": hp, "values": v, "diff_vs_persist": diff,
                                       "main_run_diff": cur["final"][fam][A.hkey(hp)][m] -
                                       cur["final"]["ms_damp"][A.hkey({"lam": 0.0})][m]}
                print(s, m, fam, hp, "diff", np.round(diff, 4), "main", round(out[str(s)][m][fam]["main_run_diff"], 4),
                      flush=True)
    with open(os.path.join(HERE, "seed_check.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
