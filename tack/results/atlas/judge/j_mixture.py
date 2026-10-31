"""JUDGE, EXPLORATORY (oracle-labelled where marked): does the McCann midpoint add anything a prev/next MIXTURE cannot?

union is the endpoint mixture of the same coupling (each OT pair contributes one of its endpoints with prob 1/2;
the balanced plan has uniform marginals), ot_pairs places the pair at its midpoint. So "ot_pairs < union" means the held
cells look more like midpoints than like endpoints. For every held stage (atlas k = 1..7, erythroid k = 2..5), on the
3 frozen fit subsamples (seeds 1000+r / 2000+r, frozen seed-1 target), I compute:
  - ED_U of the weighted mixture alpha * prev + (1 - alpha) * next for alpha in 0..1 (step 0.05) [ORACLE: best alpha];
  - ED_U of McCann points (1 - t) x_i + t y_j, pairs drawn from the frozen balanced plan (eps_rel 0.05), t in 0..1
    (step 0.1) [ORACLE: best t]; t = 0.5 is the pre-registered ot_pairs (own pair draw, rng 9 + r);
  - spread ratio (RMS radius about the own mean / held RMS radius) for union (alpha 0.5), McCann t = 0.5, indep;
  - mixture identity check: ED(mix_a, H) = a ED(P,H) + (1-a) ED(N,H) - a (1-a) ED(P,N) (exact for V-statistics).
Key model-free comparison: McCann t = 0.5 vs the ORACLE-best mixture, and ORACLE-best t vs ORACLE-best alpha.
Usage: python j_mixture.py -> j_mixture.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
SETS = {"atlas": (os.path.join(SCR, "atlas"), range(1, 8)), "ery": (os.path.join(SCR, "ery"), range(2, 6))}
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
TS = np.round(np.arange(0, 1.0001, 0.1), 1)


def spread(Z, w, ref):
    mu = w @ Z
    return float(np.sqrt(w @ ((Z - mu) ** 2).sum(1)) / np.sqrt(((ref - ref.mean(0)) ** 2).sum(1).mean()))


def main():
    res = {}
    for name, (data, ks) in SETS.items():
        res[name] = {}
        for k in ks:
            t0 = time.time()
            d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
            st = d["stage_fit"]
            Zf, Zh = d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
            A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
            S = IB.med_scale(Zf)
            n_pred = min(len(Zh), IB.NSUB)
            it = IB.sub(len(Zh), seed=1)
            tgt = IB.Target(Zh[it], None, S, with_sd=False)
            reps = []
            for r in range(3):
                iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
                ZA, ZB = Zf[A[iA]], Zf[B[iB]]
                nA, nB = len(ZA), len(ZB)
                Zm = np.vstack([ZA, ZB])
                mix = {}
                for a in ALPHAS:
                    w = np.r_[np.full(nA, a / nA), np.full(nB, (1 - a) / nB)]
                    keep = w > 0
                    mix[float(a)] = IB.ed_uv(Zm[keep], w[keep] / w[keep].sum(), tgt)[0]
                # identity check (V-statistic pieces)
                ePH = IB.ed_uv(ZA, IB.unif(nA), tgt)[1]
                eNH = IB.ed_uv(ZB, IB.unif(nB), tgt)[1]
                tB = IB.Target(ZB, None, S, with_sd=False)
                ePN_v = IB.ed_uv(ZA, IB.unif(nA), tB)[1]
                w5 = np.r_[np.full(nA, 0.5 / nA), np.full(nB, 0.5 / nB)]
                mix5_v = IB.ed_uv(Zm, w5, tgt)[1]
                ident = 0.5 * ePH + 0.5 * eNH - 0.25 * ePN_v
                P, med = IB.plan(ZA, ZB, 0.05)
                i, j = IB.draw_pairs(P, n_pred, np.random.default_rng(9 + r))
                mc = {}
                for t in TS:
                    Zt = (1 - t) * ZA[i] + t * ZB[j]
                    mc[float(t)] = IB.ed_uv(Zt, IB.unif(n_pred), tgt)[0]
                rng = np.random.default_rng(77 + r)
                ii, jj = rng.integers(0, nA, n_pred), rng.integers(0, nB, n_pred)
                Zind = 0.5 * ZA[ii] + 0.5 * ZB[jj]
                ba = max(mix, key=lambda x: -mix[x])
                bt = max(mc, key=lambda x: -mc[x])
                reps.append({"mix": mix, "mccann": mc, "best_alpha": ba, "best_mix": mix[ba], "best_t": bt,
                             "best_t_ED_U": mc[bt], "mix_0.5": mix[0.5], "mccann_0.5": mc[0.5],
                             "identity_V": {"direct": mix5_v, "identity": ident, "abs_err": abs(mix5_v - ident)},
                             "spread": {"union_a0.5": spread(Zm, w5, tgt.Z), "mccann_0.5": spread((ZA[i] + ZB[j]) / 2, IB.unif(n_pred), tgt.Z),
                                        "indep": spread(Zind, IB.unif(n_pred), tgt.Z), "prev": spread(ZA, IB.unif(nA), tgt.Z),
                                        "next": spread(ZB, IB.unif(nB), tgt.Z)},
                             "ED_V_prev_next_over4": ePN_v / 4})
            def m(f):
                return float(np.mean([f(rp) for rp in reps]))
            mixc = {a: float(np.mean([rp["mix"][a] for rp in reps])) for a in reps[0]["mix"]}
            mcc = {t: float(np.mean([rp["mccann"][t] for rp in reps])) for t in reps[0]["mccann"]}
            ba = min(mixc, key=mixc.get)
            bt = min(mcc, key=mcc.get)
            out = {"mix_curve_mean": mixc, "mccann_curve_mean": mcc, "best_alpha_on_prev": ba, "best_mix": mixc[ba],
                   "best_t": bt, "best_t_ED_U": mcc[bt], "mix_0.5": mixc[0.5], "mccann_0.5": mcc[0.5],
                   "mccann_0.5_lt_mix_0.5": mcc[0.5] < mixc[0.5],
                   "mccann_0.5_lt_ORACLE_best_mix": mcc[0.5] < mixc[ba],
                   "ORACLE_best_t_lt_ORACLE_best_mix": mcc[bt] < mixc[ba],
                   "spread_mean": {key: m(lambda rp, key=key: rp["spread"][key]) for key in reps[0]["spread"]},
                   "identity_max_abs_err_V": max(rp["identity_V"]["abs_err"] for rp in reps),
                   "convexity_bonus_ED_V_prev_next_over4": m(lambda rp: rp["ED_V_prev_next_over4"]), "reps": reps}
            res[name][str(k)] = out
            print(name, "k", k, "| mix a=0.5", round(out["mix_0.5"], 3), "best a", ba, round(out["best_mix"], 3),
                  "| mccann t=0.5", round(out["mccann_0.5"], 3), "best t", bt, round(out["best_t_ED_U"], 3),
                  "| mc0.5<bestmix", out["mccann_0.5_lt_ORACLE_best_mix"], "| bestt<bestmix", out["ORACLE_best_t_lt_ORACLE_best_mix"],
                  "| spread", {key: round(v, 3) for key, v in out["spread_mean"].items()},
                  "| ident err", f"{out['identity_max_abs_err_V']:.1e}", f"[{time.time() - t0:.0f}s]", flush=True)
        ks_ = list(res[name])
        res[name + "_summary"] = {
            "n_units": len(ks_),
            "mccann0.5_lt_mix0.5": sum(res[name][k]["mccann_0.5_lt_mix_0.5"] for k in ks_),
            "mccann0.5_lt_ORACLE_best_mix": sum(res[name][k]["mccann_0.5_lt_ORACLE_best_mix"] for k in ks_),
            "ORACLE_best_t_lt_ORACLE_best_mix": sum(res[name][k]["ORACLE_best_t_lt_ORACLE_best_mix"] for k in ks_),
            "best_alpha_on_prev": {k: res[name][k]["best_alpha_on_prev"] for k in ks_},
            "best_t": {k: res[name][k]["best_t"] for k in ks_},
            "gap_mix0.5_minus_mccann0.5": {k: round(res[name][k]["mix_0.5"] - res[name][k]["mccann_0.5"], 4) for k in ks_},
            "gap_bestmix_minus_mccann0.5": {k: round(res[name][k]["best_mix"] - res[name][k]["mccann_0.5"], 4) for k in ks_},
            "gap_bestmix_minus_bestt": {k: round(res[name][k]["best_mix"] - res[name][k]["best_t_ED_U"], 4) for k in ks_},
        }
        print(name, json.dumps(res[name + "_summary"]), flush=True)
    json.dump(res, open(os.path.join(HERE, "j_mixture.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
