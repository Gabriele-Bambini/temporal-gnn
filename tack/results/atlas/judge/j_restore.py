"""JUDGE, EXPLORATORY: is the McCann loss on the atlas an UNDER-DISPERSION artefact, or geometry?

A McCann point (x_i + y_j) / 2 averages two cells, so cell-level residual variance is halved. sb_half restores variance
with ISOTROPIC Gaussian noise (wrong shape in a 30-d PCA). Here the deficit is restored with DATA-SHAPED noise:
  z = (x_i + y_j) / 2 + (d_i + d_j) / (2 sqrt 2),  d_i = x_i - x_a, x_a a random one of the 10 nearest neighbours of
  x_i in its own fit-stage subsample (same for y_j), so Var(added) ~ Sigma / 2 if neighbours share the local law.
Calibration (uses only the held stage, descriptive): held half A vs half B (frozen own_halves split, rng 4) for
  raw A, A averaged with a random 10-NN of itself (pure "denoising", no time), and that average + the same restoration.
Also scale-free separation of the neighbours: ED_U(prev, next) / E|H - H'| and |mu_next - mu_prev| / RMS radius(H).
Both datasets (atlas k = 1..7, erythroid k = 2..5), 3 frozen fit subsamples, frozen seed-1 target, own pair draws.
Usage: python j_restore.py -> j_restore.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
SETS = {"atlas": (os.path.join(SCR, "atlas"), range(1, 8), os.path.join(HERE, "..", "confirm.json")),
        "ery": (os.path.join(SCR, "ery"), range(2, 6), os.path.join(HERE, "..", "parity_erythroid.json"))}
NN = 10


def knn_diff(Z, idx, rng):
    nn = NearestNeighbors(n_neighbors=NN + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    pick = nn[idx, rng.integers(0, NN, len(idx))]
    return Z[idx] - Z[pick], pick


def spread(Z, ref):
    return float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean() / ((ref - ref.mean(0)) ** 2).sum(1).mean()))


def main():
    res = {}
    for name, (data, ks, stored_path) in SETS.items():
        stored = json.load(open(stored_path))["interp"]
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
                rng = np.random.default_rng(500 + r)
                iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
                ZA, ZB = Zf[A[iA]], Zf[B[iB]]
                P, _ = IB.plan(ZA, ZB, 0.05)
                i, j = IB.draw_pairs(P, n_pred, np.random.default_rng(9 + r))
                mid = 0.5 * ZA[i] + 0.5 * ZB[j]
                dA, _ = knn_diff(ZA, i, rng)
                dB, _ = knn_diff(ZB, j, rng)
                rest = mid + (dA + dB) / (2 * np.sqrt(2))
                u = IB.unif(n_pred)
                reps.append({"mccann": IB.ed_uv(mid, u, tgt)[0], "mccann_restored": IB.ed_uv(rest, u, tgt)[0],
                             "spread_mccann": spread(mid, tgt.Z), "spread_restored": spread(rest, tgt.Z)})
            # separation statistics (frozen rep-0 subsamples)
            iA, iB = IB.sub(len(A), seed=1000), IB.sub(len(B), seed=2000)
            ZA, ZB = Zf[A[iA]], Zf[B[iB]]
            tB = IB.Target(ZB, None, S, with_sd=False)
            ed_pn = IB.ed_uv(ZA, IB.unif(len(ZA)), tB)[0]
            within_h = tgt.wdw / (1 - tgt.sw2)
            sep_ed = ed_pn / within_h
            sep_mu = float(np.linalg.norm(ZB.mean(0) - ZA.mean(0)) / np.sqrt(((tgt.Z - tgt.Z.mean(0)) ** 2).sum(1).mean()))
            # calibration on the held stage only
            rng_h = np.random.default_rng(4)
            perm = rng_h.permutation(len(Zh))
            hA, hB = perm[: len(Zh) // 2][:IB.NSUB], perm[len(Zh) // 2:][:IB.NSUB]
            tHB = IB.Target(Zh[hB], None, S, with_sd=False)
            ZhA = Zh[hA]
            rng_c = np.random.default_rng(11)
            idx = np.arange(len(ZhA))
            d1, pick = knn_diff(ZhA, idx, rng_c)
            avg = 0.5 * (ZhA + ZhA[pick])
            d2, _ = knn_diff(ZhA, pick, rng_c)
            avg_rest = avg + (d1 + d2) / (2 * np.sqrt(2))
            uu = IB.unif(len(ZhA))
            calib = {"raw_half": IB.ed_uv(ZhA, uu, tHB)[0], "half_averaged_with_knn": IB.ed_uv(avg, uu, tHB)[0],
                     "half_averaged_restored": IB.ed_uv(avg_rest, uu, tHB)[0],
                     "spread_avg": spread(avg, Zh[hB]), "spread_avg_restored": spread(avg_rest, Zh[hB])}
            mean = {key: float(np.mean([rp[key] for rp in reps])) for key in reps[0]}
            sm = stored[str(k)]["mean"]
            out = {"mean": mean, "frozen_union": sm["union"]["ED_U"], "frozen_ot_pairs": sm["ot_pairs"]["ED_U"],
                   "frozen_sb_half": sm["sb_half"]["ED_U"], "restored_lt_union": mean["mccann_restored"] < sm["union"]["ED_U"],
                   "sep_ED_prev_next_over_within_held": sep_ed, "sep_mean_shift_over_rms_held": sep_mu,
                   "rel_gain_ot_pairs_vs_union": (sm["union"]["ED_U"] - sm["ot_pairs"]["ED_U"]) / sm["union"]["ED_U"],
                   "calibration_held_halves": calib, "reps": reps}
            res[name][str(k)] = out
            print(name, "k", k, "| union", round(out["frozen_union"], 3), "ot_pairs", round(out["frozen_ot_pairs"], 3),
                  "sb_half", round(out["frozen_sb_half"], 3), "| own mccann", round(mean["mccann"], 3), "restored",
                  round(mean["mccann_restored"], 3), "spread", round(mean["spread_mccann"], 3), "->", round(mean["spread_restored"], 3),
                  "| sep_ed", round(sep_ed, 2), "sep_mu", round(sep_mu, 2), "| calib", {a: round(b, 3) for a, b in calib.items()},
                  f"[{time.time() - t0:.0f}s]", flush=True)
    allu = [(n, k) for n in SETS for k in res[n]]
    g = [res[n][k]["rel_gain_ot_pairs_vs_union"] for n, k in allu]
    res["summary"] = {
        "restored_lt_union": {n: f"{sum(res[n][k]['restored_lt_union'] for k in res[n])}/{len(res[n])}" for n in SETS},
        "spearman_rel_gain_vs_sep_ed_all11": float(spearmanr(g, [res[n][k]["sep_ED_prev_next_over_within_held"] for n, k in allu])[0]),
        "spearman_rel_gain_vs_sep_mu_all11": float(spearmanr(g, [res[n][k]["sep_mean_shift_over_rms_held"] for n, k in allu])[0]),
        "sep_ed": {f"{n}{k}": round(res[n][k]["sep_ED_prev_next_over_within_held"], 3) for n, k in allu},
        "sep_mu": {f"{n}{k}": round(res[n][k]["sep_mean_shift_over_rms_held"], 3) for n, k in allu},
        "rel_gain": {f"{n}{k}": round(v, 3) for (n, k), v in zip(allu, g)},
    }
    print(json.dumps(res["summary"], indent=0))
    json.dump(res, open(os.path.join(HERE, "j_restore.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
