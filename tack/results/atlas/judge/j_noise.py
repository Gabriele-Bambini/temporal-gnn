"""JUDGE, EXPLORATORY: is "union beats ot_pairs in 7/7 held stages" robust to the noise the 3 reps do not cover?

The frozen reps vary only the fit-side subsamples and pair draws; the held-stage target subsample (seed 1) is fixed.
Here, for every held stage k = 1..7 of the atlas (and k = 2..5 of the erythroid parity products):
  (a) the frozen loop of atlas_confirm.run_interp is replayed exactly (same rng = default_rng(1000k + r), same fit
      subsample seeds, IB.subsample_pred called for every method in IB.methods' order) and checked against the stored
      rep values (atlas: confirm.json; erythroid: parity_erythroid.json) at the seed-1 target;
  (b) the same predictions are rescored against 10 other target subsamples (seeds 2..11) and against the
      COMPLEMENT of the seed-1 subsample (held cells never used by the frozen target);
  (c) the held stage is scored leave-one-pool-out (all cells of the held stage except pool p, <= 1500, seed 1);
  (d) 17 fresh fit-side reps (r = 3..19, same seed conventions) are added, so 20 reps x 12 targets per k.
Only ot_pairs, uot, union, sb_half and indep are scored. Imports the frozen modules read-only.
Usage: python j_noise.py [--which atlas|ery|both]   -> j_noise.json
"""
import argparse
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
DATA = {"atlas": os.path.join(SCR, "atlas"), "ery": os.path.join(SCR, "ery")}
STORED = {"atlas": os.path.join(HERE, "..", "confirm.json"), "ery": os.path.join(HERE, "..", "parity_erythroid.json")}
KS = {"atlas": list(range(1, 8)), "ery": [2, 3, 4, 5]}
SCORE = ("union", "ot_pairs", "uot", "sb_half", "indep")
NREP = 20
TSEEDS = list(range(1, 12))


def load(data, k):
    d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh = d["Z_held"].astype(np.float64)
    comps = d["pca_comps"].astype(np.float64)
    smp_held = d["sample_held"].astype(str)
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in set(IB.S_GENES + IB.G2M_GENES) for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    cc_fit = (Xcc - Xcc.mean(0)).mean(1)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    return Zf, Xf, Zh, comps, smp_held, cc_fit, A, B


def run(which):
    stored = json.load(open(STORED[which]))["interp"]
    res = {}
    for k in KS[which]:
        t0 = time.time()
        Zf, Xf, Zh, comps, smp_held, cc_fit, A, B = load(DATA[which], k)
        S = IB.med_scale(Zf)
        n_pred = min(len(Zh), IB.NSUB)
        targets = {}
        for ts in TSEEDS:
            it = IB.sub(len(Zh), seed=ts)
            targets[f"seed{ts}"] = IB.Target(Zh[it], None, S, with_sd=False)
        it1 = IB.sub(len(Zh), seed=1)
        comp = np.setdiff1d(np.arange(len(Zh)), it1)
        if len(comp) >= 100:
            targets["complement"] = IB.Target(Zh[comp[IB.sub(len(comp), seed=1)]], None, S, with_sd=False)
        pools = [p for p in sorted(set(smp_held)) if (smp_held == p).sum() >= 20]
        if len(pools) > 1:
            for p in pools:
                rest = np.where(smp_held != p)[0]
                targets[f"lopo_{p}"] = IB.Target(Zh[rest[IB.sub(len(rest), seed=1)]], None, S, with_sd=False)
        sc = {t: {m: [] for m in SCORE} for t in targets}
        repro = []
        for r in range(NREP):
            rng = np.random.default_rng(1000 * k + r)
            iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
            ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
            M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
            M.pop("_plan", None)
            M.pop("_uot", None)
            preds = {}
            for name, (Z, X, w, gm) in M.items():
                Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
                if name in SCORE:
                    preds[name] = (Zs, ws)
            for t, tg in targets.items():
                for m in SCORE:
                    sc[t][m].append(IB.ed_uv(preds[m][0], preds[m][1], tg)[0])
            if r < 3:
                repro.append(max(abs(sc["seed1"][m][r] - stored[str(k)]["reps"][r][m]["ED_U"]) for m in SCORE))
        # summaries
        def frac(a, b, tkeys, reps):
            x = np.array([[sc[t][a][r] - sc[t][b][r] for r in reps] for t in tkeys])
            return {"n": int(x.size), "frac_A_lt_B": float((x < 0).mean()), "mean_gap": float(x.mean()),
                    "sd_gap": float(x.std(ddof=1)) if x.size > 1 else 0.0, "min_gap": float(x.min()), "max_gap": float(x.max())}
        seed_keys = [f"seed{ts}" for ts in TSEEDS]
        out = {"n_held": int(len(Zh)), "pools": pools, "repro_maxdiff_frozen_reps": float(max(repro)),
               "targets": list(targets)}
        for a in ("ot_pairs", "uot", "sb_half"):
            out[f"{a}_vs_union"] = {
                "frozen_reps_seed1": frac(a, "union", ["seed1"], range(3)),
                "frozen_reps_seeds1_11": frac(a, "union", seed_keys, range(3)),
                "frozen_reps_complement": frac(a, "union", ["complement"], range(3)) if "complement" in targets else None,
                "all20_reps_seeds1_11": frac(a, "union", seed_keys, range(NREP)),
                "all20_reps_complement": frac(a, "union", ["complement"], range(NREP)) if "complement" in targets else None,
                "lopo_frozen_reps": {t: frac(a, "union", [t], range(3)) for t in targets if t.startswith("lopo_")},
            }
        # unit-level means (as the frozen verdict uses: mean over 3 reps), per target
        out["unit_mean_per_target_frozen_reps"] = {t: {m: float(np.mean(sc[t][m][:3])) for m in SCORE} for t in targets}
        out["unit_mean_per_target_all20"] = {t: {m: float(np.mean(sc[t][m])) for m in SCORE} for t in targets}
        out["sd_over_20reps_seed1"] = {m: float(np.std(sc["seed1"][m], ddof=1)) for m in SCORE}
        out["sd_over_targets_rep0"] = {m: float(np.std([sc[t][m][0] for t in seed_keys], ddof=1)) for m in SCORE}
        res[str(k)] = out
        g = out["ot_pairs_vs_union"]
        print(which, "k", k, "repro", f"{out['repro_maxdiff_frozen_reps']:.1e}", "| ot<union frac: frozen x seeds",
              round(g["frozen_reps_seeds1_11"]["frac_A_lt_B"], 3), "complement",
              None if g["frozen_reps_complement"] is None else round(g["frozen_reps_complement"]["frac_A_lt_B"], 3),
              "all20 x seeds", round(g["all20_reps_seeds1_11"]["frac_A_lt_B"], 3), "mean gap",
              round(g["all20_reps_seeds1_11"]["mean_gap"], 4), "sd", round(g["all20_reps_seeds1_11"]["sd_gap"], 4),
              "| lopo", {t[5:]: round(v["mean_gap"], 3) for t, v in g["lopo_frozen_reps"].items()},
              f"[{time.time() - t0:.0f}s]", flush=True)
    return res


def verdict_under_targets(res):
    """EXPLORATORY: the H1 count (ot_pairs vs union only, 3 frozen reps) recomputed under each alternative target."""
    ks = sorted(res, key=int)
    tk = [t for t in res[ks[0]]["targets"] if t.startswith("seed") or t == "complement"]
    out = {}
    for t in tk:
        if not all(t in res[k]["unit_mean_per_target_frozen_reps"] for k in ks):
            continue
        u = {k: res[k]["unit_mean_per_target_frozen_reps"][t] for k in ks}
        out[t] = {"ot_pairs_lt_union": sum(u[k]["ot_pairs"] < u[k]["union"] for k in ks),
                  "uot_lt_union": sum(u[k]["uot"] < u[k]["union"] for k in ks), "of": len(ks)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both")
    args = ap.parse_args()
    out = {}
    for w in (["atlas", "ery"] if args.which == "both" else [args.which]):
        out[w] = run(w)
        out[w + "_counts_by_target"] = verdict_under_targets(out[w])
        print(w, "counts by target", out[w + "_counts_by_target"], flush=True)
    json.dump(out, open(os.path.join(HERE, "j_noise.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
