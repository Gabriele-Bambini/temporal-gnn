"""JUDGE, EXPLORATORY (ORACLE-labelled: settings judged against the held stage, never a verdict).
(a) Is the ot_pairs loss an entropic-blur artefact? ot_pairs at eps_rel 0.01 / 0.02 / 0.1 / 0.2 (frozen 0.05).
(b) Does the thin S/G2M prior (3-8 HVG genes) cost uot (H1b)? uot with the stored 96-gene score (interp npz cc_fit,
    centred on fit cells) and with no growth prior (beta 0), frozen tau 1.
Atlas k = 1..7, 3 frozen fit subsamples (seeds 1000+r / 2000+r), frozen seed-1 target, IB.methods(which=...) with
rng default_rng(1000 k + r) (pair draws differ from the frozen run), compared with the frozen union / prev / next.
Usage: python j_eps_cc.py -> j_eps_cc.json
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
DATA = os.path.join(SCR, "atlas")
C = json.load(open(os.path.join(HERE, "..", "confirm.json")))["interp"]
EPS = (0.01, 0.02, 0.1, 0.2)


def main():
    res = {}
    for k in range(1, 8):
        t0 = time.time()
        d = np.load(os.path.join(DATA, f"interp{k}.npz"), allow_pickle=True)
        st = d["stage_fit"]
        Zf, Xf, Zh = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32), d["Z_held"].astype(np.float64)
        comps = d["pca_comps"].astype(np.float64)
        cc96 = d["cc_fit"].astype(np.float64)
        A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
        S = IB.med_scale(Zf)
        n_pred = min(len(Zh), IB.NSUB)
        tgt = IB.Target(Zh[IB.sub(len(Zh), seed=1)], None, S, with_sd=False)
        sc = {f"ot_pairs_eps{e}": [] for e in EPS}
        sc.update({"uot_cc96": [], "uot_beta0": []})
        for r in range(3):
            iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
            ZA, XA = Zf[A[iA]], Xf[A[iA]].astype(np.float64)
            ZB, XB = Zf[B[iB]], Xf[B[iB]].astype(np.float64)
            for e in EPS:
                rng = np.random.default_rng(1000 * k + r)
                M = IB.methods(ZA, XA, ZB, XB, cc96[A[iA]], comps, n_pred, rng, which=("ot_pairs",), eps_rel=e)
                Z, X, w, _ = M["ot_pairs"]
                sc[f"ot_pairs_eps{e}"].append(IB.ed_uv(Z, w, tgt)[0])
            for lab, cc, beta in (("uot_cc96", cc96[A[iA]], 2.0), ("uot_beta0", np.zeros(len(iA)), 0.0)):
                rng = np.random.default_rng(1000 * k + r)
                M = IB.methods(ZA, XA, ZB, XB, cc, comps, n_pred, rng, which=("uot",), beta=beta)
                Z, X, w, _ = M["uot"]
                sc[lab].append(IB.ed_uv(Z, w, tgt)[0])
        mean = {m: float(np.mean(v)) for m, v in sc.items()}
        fr = {m: C[str(k)]["mean"][m]["ED_U"] for m in ("prev", "next", "union", "ot_pairs", "uot")}
        res[str(k)] = {"mean": mean, "frozen": fr}
        print("k", k, {m: round(v, 3) for m, v in mean.items()}, "| frozen", {m: round(v, 3) for m, v in fr.items()},
              f"[{time.time() - t0:.0f}s]", flush=True)
    ks = [str(k) for k in range(1, 8)]
    summ = {}
    for m in list(res["1"]["mean"]):
        vals = {k: res[k]["mean"][m] for k in ks}
        fr = {k: res[k]["frozen"] for k in ks}
        summ[m] = {"lt_union": sum(vals[k] < fr[k]["union"] for k in ks),
                   "beats_prev_next_union": sum(all(vals[k] < fr[k][b] for b in ("prev", "next", "union")) for k in ks),
                   "mean": float(np.mean(list(vals.values()))),
                   "mean_union": float(np.mean([fr[k]["union"] for k in ks]))}
    res["summary"] = summ
    print(json.dumps(summ, indent=0))
    json.dump(res, open(os.path.join(HERE, "j_eps_cc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
