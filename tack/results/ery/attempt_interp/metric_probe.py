"""DESCRIPTIVE metric probe (uses the held stage; not a prediction method): do the metrics of interp_bench.py reward
a CONTRACTED cloud at these sample sizes?

For each held stage k: split the held cells into two random halves H1, H2 (each <= 1500). 'Prediction' = H1 shrunk
toward its own mean by lambda (lambda = 1 is a perfect replicate sample of the true stage; lambda = 0 puts all mass
at one point). Scored against H2 with sd_fixed (fixed scale from the fit cells, as in the benchmark), ED_U, ED_V,
W2 and W1 (exact assignment). A metric whose minimum sits at lambda < 1 rewards contraction at this sample size
(finite-sample bias), so its ranking of contracted interpolants (barycentric maps, independent midpoints) is not
evidence of accuracy. Also scored: the STRICT degenerate predictor 'midmean' (all mass at the midpoint of the stage
k-1 and k+1 means; uses fit cells only) against the benchmark's own target subsample.
Usage: python -I metric_probe.py --data <folder with interp*.npz>; writes metric_probe.csv, metric_probe.log here.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import interp_bench as ib  # noqa: E402

LAMBDAS = [0.0, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    ib.LOGF = os.path.join(HERE, "metric_probe.log")
    open(ib.LOGF, "w").close()
    rows = []
    for k in (2, 3, 4, 5):
        d = np.load(os.path.join(args.data, f"interp{k}.npz"), allow_pickle=True)
        Zf, Zh, st = d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64), d["stage_fit"]
        S = ib.med_scale(Zf)
        perm = np.random.default_rng(4).permutation(len(Zh))
        h1, h2 = perm[: len(Zh) // 2][: ib.NSUB], perm[len(Zh) // 2:][: ib.NSUB]
        m_ = min(len(h1), len(h2))
        h1, h2 = h1[:m_], h2[:m_]
        tgt = ib.Target(Zh[h2], None, S)
        H1 = Zh[h1]
        mu = H1.mean(0)
        for lam in LAMBDAS:
            P = mu + lam * (H1 - mu)
            sc = ib.basic_scores(P, ib.unif(len(P)), tgt) if lam > 0 else None
            if lam == 0:  # one point carrying all the mass
                Pp = mu[None, :]
                eu, ev = ib.ed_uv(np.repeat(Pp, 2, 0), np.array([0.5, 0.5]), tgt)  # two coincident points
                w1, w2 = ib.w_exact(Pp, np.array([1.0]), tgt.Z)
                sc = {"sd_fixed": ib.sd_fixed(Pp, np.array([1.0]), tgt), "ED_U": eu, "ED_V": ev, "W1": w1, "W2": w2}
            rows.append({"k": k, "probe": "true_half_shrunk", "lambda": lam, "n_each": int(m_), **sc})
        # strict degenerate predictor against the benchmark target (same subsample seed as interp_bench)
        it = ib.sub(len(Zh), seed=1)
        tgt_b = ib.Target(Zh[it], None, S)
        mid = 0.5 * (Zf[st == k - 1].mean(0) + Zf[st == k + 1].mean(0))[None, :]
        eu, ev = ib.ed_uv(np.repeat(mid, 2, 0), np.array([0.5, 0.5]), tgt_b)
        w1, w2 = ib.w_exact(mid, np.array([1.0]), tgt_b.Z)
        rows.append({"k": k, "probe": "midmean_point_mass (strict)", "lambda": np.nan, "n_each": int(len(it)),
                     "sd_fixed": ib.sd_fixed(mid, np.array([1.0]), tgt_b), "ED_U": eu, "ED_V": ev, "W1": w1, "W2": w2})
        ib.log(pd.DataFrame([r for r in rows if r["k"] == k]).round(3).to_string())
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "metric_probe.csv"), index=False)


if __name__ == "__main__":
    main()
