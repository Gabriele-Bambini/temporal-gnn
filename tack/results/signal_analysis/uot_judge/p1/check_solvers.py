"""Unit checks for the sink / KL-UOT solvers and timing of the strict fits (run before uot_sink.py)."""
import sys
import time

import numpy as np

import uot_sink as U

rng = np.random.default_rng(0)
K = 6
C = rng.random((K, K))
C = (C + C.T) / 2
np.fill_diagonal(C, 0)
C /= C[~np.eye(K, dtype=bool)].mean()
a = rng.random((2, K)) * np.array([[100.0], [30.0]])
b = rng.random((2, K)) * np.array([[40.0], [90.0]])
P, aa, bb = U.plans_sink(a, b, C, 1.0, 0.05)
print("sink: death-birth", P[:, :K, K].sum(1) - P[:, K, :K].sum(1), "N_src-N_dst", aa.sum(1) - bb.sum(1))
print("sink: row marg err", np.abs(P[:, :K, :].sum(2) - aa).max(), "col marg err", np.abs(P[:, :, :K].sum(1) - bb).max())
# equal totals, very expensive sink -> balanced plan
pa = a / a.sum(1, keepdims=True)
pb = b / b.sum(1, keepdims=True)
Ps, _, _ = U.plans_sink(pa, pb, C, 50.0, 0.05)
Pb = U.plans_balanced(pa, pb, C, 0.05)
print("sink(lam=50) vs balanced, max |diff| of transport block", np.abs(Ps[:, :K, :K] - Pb).max(),
      "death mass", Ps[:, :K, K].sum(1))
# KL-UOT with large tau and equal totals -> balanced
Pk, ka, kb = U.plans_kl(pa, pb, C, 1e4, 0.05)
print("kl(tau=1e4) vs balanced", np.abs(Pk - Pb).max(), "row sums", Pk.sum(2)[0][:3], pa[0][:3])
Pk, ka, kb = U.plans_kl(a, b, C, 1.0, 0.05)
print("kl(tau=1) totals", Pk.sum((1, 2)), "src", a.sum(1), "dst", b.sum(1))
# cheap sink: more turnover
for lam in (0.25, 0.5, 1, 2):
    Ps, aa, _ = U.plans_sink(pa, pb, C, lam, 0.05)
    print("sink_norm lam", lam, "turnover share", Ps[:, :K, K].sum(1) / aa.sum(1))

name = sys.argv[1] if len(sys.argv) > 1 else "pancreas-human"
t0 = time.time()
d = U.Data(name)
print(name, "load", round(time.time() - t0, 1), "s; cells", np.round(d.n).astype(int).tolist())
for K in (4, 8, 16):
    t0 = time.time()
    i = d.T - 1
    fit = U.Fit(d, i, min(K, i))
    rec = fit.pi @ fit.A
    err = np.abs(rec - d.m[:, :i + 1].T).sum(1) / d.m[:, :i + 1].T.sum(1)
    print("K", K, "K_eff", fit.K, "fit", round(time.time() - t0, 2), "s; rel L1 recon err per bin", np.round(err, 3)[:6],
          "cell ratio", np.round(fit.cell_sum_ratio, 2)[:6], "UMI/cell", np.round(1 / fit.c, 0))
    print("   pi last 3 bins", np.round(fit.pi[-3:], 3))
