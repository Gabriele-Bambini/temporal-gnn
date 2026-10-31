"""Kernel-domain Sinkhorn (ery_forecast_fixed.sinkhorn) vs the original log-domain solver, on real couplings.

Cases: the ot / uot_growth couplings of origins 2 and 3 (balanced; tau 0.1 / 1.0 with beta 0 / 5), one metric-sized
balanced problem, the symmetric self-term solver against the alternating one, and a forced underflow case that must
take the log-domain fallback. Prints max |dP| / max P, the barycentric drift difference and the solver path.
"""
import sys
import time

import numpy as np

import ery_forecast_fixed as ef

worst = 0.0
for s in (2, 3):
    d = ef.load(ef.DATA, s)
    st = d["stage_past"]
    Zp, Zn = d["Z_past"][st == s - 1], d["Z_past"][st == s]
    cc = d["cc_past"][st == s - 1]
    for tau, beta in ((None, 0.0), (0.1, 0.0), (0.1, 5.0), (1.0, 5.0)):
        a = None if tau is None else np.exp(beta * cc * ef.DT)
        C = ef.sqdist(Zp, Zn)
        scale = np.median(C)
        aa = np.full(len(Zp), 1 / len(Zp)) if a is None else a / a.sum()
        b = np.full(len(Zn), 1 / len(Zn))
        t = time.time()
        P0 = ef.sinkhorn_log(aa, b, C / scale, 0.05, tau)
        t0 = time.time() - t
        fb = ef.STATS["log_fallbacks"]
        t = time.time()
        P1 = ef.sinkhorn(aa, b, C / scale, 0.05, tau)
        t1 = time.time() - t
        rel = np.abs(P1 - P0).max() / P0.max()
        T0 = (P0 @ Zn) / P0.sum(1)[:, None]
        T1 = (P1 @ Zn) / P1.sum(1)[:, None]
        worst = max(worst, rel)
        print(f"origin {s} {Zp.shape[0]}x{Zn.shape[0]} tau={tau} beta={beta}: max|dP|/maxP={rel:.2e} "
              f"max|d drift|={np.abs(T1 - T0).max():.2e} fallback={ef.STATS['log_fallbacks'] > fb} "
              f"log {t0:.1f}s kernel {t1:.2f}s", flush=True)

# symmetric self term vs the alternating solver (metric sized)
d = ef.load(ef.DATA, 4)
st = d["stage_past"]
X = d["Z_past"][st == 4][ef.sub((st == 4).sum())]
S = ef.past_scale(d["Z_past"])
w = ef.unif(len(X))
C = ef.sqdist(X, X)
alt = float((ef.sinkhorn(w, w, C, 0.05 * S) * C).sum())
sym = ef.ot_self(X, w, 0.05 * S)
print(f"self term origin 4 (S={S:.2f}): alternating {alt:.8f} symmetric {sym:.8f} rel diff {abs(alt - sym) / sym:.1e}")

# forced underflow: two far clouds -> must fall back to the log domain and agree with it
r = np.random.default_rng(0)
A, B = r.normal(0, 1, (200, 5)), r.normal(40, 1, (300, 5))
C = ef.sqdist(A, B)
eps = 0.05 * np.median(ef.sqdist(A, A))
fb = ef.STATS["log_fallbacks"]
P1 = ef.sinkhorn(ef.unif(200), ef.unif(300), C, eps)
P0 = ef.sinkhorn_log(ef.unif(200), ef.unif(300), C, eps)
print(f"forced underflow: fallback taken={ef.STATS['log_fallbacks'] > fb} identical={np.array_equal(P0, P1)}")
print("WORST max|dP|/maxP over real couplings:", f"{worst:.2e}", "PASS" if worst < 1e-8 else "FAIL")
sys.exit(0 if worst < 1e-8 else 1)
