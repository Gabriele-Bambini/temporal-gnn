"""Quick look 6: what predicts the change in the transitions where NASS fails (gastr early, pancreas early)?
Per transition, corr of y_i with lag1, lag2, acceleration (lag1 - lag2), on genes with CP10k in (1, 100] at bin i
(the persistent classes) and on all genes; plus corr(lag1, lag2) of the TRUE-ish change in those genes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quicklook import load, corr
for name in ["gastrulation-human", "pancreas-human"]:
    m, med, s, f, tot, n = load(name)
    G, B = m.shape; T = B - 1
    lib = m.sum(0); kap = 1e4 / lib; CP = m * kap[None, :]; L = np.log1p(CP)
    Y = np.diff(L, axis=1).T
    print(f"===== {name}")
    print("   i  n_i->n_i+1   | hi-genes: lag1   lag2   accel  | all: lag1  lag2  | #hi")
    for i in range(2, T):
        hi = (CP[:, i] > 1) & (CP[:, i] <= 100) & (CP[:, i - 1] > 1)
        a = corr(Y[i][hi], Y[i - 1][hi]); b = corr(Y[i][hi], Y[i - 2][hi]); c = corr(Y[i][hi], (Y[i - 1] - Y[i - 2])[hi])
        print(f"  {i:2d}  {int(n[i]):4d}->{int(n[i+1]):4d}   |   {a:+.2f}  {b:+.2f}  {c:+.2f}  |  {corr(Y[i], Y[i-1]):+.2f} {corr(Y[i], Y[i-2]):+.2f} | {hi.sum()}")
