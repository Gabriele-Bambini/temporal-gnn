import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
for name in SNAPS:
    d = load(name)
    print(name, d["m"].shape, "cells", d["n"].astype(int).tolist())
    print("  all-zero genes over snapshots:", int((d["m"].max(1) == 0).sum()))
    for target in TARGETS:
        Y = targets(d, target, name)
        P = np.zeros_like(Y); P[1:] = Y[:-1]
        sc = Scorer(Y.shape[1], len(Y), n_draws=10)
        r = sc.summary(Y, P)
        print("  trend", target, round(r["pcc"], 3), round(r["p200_up"], 3), round(r["p200_down"], 3))
