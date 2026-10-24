"""Which mask draw reproduces the CONTEXT trend numbers (l1: gastr -0.169 / 0.102 / 0.092, pancr 0.159 / 0.223 /
0.220)? Variants of 'a fixed random 25 % of genes per transition, default_rng(1000 s + i)'."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uot_core as U  # noqa: E402

sys.path.insert(0, U.HPC)
import dileo_protocol  # noqa: E402
import dileo_rescore  # noqa: E402

for name in ("pancreas-human", "gastrulation-human"):
    X, trip, n, m = U.load(name)
    T, G = len(trip) - 1, trip[0].shape[0]
    nt = int(round(0.25 * G))
    for tg in ("l1", "linear"):
        Y = np.stack([dileo_protocol.TARGET_NP[tg](m[i], m[i + 1]) for i in range(T)])
        variants = {
            "perm[:n]": lambda r: r.permutation(G)[:nt],
            "perm[G-n:]": lambda r: r.permutation(G)[G - nt:],
            "choice": lambda r: r.choice(G, nt, replace=False),
            "random<0.25": lambda r: np.where(r.random(G) < 0.25)[0],
        }
        for vn, fn in variants.items():
            for seed_rule in ("1000s+i", "s*1000+i,s=1..3"):
                ss = (0, 1, 2) if seed_rule == "1000s+i" else (1, 2, 3)
                rows = []
                for s in ss:
                    r_s = []
                    for i in range(1, T):
                        t = np.sort(fn(np.random.default_rng(1000 * s + i)))
                        mm = dileo_rescore.metrics(Y[i][t], Y[i - 1][t])
                        r_s.append([mm["pearson"], mm["p_ta_up"], mm["p_ta_down"], mm["p_at_200_up"], mm["p_at_200_down"]])
                    rows.append(np.nanmean(r_s, 0))
                v = np.mean(rows, 0)
                print(f"{name} {tg} {vn:12s} {seed_rule:16s} trend {v[0]:+.3f} / {v[1]:.3f} / {v[2]:.3f} [{v[3]:.3f} / {v[4]:.3f}]")
        # all-gene and per-transition reference
        allg = np.nanmean([np.corrcoef(Y[i], Y[i - 1])[0, 1] for i in range(1, T)])
        print(f"{name} {tg} all genes trend PCC {allg:+.3f}")
