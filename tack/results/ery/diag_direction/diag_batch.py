"""Batch check for the mean shifts (DESCRIPTIVE: uses the true stage s+1; not a forecast).

The stage-to-stage mean shifts of the erythroid lineage zig-zag (consecutive shifts nearly orthogonal, shifts two
stages apart aligned). Stages mix sequencing batches differently (E7.75: batches 1+2; E8.0: 2+3; E8.25: 2 only;
E8.5: 2+3), and sequencing batch 2 is the only batch present at EVERY stage. Here, in each origin's PCA space:
  batch vectors  mean(batch b) - mean(batch b') inside one stage (batches with >= 50 cells), their size relative to
                 the stage shift and their cosine with the shift into / out of that stage;
  batch-2 chain  mean shifts computed on batch-2 cells only (same sequencing batch on both sides), their
                 consecutive cosines (vs the all-cell ones) and the cosine of the mean velocity of the batch-2
                 stage-s cells with the batch-2 shift s -> s+1.
Output: batch.json next to this file.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_direction as dd  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ERY_DATA", "")
MINB = 50


def main():
    obs = pd.read_csv(os.path.join(DATA, "obs.csv"))
    ostage = np.array([dd.STAGES.index(x) for x in obs["stage"].astype(str)])
    out = {"note": "DESCRIPTIVE (uses stage s+1)", "origins": {}}
    for s in range(1, 6):
        f = np.load(os.path.join(DATA, f"origin{s}.npz"), allow_pickle=True)
        st = f["stage_past"]
        Zp, Zf = f["Z_past"].astype(np.float64), f["Z_fut"].astype(np.float64)
        bp = obs["sequencing.batch"].to_numpy()[ostage <= s]
        bf = obs["sequencing.batch"].to_numpy()[ostage == s + 1]
        Z = {k: Zp[st == k] for k in range(s + 1)}
        Z[s + 1] = Zf
        B = {k: bp[st == k] for k in range(s + 1)}
        B[s + 1] = bf
        mu = {k: Z[k].mean(0) for k in Z}
        sh = {k: mu[k + 1] - mu[k] for k in range(s + 1)}           # shift k -> k+1, all cells
        o = {"batch_counts": {dd.STAGES[k]: {int(b): int((B[k] == b).sum()) for b in np.unique(B[k])} for k in Z}}
        # batch vectors inside a stage
        bv = {}
        for k in (s, s + 1):
            bs = [b for b in np.unique(B[k]) if (B[k] == b).sum() >= MINB]
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    v = Z[k][B[k] == bs[j]].mean(0) - Z[k][B[k] == bs[i]].mean(0)
                    bv[f"{dd.STAGES[k]}: b{bs[j]}-b{bs[i]}"] = {
                        "norm_over_shift_next": float(np.linalg.norm(v) / np.linalg.norm(sh[s])),
                        "cos_with_shift_next": dd.cosv(v, sh[s]),
                        "cos_with_shift_prev": dd.cosv(v, sh[s - 1])}
        o["batch_vectors"] = bv
        # batch-2-only shifts
        has2 = [k for k in Z if (B[k] == 2).sum() >= 20]
        mu2 = {k: Z[k][B[k] == 2].mean(0) for k in has2}
        if all(k in mu2 for k in (s - 1, s, s + 1)):
            p2, n2 = mu2[s] - mu2[s - 1], mu2[s + 1] - mu2[s]
            o["batch2"] = {"n": {dd.STAGES[k]: int((B[k] == 2).sum()) for k in (s - 1, s, s + 1)},
                           "cos_prev_next_b2": dd.cosv(p2, n2), "cos_prev_next_all": dd.cosv(sh[s - 1], sh[s]),
                           "cos_next_b2_vs_all": dd.cosv(n2, sh[s]),
                           "norm_next_b2_over_all": float(np.linalg.norm(n2) / np.linalg.norm(sh[s])),
                           "norm_next_b2_over_rms_b2": float(np.linalg.norm(n2) / np.sqrt(
                               ((Z[s][B[s] == 2] - mu2[s]) ** 2).sum(1).mean()))}
            rows = np.where(st == s)[0][B[s] == 2]
            vel = dd.velocity_pca(f, rows, f["pca_comps"].astype(np.float64), f["murk"].astype(bool))
            for (vk, mk), (Vz, _) in vel.items():
                o["batch2"][f"cos_mean_vel_b2shift|{vk}|{mk}"] = dd.cosv(Vz.mean(0), n2)
                o["batch2"][f"median_cell_cos_vel_b2shift|{vk}|{mk}"] = float(np.median(dd.cos_rows(Vz, n2[None])))
        if s == 5:
            ks = sorted(mu2)
            c2 = [mu2[ks[i + 1]] - mu2[ks[i]] for i in range(len(ks) - 1)]
            o["chain_batch2"] = {"transitions": [f"{dd.STAGES[ks[i]]}->{dd.STAGES[ks[i + 1]]}" for i in range(len(ks) - 1)],
                                 "n": [int((B[k] == 2).sum()) for k in ks],
                                 "cos_consecutive": [dd.cosv(c2[i], c2[i + 1]) for i in range(len(c2) - 1)],
                                 "cos_matrix": [[round(dd.cosv(a, b), 3) for b in c2] for a in c2],
                                 "norm": [float(np.linalg.norm(x)) for x in c2]}
        out["origins"][s] = o
    with open(os.path.join(HERE, "batch.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
