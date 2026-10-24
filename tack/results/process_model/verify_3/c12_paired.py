"""C12 paired check: per-transition PCC (mean over the 3 masks) of each p_hat variant minus base, rolling pools, i >= 1."""
import json
import os

import numpy as np
from scipy import stats

import common as C

out = {}
for name in C.NAMES:
    base = np.load(os.path.join(C.HERE, f"c12_rows_base_{name}.npy"))
    o = {}
    for v in ("A_phat", "B_phat+post0", "C_post_all", "D_base+post0"):
        r = np.load(os.path.join(C.HERE, f"c12_rows_{v}_{name}.npy"))
        for j, k in ((0, "pcc"), (1, "up"), (2, "down")):
            dl = r[:, 1:, j].mean(0) - base[:, 1:, j].mean(0)
            o[f"{v} - base {k}"] = {"mean": round(float(dl.mean()), 4), "better": int((dl > 1e-9).sum()),
                                    "worse": int((dl < -1e-9).sum()), "wilcoxon_p": round(float(stats.wilcoxon(dl).pvalue), 4)}
    out[name] = o
    for k, x in o.items():
        print(C.SHORT[name], k, x)
json.dump(out, open(os.path.join(C.HERE, "c12_paired.json"), "w"), indent=1)
