"""Extra paired tests from results.json (per-transition, mask-averaged PCC and J = PCC + P@200 up + down):
the primary strict model against the best reproduction of the variance analyst's recency-weighted filter
(regime with fixed h = 0.5) and against the trend. Writes extra_paired.json."""
import json
import os

import numpy as np
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
PAIRS = (("full", "regime[h=0.5 fixed]"), ("full+grn", "regime[h=0.5 fixed]"), ("full", "trend"),
         ("full+grn", "trend"), ("full+grn", "full"), ("full+ot", "full"), ("full+grn", "full+grn_null"))


def test(a, b):
    out = {}
    for k, va, vb in (("pcc", a[:, 0], b[:, 0]), ("J", a.sum(1), b.sum(1))):
        dl = np.nan_to_num(va) - np.nan_to_num(vb)
        nz = dl[np.abs(dl) > 1e-12]
        out[k] = {"mean_delta": round(float(dl.mean()), 4), "better": int((dl > 1e-12).sum()),
                  "worse": int((dl < -1e-12).sum()),
                  "wilcoxon_p": round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None}
    return out


res = json.load(open(os.path.join(OUT, "results.json")))
out = {}
for name, r in res.items():
    for tg, t in r["targets"].items():
        pt = {k: np.array(v, float) for k, v in t["per_transition_pcc_up_down"].items()}
        for start in (2, 3):
            for a, b in PAIRS:
                if a in pt and b in pt:
                    v = test(pt[a][start:], pt[b][start:])
                    out[f"{name}|{tg}|i>={start}|{a} vs {b}"] = v
                    print(f"{name[:5]} {tg:6s} i>={start} {a:9s} vs {b:20s} dPCC {v['pcc']['mean_delta']:+.3f} "
                          f"({v['pcc']['better']}/{v['pcc']['worse']}, p={v['pcc']['wilcoxon_p']})  dJ {v['J']['mean_delta']:+.3f} "
                          f"({v['J']['better']}/{v['J']['worse']}, p={v['J']['wilcoxon_p']})")
json.dump(out, open(os.path.join(OUT, "extra_paired.json"), "w"), indent=1)
