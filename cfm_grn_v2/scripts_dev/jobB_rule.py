"""Pre-registration v0.2 Step 3 rule applied by code to the Job B outputs (frozen v0.1.10 on EB windows 1 and 3).
Run from cfm_grn_v2/:  python scripts_dev/jobB_rule.py  -> outputs/jobB_verdict.json"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path("outputs")
NOT_CONTROLS = {"flow_matching", "flow_matching_shuffled_time", "noise_floor"}
out = {}
for name, margin in (("eb_v0110_t1", -0.04), ("eb_v0110_t3", -0.04), ("eb_v0110_pca50_t1", 0.0),
                     ("eb_v0110_pca50_t3", 0.0)):
    df = pd.read_csv(HERE / name / "results.csv")
    ed = df.pivot_table(index="seed", columns="model", values="energy_distance")
    flow = ed["flow_matching"]
    controls = [c for c in ed.columns if c not in NOT_CONTROLS]
    means = ed.mean()
    best = min(controls, key=lambda c: means[c])

    def upper(diff):
        diff = np.asarray(diff, float)
        n = len(diff)
        return float(diff.mean() + stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n))

    below_all = bool(all(means["flow_matching"] < means[c] for c in controls))
    ub_best = upper(flow - ed[best])
    ub_shuf = upper(flow - ed["flow_matching_shuffled_time"])
    passes = below_all and ub_best < margin and ub_shuf < 0
    out[name] = {"flow": round(float(means["flow_matching"]), 4), "best_control": best,
                 "best_control_ed": round(float(means[best]), 4),
                 "diff_best": round(float((flow - ed[best]).mean()), 4), "upper_ci_best": round(ub_best, 4),
                 "margin": margin, "shuffled": round(float(means["flow_matching_shuffled_time"]), 4),
                 "upper_ci_shuffled": round(ub_shuf, 4), "below_every_control": below_all, "passes": bool(passes),
                 "all_means": {k: round(float(v), 4) for k, v in means.sort_values().items()},
                 "n_seeds": int(len(flow))}
gene = [out["eb_v0110_t1"]["passes"], out["eb_v0110_t3"]["passes"]]
if all(gene):
    wording = "beats every control on each inner EB window in gene space (windows 1 and 3 were earlier selection folds)"
elif any(gene):
    k = 1 if not gene[0] else 3
    wording = f"positive on 2 of 3 windows; fails on window {k}"
else:
    wording = "the EB positive does not extend beyond days 12-15"
out["wording_gene_space"] = wording
print(json.dumps(out, indent=1))
(HERE / "jobB_verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
