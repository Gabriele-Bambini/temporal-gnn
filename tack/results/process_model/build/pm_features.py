"""Step 2: the new l1 features (pm_lib.add_new) and their strictness checks.

Checks per dataset:
  (1) L used in fgap equals log1p(1e4 m / lib) of each bin (within 1e-9);
  (2) NaN strictness: every row i >= 1 recomputed with bins > i set to NaN and graphs of bins > i removed equals the
      stored row exactly (pm_lib.strict_nan_check), and lag1 row i equals Y[i-1] exactly;
  (3) descriptive stats: share of targets with a regulator (has_reg, has_regbar), per-transition correlation of
      grnbar_lag1 with grn_lag1 and with its null, fgap / fmom spread.
Output: pm_features.json, pm_features.log."""
import json
import os
import time

import numpy as np

import pm_lib as L

D, NS = L.D, L.NS


def main():
    log = open(os.path.join(L.B, "pm_features.log"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    out = {}
    for name in L.NAMES:
        t0 = time.time()
        d = D.Data(name)
        graph = NS.load_graph(d)
        Y = d.Y("l1")
        F, cls = NS.build_features(d, "l1", Y, graph, None)
        L.add_new(d, F, graph)
        Lchk = np.log1p(1e4 * d.m / d.lib[None, :])
        e1 = float(np.max(np.abs(Lchk - d.L)))
        assert e1 < 1e-9
        worst = L.strict_nan_check(d, F, graph)
        rows = range(1, d.T)
        cor = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        o = {"max_abs_L_minus_log1p_CP": e1, "nan_strictness_max_abs_diff": worst,
             "has_reg_share": [round(float(F["has_reg"][i].mean()), 4) for i in rows],
             "has_regbar_share": [round(float(F["has_regbar"][i].mean()), 4) for i in rows],
             "corr_grnbar_vs_grn_lag1": [round(cor(F["grnbar_lag1"][i], F["grn_lag1"][i]), 4) for i in range(2, d.T)],
             "corr_grnbar_vs_null": [round(cor(F["grnbar_lag1"][i], F["grnbarnull_lag1"][i]), 4) for i in range(2, d.T)],
             "fgap_sd": [round(float(F["fgap"][i].std()), 4) for i in rows],
             "fmom_sd": [round(float(F["fmom"][i].std()), 4) for i in rows],
             "seconds": round(time.time() - t0)}
        out[name] = o
        say(f"== {name}: L check {e1:.2e}; NaN strictness max diff {worst} (rows 1..{d.T - 1} exact) ({o['seconds']}s)")
        say(f"  has_reg share mean {np.mean(o['has_reg_share']):.3f}, has_regbar {np.mean(o['has_regbar_share']):.3f}")
        say(f"  corr(grnbar, grn_lag1) mean {np.nanmean(o['corr_grnbar_vs_grn_lag1']):.3f}; "
            f"corr(grnbar, null) mean {np.nanmean(o['corr_grnbar_vs_null']):.3f}")
        say(f"  fgap sd mean {np.mean(o['fgap_sd']):.4f}, fmom sd mean {np.mean(o['fmom_sd']):.4f}")
        del F, graph
    json.dump(out, open(os.path.join(L.B, "pm_features.json"), "w"), indent=1)
    log.close()


if __name__ == "__main__":
    main()
