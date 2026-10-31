"""EXPLORATORY (not pre-registered): the wider-gap test of diag_gap.py on the zebrafish GAP14 cells (zf_prep.load).
For gap g = 1..5 (2..10 h) and held timepoint k (g <= k <= 10 - g): fit = k-g and k+g, held = k; the frozen methods
and the same extra numbers as diag_gap.unit (separation, oracle mixture, per-embryo floor, per-type separation).
Pool = embryo. Output ~/tack/out/zf/diag_gap_zf.json.
"""
import importlib
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_gap as DG  # noqa: E402
import zf_prep as ZP  # noqa: E402

OUT = os.path.expanduser("~/tack/out/zf/diag_gap_zf.json")


def main():
    X, obs, genes = ZP.load()
    stage = np.array([ZP.TPS.index(t) for t in obs["timepoint"]])
    pool = obs["embryo"].astype(str).to_numpy()
    ct = obs["cell_type_broad"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc_genes = set(str(x).capitalize() for x in list(s_g) + list(g2m))
    res = {"units": []}
    for g in (1, 2, 3, 4, 5):
        for k in range(g, 11 - g):
            u = DG.unit(X, stage, pool, ct, genes, cc_genes, k, g)
            res["units"].append(u)
            print("g", g, "k", k, "sep", round(u["separation"], 2), {m: round(v, 3) for m, v in u["mean"].items()},
                  "oracle_mix", u["oracle_mix"], f"[{u['sec']}s]", flush=True)
            with open(OUT, "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
