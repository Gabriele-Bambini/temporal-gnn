"""Do rarely detected genes dominate the TRAINING space (standardised on the fit cells, as train_flow does)?

A gene detected in a handful of fit cells has a tiny standard deviation there; its few expressing cells then sit
hundreds of units away from everyone else, and both the optimal-transport couplings (squared Euclidean cost) and
the regression loss are dominated by that one coordinate. This measures, on the fit cells of the reported design:
  - how many selected genes are detected in fewer than 1% of the fit cells (the SCENIC default filter);
  - the largest |z| any fit cell reaches, and the share of the total squared norm of the fit cells carried by the
    rare genes;
before and after the 1% filter. Uses only the fit cells: nothing of the held-out or validation stage is read.

  python scripts_dev/check_training_scale.py eb          (local, embryoid body)
  python scripts_dev/check_training_scale.py argelaguet  (HPC)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.training.search import interpolation_design

which = sys.argv[1]
if which == "eb":
    import eb_benchmark as B
    X, obs, symbols, _ = B.load_cache(B.CACHE)
    regs = B.prior_regulators(B.PRIOR)
    test, max_cells = 2.0, 2000
else:
    import argelaguet_benchmark as B
    X, obs, symbols, _ = B.load_cache(Path("~/rafm_data/processed/argelaguet2022_rna.npz").expanduser())
    course = (~obs["sample"].str.contains("CRISPR")) & (obs["genotype"] == "WT")
    X, obs = X[course.to_numpy()], obs.loc[course].reset_index(drop=True)
    obs["time"] = (obs["e_day"] - B.DAY0) / B.UNIT
    obs["replicate"], obs["condition"] = "pool", "WT"
    regs = set(pd.read_csv(Path("~/rafm_data/priors/collectri_mouse.tsv").expanduser(), sep="\t")
               ["source_genesymbol"].astype(str))
    test, max_cells = 2.0, 3000

times = sorted(float(t) for t in np.unique(obs["time"]))
d = interpolation_design(times, test)
Xn = normalize_log1p(X, keep_sparse=True)
fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
sel = select_genes(Xn[fit_cells], symbols, n_hvg=1000, must_keep=regs)
s = snapshot_set_from_arrays(Xn, obs, symbols, time_key="time", replicate_key="replicate", condition_key="condition",
                             gene_idx=sel, max_cells_per_group=max_cells, min_cells=200, rng=np.random.default_rng(0))
fit = np.isin(s.biological_time, d["fit_times"])
xf = s.x[fit]
detect = (xf > 0).mean(0)
Z = Standardizer().fit(xf).transform(xf)
rare = detect < 0.01
sq = (Z.astype(np.float64) ** 2)
print(f"{which}: {len(sel)} genes, {int(fit.sum())} fit cells")
print(f"  genes detected in < 1% of fit cells: {int(rare.sum())} "
      f"(regulators {sum(1 for g, r in zip(s.gene_names, rare) if r and g in regs)})")
print(f"  max |z| in the fit cells: all genes {np.abs(Z).max():.1f}; without the rare ones {np.abs(Z[:, ~rare]).max():.1f}")
print(f"  share of the total squared norm carried by the rare genes: {sq[:, rare].sum() / sq.sum():.3f} "
      f"(they are {rare.mean():.3f} of the genes)")
top = np.argsort(-np.abs(Z).max(0))[:8]
print("  genes with the largest |z|:", [(s.gene_names[i], round(float(np.abs(Z[:, i]).max()), 1),
                                         round(float(detect[i]), 4)) for i in top])
row = sq.sum(1)
print(f"  squared norm per cell: median {np.median(row):.0f}, max {row.max():.0f}; cells whose norm is > 10x the "
      f"median: {int((row > 10 * np.median(row)).sum())}")
