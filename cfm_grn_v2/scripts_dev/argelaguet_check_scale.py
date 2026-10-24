"""Is the Argelaguet result dominated by a few genes that explode under standardisation?

The noise floor of the benchmark (two halves of the SAME held-out stage) has an error of the mean of 106 in the
evaluation space, against 1.6 on the embryoid body: two halves of one sample should agree on their mean. A gene
that is almost never expressed in the observed stages has a tiny standard deviation there (the Standardizer only
floors it at 1e-6), so the few cells that express it get standardised values in the hundreds, and those values then
dominate every distance. The regulator list is the obvious source: a transcription factor is kept if it is
expressed in at least one fit cell.

This rebuilds the benchmark's evaluation space for the reported fold and reports, per gene, the standard deviation
used for standardisation and the largest standardised value reached in the held-out stage; then the noise floor and
the mean-interpolation control recomputed with and without the offending genes. No model is trained.

Run on the HPC from ~/cfm_grn_v2:  PYTHONPATH=src python scripts_dev/argelaguet_check_scale.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from argelaguet_benchmark import DAY0, UNIT, load_cache
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.metrics import energy_distance
from cfm_grn_v2.training.search import interpolation_design

CACHE = Path("~/rafm_data/processed/argelaguet2022_rna.npz").expanduser()
PRIOR = Path("~/rafm_data/priors/collectri_mouse.tsv").expanduser()
OUT = Path("outputs/argelaguet_check_scale")
OUT.mkdir(parents=True, exist_ok=True)


def ed(a, b, seed=0, n=1000):
    r = np.random.default_rng(seed)
    return float(energy_distance(a[r.choice(len(a), min(n, len(a)), replace=False)],
                                 b[r.choice(len(b), min(n, len(b)), replace=False)]))


X, obs, symbols, _ = load_cache(CACHE)
course = (~obs["sample"].str.contains("CRISPR")) & (obs["genotype"] == "WT")
X, obs = X[course.to_numpy()], obs.loc[course].reset_index(drop=True)
obs["time"] = (obs["e_day"] - DAY0) / UNIT
obs["replicate"], obs["condition"] = "pool", "WT"
times = sorted(float(t) for t in np.unique(obs["time"]))
d = interpolation_design(times, 2.0)
Xn = normalize_log1p(X, keep_sparse=True)
fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
regs = set(pd.read_csv(PRIOR, sep="\t")["source_genesymbol"].astype(str))
sel = select_genes(Xn[fit_cells], symbols, n_hvg=1000, must_keep=regs)
hvg_only = set(select_genes(Xn[fit_cells], symbols, n_hvg=1000, must_keep=()).tolist())
s = snapshot_set_from_arrays(Xn, obs, symbols, time_key="time", replicate_key="replicate", condition_key="condition",
                             gene_idx=sel, max_cells_per_group=3000, min_cells=200, rng=np.random.default_rng(0))
t = s.biological_time
observed = np.isin(t, d["obs_times"])
std = Standardizer().fit(s.x[observed])
Z = std.transform(s.x)
truth = Z[t == d["test_time"]]
src = Z[t == d["t_prev"]]
nxt = Z[t == d["t_next"]]
mean_interp = src + d["frac"] * (nxt.mean(0) - src.mean(0))

genes = pd.DataFrame({"gene": s.gene_names, "std_observed": std.std_,
                      "max_abs_z_heldout": np.abs(truth).max(0), "is_regulator": [g in regs for g in s.gene_names],
                      "is_hvg": [i in hvg_only for i in sel],
                      "frac_expressing_observed": (s.x[observed] > 0).mean(0),
                      "frac_expressing_heldout": (s.x[t == d["test_time"]] > 0).mean(0)})
genes = genes.sort_values("std_observed")
genes.to_csv(OUT / "genes.csv", index=False)
print(genes.head(15).round(5).to_string(index=False))

rows = []
for cut in (0.0, 0.01, 0.03, 0.1):
    keep = (std.std_ >= cut) if cut > 0 else np.ones(len(sel), dtype=bool)
    half = np.random.default_rng(0).permutation(len(truth))
    a_, b_ = truth[half[: len(truth) // 2]][:, keep], truth[half[len(truth) // 2:]][:, keep]
    rows.append({"min_std_kept": cut, "genes_kept": int(keep.sum()),
                 "noise_floor_ED": ed(a_, b_), "noise_floor_mean_error": float(np.linalg.norm(a_.mean(0) - b_.mean(0))),
                 "mean_interp_ED": ed(mean_interp[:, keep], truth[:, keep]),
                 "persistence_ED": ed(src[:, keep], truth[:, keep]),
                 "max_abs_z_heldout": float(np.abs(truth[:, keep]).max())})
tab = pd.DataFrame(rows)
print()
print(tab.round(4).to_string(index=False))
summary = {"n_genes": int(len(sel)), "n_regulators": int(genes.is_regulator.sum()),
           "genes_std_below_0.01": int((std.std_ < 0.01).sum()), "genes_std_below_0.03": int((std.std_ < 0.03).sum()),
           "regulators_among_std_below_0.03": int(genes.loc[genes.std_observed < 0.03, "is_regulator"].sum()),
           "table": tab.to_dict(orient="records")}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=1))
