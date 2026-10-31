"""Describe the design of the Saunders 2023 zebrafish reference (GSE202639_reference_cell_metadata.csv.gz): cells and
embryos per timepoint x experiment x temperature x gene_target. Prints tables; writes ~/rafm_data/saunders2023/design.json."""
import json
import os

import pandas as pd

D = os.path.expanduser("~/rafm_data/saunders2023")
m = pd.read_csv(os.path.join(D, "GSE202639_reference_cell_metadata.csv.gz"),
                usecols=["cell", "timepoint", "expt", "embryo", "temp", "gene_target", "cell_type_broad", "n.umi"])
print("cells", len(m), "embryos", m["embryo"].nunique())
for c in ("timepoint", "expt", "temp", "gene_target"):
    print(c, m[c].value_counts().sort_index().to_dict())
emb = m.groupby("embryo").agg(timepoint=("timepoint", "first"), expt=("expt", "first"), temp=("temp", "first"),
                              target=("gene_target", "first"), n=("cell", "size"), ntp=("timepoint", "nunique"))
print("embryos with >1 timepoint:", int((emb["ntp"] > 1).sum()))
tab = emb.groupby(["timepoint", "expt"]).size().unstack(fill_value=0)
print(tab.to_string())
print(emb.groupby("temp").size().to_dict())
print(emb.groupby(["expt", "target"]).size().unstack(fill_value=0).to_string())
print(emb[emb["expt"] == "GAP14"].groupby(["timepoint", "target"]).size().unstack(fill_value=0).to_string())
print("cells per embryo: median", emb["n"].median(), "quantiles", emb["n"].quantile([0.1, 0.5, 0.9]).to_dict())
json.dump({"n_cells": int(len(m)), "embryos_by_tp_expt": {str(k): v for k, v in tab.to_dict(orient="index").items()},
           "temp": emb.groupby("temp").size().to_dict(), "target": emb.groupby("target").size().to_dict(),
           "cells_per_embryo_median": float(emb["n"].median())},
          open(os.path.join(D, "design.json"), "w"), indent=1, default=str)
