"""Metadata-only data card for sci-FATE2 (GSE236512). Reads h5ad files in backed mode and reports structure:
shape, obs columns (value counts for low-cardinality columns), var columns, layer / obsm / uns keys.
It never reads expression values, so the pre-registration for the decisive test can be frozen afterwards.

Usage: python datacard_scifate2.py file1.h5ad [file2.h5ad ...] > datacard.txt
"""
import sys

import anndata
import pandas as pd

for path in sys.argv[1:]:
    print(f"\n######## {path}")
    ad = anndata.read_h5ad(path, backed="r")
    print("shape", ad.shape)
    print("layers", list(ad.layers.keys()))
    print("obsm", list(ad.obsm.keys()), "| varm", list(ad.varm.keys()), "| obsp", list(ad.obsp.keys()))
    print("uns", list(ad.uns.keys()))
    print("var columns", list(ad.var.columns), "| first var names", list(ad.var_names[:5]))
    print("obs columns", list(ad.obs.columns))
    for col in ad.obs.columns:
        s = ad.obs[col]
        nun = s.nunique()
        print(f"\n-- obs[{col}] dtype={s.dtype} n_unique={nun}")
        if nun <= 60:
            print(s.value_counts(dropna=False).sort_index().to_string())
        elif pd.api.types.is_numeric_dtype(s):
            print(s.describe().to_string())
    # cross-tab of the most likely time and replicate columns
    lower = {c.lower(): c for c in ad.obs.columns}
    tcols = [lower[k] for k in lower if any(w in k for w in ("day", "time", "timepoint", "hour"))]
    rcols = [lower[k] for k in lower if any(w in k for w in ("rep", "batch", "sample", "experiment"))]
    for tc in tcols[:2]:
        for rc in rcols[:3]:
            if ad.obs[tc].nunique() <= 60 and ad.obs[rc].nunique() <= 60:
                print(f"\n== crosstab {tc} x {rc}")
                print(pd.crosstab(ad.obs[tc], ad.obs[rc]).to_string())
    ad.file.close()
