"""Read the Argelaguet et al. (2022) mouse gastrulation multiome from GEO (GSE205117) and cache the RNA part.

Each GEO sample ships one 10x MatrixMarket matrix that holds BOTH modalities: 32 285 "Gene Expression" features
and 199 885 "Peaks". The barcodes are the cells the authors kept after their own quality control (they match
their metadata table one to one), so no further cell filter is applied here. This script keeps the gene rows only,
streams each matrix in chunks so a 500 MB gzip never has to be expanded in memory, and writes one sparse cache
with the cell metadata. The peak rows are left in the raw files for the chromatin phase.

Stages are converted to embryonic days (E7.5 -> 7.5 ...). Genotype and sample are kept, including the Brachyury
(T) CRISPR knockout and its matched wild type at E8.5: they belong to the perturbation phase and are filtered
out of the time-course task by the benchmark, not here.

Run on the HPC:  python scripts_dev/argelaguet_prepare.py --geo ~/rafm_data/argelaguet2022/geo \
                     --meta ~/rafm_data/argelaguet2022/meta/metadata.txt.gz --out ~/rafm_data/processed/argelaguet2022_rna.npz
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

STAGE_DAY = {"E7.5": 7.5, "E7.75": 7.75, "E8.0": 8.0, "E8.5": 8.5, "E8.75": 8.75}


def sample_files(geo: Path) -> dict:
    """{sample: {"matrix", "barcodes", "features"}} from names like GSM6205416_E7.5_rep1_GEX_matrix.mtx.gz."""
    out: dict = {}
    for f in sorted(geo.glob("GSM*_GEX_*")):
        m = re.match(r"GSM\d+_(.+)_GEX_(matrix|barcodes|features)\.", f.name)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = f
    bad = [s for s, d in out.items() if set(d) != {"matrix", "barcodes", "features"}]
    if bad:
        raise FileNotFoundError(f"incomplete sample(s): {bad}")
    return out


def read_gene_matrix(files: dict, chunk: int = 20_000_000):
    """Stream a 10x multiome MatrixMarket file and keep the Gene Expression rows. Returns CSR (cells x genes),
    gene ids, gene symbols, barcodes."""
    feats = pd.read_csv(files["features"], sep="\t", header=None, dtype=str)
    is_gene = (feats[2] == "Gene Expression").to_numpy()
    gene_pos = np.full(len(feats), -1, dtype=np.int64)
    gene_pos[is_gene] = np.arange(int(is_gene.sum()))
    barcodes = pd.read_csv(files["barcodes"], header=None, dtype=str)[0].tolist()
    with gzip.open(files["matrix"], "rt") as fh:
        line = fh.readline()
        if not line.startswith("%%MatrixMarket"):
            raise ValueError(f"{files['matrix']} is not a MatrixMarket file")
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n_feat, n_cells, nnz = (int(v) for v in line.split())
        if n_feat != len(feats) or n_cells != len(barcodes):
            raise ValueError(f"{files['matrix'].name}: header {n_feat}x{n_cells} disagrees with features/barcodes")
        rows, cols, vals, seen = [], [], [], 0
        for part in pd.read_csv(fh, sep=" ", header=None, names=["f", "c", "v"], dtype={"f": np.int64, "c": np.int64, "v": np.float32},
                                chunksize=chunk, engine="c"):
            seen += len(part)
            g = gene_pos[part["f"].to_numpy() - 1]
            keep = g >= 0
            rows.append(part["c"].to_numpy()[keep] - 1)
            cols.append(g[keep])
            vals.append(part["v"].to_numpy()[keep])
    if seen != nnz:
        raise ValueError(f"{files['matrix'].name}: read {seen} entries, header says {nnz}")
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_cells, int(is_gene.sum())), dtype=np.float32)
    return X, feats.loc[is_gene, 0].tolist(), feats.loc[is_gene, 1].tolist(), barcodes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--geo", required=True)
    p.add_argument("--meta", required=True, help="the authors' cell metadata (cell, sample, stage, genotype, QC)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-cells", type=int, default=20, help="keep genes detected in at least this many cells")
    a = p.parse_args()
    geo, out = Path(a.geo).expanduser(), Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(a.meta).expanduser(), sep="\t").set_index("cell")
    files = sample_files(geo)
    print(f"{len(files)} samples: {sorted(files)}", flush=True)

    t0, blocks, obs, ids, symbols = time.time(), [], [], None, None
    for sample in sorted(files):
        X, gid, gsym, bc = read_gene_matrix(files[sample])
        if ids is None:
            ids, symbols = gid, gsym
        elif gid != ids:                                  # same genes in another order: align by gene id
            if set(gid) != set(ids) or len(gid) != len(ids):
                raise ValueError(f"{sample}: gene set differs from the first sample")
            pos = {g: i for i, g in enumerate(gid)}
            X = X[:, [pos[g] for g in ids]]
        cells = [f"{sample}#{b}" for b in bc]
        known = np.array([c in meta.index for c in cells])
        X = X[known]
        m = meta.loc[[c for c, k in zip(cells, known) if k]]
        blocks.append(X)
        obs.append(pd.DataFrame({"cell": m.index, "sample": m["sample"].to_numpy(), "stage": m["stage"].to_numpy(),
                                 "genotype": m["genotype"].to_numpy(), "n_counts_authors": m["nCount_RNA"].to_numpy()}))
        print(f"{sample}: {X.shape[0]} cells kept of {len(bc)} barcodes ({int((~known).sum())} not in the metadata), "
              f"{X.nnz} gene entries ({time.time() - t0:.0f}s)", flush=True)

    X = sp.vstack(blocks, format="csr")
    obs = pd.concat(obs, ignore_index=True)
    obs["e_day"] = obs["stage"].map(STAGE_DAY).astype(float)
    if obs["e_day"].isna().any():
        raise ValueError(f"unknown stage(s): {sorted(set(obs.loc[obs['e_day'].isna(), 'stage']))}")
    keep_gene = np.asarray((X > 0).sum(0)).ravel() >= a.min_cells
    X = X[:, keep_gene].tocsr()
    ids = [g for g, k in zip(ids, keep_gene) if k]
    symbols = [g for g, k in zip(symbols, keep_gene) if k]
    qc = {"cells": int(X.shape[0]), "genes_kept": int(X.shape[1]), "genes_raw": int(len(keep_gene)),
          "min_cells": a.min_cells, "per_sample": obs.groupby("sample").size().to_dict(),
          "per_stage_genotype": {f"{k[0]}|{k[1]}": int(v) for k, v in obs.groupby(["stage", "genotype"]).size().items()}}
    np.savez_compressed(out, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                        gene_ids=np.array(ids), gene_symbols=np.array(symbols),
                        cell=obs["cell"].to_numpy(dtype=str), sample=obs["sample"].to_numpy(dtype=str),
                        stage=obs["stage"].to_numpy(dtype=str), genotype=obs["genotype"].to_numpy(dtype=str),
                        e_day=obs["e_day"].to_numpy(), qc_json=np.array(json.dumps(qc)))
    print(json.dumps(qc, indent=1))
    print(f"saved {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
