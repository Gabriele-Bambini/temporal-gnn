"""Read the Moon et al. (2019) embryoid-body time course and cache it as one sparse matrix plus cell metadata.

The archive holds five 10x samples, one per time window of the 27-day differentiation
(T0_1A = days 0-3, T2_3B = days 6-9, T4_5C = days 12-15, T6_7D = days 18-21, T8_9E = days 24-27).
This is the series on which TrajectoryNet, MIOFlow and MMFM report held-out-timepoint results, so it is
the one place where our number can be read next to published ones.

Quality control is deliberately plain and is declared in the data card: a cell is kept when it has at
least ``min_counts`` UMIs, at least ``min_genes`` detected genes and a mitochondrial fraction below
``max_mito``; a gene is kept when it is detected in at least ``min_cells`` of the kept cells. Nothing
here uses the time label, so the filter cannot favour one snapshot over another.

Run from cfm_grn_v2/:  python scripts_dev/eb_prepare.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SAMPLES = [("T0_1A", 0, "days 0-3"), ("T2_3B", 1, "days 6-9"), ("T4_5C", 2, "days 12-15"),
           ("T6_7D", 3, "days 18-21"), ("T8_9E", 4, "days 24-27")]


def read_mtx(folder: Path) -> tuple[sp.csr_matrix, list[str], list[str], list[str]]:
    """10x MatrixMarket triplet file (genes x cells) -> CSR (cells x genes), gene ids, symbols, barcodes."""
    genes = pd.read_csv(folder / "genes.tsv", sep="\t", header=None, names=["id", "symbol"], dtype=str)
    barcodes = pd.read_csv(folder / "barcodes.tsv", header=None, names=["bc"], dtype=str)["bc"].tolist()
    with open(folder / "matrix.mtx", "rb") as fh:
        line = fh.readline()
        if not line.startswith(b"%%MatrixMarket"):
            raise ValueError(f"{folder}/matrix.mtx is not a MatrixMarket file")
        while True:
            line = fh.readline()
            if not line.startswith(b"%"):
                break
        n_genes, n_cells, nnz = (int(v) for v in line.split())
        trip = np.loadtxt(fh, dtype=np.int64)
    if trip.shape != (nnz, 3):
        raise ValueError(f"{folder}: expected {nnz} triplets, read {trip.shape}")
    if n_genes != len(genes) or n_cells != len(barcodes):
        raise ValueError(f"{folder}: header {n_genes}x{n_cells} disagrees with genes.tsv / barcodes.tsv")
    X = sp.csr_matrix((trip[:, 2].astype(np.float32), (trip[:, 1] - 1, trip[:, 0] - 1)), shape=(n_cells, n_genes))
    return X, genes["id"].tolist(), genes["symbol"].tolist(), barcodes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=r"C:\Users\Gabriele\data\rafm\raw\moon2019\scRNAseq")
    p.add_argument("--out", default=r"C:\Users\Gabriele\data\rafm\processed\moon2019_eb.npz")
    p.add_argument("--min-counts", type=int, default=1000)
    p.add_argument("--min-genes", type=int, default=500)
    p.add_argument("--max-mito", type=float, default=0.20)
    p.add_argument("--min-cells", type=int, default=20)
    a = p.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    blocks, obs_rows, ids, symbols = [], [], None, None
    t0 = time.time()
    for folder, t_index, window in SAMPLES:
        X, gid, gsym, bc = read_mtx(raw / folder)
        if ids is None:
            ids, symbols = gid, gsym
        elif gid != ids:
            raise ValueError(f"{folder} has a different gene list")
        blocks.append(X)
        obs_rows.append(pd.DataFrame({"barcode": bc, "sample": folder, "time": float(t_index), "window": window}))
        print(f"{folder}: {X.shape[0]} cells, {X.nnz} non-zeros ({time.time() - t0:.0f}s)", flush=True)

    X = sp.vstack(blocks, format="csr")
    obs = pd.concat(obs_rows, ignore_index=True)
    n_raw = X.shape[0]

    counts = np.asarray(X.sum(1)).ravel()
    detected = np.asarray((X > 0).sum(1)).ravel()
    mito = np.array([s.startswith("MT-") for s in symbols])
    mito_frac = np.asarray(X[:, mito].sum(1)).ravel() / np.maximum(counts, 1.0)
    keep_cell = (counts >= a.min_counts) & (detected >= a.min_genes) & (mito_frac < a.max_mito)
    X, obs = X[keep_cell], obs.loc[keep_cell].reset_index(drop=True)
    obs["n_counts"], obs["n_genes"], obs["mito_frac"] = counts[keep_cell], detected[keep_cell], mito_frac[keep_cell]

    keep_gene = np.asarray((X > 0).sum(0)).ravel() >= a.min_cells
    X = X[:, keep_gene].tocsr()
    ids = [g for g, k in zip(ids, keep_gene) if k]
    symbols = [g for g, k in zip(symbols, keep_gene) if k]

    qc = {"cells_raw": int(n_raw), "cells_kept": int(X.shape[0]), "genes_raw": int(len(keep_gene)),
          "genes_kept": int(X.shape[1]), "filters": {"min_counts": a.min_counts, "min_genes": a.min_genes,
                                                     "max_mito": a.max_mito, "min_cells": a.min_cells},
          "per_sample": obs.groupby("sample").size().to_dict(),
          "median_counts": float(np.median(obs["n_counts"])), "median_genes": float(np.median(obs["n_genes"]))}
    np.savez_compressed(out, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                        gene_ids=np.array(ids), gene_symbols=np.array(symbols),
                        time=obs["time"].to_numpy(), sample=obs["sample"].to_numpy(dtype=str),
                        barcode=obs["barcode"].to_numpy(dtype=str), n_counts=obs["n_counts"].to_numpy(),
                        n_genes=obs["n_genes"].to_numpy(), mito_frac=obs["mito_frac"].to_numpy(),
                        qc_json=np.array(json.dumps(qc)))
    print(json.dumps(qc, indent=1))
    print(f"saved {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
