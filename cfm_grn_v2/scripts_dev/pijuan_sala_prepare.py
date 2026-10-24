"""Read the Pijuan-Sala et al. (2019) mouse gastrulation atlas (raw counts from the authors' atlas_data share) and
cache the wild-type time course E6.5-E8.5 as one sparse matrix of raw counts plus cell metadata (pre-registration
v0.2, Step 5).

The deposit is one MatrixMarket coordinate file of GENES x CELLS (29 452 x 139 331, about 4.8e8 entries, 1.5 GB
gzip), genes.tsv (Ensembl id, symbol) and meta.tab, whose rows are the matrix columns in order. The matrix is
streamed in chunks and only the entries of kept cells are held, as int32 cell/gene indices and float32 counts, then
turned into CSR (cells x genes); peak memory is a few times the kept entries (about 4.5 GB each copy), far below 96 GB.

Cells are dropped when meta.tab flags them as doublet or stripped nucleus, or when their stage is
'mixed_gastrulation' (pooled embryos without a single stage). No gene filter: genes are chosen later on each fold's
fit cells. Biological time is in quarter days from E6.5 (E6.5 -> 0, E6.75 -> 1, ..., E8.5 -> 8).

Replicates. The embryo pools of a stage are different embryos at every stage and do not link across stages, so the
training pools them (replicate = 'all'), exactly as Argelaguet's wild_type_course does; the pool label is kept in the
'pool' field for the replicate floors and the per-replicate robustness checks of Step 7, together with the 10x sample
and the sequencing batch (which is partly tied to stage: the E8.0 pools were sequenced in different batches).
replicate = 'all' is a documented DEVIATION from the literal Step-5 text "replicate = pool" (docs/superpowers/
decisions.md, 2026-09-26 afternoon): with replicate = pool no stage transition stays inside one replicate, so there
would be no training pair; the pool is kept for the replicate floor and the per-pool ROBUST check, which do not change.

Checks, all by code: --out must end in .npz (checked first); the inputs against the pre-registered sizes and SHA-256;
matrix rows = genes.tsv rows; matrix columns = meta.tab rows (else stop); every stage known; doublet/stripped flags
TRUE/FALSE; entries within the declared dimensions, as many as the header declares, no duplicate among the kept ones;
non-integer counts (counted outside E8.0 only), duplicated or empty gene symbols (read literally, so a gene called
'NA' stays 'NA'; the loader makes names unique) and whether meta.tab's cell names run cell_1 ... cell_N in column
order (all reported); kept cells per stage against the counts recorded before approval. A count difference makes the
run exit with an error AFTER the cache and its report are written, and load_pijuan_sala refuses such a cache unless
strict=False.

Test stage (E8.0), pre-registration Step 5 "Parsers print only per-time cell counts. Test-stage expression is not
inspected before Step 7": no statistic of expression values that includes an E8.0 cell is computed, printed or stored
(the kept-entry and non-integer counts leave E8.0 out). Printed: the input sizes and hashes, the matrix dimensions
and declared entries from its header, the filter counts, cells per stage, the gene count, the metadata alignment
witness, the stages whose pools span several sequencing batches, and the non-integer entries outside E8.0.

Run on the HPC (from ~/cfm_grn_v2):  python scripts_dev/pijuan_sala_prepare.py --raw ~/rafm_data/pijuan_sala2019 \
                                         --out ~/rafm_data/processed/pijuan_sala2019_atlas.npz
Runtime: the stream was measured locally at about 1.6e6 entries per second, i.e. about 5-6 min for 4.8e8 entries,
plus SHA-256 of 1.5 GB and the compressed save of a CSR of about 3 GB. Submit it as its own Slurm job with
--time=02:00:00 (not the pre-registration's "about 30 min" estimate for both parsers together). The cache is written
to a temporary name and renamed only when complete, so a killed job leaves no partial .npz under --out.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from veres_prepare import (check_out, compare_counts, input_records, non_integer, read_cache,  # noqa: E402
                           save_cache, stage_gate, time_key)

INPUTS = {"counts": ("raw_counts.mtx.gz", 1527653430,
                     "a032d82e4f948214e69bda04ac954fa0a322b5bc0c3928a7ef61a8ffdb89c809"),
          "genes": ("genes.tsv.gz", 228898, "e1dbb4662b224fd77f6c3ac52c5882da2abd1013daf869cb2fb99a004044f85f"),
          "meta": ("meta.tab.gz", 4941002, "1b5e09b6c16803278f9b2d427dff0888f20d41889de3369eebbae840cc886988")}
EXPECTED = {"E6.5": 3484, "E6.75": 2075, "E7.0": 14749, "E7.25": 13537, "E7.5": 10994, "E7.75": 14493,
            "E8.0": 16681, "E8.25": 15935, "E8.5": 16909}
DROP_STAGE = "mixed_gastrulation"
DAY0, UNIT, LAST = 6.5, 0.25, 8
CONDITION = "WT"
TEST_TIME = 6.0                 # primary test stage: E8.0 (from E7.75)
MIN_STAGE_CELLS = 500           # QC gate: a stage enters only with >= 500 cells
MAX_CELLS_PER_GROUP = 3000      # at most 3000 cells per stage, applied by the fold scripts
DEFAULT_RAW = "~/rafm_data/pijuan_sala2019"
DEFAULT_OUT = "~/rafm_data/processed/pijuan_sala2019_atlas.npz"     # main's --out and load_pijuan_sala's path


def stage_time(stage: str) -> float:
    """'E7.25' -> 3.0: quarter days from E6.5; only E6.5 ... E8.5 exist in this course."""
    m = re.fullmatch(r"E(\d+(?:\.\d+)?)", str(stage))
    if m is None:
        raise ValueError(f"unknown stage {stage!r}")
    q = (float(m.group(1)) - DAY0) / UNIT
    if abs(q - round(q)) > 1e-9 or not 0 <= round(q) <= LAST:
        raise ValueError(f"stage {stage!r} is not a quarter day between E6.5 and E8.5")
    return float(round(q))


def flag(meta: pd.DataFrame, col: str) -> np.ndarray:
    v = meta[col].astype("string").str.strip().str.upper()
    bad = ~v.isin(["TRUE", "FALSE"]).fillna(False).to_numpy(dtype=bool)
    if bad.any():
        raise ValueError(f"meta.tab column {col}: {int(bad.sum())} value(s) other than TRUE/FALSE, "
                         f"e.g. {meta[col].to_numpy()[bad][:3].tolist()}")
    return (v == "TRUE").to_numpy(dtype=bool)


def mtx_header(path: Path) -> tuple[int, int, int, int]:
    """(rows, columns, entries, number of header lines) of a gzipped MatrixMarket coordinate file."""
    with gzip.open(path, "rt") as fh:
        line, n = fh.readline(), 1
        if not line.startswith("%%MatrixMarket") or "coordinate" not in line:
            raise ValueError(f"{path.name} is not a MatrixMarket coordinate file")
        line, n = fh.readline(), n + 1
        while line.startswith("%"):
            line, n = fh.readline(), n + 1
    r, c, e = (int(v) for v in line.split())
    return r, c, e, n


def stream_kept(path: Path, n_genes: int, n_cells: int, nnz: int, n_head: int, col_pos: np.ndarray, chunk: int):
    """Entries of the kept columns: (kept-cell index int32, gene index int32, count float32)."""
    rows, cols, vals, seen = [], [], [], 0
    with gzip.open(path, "rt") as fh:
        for _ in range(n_head):
            fh.readline()
        for part in pd.read_csv(fh, sep=r"\s+", header=None, names=["g", "c", "v"], chunksize=chunk, engine="c",
                                dtype={"g": np.int64, "c": np.int64, "v": np.float64}):
            g, c = part["g"].to_numpy(), part["c"].to_numpy()
            if len(part) and (g.min() < 1 or g.max() > n_genes or c.min() < 1 or c.max() > n_cells):
                raise ValueError(f"{path.name}: entry outside the declared {n_genes} x {n_cells}")
            seen += len(part)
            k = col_pos[c - 1]
            keep = k >= 0
            rows.append(k[keep].astype(np.int32))
            cols.append((g[keep] - 1).astype(np.int32))
            vals.append(part["v"].to_numpy()[keep].astype(np.float32))
    if seen != nnz:
        raise ValueError(f"{path.name}: read {seen} entries, the header declares {nnz}")
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def prepare(raw: Path, out: Path, chunk: int = 20_000_000, expected: dict | None = None, verify: bool = True) -> dict:
    """Check, filter, stream and cache. Returns the QC report (also stored in the cache and next to it)."""
    raw, out, t0 = Path(raw), check_out(out), time.time()
    expected = EXPECTED if expected is None else expected
    inputs = input_records(raw, INPUTS, verify)
    genes = pd.read_csv(raw / INPUTS["genes"][0], sep="\t", header=None, dtype=str, keep_default_na=False)
    meta = pd.read_csv(raw / INPUTS["meta"][0], sep="\t", dtype=str, keep_default_na=False)
    need = {"cell", "sample", "pool", "stage", "sequencing.batch", "doublet", "stripped"}
    if not need <= set(meta.columns):
        raise ValueError(f"meta.tab lacks columns {sorted(need - set(meta.columns))}")
    if genes.shape[1] < 2:
        raise ValueError(f"genes.tsv has {genes.shape[1]} column(s); expected Ensembl id and symbol")
    n_genes, n_cells, nnz, n_head = mtx_header(raw / INPUTS["counts"][0])
    if n_genes != len(genes):
        raise ValueError(f"the matrix has {n_genes} rows but genes.tsv lists {len(genes)} genes")
    if n_cells != len(meta):
        raise ValueError(f"the matrix has {n_cells} columns (cells) but meta.tab has {len(meta)} rows: "
                         f"matrix and metadata are not aligned")
    if meta["cell"].duplicated().any():
        raise ValueError("meta.tab: duplicate cell names")
    if (meta["stage"].str.strip() == "").any():
        raise ValueError(f"meta.tab: {int((meta['stage'].str.strip() == '').sum())} cell(s) without a stage")
    sequential = bool((meta["cell"].to_numpy() == np.array([f"cell_{i + 1}" for i in range(len(meta))])).all())
    doublet, stripped = flag(meta, "doublet"), flag(meta, "stripped")
    mixed = (meta["stage"] == DROP_STAGE).to_numpy(dtype=bool)
    tmap = {s: stage_time(s) for s in sorted(set(meta.loc[~mixed, "stage"]))}
    keep = ~doublet & ~stripped & ~mixed
    col_pos = np.full(n_cells, -1, dtype=np.int64)
    col_pos[keep] = np.arange(int(keep.sum()))
    print(f"matrix {n_genes} genes x {n_cells} cells, {nnz} entries; {int(keep.sum())} cells kept "
          f"({int(doublet.sum())} doublet, {int(stripped.sum())} stripped, {int(mixed.sum())} {DROP_STAGE}; "
          f"flags overlap)", flush=True)

    r, c, v = stream_kept(raw / INPUTS["counts"][0], n_genes, n_cells, nnz, n_head, col_pos, chunk)
    print(f"streamed {nnz} entries ({time.time() - t0:.0f}s)", flush=True)
    X = sp.coo_matrix((v, (r, c)), shape=(int(keep.sum()), n_genes)).tocsr()
    n_kept_entries = len(v)
    del r, c, v
    X.sum_duplicates()                  # canonical CSR whatever scipy's tocsr does, so repeated entries show up
    if X.nnz != n_kept_entries:
        raise ValueError(f"{n_kept_entries - X.nnz} duplicate (gene, cell) entries among the kept cells")

    m = meta.loc[keep].reset_index(drop=True)
    obs = pd.DataFrame({"cell": m["cell"].astype(str), "stage": m["stage"].astype(str),
                        "time": m["stage"].map(tmap).astype(float), "replicate": "all",
                        "sample": m["sample"].astype(str), "pool": m["pool"].astype(str),
                        "batch": m["sequencing.batch"].astype(str)})
    order = sorted(tmap, key=tmap.get)
    per_stage = {s: int(n) for s, n in obs.groupby("stage").size().items()}
    pools = {s: {p: int(n) for p, n in g.groupby("pool").size().items()} for s, g in obs.groupby("stage")}
    batches = {s: {p: sorted(set(gp["batch"])) for p, gp in g.groupby("pool")} for s, g in obs.groupby("stage")}
    several = [s for s in order if s in per_stage and obs.loc[obs["stage"] == s, "batch"].nunique() > 1]
    X_nt = X[(obs["time"] != TEST_TIME).to_numpy()]        # the test stage's expression is not inspected
    nnz_nt, non_int_nt = int(X_nt.nnz), non_integer(X_nt)
    del X_nt
    qc = {"dataset": "pijuan_sala2019_atlas", "inputs": inputs, "test_time": TEST_TIME,
          "matrix": {"genes": n_genes, "cells": n_cells, "entries": nnz, "entries_kept_excl_test": nnz_nt},
          "filters": {"rule": "drop doublet == TRUE, stripped == TRUE and stage == mixed_gastrulation",
                      "flagged_doublet": int(doublet.sum()), "flagged_stripped": int(stripped.sum()),
                      "stage_mixed_gastrulation": int(mixed.sum()), "removed": int((~keep).sum())},
          "cells_kept": int(X.shape[0]), "genes": int(n_genes),
          "per_time": {time_key(tmap[s]): per_stage[s] for s in order if s in per_stage}, "per_stage": per_stage,
          "per_stage_pool": pools, "stage_pool_batches": batches, "stages_with_pools_in_several_batches": several,
          "expected_counts": compare_counts(per_stage, expected),
          "non_integer_entries_excl_test": non_int_nt, "meta_cell_names_sequential": sequential,
          "duplicated_gene_symbols": int(genes.loc[genes[1] != "", 1].duplicated().sum()),
          "empty_gene_symbols": int((genes[1] == "").sum()),
          "replicate": "all (the pools of a stage are pooled; 'pool' kept for replicate floors and robustness); "
                       "documented deviation from Step 5 'replicate = pool', decisions.md 2026-09-26 afternoon",
          "time_unit": "quarter day from E6.5", "chunk_entries": int(chunk)}
    save_cache(out, X, {k: obs[k].to_numpy(dtype=str) for k in ("cell", "stage", "replicate", "sample", "pool",
                                                                   "batch")} | {"time": obs["time"].to_numpy()},
               genes[1].astype(str).tolist(), genes[0].astype(str).tolist(), qc)

    print(f"genes: {n_genes}; cells kept: {X.shape[0]}; non-integer entries outside the test stage (t={TEST_TIME:g}): "
          f"{qc['non_integer_entries_excl_test']}; meta.tab cell names are cell_1..cell_{n_cells} in matrix-column "
          f"order: {sequential}", flush=True)
    print("cells per stage: " + ", ".join(f"{s} (t={tmap[s]:g})={per_stage.get(s, 0)}" for s in order), flush=True)
    print(f"stages whose pools span several sequencing batches: {several}", flush=True)
    if not qc["expected_counts"]["ok"]:
        print(f"MISMATCH against the expected counts: {qc['expected_counts']['mismatches']}", flush=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s", flush=True)
    return qc


def load_pijuan_sala(path=DEFAULT_OUT, min_stage_cells: int = MIN_STAGE_CELLS, strict: bool = True):
    """(X csr float32 raw counts, obs[cell, stage, time, replicate, condition, sample, pool, batch], gene symbols,
    qc), after the pre-registered stage gate; mirrors eb_benchmark.load_cache. The default path is the parser's --out
    default, so the fold scripts can call loader() with neither --data nor --cache."""
    X, z, symbols, qc = read_cache(path)
    obs = pd.DataFrame({"cell": z["cell"].astype(str), "stage": z["stage"].astype(str),
                        "time": z["time"].astype(float), "replicate": z["replicate"].astype(str),
                        "condition": CONDITION, "sample": z["sample"].astype(str), "pool": z["pool"].astype(str),
                        "batch": z["batch"].astype(str)})
    X, obs, qc = stage_gate(X, obs, qc, min_stage_cells, TEST_TIME, strict)
    return X, obs, symbols, qc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=DEFAULT_RAW)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--chunk", type=int, default=20_000_000, help="matrix entries parsed at a time")
    p.add_argument("--no-verify", action="store_true", help="skip the pre-registered size/SHA-256 check (tests)")
    a = p.parse_args()
    qc = prepare(Path(a.raw).expanduser(), Path(a.out).expanduser(), chunk=a.chunk, verify=not a.no_verify)
    if not qc["expected_counts"]["ok"]:
        sys.exit(f"STOP: kept cells per stage differ from the expected counts: {qc['expected_counts']['mismatches']}")


if __name__ == "__main__":
    main()
