"""Read the Veres et al. (2019) Stage-5 time course (GEO GSE114412) and cache it as one sparse matrix of raw counts
plus cell metadata (pre-registration v0.2, Step 5).

Stage 5 of the in-vitro beta-cell differentiation was sampled every day (days 0-7) in two independent
differentiations, each day split into two technical library preparations (b1, b2) and sequenced with inDrops. The
deposit is a DENSE tab-separated cells x genes table of UMIFM counts (header expected to be 'library.barcode',
possibly written '# library.barcode', then the gene symbols; the layout was not opened before approval, so either
form is accepted and the first field is recorded) and a metadata table keyed on the same library.barcode. The table
is streamed in row chunks, joined on the barcode and stored sparse. No cell or gene filter is applied: genes are
chosen later on each fold's fit cells, so a filter here would let the test stage shape the gene space.

Replicate = the differentiation ('diff1', 'diff2'): a biological unit followed across days, so couplings stay within
one. The two library splits of a day are technical and are kept only as the 'library' field (the loader copies it
into 'pool', the column the fold scripts' generic cache carries); they are never a unit of analysis.

Checks, all by code: --out must end in .npz (checked first, so the name used is the name given); the inputs against
the pre-registered sizes and SHA-256; the header's first field is library.barcode (the orientation: cells in rows);
complete rows and unique barcodes; every count row against the metadata (mismatches are reported and the unmatched
rows dropped); the library label against the day and the differentiation; non-integer counts, counted on the
non-test days only (reported); duplicated gene symbols (reported; the loader makes them unique); the cells per
(differentiation, day) against the counts recorded before approval. A count difference makes the run exit with an
error AFTER the cache and its report are written, and load_veres refuses such a cache unless strict=False.

Test stage (day 4), pre-registration Step 5 "Parsers print only per-time cell counts. Test-stage expression is not
inspected before Step 7": no statistic of expression values that includes a day-4 cell is computed, printed or
stored. Printed: the input sizes and hashes, the table's dimensions, cells per day and per (differentiation, day),
the gene count, and metadata or format checks (join mismatches, library-label disagreements, non-integer entries
outside day 4).

Run on the HPC (from ~/cfm_grn_v2):  python scripts_dev/veres_prepare.py --raw ~/rafm_data/veres2019 \
                                         --out ~/rafm_data/processed/veres2019_stage5.npz
Runtime: parsing a full-width synthetic table (16 224 gene columns, about 14 % non-zero) took 19-54 s per 4 000 rows
in local runs, i.e. up to about 12 min for 51 274 rows, plus hashing and the compressed save. Submit it as its own
Slurm job with --time=02:00:00 (not the pre-registration's "about 30 min" estimate for both parsers together). The
cache is written to a temporary name and renamed only when complete, so a killed job leaves no partial .npz under
--out.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

INPUTS = {"counts": ("GSE114412_Stage_5.all.processed_counts.tsv.gz", 108394226,
                     "9304c5ed767f72af3108e303176c8487737b0ede25965be66e2455b708cfba15"),
          "meta": ("GSE114412_Stage_5.all.cell_metadata.tsv.gz", 1764292,
                   "5006773f740191732bc1b4a6c3f086cc63d00a601cdeedc7cfcb4fa126facda7")}
EXPECTED = {"diff1": [4563, 3406, 1942, 2901, 3337, 3346, 2448, 2141],
            "diff2": [1627, 2771, 1620, 9576, 3143, 3058, 3565, 1830]}
CONDITION = "stage5"
TEST_TIME = 4.0                 # primary test stage: day 4 (from day 3)
MIN_STAGE_CELLS = 500           # QC gate: a stage enters only with >= 500 cells
MAX_CELLS_PER_GROUP = 1500      # at most 1500 cells per (day, differentiation), applied by the fold scripts
LIBRARY = re.compile(r"stg5diff(\d+)_S5d(\d+)_b\d+")
DEFAULT_RAW = "~/rafm_data/veres2019"
DEFAULT_OUT = "~/rafm_data/processed/veres2019_stage5.npz"     # main's --out and load_veres's path, one value


def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def input_records(raw: Path, inputs: dict, verify: bool) -> dict:
    """Size and SHA-256 of every input; with verify, both must equal the pre-registered values."""
    out = {}
    for key, (name, size, digest) in inputs.items():
        path = raw / name
        if not path.exists():
            raise FileNotFoundError(path)
        n = path.stat().st_size
        if verify and n != size:
            raise ValueError(f"{name}: {n} B, the pre-registered size is {size} B")
        h = sha256(path)
        if verify and h != digest:
            raise ValueError(f"{name}: sha256 {h} differs from the pre-registered {digest}")
        out[key] = {"file": name, "bytes": int(n), "sha256": h, "matches_prereg": bool(n == size and h == digest)}
        print(f"{name}: {n} B sha256 {h} ({'matches' if out[key]['matches_prereg'] else 'DIFFERS from'} the "
              f"pre-registered value)", flush=True)
    return out


def compare_counts(observed: dict, expected: dict) -> dict:
    mism = {k: {"expected": int(expected.get(k, 0)), "observed": int(observed.get(k, 0))}
            for k in sorted(set(observed) | set(expected)) if int(observed.get(k, 0)) != int(expected.get(k, 0))}
    return {"ok": not mism, "mismatches": mism, "expected": {k: int(v) for k, v in expected.items()}}


def time_key(t: float) -> str:
    return f"{float(t):g}"


def non_integer(X) -> int:
    """Stored entries that are not whole numbers (raw counts should have none; they are kept and reported). Callers
    pass the non-test rows only: the test stage's expression is not inspected before Step 7."""
    return int((X.data != np.round(X.data)).sum())


def check_out(out: Path) -> Path:
    """The cache path must end in .npz: np.savez_compressed would otherwise append it, and the side report, the final
    size print and the loader would all use a different name. Checked before any input is read."""
    if Path(out).suffix != ".npz":
        raise ValueError(f"--out must end with .npz: {out}")
    return Path(out)


def save_cache(out: Path, X, fields: dict, gene_symbols, gene_ids, qc: dict) -> None:
    """Written to '<name>.partial.npz' and renamed when complete: a job killed during the save leaves no .npz under
    the final name for a loader to pick up."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez_compressed(tmp, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
                        gene_symbols=np.array(gene_symbols, dtype=str), gene_ids=np.array(gene_ids, dtype=str),
                        qc_json=np.array(json.dumps(qc)), **{k: np.asarray(v) for k, v in fields.items()})
    tmp.replace(out)
    out.with_suffix(".qc.json").write_text(json.dumps(qc, indent=1), encoding="utf-8")


def unique_symbols(symbols, ids) -> tuple[list, list]:
    """Gene names made unique, because the fold scripts index genes (and the CollecTRI prior) by symbol: the first
    occurrence of a symbol keeps it, a later one becomes 'symbol|id' ('symbol|k' when the id is the symbol itself), an
    empty symbol becomes its id. Returns the names and the ones that changed."""
    taken, seen, names, renamed = set(symbols), {}, [], []
    for s, i in zip(symbols, ids):
        if not s:
            base = i
        else:
            k = seen[s] = seen.get(s, -1) + 1
            base = s if k == 0 else (f"{s}|{i}" if i and i != s else f"{s}|{k}")
        name, n = base, 1
        while name != s and name in taken:
            n += 1
            name = f"{base}.{n}"
        if name != s:
            taken.add(name)
            renamed.append(name)
        names.append(name)
    return names, renamed


def read_cache(path) -> tuple[sp.csr_matrix, dict, list, dict]:
    """(X, the other cached arrays by name, unique gene names, qc) of a cache written by save_cache. The cache keeps
    the deposit's symbols; the renamed ones are listed in qc['gene_symbols_renamed']."""
    with np.load(Path(path).expanduser(), allow_pickle=False) as npz:
        z = {k: npz[k] for k in npz.files}
    X = sp.csr_matrix((z.pop("data").astype(np.float32, copy=False), z.pop("indices"), z.pop("indptr")),
                      shape=tuple(z.pop("shape")))
    names, renamed = unique_symbols(z["gene_symbols"].astype(str).tolist(), z["gene_ids"].astype(str).tolist())
    return X, z, names, {**json.loads(str(z["qc_json"])), "gene_symbols_renamed": renamed}


def stage_gate(X, obs: pd.DataFrame, qc: dict, min_stage_cells: int, test_time: float, strict: bool):
    """Pre-registered QC gate: a time point enters only with >= min_stage_cells cells (replicates pooled); a test
    stage that fails is dropped and reported, never replaced. With strict, a cache whose cell counts differ from the
    expected ones is refused."""
    if strict and not qc["expected_counts"]["ok"]:
        raise ValueError(f"cell counts differ from the expected ones: {qc['expected_counts']['mismatches']} "
                         f"(strict=False loads it anyway)")
    n = obs.groupby("time").size()
    dropped = sorted(float(t) for t, k in n.items() if k < min_stage_cells)
    keep = ~obs["time"].isin(dropped).to_numpy()
    if not keep.all():
        X, obs = X[keep], obs.loc[keep].reset_index(drop=True)
    qc = {**qc, "qc_gate": {"min_stage_cells": int(min_stage_cells), "dropped_times": dropped,
                            "test_time": float(test_time), "test_time_present": bool(float(test_time) in set(n.index)),
                            "test_time_dropped": bool(float(test_time) in dropped)}}
    return X, obs, qc


def read_counts(path: Path, chunk: int) -> tuple[sp.csr_matrix, np.ndarray, list, str]:
    """Dense tab-separated cells x genes table -> CSR (cells x genes, float32), barcodes, gene symbols, the header's
    first field as written. The first field must be library.barcode, with or without a leading '#': that fixes the
    orientation (cells in rows); the file itself is pinned by its size and SHA-256."""
    with gzip.open(path, "rt") as fh:
        head = fh.readline().rstrip("\r\n").split("\t")
        if head[0].lstrip("#").strip() != "library.barcode":
            raise ValueError(f"{path.name}: the header's first field must be 'library.barcode' (cells in rows), "
                             f"found {head[0]!r}")
        genes = head[1:]
        if not genes or any(g == "" for g in genes):
            raise ValueError(f"{path.name}: empty gene name in the header")
        names = ["cell"] + [f"g{i}" for i in range(len(genes))]
        blocks, cells = [], []
        for part in pd.read_csv(fh, sep="\t", header=None, names=names, dtype={"cell": str}, chunksize=chunk,
                                engine="c"):
            if not isinstance(part.index, pd.RangeIndex):
                raise ValueError(f"{path.name}: rows with more fields than the header")
            vals = part.iloc[:, 1:].to_numpy(dtype=np.float32)
            bad = np.isnan(vals).any(1) | part["cell"].isna().to_numpy()
            if bad.any():
                raise ValueError(f"{path.name}: missing values in row {int(part.index[np.argmax(bad)]) + 2} "
                                 f"(a short or malformed row)")
            blocks.append(sp.csr_matrix(vals))
            cells += part["cell"].tolist()
    cells = np.array(cells, dtype=str)
    dup = pd.Index(cells).duplicated()
    if dup.any():
        raise ValueError(f"{path.name}: {int(dup.sum())} duplicate barcode(s), e.g. {cells[dup][:3].tolist()}")
    return sp.vstack(blocks, format="csr"), cells, genes, head[0]


def read_meta(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t", dtype=str)
    meta.columns = [c.lstrip("#").strip() for c in meta.columns]
    need = ["library.barcode", "Differentiation", "CellWeek", "Lib_prep_batch"]
    if not set(need) <= set(meta.columns):
        raise ValueError(f"{path.name} lacks columns {sorted(set(need) - set(meta.columns))}")
    if meta["library.barcode"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate library.barcode values")
    return meta.set_index("library.barcode")


def integer_column(m: pd.DataFrame, col: str) -> np.ndarray:
    v = pd.to_numeric(m[col], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(v) | (v != np.round(v))
    if bad.any():
        raise ValueError(f"metadata column {col}: {int(bad.sum())} non-integer value(s), "
                         f"e.g. {m[col].to_numpy()[bad][:3].tolist()}")
    return v.astype(int)


def prepare(raw: Path, out: Path, chunk: int = 2000, expected: dict | None = None, verify: bool = True) -> dict:
    """Parse, join, check and cache. Returns the QC report (also stored in the cache and next to it)."""
    raw, out, t0 = Path(raw), check_out(out), time.time()
    expected = EXPECTED if expected is None else expected
    inputs = input_records(raw, INPUTS, verify)
    meta = read_meta(raw / INPUTS["meta"][0])
    X, cells, genes, first_field = read_counts(raw / INPUTS["counts"][0], chunk)
    print(f"counts: {X.shape[0]} rows x {len(genes)} genes ({time.time() - t0:.0f}s)", flush=True)

    has = pd.Index(cells).isin(meta.index)
    orphans = cells[~has].tolist()
    ghosts = int((~meta.index.isin(cells)).sum())
    X, cells = X[has], cells[has]
    m = meta.loc[cells]
    diff, day = integer_column(m, "Differentiation"), integer_column(m, "CellWeek")
    lib = m["Lib_prep_batch"].astype(str).to_numpy()
    parsed = [LIBRARY.fullmatch(s) for s in lib]
    disagree = int(sum(p is None or (int(p.group(1)), int(p.group(2))) != (d, t)
                       for p, d, t in zip(parsed, diff, day)))
    obs = pd.DataFrame({"cell": cells, "time": day.astype(float), "replicate": [f"diff{d}" for d in diff],
                        "sample": [f"diff{d}_d{t}" for d, t in zip(diff, day)], "library": lib})

    per_rt = {f"{r}|{t:g}": int(n) for (r, t), n in obs.groupby(["replicate", "time"]).size().items()}
    exp_flat = {f"{r}|{t}": int(n) for r, v in expected.items() for t, n in enumerate(v)}
    non_test = (obs["time"] != TEST_TIME).to_numpy()        # the test stage's expression is not inspected
    qc = {"dataset": "veres2019_stage5", "inputs": inputs, "header_first_field": first_field,
          "count_rows": int(len(has)), "genes": int(len(genes)),
          "cells_kept": int(X.shape[0]), "count_rows_without_meta": len(orphans),
          "count_rows_without_meta_examples": orphans[:10], "meta_rows_without_counts": ghosts,
          "library_label_disagreements": disagree, "test_time": TEST_TIME,
          "non_integer_entries_excl_test": non_integer(X[non_test]),
          "per_time": {time_key(t): int(n) for t, n in obs.groupby("time").size().items()},
          "per_replicate_time": per_rt, "expected_counts": compare_counts(per_rt, exp_flat),
          "duplicated_gene_symbols": int(pd.Index(genes).duplicated().sum()),
          "filters": "none: every cell of the deposit that has metadata, every gene",
          "replicate": "differentiation (diff1, diff2), biological and linked across days; libraries b1/b2 technical",
          "time_unit": "Stage-5 day (CellWeek)", "chunk_rows": int(chunk)}
    save_cache(out, X, {k: obs[k].to_numpy(dtype=str) for k in ("cell", "replicate", "sample", "library")} |
               {"time": obs["time"].to_numpy(dtype=float)}, genes, genes, qc)

    print(f"genes: {len(genes)}; cells kept: {X.shape[0]}; count rows without metadata: {len(orphans)} "
          f"{orphans[:10]}; metadata rows without counts: {ghosts}; library labels disagreeing with "
          f"(differentiation, day): {disagree}; non-integer entries outside the test day {TEST_TIME:g}: "
          f"{qc['non_integer_entries_excl_test']}", flush=True)
    for r in sorted(expected):
        print(f"{r} cells per day: " + ", ".join(f"d{t}={per_rt.get(f'{r}|{t}', 0)}" for t in range(len(expected[r]))),
              flush=True)
    print("cells per day: " + ", ".join(f"d{k}={v}" for k, v in qc["per_time"].items()), flush=True)
    if not qc["expected_counts"]["ok"]:
        print(f"MISMATCH against the expected counts: {qc['expected_counts']['mismatches']}", flush=True)
    print(f"saved {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s", flush=True)
    return qc


def load_veres(path=DEFAULT_OUT, min_stage_cells: int = MIN_STAGE_CELLS, strict: bool = True):
    """(X csr float32 raw counts, obs[cell, time, replicate, condition, sample, library, pool], gene symbols, qc),
    after the pre-registered stage gate; mirrors eb_benchmark.load_cache. The default path is the parser's --out
    default, so the fold scripts can call loader() with neither --data nor --cache. 'pool' repeats the technical
    library (b1/b2 of a day), only because the fold scripts' generic cache carries a pool column; the biological unit
    of the robustness checks is the replicate (differentiation)."""
    X, z, symbols, qc = read_cache(path)
    lib = z["library"].astype(str)
    obs = pd.DataFrame({"cell": z["cell"].astype(str), "time": z["time"].astype(float),
                        "replicate": z["replicate"].astype(str), "condition": CONDITION,
                        "sample": z["sample"].astype(str), "library": lib, "pool": lib})
    X, obs, qc = stage_gate(X, obs, qc, min_stage_cells, TEST_TIME, strict)
    return X, obs, symbols, qc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=DEFAULT_RAW)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--chunk", type=int, default=2000, help="rows of the dense table parsed at a time")
    p.add_argument("--no-verify", action="store_true", help="skip the pre-registered size/SHA-256 check (tests)")
    a = p.parse_args()
    qc = prepare(Path(a.raw).expanduser(), Path(a.out).expanduser(), chunk=a.chunk, verify=not a.no_verify)
    if not qc["expected_counts"]["ok"]:
        sys.exit(f"STOP: cells per (differentiation, day) differ from the expected counts: "
                 f"{qc['expected_counts']['mismatches']}")


if __name__ == "__main__":
    main()
