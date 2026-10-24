"""Step-5 parsers of pre-registration v0.2 (Veres 2019 Stage 5, Pijuan-Sala 2019 atlas) on tiny synthetic fixtures
written in the raw formats of the two deposits (gzip): parse correctness, filters, time mapping, alignment checks,
the loud count check and the loaders."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_dev"))
import pijuan_sala_prepare as PS  # noqa: E402
import veres_prepare as V  # noqa: E402

GENES_V = [f"G{i}" for i in range(7)]


def _veres_raw(folder: Path, header="# library.barcode", short_row=False, dup_barcode=False, bad_week=False,
               bad_lib=False, genes=GENES_V, fraction=False, text_count=False):
    """Dense cells x genes counts (one row without metadata at the end) and a shuffled metadata table (one row
    without counts). Returns barcodes, counts, cells [(barcode, differentiation, day, library)], expected counts."""
    rng = np.random.default_rng(0)
    cells = []
    for diff in (1, 2):
        for day in range(3):
            for b in (1, 2):
                lib = f"stg5diff{diff}_S5d{day}_b{b}"
                cells += [(f"{lib}.bc{k}", diff, day, lib) for k in range(1 + day + diff)]
    ids = [c[0] for c in cells] + ["orphan.bc0"]
    if dup_barcode:
        ids[1] = ids[0]
    counts = rng.poisson(0.8, size=(len(ids), len(genes)))
    lines = [header + "\t" + "\t".join(genes)]
    lines += ["\t".join([i] + [str(v) for v in row]) for i, row in zip(ids, counts)]
    if short_row:
        lines[3] = "\t".join(lines[3].split("\t")[:-2])
    if fraction:
        lines[2] = "\t".join(lines[2].split("\t")[:-1] + ["0.5"])
    if text_count:
        lines[4] = "\t".join(lines[4].split("\t")[:-1] + ["x"])
    with gzip.open(folder / V.INPUTS["counts"][0], "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    n = len(cells) + 1
    meta = pd.DataFrame({"library.barcode": [c[0] for c in cells] + ["ghost.bc9"], "Assigned_cluster": "prog",
                         "Assigned_subcluster": "prog_a", "tSNE_dim1": 0.5, "tSNE_dim2": -0.5,
                         "Differentiation": [c[1] for c in cells] + [1], "CellWeek": [c[2] for c in cells] + [0],
                         "Lib_prep_batch": [c[3] for c in cells] + ["stg5diff1_S5d0_b1"],
                         "Indrops_barcode_sequence": ["ACGTACGT"] * n})
    if bad_week:
        meta["CellWeek"] = meta["CellWeek"].astype(float)
        meta.loc[0, "CellWeek"] = 1.5
    if bad_lib:
        meta.loc[0, "Lib_prep_batch"] = "stg5diff2_S5d5_b1"
    meta.sample(frac=1.0, random_state=0).to_csv(folder / V.INPUTS["meta"][0], sep="\t", index=False,
                                                 compression="gzip")
    expected = {f"diff{d}": [sum(1 for c in cells if c[1] == d and c[2] == day) for day in range(3)] for d in (1, 2)}
    return ids, counts, cells, expected


def test_veres_parses_joins_on_barcode_and_caches(tmp_path):
    ids, counts, cells, expected = _veres_raw(tmp_path)
    out = tmp_path / "proc" / "veres.npz"
    qc = V.prepare(tmp_path, out, chunk=5, expected=expected, verify=False)
    X, obs, symbols, qc2 = V.load_veres(out, min_stage_cells=1)
    assert sp.issparse(X) and X.format == "csr" and X.dtype == np.float32
    assert symbols == GENES_V and X.shape == (len(cells), len(GENES_V))
    assert np.array_equal(X.toarray(), counts[:-1])                     # count-file order, orphan row dropped
    assert obs["cell"].tolist() == ids[:-1]
    assert obs["time"].tolist() == [float(c[2]) for c in cells]
    assert obs["replicate"].tolist() == [f"diff{c[1]}" for c in cells]
    assert obs["library"].tolist() == [c[3] for c in cells] and obs["pool"].tolist() == obs["library"].tolist()
    assert obs["sample"].tolist() == [f"diff{c[1]}_d{c[2]}" for c in cells]
    assert set(obs["condition"]) == {"stage5"}
    assert qc["non_integer_entries_excl_test"] == 0 and qc2["gene_symbols_renamed"] == []
    assert qc["header_first_field"] == "# library.barcode" and qc["test_time"] == 4.0
    assert qc["count_rows_without_meta"] == 1 and qc["meta_rows_without_counts"] == 1
    assert qc["count_rows_without_meta_examples"] == ["orphan.bc0"]
    assert qc["expected_counts"]["ok"] and qc["library_label_disagreements"] == 0
    assert qc["per_time"] == {"0": 10, "1": 14, "2": 18} and qc["genes"] == len(GENES_V)
    assert qc["per_replicate_time"]["diff2|2"] == expected["diff2"][2]
    assert qc2["qc_gate"]["dropped_times"] == [] and qc2["expected_counts"]["ok"]
    z = np.load(out, allow_pickle=False)
    need = {"data", "indices", "indptr", "shape", "gene_symbols", "gene_ids", "cell", "sample", "time", "replicate",
            "library", "qc_json"}
    assert need <= set(z.files) and z["gene_ids"].tolist() == GENES_V
    side = json.loads(out.with_suffix(".qc.json").read_text(encoding="utf-8"))
    assert side["per_time"] == qc["per_time"] and side["inputs"]["counts"]["sha256"]


def test_veres_chunk_size_does_not_change_the_cache(tmp_path):
    _, _, _, expected = _veres_raw(tmp_path)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    V.prepare(tmp_path, a, chunk=1, expected=expected, verify=False)
    V.prepare(tmp_path, b, chunk=1000, expected=expected, verify=False)
    Xa, oa, _, _ = V.load_veres(a, min_stage_cells=1)
    Xb, ob, _, _ = V.load_veres(b, min_stage_cells=1)
    assert np.array_equal(Xa.toarray(), Xb.toarray()) and oa.equals(ob)


def test_veres_count_mismatch_is_loud(tmp_path, monkeypatch):
    _, _, _, expected = _veres_raw(tmp_path)
    expected["diff2"][1] += 1
    out = tmp_path / "v.npz"
    qc = V.prepare(tmp_path, out, chunk=4, expected=expected, verify=False)
    assert not qc["expected_counts"]["ok"] and "diff2|1" in qc["expected_counts"]["mismatches"]
    with pytest.raises(ValueError, match="expected"):
        V.load_veres(out)
    assert V.load_veres(out, strict=False, min_stage_cells=1)[0].shape[0] == 42
    out2 = tmp_path / "main.npz"
    monkeypatch.setattr(sys, "argv", ["veres_prepare.py", "--raw", str(tmp_path), "--out", str(out2), "--no-verify"])
    with pytest.raises(SystemExit) as exc:                               # the real expected counts do not fit
        V.main()
    assert exc.value.code not in (0, None) and out2.exists()


def test_veres_input_checks(tmp_path):
    _veres_raw(tmp_path)
    with pytest.raises(ValueError, match="pre-registered"):
        V.prepare(tmp_path, tmp_path / "x.npz", verify=True)
    assert not (tmp_path / "x.npz").exists()
    for i, (kw, msg) in enumerate(((dict(header="barcode"), "library.barcode"), (dict(short_row=True), "missing"),
                                   (dict(dup_barcode=True), "duplicate"), (dict(bad_week=True), "CellWeek"))):
        d = tmp_path / f"case{i}"
        d.mkdir()
        _veres_raw(d, **kw)
        with pytest.raises(ValueError, match=msg):
            V.prepare(d, d / "x.npz", verify=False)
        assert not (d / "x.npz").exists()
    d = tmp_path / "text"                                               # a non-numeric count stops the parse
    d.mkdir()
    _veres_raw(d, text_count=True)
    with pytest.raises(ValueError):
        V.prepare(d, d / "x.npz", verify=False)
    assert not list(d.glob("x*.npz"))


def test_veres_library_label_is_checked_against_day_and_differentiation(tmp_path):
    _, _, _, expected = _veres_raw(tmp_path, bad_lib=True)
    qc = V.prepare(tmp_path, tmp_path / "v.npz", expected=expected, verify=False)
    assert qc["library_label_disagreements"] == 1 and qc["expected_counts"]["ok"]


def test_veres_loader_applies_the_stage_gate(tmp_path):
    _, _, _, expected = _veres_raw(tmp_path)
    out = tmp_path / "v.npz"
    V.prepare(tmp_path, out, expected=expected, verify=False)
    X, obs, _, qc = V.load_veres(out, min_stage_cells=11)               # day 0 has 10 cells
    assert qc["qc_gate"]["dropped_times"] == [0.0] and sorted(obs["time"].unique()) == [1.0, 2.0]
    assert X.shape[0] == len(obs) == 32
    assert qc["qc_gate"]["test_time"] == V.TEST_TIME == 4.0 and not qc["qc_gate"]["test_time_present"]


def test_veres_duplicate_symbols_are_made_unique_by_the_loader_and_fractions_reported(tmp_path):
    genes = ["G0", "G1", "G0", "G3", "G0"]
    _, counts, _, expected = _veres_raw(tmp_path, genes=genes, fraction=True)
    out = tmp_path / "v.npz"
    qc = V.prepare(tmp_path, out, chunk=3, expected=expected, verify=False)
    assert qc["duplicated_gene_symbols"] == 2 and qc["non_integer_entries_excl_test"] == 1
    X, _, symbols, qc2 = V.load_veres(out, min_stage_cells=1)
    assert symbols == ["G0", "G1", "G0|1", "G3", "G0|2"] and qc2["gene_symbols_renamed"] == ["G0|1", "G0|2"]
    assert np.load(out)["gene_symbols"].tolist() == genes              # the cache keeps the deposit's symbols
    assert X[1, 4] == np.float32(0.5) and X[0, 4] == counts[0, 4]


def test_veres_header_with_or_without_hash(tmp_path):
    """The layout was not opened before approval: the first field must be library.barcode (cells in rows), with or
    without '#', and it is recorded."""
    for i, h in enumerate(("# library.barcode", "library.barcode", "#library.barcode")):
        d = tmp_path / f"h{i}"
        d.mkdir()
        _, counts, _, expected = _veres_raw(d, header=h)
        qc = V.prepare(d, d / "v.npz", expected=expected, verify=False)
        assert qc["header_first_field"] == h and qc["expected_counts"]["ok"]
        assert np.array_equal(V.load_veres(d / "v.npz", min_stage_cells=1)[0].toarray(), counts[:-1])


def test_veres_expression_checks_leave_the_test_day_out(tmp_path, monkeypatch):
    """Prereg Step 5: test-stage expression is not inspected, so the non-integer count skips the test day."""
    _, _, cells, expected = _veres_raw(tmp_path, fraction=True)          # the fraction sits in a day-0 row
    assert cells[1][2] == 0
    qc = V.prepare(tmp_path, tmp_path / "a.npz", expected=expected, verify=False)
    assert qc["non_integer_entries_excl_test"] == 1 and "non_integer_entries" not in qc
    monkeypatch.setattr(V, "TEST_TIME", 0.0)                             # now day 0 is the test day
    qc = V.prepare(tmp_path, tmp_path / "b.npz", expected=expected, verify=False)
    assert qc["non_integer_entries_excl_test"] == 0 and qc["test_time"] == 0.0


@pytest.mark.parametrize("which", ["veres", "pijuan_sala"])
def test_out_must_end_in_npz(tmp_path, which):
    """np.savez_compressed would append .npz and the report, the size print and the loader would miss the file:
    the name is refused before any input is read (verify=True would otherwise fail on the fixture's hashes)."""
    mod = V if which == "veres" else PS
    _veres_raw(tmp_path) if which == "veres" else _ps_raw(tmp_path)
    for name in ("cache", "cache.npy", "cache.npz.tmp"):
        with pytest.raises(ValueError, match=r"\.npz"):
            mod.prepare(tmp_path, tmp_path / name, verify=True)
    assert not list(tmp_path.glob("cache*"))


def test_unique_symbols():
    assert V.unique_symbols(["a", "b", "a", "a|x"], ["i1", "i2", "x", "i4"]) == (["a", "b", "a|x.2", "a|x"], ["a|x.2"])
    assert V.unique_symbols(["a", "", "a"], ["i1", "i2", "i3"]) == (["a", "i2", "a|i3"], ["i2", "a|i3"])
    names, renamed = V.unique_symbols(["a"] * 3, ["a"] * 3)
    assert names == ["a", "a|1", "a|2"] and renamed == ["a|1", "a|2"]


STAGES_PS = ["E6.5", "E6.75", "E7.0", "E8.0", "E8.5", "mixed_gastrulation"]


def _ps_raw(folder: Path, meta_extra=0, gene_missing=0, nnz_extra=0, shuffle=False, stage_override=None,
            bad_flag=False, dup_entry=False, symbols=None, reverse_names=False, frac_stage=None):
    """genes.tsv.gz, meta.tab.gz and a genes x cells MatrixMarket file (column-major entries like R's writeMM, or
    shuffled). Returns the dense genes x cells matrix, the meta table aligned with it and the kept-cell mask."""
    rng = np.random.default_rng(1)
    rows = []
    for s_i, st in enumerate(STAGES_PS):
        for p in (1, 2):
            rows += [(st, 2 * s_i + p, 2 if (st == "E8.0" and p == 2) else 1) for _ in range(3 + s_i)]
    n = len(rows)
    doublet = np.arange(n) % 5 == 2
    stripped = np.arange(n) % 7 == 3
    meta = pd.DataFrame({"cell": [f"cell_{i + 1}" for i in range(n)], "barcode": [f"bc{i}" for i in range(n)],
                         "sample": [str(r[1] + 10) for r in rows], "pool": [str(r[1]) for r in rows],
                         "stage": [r[0] for r in rows], "sequencing.batch": [r[2] for r in rows], "theiler": "TS10",
                         "doub.density": 0.1, "doublet": np.where(doublet, "TRUE", "FALSE"), "cluster": 1,
                         "cluster.sub": 1, "cluster.stage": 1, "cluster.theiler": 1,
                         "stripped": np.where(stripped, "TRUE", "FALSE"), "celltype": "Epiblast", "colour": "#635547",
                         "umapX": 0.1, "umapY": -0.1})
    if stage_override:
        meta.loc[0, "stage"] = stage_override
    if bad_flag:
        meta.loc[1, "doublet"] = "NA"
    if reverse_names:
        meta["cell"] = meta["cell"].to_numpy()[::-1]
    keep = ~doublet & ~stripped & (meta["stage"] != "mixed_gastrulation").to_numpy()
    written = pd.concat([meta, meta.iloc[:meta_extra]], ignore_index=True) if meta_extra else meta
    written.to_csv(folder / PS.INPUTS["meta"][0], sep="\t", index=False, compression="gzip")
    genes = pd.DataFrame({"id": [f"ENSMUSG{i:011d}" for i in range(6)],
                          "sym": symbols or [f"Gene{i}" for i in range(6)]})
    genes.iloc[:len(genes) - gene_missing].to_csv(folder / PS.INPUTS["genes"][0], sep="\t", header=False,
                                                  index=False, compression="gzip")
    M = rng.poisson(0.7, size=(6, n))
    g, c = np.nonzero(M.T)[1], np.nonzero(M.T)[0]                      # column-major: by cell, then gene
    order = rng.permutation(len(g)) if shuffle else np.arange(len(g))
    if dup_entry:
        order = np.r_[order, order[:1]]
    entries = [f"{g[i] + 1} {c[i] + 1} {M[g[i], c[i]]}" for i in order]
    if frac_stage:                                                      # one fractional count in a kept cell
        cols = set(np.flatnonzero(keep & (meta["stage"] == frac_stage).to_numpy()).tolist())
        j = next(j for j, i in enumerate(order) if c[i] in cols)
        M = M.astype(float)
        M[g[order[j]], c[order[j]]] = 0.5
        entries[j] = f"{g[order[j]] + 1} {c[order[j]] + 1} 0.5"
    lines = ["%%MatrixMarket matrix coordinate integer general", "% written by a test",
             f"6 {n} {len(order) + nnz_extra}"] + entries
    with gzip.open(folder / PS.INPUTS["counts"][0], "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return M, meta, keep


def _ps_expected(meta, keep):
    return meta.loc[keep, "stage"].value_counts().to_dict()


def test_pijuan_sala_filters_time_and_cache(tmp_path):
    M, meta, keep = _ps_raw(tmp_path)
    out = tmp_path / "proc" / "ps.npz"
    qc = PS.prepare(tmp_path, out, chunk=7, expected=_ps_expected(meta, keep), verify=False)
    X, obs, symbols, qc2 = PS.load_pijuan_sala(out, min_stage_cells=1)
    assert sp.issparse(X) and X.format == "csr" and X.dtype == np.float32 and X.shape == (int(keep.sum()), 6)
    assert np.array_equal(X.toarray(), M.T[keep])
    assert symbols == [f"Gene{i}" for i in range(6)]
    assert obs["cell"].tolist() == meta.loc[keep, "cell"].tolist()
    tmap = {"E6.5": 0.0, "E6.75": 1.0, "E7.0": 2.0, "E8.0": 6.0, "E8.5": 8.0}
    assert obs["time"].tolist() == [tmap[s] for s in meta.loc[keep, "stage"]]
    assert set(obs["replicate"]) == {"all"} and set(obs["condition"]) == {"WT"}
    assert obs["pool"].tolist() == meta.loc[keep, "pool"].tolist()
    assert obs["sample"].tolist() == meta.loc[keep, "sample"].tolist()
    assert obs["batch"].tolist() == meta.loc[keep, "sequencing.batch"].astype(str).tolist()
    assert "mixed_gastrulation" not in set(obs["stage"])
    f = qc["filters"]
    assert f["flagged_doublet"] == int((meta["doublet"] == "TRUE").sum())
    assert f["flagged_stripped"] == int((meta["stripped"] == "TRUE").sum())
    assert f["stage_mixed_gastrulation"] == int((meta["stage"] == "mixed_gastrulation").sum())
    assert qc["cells_kept"] == int(keep.sum()) and qc["expected_counts"]["ok"]
    assert qc["non_integer_entries_excl_test"] == 0 and qc["meta_cell_names_sequential"] is True
    assert qc2["gene_symbols_renamed"] == []
    assert qc["stages_with_pools_in_several_batches"] == ["E8.0"]
    assert qc["per_time"]["6"] == int((meta.loc[keep, "stage"] == "E8.0").sum())
    assert sum(qc["per_stage_pool"]["E8.0"].values()) == qc["per_time"]["6"]
    z = np.load(out, allow_pickle=False)
    need = {"data", "indices", "indptr", "shape", "gene_symbols", "gene_ids", "cell", "sample", "time", "replicate",
            "pool", "batch", "stage", "qc_json"}
    assert need <= set(z.files) and z["gene_ids"][0] == "ENSMUSG00000000000"
    assert qc2["qc_gate"]["dropped_times"] == [] and qc2["qc_gate"]["test_time_present"]
    side = json.loads(out.with_suffix(".qc.json").read_text(encoding="utf-8"))
    assert side["cells_kept"] == qc["cells_kept"] and side["inputs"]["counts"]["bytes"] > 0


def test_pijuan_sala_expression_checks_leave_the_test_stage_out(tmp_path):
    """Prereg Step 5: test-stage expression is not inspected, so the non-integer and kept-entry counts skip E8.0;
    the cache still holds E8.0 exactly as read."""
    for stage, n_frac in (("E8.0", 0), ("E7.0", 1)):
        d = tmp_path / stage
        d.mkdir()
        M, meta, keep = _ps_raw(d, frac_stage=stage)
        qc = PS.prepare(d, d / "x.npz", chunk=4, expected=_ps_expected(meta, keep), verify=False)
        assert qc["non_integer_entries_excl_test"] == n_frac and qc["test_time"] == 6.0
        non_test = keep & (meta["stage"] != "E8.0").to_numpy()
        assert qc["matrix"]["entries_kept_excl_test"] == int(np.count_nonzero(M.T[non_test]))
        assert "non_integer_entries" not in qc and "entries_kept" not in qc["matrix"]
        assert "deviation" in qc["replicate"]
        X = PS.load_pijuan_sala(d / "x.npz", min_stage_cells=1)[0]
        assert np.array_equal(X.toarray(), M.T[keep])


def test_pijuan_sala_entry_order_and_chunks_do_not_matter(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(), b_dir.mkdir()
    M, meta, keep = _ps_raw(a_dir)
    _ps_raw(b_dir, shuffle=True)
    exp = _ps_expected(meta, keep)
    PS.prepare(a_dir, a_dir / "x.npz", chunk=3, expected=exp, verify=False)
    PS.prepare(b_dir, b_dir / "x.npz", chunk=10_000, expected=exp, verify=False)
    Xa = PS.load_pijuan_sala(a_dir / "x.npz", min_stage_cells=1)[0]
    Xb = PS.load_pijuan_sala(b_dir / "x.npz", min_stage_cells=1)[0]
    assert np.array_equal(Xa.toarray(), Xb.toarray()) and np.array_equal(Xa.toarray(), M.T[keep])


@pytest.mark.parametrize("kw,msg", [(dict(meta_extra=2), "columns"), (dict(gene_missing=1), "genes"),
                                    (dict(nnz_extra=3), "entries"), (dict(stage_override="E7.1"), "stage"),
                                    (dict(bad_flag=True), "doublet"), (dict(dup_entry=True), "duplicate")])
def test_pijuan_sala_alignment_checks_stop(tmp_path, kw, msg):
    M, meta, keep = _ps_raw(tmp_path, **kw)
    with pytest.raises(ValueError, match=msg):
        PS.prepare(tmp_path, tmp_path / "x.npz", chunk=5, expected=_ps_expected(meta, keep), verify=False)
    assert not (tmp_path / "x.npz").exists()


def test_pijuan_sala_count_mismatch_is_loud(tmp_path, monkeypatch):
    M, meta, keep = _ps_raw(tmp_path)
    exp = _ps_expected(meta, keep)
    exp["E8.0"] += 1
    out = tmp_path / "ps.npz"
    qc = PS.prepare(tmp_path, out, expected=exp, verify=False)
    assert not qc["expected_counts"]["ok"] and set(qc["expected_counts"]["mismatches"]) == {"E8.0"}
    with pytest.raises(ValueError, match="expected"):
        PS.load_pijuan_sala(out)
    with pytest.raises(ValueError, match="pre-registered"):
        PS.prepare(tmp_path, tmp_path / "v.npz", verify=True)
    assert not (tmp_path / "v.npz").exists()
    out2 = tmp_path / "main.npz"
    monkeypatch.setattr(sys, "argv", ["pijuan_sala_prepare.py", "--raw", str(tmp_path), "--out", str(out2),
                                      "--no-verify"])
    with pytest.raises(SystemExit) as exc:
        PS.main()
    assert exc.value.code not in (0, None) and out2.exists()


def test_pijuan_sala_expected_counts_are_the_registered_ones():
    assert sum(PS.EXPECTED.values()) == 108_857 and PS.EXPECTED["E8.0"] == 16_681
    assert [PS.stage_time(s) for s in PS.EXPECTED] == [float(i) for i in range(9)]
    assert sum(sum(v) for v in V.EXPECTED.values()) == 51_274 and V.EXPECTED["diff2"][3] == 9576


def test_pijuan_sala_stage_time_and_gate(tmp_path):
    assert [PS.stage_time(s) for s in ("E6.5", "E6.75", "E7.25", "E8.0", "E8.5")] == [0.0, 1.0, 3.0, 6.0, 8.0]
    for bad in ("E7.1", "E6.25", "E8.75", "late"):
        with pytest.raises(ValueError):
            PS.stage_time(bad)
    M, meta, keep = _ps_raw(tmp_path)
    out = tmp_path / "ps.npz"
    PS.prepare(tmp_path, out, expected=_ps_expected(meta, keep), verify=False)
    per_time = pd.Series(PS.load_pijuan_sala(out, min_stage_cells=1)[1]["time"]).value_counts()
    small = float(per_time.idxmin())
    X, obs, _, qc = PS.load_pijuan_sala(out, min_stage_cells=int(per_time.min()) + 1)
    assert small in qc["qc_gate"]["dropped_times"] and small not in set(obs["time"]) and X.shape[0] == len(obs)
    X, obs, _, qc = PS.load_pijuan_sala(out, min_stage_cells=int(per_time.max()) + 1)
    assert qc["qc_gate"]["test_time_dropped"] and PS.TEST_TIME not in set(obs["time"])


def test_pijuan_sala_duplicate_and_missing_symbols(tmp_path):
    syms = ["Gene0", "Pakap", "Gene2", "Pakap", "", "NA"]
    M, meta, keep = _ps_raw(tmp_path, symbols=syms, reverse_names=True)
    out = tmp_path / "ps.npz"
    qc = PS.prepare(tmp_path, out, expected=_ps_expected(meta, keep), verify=False)
    assert qc["duplicated_gene_symbols"] == 1 and qc["meta_cell_names_sequential"] is False
    X, obs, symbols, qc2 = PS.load_pijuan_sala(out, min_stage_cells=1)
    assert symbols == ["Gene0", "Pakap", "Gene2", "Pakap|ENSMUSG00000000003", "ENSMUSG00000000004", "NA"]
    assert qc2["gene_symbols_renamed"] == ["Pakap|ENSMUSG00000000003", "ENSMUSG00000000004"]
    assert np.array_equal(X.toarray(), M.T[keep]) and obs["cell"].tolist() == meta.loc[keep, "cell"].tolist()


@pytest.mark.parametrize("which", ["veres", "pijuan_sala"])
def test_loaders_meet_the_fold_scripts_contract(tmp_path, which):
    """What freshfold_select/freshfold_measure read: time on the pre-registered grid, replicate, condition, sample,
    pool, unique symbols, a raw-count CSR aligned with obs."""
    if which == "veres":
        _, _, _, expected = _veres_raw(tmp_path, genes=["A", "B", "A", "C"])
        V.prepare(tmp_path, tmp_path / "c.npz", expected=expected, verify=False)
        X, obs, symbols, qc = V.load_veres(tmp_path / "c.npz", min_stage_cells=1)
        grid, n_rep = set(range(8)), 2
    else:
        _, meta, keep = _ps_raw(tmp_path, symbols=["A", "B", "A", "C", "D", "E"])
        PS.prepare(tmp_path, tmp_path / "c.npz", expected=_ps_expected(meta, keep), verify=False)
        X, obs, symbols, qc = PS.load_pijuan_sala(tmp_path / "c.npz", min_stage_cells=1)
        grid, n_rep = set(range(9)), 1
    assert {"time", "replicate", "condition", "sample", "pool"} <= set(obs.columns)
    assert set(obs["time"]) <= {float(t) for t in grid} and obs["time"].dtype == float
    assert obs["replicate"].nunique() == n_rep and obs["condition"].nunique() == 1
    assert len(set(symbols)) == len(symbols) == X.shape[1] and X.shape[0] == len(obs)
    assert isinstance(obs.index, pd.RangeIndex) and json.dumps(qc)


@pytest.mark.parametrize("which", ["veres", "pijuan_sala"])
def test_fold_scripts_accept_the_loaders_output(tmp_path, which):
    """The loaders' output passes freshfold_select's own contract check (with the robustness key freshfold_measure
    reads), survives its generic cache and its per-group subsampling with the metadata still attached."""
    import freshfold_measure as fm
    import freshfold_select as fs
    if which == "veres":
        _, _, _, expected = _veres_raw(tmp_path, genes=["A", "B", "A", "C"])
        V.prepare(tmp_path, tmp_path / "c.npz", expected=expected, verify=False)
        X, obs, symbols, qc = V.load_veres(tmp_path / "c.npz", min_stage_cells=1)
    else:
        _, meta, keep = _ps_raw(tmp_path, symbols=["A", "B", "A", "C", "", "NA"])
        PS.prepare(tmp_path, tmp_path / "c.npz", expected=_ps_expected(meta, keep), verify=False)
        X, obs, symbols, qc = PS.load_pijuan_sala(tmp_path / "c.npz", min_stage_cells=1)
    fs.check_contract(obs, symbols, which, need=("time", "replicate", "condition", fm.ROBUST_KEY[which]))
    fs.write_npz_cache(tmp_path / "generic.npz", X, obs, symbols, qc)
    X2, obs2, sym2, qc3 = fs.load_npz_cache(tmp_path / "generic.npz")
    assert np.array_equal(X2.toarray(), X.toarray()) and sym2 == symbols and qc3 == json.loads(json.dumps(qc))
    for k in ("time", "replicate", "condition", "sample", "pool"):
        assert obs2[k].tolist() == obs[k].tolist()
    groups = obs.groupby(["condition", "replicate", "time"]).size()
    kept = fs.subsample_mask(obs, cap=int(groups.max()) - 1, min_cells=0, seed=0)
    assert kept.sum() == sum(min(int(n), int(groups.max()) - 1) for n in groups)
    assert obs.loc[kept, fm.ROBUST_KEY[which]].notna().all()


@pytest.mark.parametrize("which", ["veres", "pijuan_sala"])
def test_loader_default_path_is_the_parser_default_out(tmp_path, monkeypatch, which):
    """freshfold_select.load_data calls loader() when neither --data nor --cache is given: the loader must default to
    the parser's --out default (one constant), and the cache must appear only under its final name."""
    import inspect
    import freshfold_select as fs
    mod, load = (V, V.load_veres) if which == "veres" else (PS, PS.load_pijuan_sala)
    assert inspect.signature(load).parameters["path"].default == mod.DEFAULT_OUT
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    raw, out = tmp_path / "raw", Path(mod.DEFAULT_OUT).expanduser()
    assert out.is_relative_to(tmp_path)
    raw.mkdir()
    if which == "veres":
        _, _, _, expected = _veres_raw(raw)
    else:
        _, meta, keep = _ps_raw(raw)
        expected = _ps_expected(meta, keep)
    monkeypatch.setattr(mod, "EXPECTED", expected)
    monkeypatch.setattr(sys, "argv", [f"{which}_prepare.py", "--raw", str(raw), "--no-verify"])
    mod.main()                                                          # --out left at its default
    assert out.exists() and not list(out.parent.glob("*.partial.npz"))
    X, obs, symbols, qc = fs.load_data(which)                          # no --data, no --cache
    ref = load(out)
    assert X.shape == ref[0].shape and obs.equals(ref[1]) and symbols == ref[2]
    assert qc["qc_gate"]["min_stage_cells"] == mod.MIN_STAGE_CELLS
