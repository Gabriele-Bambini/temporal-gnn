# TGL-GRN Continuous Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small from-scratch project that uses Sottoriva's distributed pancreas and gastrulation GRN files to test whether confidence-aware continuous pseudotime modelling recovers temporal signal beyond persistence and temporal nulls.

**Architecture:** A streaming data layer converts the large edge CSVs into compact top-K graph snapshots and aligns them with per-gene expression summaries. A causal confidence-aware context feeds the same directed graph vector field into a direct Temporal GNN and a Graph Flow Matching model. Whole future snapshots are evaluated against simple baselines and shuffled-time controls, producing a machine-readable and human-readable verdict.

**Tech Stack:** Python 3.12, NumPy, pandas, PyArrow, SciPy, scikit-learn, PyTorch, pytest, matplotlib. PyTorch Geometric is intentionally not required for the core directed aggregation.

---

## File structure

- `tgl-grn-continuous/pyproject.toml`: package metadata, dependencies, pytest configuration and CLI entry point.
- `tgl-grn-continuous/README.md`: scientific scope, commands, interpretation and limitations.
- `tgl-grn-continuous/src/tgl_grn_continuous/data.py`: schema parsing, node timeline loading and bounded-memory edge cache creation.
- `tgl-grn-continuous/src/tgl_grn_continuous/temporal.py`: causal confidence-aware context and chronological splits.
- `tgl-grn-continuous/src/tgl_grn_continuous/graph.py`: graph normalization and structural per-node features.
- `tgl-grn-continuous/src/tgl_grn_continuous/models.py`: shared directed graph encoder, recurrent history, direct model and flow-matching model.
- `tgl-grn-continuous/src/tgl_grn_continuous/training.py`: deterministic training, integration and checkpoint selection.
- `tgl-grn-continuous/src/tgl_grn_continuous/evaluation.py`: baselines, metrics, null controls and verdict.
- `tgl-grn-continuous/src/tgl_grn_continuous/experiment.py`: end-to-end orchestration and report generation.
- `tgl-grn-continuous/src/tgl_grn_continuous/__main__.py`: CLI.
- `tgl-grn-continuous/tests/`: behavioral and integration tests built from synthetic CSV fixtures.

### Task 1: Scaffold the isolated project and define contracts

**Files:**
- Create: `tgl-grn-continuous/pyproject.toml`
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/__init__.py`
- Create: `tgl-grn-continuous/tests/conftest.py`
- Test: `tgl-grn-continuous/tests/test_import.py`

- [ ] **Step 1: Write the failing import test**

```python
def test_package_exposes_version():
    import tgl_grn_continuous
    assert isinstance(tgl_grn_continuous.__version__, str)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_import.py -q`
Expected: FAIL because `tgl_grn_continuous` does not exist.

- [ ] **Step 3: Add the package scaffold**

Create a `src`-layout package, declare Python `>=3.11`, the listed dependencies, and `__version__ = "0.1.0"`. Add a `tgl-grn-continuous` console script pointing to `tgl_grn_continuous.__main__:main`.

- [ ] **Step 4: Run the test and verify GREEN**

Run: `python -m pytest tests/test_import.py -q`
Expected: `1 passed`.

### Task 2: Load the node timeline without temporal leakage

**Files:**
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/data.py`
- Test: `tgl-grn-continuous/tests/test_data.py`

- [ ] **Step 1: Write failing tests for the distributed column schema**

The fixture must contain `node_id`, `gene`, two timestamps and all five statistics. Assert that `load_node_timeline()` returns shape `[time, node, 4]`, excludes `sum` by default, preserves gene order and returns integer timestamp labels.

```python
timeline = load_node_timeline(csv_path)
assert timeline.values.shape == (2, 2, 4)
assert timeline.feature_names == ("mean", "median", "std", "frac_nonzero")
assert timeline.timestamps.tolist() == [2, 3]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_data.py::test_load_node_timeline_excludes_cell_count_sum -q`
Expected: FAIL because the loader is missing.

- [ ] **Step 3: Implement schema parsing and validation**

Parse names matching `tNNN_sNN_stat`, align requested timestamps, cast numeric data to `float32`, reject duplicate node IDs and expose the omitted sum separately for diagnostics.

- [ ] **Step 4: Verify GREEN and malformed-schema behavior**

Run: `python -m pytest tests/test_data.py -q`
Expected: all node-loader tests pass, including a clear error for missing statistics.

### Task 3: Build compact graph snapshots from the giant CSVs

**Files:**
- Modify: `tgl-grn-continuous/src/tgl_grn_continuous/data.py`
- Test: `tgl-grn-continuous/tests/test_edge_cache.py`

- [ ] **Step 1: Write failing tests for bounded top-K extraction**

Create a small edge CSV with two grouped timestamps, descending and deliberately unsorted weights. Assert exact top targets per regulator, parsed `bin_mid`, parsed `n_cells_bin`, and deterministic cache reload.

```python
cache = build_edge_cache(edge_csv, cache_path, max_targets_per_source=2)
assert max(np.bincount(cache.snapshots[2].source)) <= 2
assert cache.bin_mid[2] == pytest.approx(0.2)
assert cache.n_cells[2] == 10
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_edge_cache.py -q`
Expected: FAIL because cache extraction is missing.

- [ ] **Step 3: Implement streaming extraction**

Use `pyarrow.csv.open_csv()` with selected columns. Maintain only the requested number of targets per `(timestamp, source)`, parse one metadata blob per timestamp, normalize dtypes and save compressed NumPy arrays plus JSON metadata. Never load the whole edge table.

- [ ] **Step 4: Verify GREEN and memory contract**

Run: `python -m pytest tests/test_edge_cache.py -q`
Expected: all extraction, unsorted-input and cache-roundtrip tests pass.

### Task 4: Compute trustworthy graph features

**Files:**
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/graph.py`
- Test: `tgl-grn-continuous/tests/test_graph.py`

- [ ] **Step 1: Write failing directed-graph tests**

Use a three-node directed graph with analytically checkable in-strength, out-strength and PageRank ordering. Test robust per-snapshot weight normalization and confirm source/target direction is not reversed.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_graph.py -q`
Expected: FAIL because graph utilities are missing.

- [ ] **Step 3: Implement graph normalization and node statistics**

Normalize positive weights by a robust high quantile, clip extreme values, calculate strengths with `numpy.add.at`, and implement convergent weighted PageRank with dangling-node handling.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_graph.py -q`
Expected: all directed graph tests pass.

### Task 5: Recover context causally and create honest time splits

**Files:**
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/temporal.py`
- Test: `tgl-grn-continuous/tests/test_temporal.py`

- [ ] **Step 1: Write failing causal-smoothing tests**

Assert that changing a future snapshot cannot alter an earlier context; a low-cell bin borrows more from history than a high-cell bin; and an explicit change point prevents averaging across the boundary.

```python
ctx_a = causal_confidence_context(values, times, cells, min_effective_cells=100)
values[-1] *= 1000
ctx_b = causal_confidence_context(values, times, cells, min_effective_cells=100)
np.testing.assert_allclose(ctx_a[:-1], ctx_b[:-1])
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_temporal.py -q`
Expected: FAIL because temporal preprocessing is missing.

- [ ] **Step 3: Implement causal adaptive windows and chronological splits**

Use only present/past observations, distance decay and square-root cell-count confidence. Detect large robust adjacent changes and do not cross them. Provide train, validation and test transition indices with no shared target snapshot.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_temporal.py -q`
Expected: all leakage, confidence and split tests pass.

### Task 6: Implement the shared directed field and both temporal objectives

**Files:**
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/models.py`
- Test: `tgl-grn-continuous/tests/test_models.py`

- [ ] **Step 1: Write failing model-contract tests**

Test output shapes, separate incoming/outgoing source-to-target messages, recurrent-state propagation, gradients through a history window, exact zero-step identity, and an FM target scaled by actual pseudotime interval.

```python
pred = model.step(x, edge_index, edge_weight, s=0.2, delta_s=0.1)
assert pred.shape == x.shape
torch.testing.assert_close(model.step(x, edge_index, edge_weight, s=0.2, delta_s=0.0), x)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_models.py -q`
Expected: FAIL because models are missing.

- [ ] **Step 3: Implement minimal models**

Implement a pure-PyTorch directed weighted aggregator with `index_add_`, a shared per-gene GRU over 3–4 encoded snapshots, a residual vector field, `DirectTemporalGNN.step()`, `GraphFlowMatcher.loss()` and midpoint/RK4 integration. Implement the no-graph ablation by disabling messages without changing the recurrent and decoder capacity.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_models.py -q`
Expected: all shape, direction, integration and gradient tests pass.

### Task 7: Train deterministically and select on future validation snapshots

**Files:**
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/training.py`
- Test: `tgl-grn-continuous/tests/test_training.py`

- [ ] **Step 1: Write a failing learnable-toy test**

Construct a two-node process with a known linear directed evolution. Assert that training lowers validation error and that the best checkpoint is restored rather than the last epoch.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_training.py -q`
Expected: FAIL because the trainer is missing.

- [ ] **Step 3: Implement training loops**

Set all seeds, normalize using training snapshots only, weight pair losses by source/target confidence, train on full chronological transitions, clip gradients, use patience-based validation and restore a deep-copied best state dict.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_training.py -q`
Expected: the toy dynamics are learned reproducibly.

### Task 8: Add baselines, null controls and a conservative verdict

**Files:**
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/evaluation.py`
- Test: `tgl-grn-continuous/tests/test_evaluation.py`

- [ ] **Step 1: Write failing metric and verdict tests**

Test persistence, unequal-gap momentum, delta Spearman, top-change precision, hub-turnover precision and all three verdict states. Include constant-vector behavior without NaNs.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_evaluation.py -q`
Expected: FAIL because evaluation is missing.

- [ ] **Step 3: Implement baselines and nulls**

Add persistence, momentum, no-graph FM and deterministic target-time permutation. Aggregate metrics across late transitions and seeds. Require improvement over both persistence and shuffled time before emitting `signal_recovered`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_evaluation.py -q`
Expected: all metric edge cases and verdict tests pass.

### Task 9: Wire the experiment and CLI

**Files:**
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/experiment.py`
- Create: `tgl-grn-continuous/src/tgl_grn_continuous/__main__.py`
- Test: `tgl-grn-continuous/tests/test_cli.py`

- [ ] **Step 1: Write a failing end-to-end fixture test**

Run the CLI against tiny node and edge CSV fixtures. Assert creation of `metrics.json`, `report.md`, `predictions.npz` and a report containing every baseline and the limitations statement.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_cli.py -q`
Expected: FAIL because orchestration is missing.

- [ ] **Step 3: Implement prepare/run/report commands**

Support `--quick`, `--datasets`, `--data-root`, `--output-dir`, `--top-k-edges`, `--epochs` and `--seeds`. In quick mode use bounded epochs and one seed but never weaken the chronological split or remove baselines.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_cli.py -q`
Expected: the fixture experiment completes and all artifacts are readable.

### Task 10: Document and run the real quick benchmark

**Files:**
- Create: `tgl-grn-continuous/README.md`
- Create: `tgl-grn-continuous/results/.gitkeep`
- Modify: `tgl-grn-continuous/README.md`

- [ ] **Step 1: Document the scientific contract before running**

Explain preserved ideas, corrections, identity matching versus OT, commands, expected runtime, metrics, and the hard limitation that no cell-level dynamics can be recovered from aggregated files.

- [ ] **Step 2: Run the full automated suite**

Run: `python -m pytest -q`
Expected: zero failures.

- [ ] **Step 3: Run a real pancreas quick benchmark**

Run: `python -m tgl_grn_continuous run --data-root ../tgl-grn-1CCD/GRN --datasets pancreas --quick --output-dir results/quick-pancreas`
Expected: a complete report or an explicit resource/runtime failure with resumable edge cache.

- [ ] **Step 4: Run a real gastrulation quick benchmark**

Run: `python -m tgl_grn_continuous run --data-root ../tgl-grn-1CCD/GRN --datasets gastrulation --quick --output-dir results/quick-gastrulation`
Expected: a complete report or an explicit resource/runtime failure with resumable edge cache.

- [ ] **Step 5: Perform final verification**

Run: `python -m pytest -q` and inspect both generated reports, cache sizes and `git diff --check` if the project has been initialized as a repository.
Expected: zero test failures, no malformed output and no claim stronger than the evidence.

## Plan self-review

- The design preserves the original biological object and datasets while avoiding code copying.
- Every production behavior has a preceding failing test.
- The graph CSV path is bounded-memory and appropriate for the measured 1.81 GB free RAM.
- The primary target is continuous and future-aligned; binary link prediction is not silently retained.
- Evaluation contains persistence and temporal nulls and never splits nodes inside the same future snapshot.
- The plan does not claim cell-level, lineage or causal recovery from aggregate data.
