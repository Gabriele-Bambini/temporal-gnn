---
status: mapped
mapped_at: 2026-09-30T15:52:22+02:00
mapped_by: gpt-5.6-luna
confidence: low
---
# Map - Temporal GNN

## One-liner
Research workspace containing temporal-GRN/temporal-GNN snapshots and a smaller conditional-flow-matching GRN prototype. This bounded tier-0/Graphify map is not implementation-verified.

## How to run
- Tier-0 entry points include `cfm_grn/run_benchmark_v2.py`, `cfm_grn/run_benchmark_v3.py`, `cfm_grn/run_experiments_comparison.py`, `cfm_grn/train.py`, and `cfm_grn/tests/*`.
- The `tgl-grn-*` variants expose training, evaluation, download, plotting, and HPC scripts; see `.projectos/scan.md`.
- Exact environments, commands, and test status are unverified; do not treat notebooks/results/HPC scripts as validated instructions.

## Architecture
- Temporal GRN snapshots are loaded into PyG-style graph representations; dataset/GRN/PPI loaders recur across `tgl-grn-*` variants.
- Directed/recurrent GNN model zoos feed future link prediction, gene-expression forecasting, and out-degree-centrality forecasting.
- Pipelines save checkpoints/logs/results and include HPC orchestration and visualization utilities.
- `cfm_grn` is a separate conditional-flow-matching line: synthetic perturbation data, flow matcher/ODE solver, graph vector fields, regulatory masks/priors, baselines, evaluation, and runners.
- `tgl-grn-continuous` is a small prototype (8 files, ~305 LOC); edges/graph/models/temporal/evaluation/training/experiment areas are under construction and unverified.
- Graphify shows coupling around `DirSAGEGRU`, `DirGATGRU`, dataset loading, and centrality/link/gene training, with duplicated historical snapshots.

## Modules
### `tgl-grn-recurrent`, `tgl-grn-1CCD`, `tgl-grn-1CCD-fix`, `tgl-grn-exp`, `tgl-grn-2` - temporal GRN variants
- Dataset loaders, directed layers, model zoos, training/evaluation, centrality/heatmaps, and HPC scripts.
- Repeated symbols include `load_dataset`, `load_grn`, `load_ppi`, `DirSAGEGRU`, `DirGATGRU`, `GCRNGRU`, `load_model`, and `train_live_model*` (scan/Graphify; not source-verified).
- Exact differences and canonical snapshot remain open.

### `cfm_grn` - conditional flow-matching prototype
- `cfm/flow_matcher.py`, `cfm/solver.py`: flow-matching engine and ODE inference.
- `models/graph_vector_field.py`, `models/baselines.py`, `models/regulatory_mask.py`: graph fields, controls, and regulatory prior/mask components.
- `data/dataset.py`, `data/synthetic_grn.py`, `evaluate.py`: perturbation-pair data, simulator, and metrics.
- Scientific validity and real-data readiness are not established by this map.

### `tgl-grn-continuous` - continuous prototype (under construction)
- Small prototype organized around edges, graph construction, models, temporal handling, evaluation, training, and experiments.
- Parent agent is actively building these areas; interfaces and behavior are provisional until tested.

### `fatto da me` - utility area
- Minimal data-download script according to tier-0 scan; role/provenance unverified.

## Data & artifacts
- Inputs recur under GRN/PPI paths and project download scripts. Generated areas include `tmp/`, `cfm_grn/output/`, `*/results/`, and `tgl-grn-1CCD/GRN/data/`.
- Do not read data, binaries, notebook outputs, logs, checkpoints, or result directories wholesale during mapping.
- Graphify flags duplicated `AUTOPSY.md`/`results.md` and a likely trivial/ill-posed out-degree setup; this is context, not fresh validation.

## Conventions & gotchas
- Workspace is large (770 files, ~57k LOC); this map intentionally avoids broad source inspection.
- Snapshot names indicate lineage/experiments, not confirmed release boundaries.
- Graphify includes inferred/ambiguous edges; local source wins when later verified.
- Separate scientific-validity questions from execution bugs before interpreting benchmarks.

## Open questions (unverified)
- Which `tgl-grn-*` snapshot is canonical?
- What exact API and benchmark does `tgl-grn-continuous` expose?
- Do tests cover leakage, hard negatives, chronological splits, and persistence/degree baselines?
- Which data/environments are local versus HPC-only?
- Are flow matching and temporal-GNN lines intended to merge, and under what validated interface?
