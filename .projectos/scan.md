# Scan - Temporal GNN
generated 2026-09-30T15:50:01+02:00 by `pos scan` (tier-0, no LLM). files 770 | code files 240 | ~57410 LOC | truncated: no
size class **M** -> mapping plan: 3x pos-mapper (cheap tier) in parallel, one per group tgl-grn-1CCD-fix,tgl-grn-2 | tgl-grn-1CCD,tgl-grn-exp | tgl-grn-recurrent,cfm_grn,tgl-grn-continuous,fatto da me; then 1x pos-mapper merges parts into MAP.md

## Layout
| group | files | code | ~LOC | langs |
|---|---|---|---|---|
| tgl-grn-recurrent | 170 | 46 | 13821 | Python, Notebook, Shell |
| tgl-grn-1CCD-fix | 142 | 35 | 10964 | Python, Shell, PowerShell |
| tgl-grn-1CCD | 283 | 44 | 10078 | Python, Notebook, Shell |
| tgl-grn-exp | 46 | 35 | 8949 | Python, Shell |
| tgl-grn-2 | 44 | 33 | 8427 | Python, Shell |
| cfm_grn | 44 | 39 | 4833 | Python, Notebook |
| tgl-grn-continuous | 8 | 7 | 305 | Python |
| fatto da me | 2 | 1 | 33 | Python |
| . | 9 | 0 | 0 |  |
| docs | 4 | 0 | 0 |  |
| sources | 18 | 0 | 0 |  |

Data/artifact dirs (NOT scanned - never read wholesale): tmp/ (1 entries), cfm_grn/output/ (6 entries), tgl-grn-exp/results/ (1 entries), tgl-grn-2/results/ (1 entries), tgl-grn-1CCD-fix/results/ (1 entries), tgl-grn-1CCD/results/ (1 entries), tgl-grn-1CCD/GRN/data/ (2 entries)

## Languages
Python 201 files ~54860 LOC; Notebook 21 files ~2050 LOC; Shell 15 files ~380 LOC; PowerShell 3 files ~120 LOC

## Entry points
- cfm_grn/run_benchmark_v2.py (__main__)
- cfm_grn/run_benchmark_v3.py (__main__)
- cfm_grn/run_experiments_comparison.py (__main__)
- cfm_grn/tests/test_cfm.py (__main__)
- cfm_grn/tests/test_data.py (__main__)
- cfm_grn/tests/test_models.py (__main__)
- cfm_grn/train.py (__main__)
- fatto da me/download_data.py (__main__)
- tgl-grn-1CCD-fix/backup_local_20260917/train_lp.py (__main__)
- tgl-grn-1CCD-fix/cancel_and_resubmit.py (__main__)
- tgl-grn-1CCD-fix/check_plots_job.py (__main__)
- tgl-grn-1CCD-fix/check_plots_logs.py (__main__)
- tgl-grn-1CCD-fix/check_status.py (__main__)
- tgl-grn-1CCD-fix/download_and_check.py (__main__)
- tgl-grn-1CCD-fix/download_data.py (__main__)
- tgl-grn-1CCD-fix/export_top_hubs.py (__main__)

## Tests
7 test files

## Existing context files
AGENTS.md, CLAUDE.md, graphify-out/GRAPH_REPORT.md, docs (graphify report: read its summary / god-nodes first)

## Hotspots
largest code files: tgl-grn-1CCD-fix/backup_local_20260917/train_models.py (1724), tgl-grn-1CCD-fix/train_models.py (1552), tgl-grn-recurrent/train_models.py (1514), tgl-grn-exp/train_future_node_centrality.py (1372), tgl-grn-recurrent/train_future_node_centrality.py (1367), tgl-grn-recurrent/_backup_pre_p0p1/train_models.py (1358), tgl-grn-recurrent/_backup_pre_p0p1/train_future_node_centrality.py (1339), tgl-grn-exp/models.py (1331)

## Symbol index (top-level defs; notebooks: headings/imports)
### tgl-grn-recurrent
- tgl-grn-recurrent/Jupyter Notebooks tgl-grn-1CCD/01_hpc_setup_and_data.ipynb (276)
- tgl-grn-recurrent/Jupyter Notebooks tgl-grn-1CCD/02_link_prediction.ipynb (109)
- tgl-grn-recurrent/Jupyter Notebooks tgl-grn-1CCD/03_gene_expression_forecasting.ipynb (107)
- tgl-grn-recurrent/Jupyter Notebooks tgl-grn-1CCD/04_centrality_forecasting.ipynb (104)
- tgl-grn-recurrent/Jupyter Notebooks tgl-grn-1CCD/05_hub_analysis_and_plots.ipynb (105)
- tgl-grn-recurrent/_backup_pre_p0p1/dataset.py (517)
- tgl-grn-recurrent/_backup_pre_p0p1/train_future_node_centrality.py (1339)
- tgl-grn-recurrent/_backup_pre_p0p1/train_lp.py (133)
- tgl-grn-recurrent/_backup_pre_p0p1/train_models.py (1358)
- tgl-grn-recurrent/aggregate_results.py (203)
- tgl-grn-recurrent/cancel_and_resubmit.py (55)
- tgl-grn-recurrent/check_plots_job.py (13)
- tgl-grn-recurrent/check_plots_logs.py (26)
- tgl-grn-recurrent/check_status.py (12)
- tgl-grn-recurrent/check_target_alignment.py (104)
- tgl-grn-recurrent/dataset.py (610)
- tgl-grn-recurrent/diag_signal.py (636)
- tgl-grn-recurrent/directed_layers.py (275)
- tgl-grn-recurrent/download_and_check.py (33)
- tgl-grn-recurrent/download_data.py (23)
- tgl-grn-recurrent/export_top_hubs.py (243)
- tgl-grn-recurrent/fetch_results.py (61)
- tgl-grn-recurrent/grn_centrality.py (97)
- tgl-grn-recurrent/hpc/launch_all_on_hpc.sh (30)
- tgl-grn-recurrent/hpc_auto_submit.py (97)
- tgl-grn-recurrent/hub_heatmap.py (364)
- tgl-grn-recurrent/inspect_gene.py (6)
- tgl-grn-recurrent/inspect_json.py (8)
- tgl-grn-recurrent/inspect_tgnn.py (19)
- tgl-grn-recurrent/inspect_tgnn2.py (17)
- tgl-grn-recurrent/models.py (1293)
- tgl-grn-recurrent/models_gene.py (827)
- tgl-grn-recurrent/read_errors.py (13)
- tgl-grn-recurrent/remote_download_data.py (47)
- tgl-grn-recurrent/remote_pip_install.py (16)
- tgl-grn-recurrent/run_final_plots.py (39)
- tgl-grn-recurrent/run_top_hubs.sh (15)
- tgl-grn-recurrent/submit_and_wait.py (53)
- tgl-grn-recurrent/submit_phase1.py (52)
- tgl-grn-recurrent/sync_and_run.ps1 (40)
- tgl-grn-recurrent/sync_and_run.sh (31)
- tgl-grn-recurrent/train_future_node_centrality.py (1367)
- tgl-grn-recurrent/train_gene.py (122)
- tgl-grn-recurrent/train_lp.py (163)
- tgl-grn-recurrent/train_models.py (1514)
- tgl-grn-recurrent/train_models_gene.py (1249)
### tgl-grn-1CCD-fix
- tgl-grn-1CCD-fix/backup_local_20260917/dataset.py (577) - load_dataset(dataset, name, gene_features, use_scgpt); load_ppi(name); load_grn(name, gene_features, use_scgpt, use_scfoundation)
- tgl-grn-1CCD-fix/backup_local_20260917/train_lp.py (117) - save_results(results, model_str, dataset_str, dataset_name); main()
- tgl-grn-1CCD-fix/backup_local_20260917/train_models.py (1724) - train_val_test_split(dataset, dataset_str, dataset_name); active_negative_sampling(edge_index, num_neg_samples, undirected); degree_matched_negative_sampling(prev_edge...
- tgl-grn-1CCD-fix/cancel_and_resubmit.py (66) - main()
- tgl-grn-1CCD-fix/check_plots_job.py (18) - check()
- tgl-grn-1CCD-fix/check_plots_logs.py (33) - check()
- tgl-grn-1CCD-fix/check_status.py (17) - check()
- tgl-grn-1CCD-fix/dataset.py (656) - load_dataset(dataset, name, gene_features, use_scgpt); load_ppi(name); load_grn(name, gene_features, use_scgpt, use_scfoundation)
- tgl-grn-1CCD-fix/download_and_check.py (42) - check_and_download()
- tgl-grn-1CCD-fix/download_data.py (25) - download_dataset()
- tgl-grn-1CCD-fix/export_top_hubs.py (269) - ensure_dir(path); sanitize_name(s); build_checkpoint_path(model_str, dataset_str, dataset_name, metric); build_node_mapping_path(dataset_name); load_node_mapping(mappi...
- tgl-grn-1CCD-fix/fetch_results.py (61)
- tgl-grn-1CCD-fix/grn_centrality.py (97)
- tgl-grn-1CCD-fix/hpc/launch_all_on_hpc.sh (30)
- tgl-grn-1CCD-fix/hpc_auto_submit.py (97)
- tgl-grn-1CCD-fix/hub_heatmap.py (364)
- tgl-grn-1CCD-fix/inspect_gene.py (6)
- tgl-grn-1CCD-fix/inspect_json.py (8)
- tgl-grn-1CCD-fix/inspect_tgnn.py (19)
- tgl-grn-1CCD-fix/inspect_tgnn2.py (17)
- tgl-grn-1CCD-fix/models.py (1239)
- tgl-grn-1CCD-fix/models_gene.py (827)
- tgl-grn-1CCD-fix/read_errors.py (13)
- tgl-grn-1CCD-fix/remote_download_data.py (47)
- tgl-grn-1CCD-fix/remote_pip_install.py (16)
- tgl-grn-1CCD-fix/run_final_plots.py (39)
- tgl-grn-1CCD-fix/run_top_hubs.sh (15)
- tgl-grn-1CCD-fix/submit_and_wait.py (53)
- tgl-grn-1CCD-fix/sync_and_run.ps1 (40)
- tgl-grn-1CCD-fix/sync_and_run.sh (31)
- tgl-grn-1CCD-fix/train_future_node_centrality.py (1302)
- tgl-grn-1CCD-fix/train_gene.py (122)
- tgl-grn-1CCD-fix/train_lp.py (176)
- tgl-grn-1CCD-fix/train_models.py (1552)
- tgl-grn-1CCD-fix/train_models_gene.py (1249)
### tgl-grn-1CCD
- tgl-grn-1CCD/Jupyter Notebooks tgl-grn-1CCD/01_hpc_setup_and_data.ipynb (276)
- tgl-grn-1CCD/Jupyter Notebooks tgl-grn-1CCD/02_link_prediction.ipynb (109)
- tgl-grn-1CCD/Jupyter Notebooks tgl-grn-1CCD/03_gene_expression_forecasting.ipynb (107)
- tgl-grn-1CCD/Jupyter Notebooks tgl-grn-1CCD/04_centrality_forecasting.ipynb (104)
- tgl-grn-1CCD/Jupyter Notebooks tgl-grn-1CCD/05_hub_analysis_and_plots.ipynb (105)
- tgl-grn-1CCD/cancel_and_resubmit.py (55)
- tgl-grn-1CCD/check_plots_job.py (13)
- tgl-grn-1CCD/check_plots_logs.py (26)
- tgl-grn-1CCD/check_status.py (12)
- tgl-grn-1CCD/dataset.py (517)
- tgl-grn-1CCD/directed_layers.py (275)
- tgl-grn-1CCD/download_and_check.py (33)
- tgl-grn-1CCD/download_data.py (23)
- tgl-grn-1CCD/export_top_hubs.py (243)
- tgl-grn-1CCD/fetch_results.py (61)
- tgl-grn-1CCD/grn_centrality.py (97)
- tgl-grn-1CCD/hpc/launch_all_on_hpc.sh (30)
- tgl-grn-1CCD/hpc_auto_submit.py (97)
- tgl-grn-1CCD/hub_heatmap.py (364)
- tgl-grn-1CCD/inspect_gene.py (6)
- tgl-grn-1CCD/inspect_json.py (8)
- tgl-grn-1CCD/inspect_tgnn.py (19)
- tgl-grn-1CCD/inspect_tgnn2.py (17)
- tgl-grn-1CCD/logs-baseline/audit_learned_baseline.py (197)
- tgl-grn-1CCD/logs-baseline/audit_negative_sampling.py (400)
- tgl-grn-1CCD/logs-baseline/audit_new_edges_definition.py (61)
- tgl-grn-1CCD/logs-baseline/audit_same_source.py (195)
- tgl-grn-1CCD/logs-baseline/collect_hpc_ns_audit.py (120)
- tgl-grn-1CCD/models.py (1239)
- tgl-grn-1CCD/models_gene.py (827)
- tgl-grn-1CCD/read_errors.py (13)
- tgl-grn-1CCD/remote_download_data.py (47)
- tgl-grn-1CCD/remote_pip_install.py (16)
- tgl-grn-1CCD/run_final_plots.py (39)
- tgl-grn-1CCD/run_top_hubs.sh (15)
- tgl-grn-1CCD/submit_and_wait.py (53)
- tgl-grn-1CCD/submit_phase1.py (52)
- tgl-grn-1CCD/sync_and_run.ps1 (40)
- tgl-grn-1CCD/sync_and_run.sh (31)
- tgl-grn-1CCD/train_future_node_centrality.py (1302)
- tgl-grn-1CCD/train_gene.py (122)
- tgl-grn-1CCD/train_lp.py (133)
- tgl-grn-1CCD/train_models.py (1330)
- tgl-grn-1CCD/train_models_gene.py (1249)
### tgl-grn-exp
- tgl-grn-exp/baselines.py (178)
- tgl-grn-exp/cancel_and_resubmit.py (55)
- tgl-grn-exp/check_plots_job.py (13)
- tgl-grn-exp/check_plots_logs.py (26)
- tgl-grn-exp/check_status.py (12)
- tgl-grn-exp/dataset.py (528)
- tgl-grn-exp/directed_layers.py (275)
- tgl-grn-exp/download_and_check.py (33)
- tgl-grn-exp/download_data.py (23)
- tgl-grn-exp/eval_baselines.py (274)
- tgl-grn-exp/export_top_hubs.py (243)
- tgl-grn-exp/fetch_results.py (61)
- tgl-grn-exp/grn_centrality.py (97)
- tgl-grn-exp/hpc/launch_all_on_hpc.sh (30)
- tgl-grn-exp/hpc_auto_submit.py (97)
- tgl-grn-exp/hub_heatmap.py (364)
- tgl-grn-exp/inspect_gene.py (6)
- tgl-grn-exp/inspect_json.py (8)
- tgl-grn-exp/inspect_tgnn.py (19)
- tgl-grn-exp/inspect_tgnn2.py (17)
- tgl-grn-exp/models.py (1331)
- tgl-grn-exp/models_gene.py (827)
- tgl-grn-exp/read_errors.py (13)
- tgl-grn-exp/remote_download_data.py (47)
- tgl-grn-exp/remote_pip_install.py (16)
- tgl-grn-exp/run_final_plots.py (39)
- tgl-grn-exp/run_top_hubs.sh (15)
- tgl-grn-exp/submit_and_wait.py (53)
- tgl-grn-exp/submit_phase1.py (52)
- tgl-grn-exp/sync_and_run.sh (31)
- tgl-grn-exp/train_future_node_centrality.py (1372)
- tgl-grn-exp/train_gene.py (122)
- tgl-grn-exp/train_lp.py (132)
- tgl-grn-exp/train_models.py (1291)
- tgl-grn-exp/train_models_gene.py (1249)
### tgl-grn-2
- tgl-grn-2/cancel_and_resubmit.py (55)
- tgl-grn-2/check_plots_job.py (13)
- tgl-grn-2/check_plots_logs.py (26)
- tgl-grn-2/check_status.py (12)
- tgl-grn-2/dataset.py (528)
- tgl-grn-2/directed_layers.py (275)
- tgl-grn-2/download_and_check.py (33)
- tgl-grn-2/download_data.py (23)
- tgl-grn-2/export_top_hubs.py (243)
- tgl-grn-2/fetch_results.py (61)
- tgl-grn-2/grn_centrality.py (97)
- tgl-grn-2/hpc/launch_all_on_hpc.sh (30)
- tgl-grn-2/hpc_auto_submit.py (97)
- tgl-grn-2/hub_heatmap.py (364)
- tgl-grn-2/inspect_gene.py (6)
- tgl-grn-2/inspect_json.py (8)
- tgl-grn-2/inspect_tgnn.py (19)
- tgl-grn-2/inspect_tgnn2.py (17)
- tgl-grn-2/models.py (1331)
- tgl-grn-2/models_gene.py (827)
- tgl-grn-2/read_errors.py (13)
- tgl-grn-2/remote_download_data.py (47)
- tgl-grn-2/remote_pip_install.py (16)
- tgl-grn-2/run_final_plots.py (39)
- tgl-grn-2/run_top_hubs.sh (15)
- tgl-grn-2/submit_and_wait.py (53)
- tgl-grn-2/submit_phase1.py (52)
- tgl-grn-2/sync_and_run.sh (31)
- tgl-grn-2/train_future_node_centrality.py (1302)
- tgl-grn-2/train_gene.py (122)
- tgl-grn-2/train_lp.py (132)
- tgl-grn-2/train_models.py (1291)
- tgl-grn-2/train_models_gene.py (1249)
### cfm_grn
- cfm_grn/_backup_pre_fixes/cfm/flow_matcher.py (191) - "Conditional Flow Matching (CFM) Engine." | class ConditionalFlowMatcher()[3m]; class OptimalTransportCFM(ConditionalFlowMatcher)[2m]
- cfm_grn/_backup_pre_fixes/data/synthetic_grn.py (187) - "Simulatore Sintetico di Dinamiche di Regolazione Genica (Hill Kinet..." | class SyntheticGRNSimulator()[3m]; class SyntheticDataset(Dataset)
- cfm_grn/_backup_pre_fixes/evaluate.py (88) - "Metriche di Valutazione per Previsioni di Perturbazione a Singola C..." | compute_energy_distance(x_pred, x_true); compute_evaluation_metrics(x_pred, x_true, target_g...
- cfm_grn/_backup_pre_fixes/models/baselines.py (102) - "Modelli di Controllo e Baseline per Ablazione Scientifica." | class MLPVectorField(nn.Module)[1m]; class StaticGraphVectorField(GraphVectorField); class PersistenceBa...
- cfm_grn/_backup_pre_fixes/models/graph_vector_field.py (227) - "Graph Vector Field: Rete Neurale MPNN per la predizione del campo v..." | class DirectedGeneConv(nn.Module)[1m]; class GraphVectorField(nn.Module)[2m]
- cfm_grn/_backup_pre_fixes/models/regulatory_mask.py (168) - "Modulo di Prior Regolatorio e Maschera di Attivita' Contestuale (GR..." | class SparseAdjacencyPrior()[2m]; class RegulatoryActivityMask(nn.Module)[2m]
- cfm_grn/aggregate_v2.py (137) - "Aggregate results_v2/*.json into summary tables and pre-registered..." | ci95(x); paired(sc, a, b, ot_a)
- cfm_grn/aggregate_v3.py (183) - "Aggregate results_v3 (and results_v2 for pairing) into tables + pre..." | load(path, v2); ci95(x); sel(); paired(a_kw, b_kw, metric, sign); add(tag, label, a_kw, b_kw)
- cfm_grn/aggregate_v4.py (132) - "Degradation curves from results_v3 + results_v4 (identical data/fol..." | ci95(x); curve(sub, xcol)
- cfm_grn/cfm/__init__.py (11) - "Conditional Flow Matching (CFM) engine." |
- cfm_grn/cfm/flow_matcher.py (219) - "Conditional Flow Matching (CFM) Engine." | class ConditionalFlowMatcher()[3m]; class OptimalTransportCFM(ConditionalFlowMatcher)[2m]
- cfm_grn/cfm/solver.py (97) - "ODE Solvers per l'inferenza e la generazione con Flow Matching." | class ODESolver()[1m]
- cfm_grn/data/__init__.py (11) - "Data package for cfm_grn." |
- cfm_grn/data/dataset.py (99) - "Dataset e DataLoaders per profili di singola cellula sotto perturba..." | class PerturbationPairDataset(Dataset); create_dataloaders(benchmark_data, batch_size, held_...
- cfm_grn/data/synthetic_grn.py (362) - "Simulatore Sintetico di Dinamiche di Regolazione Genica (Hill Kinet..." | class SyntheticGRNSimulator()[9m]; class SyntheticDataset(Dataset)
- cfm_grn/evaluate.py (167) - "Metriche di Valutazione per Previsioni di Perturbazione a Singola C..." | compute_energy_distance(x_pred, x_true); compute_delta_metrics(x_pred, x_true, x_ctrl, targe...
- cfm_grn/models/__init__.py (18) - "Models package for cfm_grn." |
- cfm_grn/models/baselines.py (144) - "Modelli di Controllo e Baseline per Ablazione Scientifica." | class MLPVectorField(nn.Module)[1m]; class StaticGraphVectorField(GraphVectorField); class LearnedStatic...
- cfm_grn/models/graph_vector_field.py (245) - "Graph Vector Field: Rete Neurale MPNN per la predizione del campo v..." | class DirectedGeneConv(nn.Module)[1m]; class GraphVectorField(nn.Module)[3m]
- cfm_grn/models/regulatory_mask.py (188) - "Modulo di Prior Regolatorio e Maschera di Attivita' Contestuale (GR..." | class SparseAdjacencyPrior()[2m]; class RegulatoryActivityMask(nn.Module)[2m]
- cfm_grn/notebooks/01_flow_matcher.ipynb (204) - nb 17code/18md | Lezione 1: Il Motore del Flow Matching; 0. Setup e Import; 1. Il Problema Biologico: Perché Serve un Modello Generativo; Creiamo un esempio giocattolo...
- cfm_grn/notebooks/02_ode_solver.ipynb (60) - nb 5code/10md | Lezione 2: L'Integratore ODE (Test-Time Inference); 1. Dalla Velocità alla Traiettoria: La Teoria dell'Integrazi; 1.1 Metodo di Eulero (1° Ordine); 1.2...
... 25 more code files not listed (use Grep/Glob).
