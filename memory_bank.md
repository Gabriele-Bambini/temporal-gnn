# Memory Bank — Temporal GNN Project

## User Preferences & Credentials
- **HPC Cluster**: HPC Bocconi (`slogin.hpc.unibocconi.it`)
- **Username**: `3393519` (or `3393523` / `3393519@studbocconi.it`)
- **SSH Key**: `~/.ssh/id_bocconi`
- **SSH Key Passphrase**: `barracuda`
- **Slurm Queues/Partition**: `--partition=stud --qos=stud`
- **Preferred GPU**: `--gres=gpu:1`

## Project Architecture & Notes
- Paper: *"Applications of temporal graph learning for predicting the dynamics of biological systems"* (Dileo & Sottoriva, arXiv:2605.28659)
- Repository: `tgl-grn-1CCD`
- Dataset HuggingFace: `manuel-dileo/temporal-grn`
- Models: `GCRN-GRU`, `ChebNet`, `EvolveGCN-O/H`, `ROLAND`, `GCN`, `GAT`, `scGPT`, `scFoundation`, `MLP`, `Linear`
- Tasks:
  1. Future Gene Expression Forecasting
  2. Future Link Prediction
  3. Future Out-Degree Centrality Prediction

## PhD Project — Conditional Flow Matching + GRN (`cfm_grn`)
- Directory: `cfm_grn`
- Concept: Learning Context-Specific Regulatory Activity for Generalizable Single-Cell Perturbation Prediction
- Framework: Optimal Transport Conditional Flow Matching (OT-CFM) + Directed GNN (DirectedGeneConv) + Activity Mask M_theta
- Baselines: Static Graph GNN, MLP Vector Field, Persistence Control
- Evaluation: Energy Distance, Pearson correlation, target knockdown accuracy on held-out genes
- Status: Fully implemented, 8/8 unit tests passed, benchmark comparison completed.
