#!/bin/bash
#SBATCH --job-name=rafm_D_veres
#SBATCH --output=logs/rafm_D_veres_%j.out
#SBATCH --error=logs/rafm_D_veres_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Pre-registration v0.2, Step 6 (Job D), Veres et al. 2019 Stage 5: fresh-fold selection (test day 4 deleted first),
# pre-registered folds V1-V56, seeds 0 and 1, arms base / noft / gated / transfer from one training each (no M arm:
# outputs/mech_v02_decision.json admits none). Defaults only: --decide refuses other seeds / genes / thresholds.
# Needs the cache of hpc_parse_v02.sbatch. Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_jobD_veres.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
echo "[$(date)] Job D Veres on $(hostname)"
python -u scripts_dev/freshfold_select.py --dataset veres --data "$DATA/processed/veres2019_stage5.npz" \
    --prior "$DATA/priors/collectri_human.tsv" --mech-decision outputs/mech_v02_decision.json --out outputs/freshfold_veres
echo "[$(date)] Job D Veres done"
