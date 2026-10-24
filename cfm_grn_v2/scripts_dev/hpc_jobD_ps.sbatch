#!/bin/bash
#SBATCH --job-name=rafm_D_ps
#SBATCH --output=logs/rafm_D_ps_%j.out
#SBATCH --error=logs/rafm_D_ps_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Pre-registration v0.2, Step 6 (Job D), Pijuan-Sala et al. 2019: fresh-fold selection (test E8.0 = quarter day 6
# deleted first), folds P1-P234, seeds 0 and 1, same arms as Veres; replicate = 'all' for training (documented
# deviation, decisions.md 2026-09-26 pomeriggio). Then R-SEL and R-HOR pooled over both data sets, by code ->
# outputs/freshfold/choice.json (no human edit before Job E). Submit after Job D Veres has finished:
#   sbatch scripts_dev/hpc_jobD_ps.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs outputs/freshfold
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
echo "[$(date)] Job D Pijuan-Sala on $(hostname)"
python -u scripts_dev/freshfold_select.py --dataset pijuan_sala --data "$DATA/processed/pijuan_sala2019_atlas.npz" \
    --prior "$DATA/priors/collectri_mouse.tsv" --mech-decision outputs/mech_v02_decision.json --out outputs/freshfold_ps
echo "[$(date)] Job D Pijuan-Sala done; applying R-SEL and R-HOR"
python -u scripts_dev/freshfold_select.py --decide outputs/freshfold_veres outputs/freshfold_ps \
    --choice outputs/freshfold/choice.json
echo "[$(date)] Job D decided"
