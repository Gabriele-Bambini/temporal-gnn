#!/bin/bash
#SBATCH --job-name=rafm_E_veres
#SBATCH --output=logs/rafm_E_veres_%j.out
#SBATCH --error=logs/rafm_E_veres_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Pre-registration v0.2, Step 7 (Job E), Veres: the ONE measurement of Veres day 4 from day 3 with the Step-6 winner
# (outputs/freshfold/choice.json, re-derived by code from the Job-D outputs before any data is loaded), base, noft,
# gated, every control, the shuffled twin, 5 seeds; per-differentiation scores for ROBUST; replicate floor.
# Loader (--data), never --cache. Submit after Job D is decided:  sbatch scripts_dev/hpc_jobE_veres.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
echo "[$(date)] Job E Veres on $(hostname)"
python -u scripts_dev/freshfold_measure.py --dataset veres --data "$DATA/processed/veres2019_stage5.npz" \
    --prior "$DATA/priors/collectri_human.tsv" --choice outputs/freshfold/choice.json --out outputs/freshfold_measure_veres
echo "[$(date)] Job E Veres done"
