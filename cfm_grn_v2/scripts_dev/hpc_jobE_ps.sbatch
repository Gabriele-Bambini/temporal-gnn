#!/bin/bash
#SBATCH --job-name=rafm_E_ps
#SBATCH --output=logs/rafm_E_ps_%j.out
#SBATCH --error=logs/rafm_E_ps_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Pre-registration v0.2, Step 7 (Job E), Pijuan-Sala: the ONE measurement of E8.0 from E7.75 (primary) and of E8.0
# with E8.25 also removed (secondary), same models and controls as Veres; per-pool scores for ROBUST, replicate floor
# with the E8.0 pools' sequencing batches. Then R-CONF over both data sets (Holm over the 2 primaries), by code ->
# outputs/freshfold/verdict.json. Submit after Job E Veres has finished:  sbatch scripts_dev/hpc_jobE_ps.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs outputs/freshfold
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
echo "[$(date)] Job E Pijuan-Sala on $(hostname)"
python -u scripts_dev/freshfold_measure.py --dataset pijuan_sala --data "$DATA/processed/pijuan_sala2019_atlas.npz" \
    --prior "$DATA/priors/collectri_mouse.tsv" --choice outputs/freshfold/choice.json --out outputs/freshfold_measure_ps
echo "[$(date)] Job E Pijuan-Sala done; applying R-CONF"
python -u scripts_dev/freshfold_measure.py --decide outputs/freshfold_measure_veres outputs/freshfold_measure_ps \
    --verdict outputs/freshfold/verdict.json
echo "[$(date)] Job E decided"
