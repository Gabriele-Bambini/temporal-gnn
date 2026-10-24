#!/bin/bash
#SBATCH --job-name=rafm_P_parse
#SBATCH --output=logs/rafm_P_parse_%j.out
#SBATCH --error=logs/rafm_P_parse_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Pre-registration v0.2, Step 5: cache the two fresh data sets as sparse raw counts + cell metadata. Each parser checks
# its inputs against the pre-registered sizes / SHA-256 and the expected cells per stage, and stops on any mismatch;
# it prints only per-stage cell counts (the test stages' expression is never inspected).
# Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_parse_v02.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs "$HOME/rafm_data/processed"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
echo "[$(date)] parsers on $(hostname)"
rc=0
python -u scripts_dev/veres_prepare.py --raw "$DATA/veres2019" --out "$DATA/processed/veres2019_stage5.npz" || rc=1
echo "[$(date)] Veres done (rc=$rc)"
python -u scripts_dev/pijuan_sala_prepare.py --raw "$DATA/pijuan_sala2019" \
    --out "$DATA/processed/pijuan_sala2019_atlas.npz" || rc=1
echo "[$(date)] Pijuan-Sala done (rc=$rc)"
exit $rc
