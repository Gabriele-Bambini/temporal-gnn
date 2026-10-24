#!/bin/bash
#SBATCH --job-name=rafm_C_ebpub
#SBATCH --output=logs/rafm_C_ebpub_%j.out
#SBATCH --error=logs/rafm_C_ebpub_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Pre-registration v0.2, Step 4 (Job C, configuration C_a = frozen v0.1.10): the like-for-like published embryoid-body
# benchmark (TrajectoryNet's eb_velocity_v5.npz, 5 whitened PCs, exact W1 with POT, leave-out t = 1, 2, 3, 5 seeds),
# every control, 'next', the shuffled twin and the in-package OT-CFM equivalent; calibration gate, model check, P2a,
# P2b applied by code (summary.json, written once after the deciding primary scaler and again at the end).
# Deviations logged in decisions.md 2026-09-27 (b). One torch thread: 5-D inputs and small networks run faster so.
# Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_jobC_ebpub.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src PYTHONUNBUFFERED=1
echo "[$(date)] Job C on $(hostname)"
python -c "import ot; print('POT', ot.__version__)"
python -u scripts_dev/eb_published_protocol.py --npz "$HOME/rafm_data/eb_published/eb_velocity_v5.npz" --seeds 5 \
    --threads 1 --out outputs/eb_published
echo "[$(date)] Job C done"
