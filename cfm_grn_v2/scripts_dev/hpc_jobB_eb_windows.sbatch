#!/bin/bash
#SBATCH --job-name=rafm_B_ebwin
#SBATCH --output=logs/rafm_B_ebwin_%j.out
#SBATCH --error=logs/rafm_B_ebwin_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Pre-registration v0.2, Step 3 (Job B): the FROZEN version-0.1.10 model on the two other inner windows of the
# embryoid body (days 6-9 = window 1, days 18-21 = window 3), exactly the arguments of outputs/eb_v0110
# (--seeds 5 --n-hvg 1000, max-cells 2000, min-detect 0.01, choice outputs/eb_select/choice.json), every control and
# the shuffled-time twin; gene space (primary) and 50 PCs (does not condition the gene-space claim).
# Non-confirmatory: windows 1 and 3 were selection folds. Rule and wording are in the pre-registration.
# Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_jobB_eb_windows.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
C="$DATA/processed/moon2019_eb.npz"; P="$DATA/priors/collectri_human.tsv"
bench() { python -u scripts_dev/eb_benchmark.py --cache "$C" --prior "$P" --seeds 5 --n-hvg 1000 \
              --choice outputs/eb_select/choice.json "$@"; }
echo "[$(date)] Job B on $(hostname)"
for T in 1 3; do
    bench --test-time "$T.0" --out "outputs/eb_v0110_t$T"
    echo "[$(date)] gene space, window $T done"
done
for T in 1 3; do
    bench --test-time "$T.0" --pca 50 --out "outputs/eb_v0110_pca50_t$T"
    echo "[$(date)] 50 PCs, window $T done"
done
echo "[$(date)] Job B done"
