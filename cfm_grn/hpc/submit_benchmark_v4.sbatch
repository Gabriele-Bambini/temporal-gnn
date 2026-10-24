#!/bin/bash
#SBATCH --job-name=cfm_v4
#SBATCH --output=logs/cfm_v4_%j.out
#SBATCH --error=logs/cfm_v4_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00

# Degradation curves. vnoise 0/1 and sign_error 0 already exist in results_v3
# (identical data, folds, seeds), so only the new levels are run here.
set -uo pipefail

source /software/miniconda3/etc/profile.d/conda.sh || true
conda activate sheaf || conda activate tsgnn || conda activate base

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_v4
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")

JOBS=jobs_v4.txt
: > "$JOBS"
for seed in 0 1 2; do
  for fold in 0 1 2 3 4; do
    base="--fold $fold --seed $seed --out results_v4"
    for sc in noisy noisy_context; do
      # (a) velocity noise sweep: SNR 2, 0.5, 0.25
      for vn in 0.5 2 4; do
        for m in mech_graph mech_graph_learned; do
          echo "--family mech --scenario $sc --model $m --signed --vnoise $vn $base" >> "$JOBS"
        done
      done
      # (b) sign error sweep: 0.5 = no sign information left
      for se in 0.1 0.2 0.3 0.5; do
        echo "--family cfm --scenario $sc --model static_graph --signed --sign_error $se $base" >> "$JOBS"
        echo "--family mech --scenario $sc --model mech_graph --signed --sign_error $se $base" >> "$JOBS"
        echo "--family mech --scenario $sc --model mech_graph --signed --vnoise 1 --sign_error $se $base" >> "$JOBS"
      done
    done
  done
done
echo "runs: $(wc -l < "$JOBS") | device: $DEVICE"
date

export OMP_NUM_THREADS=1
xargs -P 6 -I{} sh -c "python -u run_benchmark_v3.py {} --device $DEVICE >> logs/cfm_v4_runs.log 2>&1" < "$JOBS"

echo "=== finished: $(ls results_v4 | wc -l) result files ==="
date
