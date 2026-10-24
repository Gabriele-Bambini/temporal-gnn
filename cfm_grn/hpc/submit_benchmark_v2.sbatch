#!/bin/bash
#SBATCH --job-name=cfm_v2
#SBATCH --output=logs/cfm_v2_%j.out
#SBATCH --error=logs/cfm_v2_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# One failed run must not stop the others -> no `-e`.
set -uo pipefail

source /software/miniconda3/etc/profile.d/conda.sh || true
conda activate sheaf || conda activate tsgnn || conda activate base

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_v2

python -c "import torch, scipy, pandas; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")

# Seeds outermost: if the job is cut short, complete seeds are still usable.
JOBS=jobs_v2.txt
: > "$JOBS"
for seed in 0 1 2; do
  for fold in 0 1 2 3 4; do
    for sc in oracle noisy noisy_context; do
      echo "--scenario $sc --model persistence --fold $fold --seed $seed" >> "$JOBS"
      for m in mlp static_graph static_learned graph_cfm graph_cfm_dense; do
        echo "--scenario $sc --model $m --fold $fold --seed $seed" >> "$JOBS"
      done
    done
    # Ablation of the OT bug (original global coupling), oracle scenario only.
    for m in mlp static_graph graph_cfm; do
      echo "--scenario oracle --model $m --ot global --fold $fold --seed $seed" >> "$JOBS"
    done
  done
done
echo "runs: $(wc -l < "$JOBS") | device: $DEVICE"
date

# Tiny models: 6 concurrent runs share the GPU. Runs are resumable (skip if JSON exists).
export OMP_NUM_THREADS=1
xargs -P 6 -I{} sh -c "python -u run_benchmark_v2.py {} --device $DEVICE >> logs/cfm_v2_runs.log 2>&1" < "$JOBS"

echo "=== finished: $(ls results_v2 | wc -l) result files ==="
date
