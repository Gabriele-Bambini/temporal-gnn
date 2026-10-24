#!/bin/bash
#SBATCH --job-name=cfm_v3
#SBATCH --output=logs/cfm_v3_%j.out
#SBATCH --error=logs/cfm_v3_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00

# One failed run must not stop the others -> no `-e`.
set -uo pipefail

source /software/miniconda3/etc/profile.d/conda.sh || true
conda activate sheaf || conda activate tsgnn || conda activate base

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_v3
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")

JOBS=jobs_v3.txt
: > "$JOBS"
for seed in 0 1 2; do
  for fold in 0 1 2 3 4; do
    base="--fold $fold --seed $seed"

    # (1)+(2) snapshot models, signed vs unsigned prior, edge diagnostics
    for sc in noisy noisy_context; do
      for m in static_graph static_learned graph_cfm; do
        echo "--family cfm --scenario $sc --model $m $base" >> "$JOBS"
        echo "--family cfm --scenario $sc --model $m --signed $base" >> "$JOBS"
      done
    done

    # (3) mechanistic route
    for sc in oracle noisy noisy_context; do
      echo "--family mech --scenario $sc --model mech_true $base" >> "$JOBS"
      for vn in 0 1; do
        echo "--family mech --scenario $sc --model mech_mlp --vnoise $vn $base" >> "$JOBS"
        for m in mech_graph mech_graph_learned; do
          echo "--family mech --scenario $sc --model $m --vnoise $vn $base" >> "$JOBS"
          if [ "$sc" != "oracle" ]; then   # oracle prior is all-activation: signed == unsigned
            echo "--family mech --scenario $sc --model $m --signed --vnoise $vn $base" >> "$JOBS"
          fi
        done
      done
    done
  done
done
echo "runs: $(wc -l < "$JOBS") | device: $DEVICE"
date

export OMP_NUM_THREADS=1
xargs -P 6 -I{} sh -c "python -u run_benchmark_v3.py {} --device $DEVICE >> logs/cfm_v3_runs.log 2>&1" < "$JOBS"

echo "=== finished: $(ls results_v3 | wc -l) result files ==="
date
