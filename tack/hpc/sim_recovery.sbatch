#!/bin/bash
#SBATCH --job-name=simrec
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=20:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
# Simulation recovery: MODE=smoke (all scenarios quick) or full (self, fast, double in sequence).
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate chrono
cd ~/tack/code
MODE=${MODE:-full}
echo "MODE=$MODE host=$(hostname) start=$(date -Is)"
for S in self fast double; do
  if [ "$MODE" = "smoke" ]; then
    python sim_recovery.py --scenario $S --quick --procs ${SLURM_CPUS_PER_TASK:-8} || exit 1
  else
    python sim_recovery.py --scenario $S --procs ${SLURM_CPUS_PER_TASK:-8} || exit 1
  fi
done
echo "end=$(date -Is)"
