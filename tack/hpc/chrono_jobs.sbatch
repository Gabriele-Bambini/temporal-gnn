#!/bin/bash
#SBATCH --job-name=chrono_tack
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
# TACK step 0 on the HPC. MODE=smoke: quick timing of both scripts; MODE=full: reproduction then probe.
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate chrono
cd ~/tack/code
MODE=${MODE:-full}
echo "MODE=$MODE host=$(hostname) start=$(date -Is)"
if [ "$MODE" = "smoke" ]; then
  python chrono_repro.py --out ~/tack/out/repro_quick --quick --procs ${SLURM_CPUS_PER_TASK:-8} || exit 1
  python chrono_prior_probe.py --prep ~/tack/out/repro_quick/erythroid_prep.npz --out ~/tack/out/probe_quick --quick --procs ${SLURM_CPUS_PER_TASK:-8} || exit 1
else
  python chrono_repro.py --out ~/tack/out/repro --procs ${SLURM_CPUS_PER_TASK:-8} --n-random 100 || exit 1
  python chrono_prior_probe.py --prep ~/tack/out/repro/erythroid_prep.npz --out ~/tack/out/probe --procs ${SLURM_CPUS_PER_TASK:-8} || exit 1
fi
echo "end=$(date -Is)"
