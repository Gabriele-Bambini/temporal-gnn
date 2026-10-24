#!/bin/bash
#SBATCH --job-name=sf2clock
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
# sci-FATE2 clock probe (prereg v2, sha256 10d74859...). MODE=smoke or full.
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate chrono
cd ~/tack/code
MODE=${MODE:-full}
echo "MODE=$MODE host=$(hostname) start=$(date -Is)"
if [ "$MODE" = "smoke" ]; then
  python scifate2_clock_probe.py --out ~/tack/out/scifate2_clock_smoke --quick --procs ${SLURM_CPUS_PER_TASK:-8} || exit 1
else
  python scifate2_clock_probe.py --out ~/tack/out/scifate2_clock --procs ${SLURM_CPUS_PER_TASK:-8} || exit 1
fi
echo "end=$(date -Is)"
