#!/bin/bash
#SBATCH --job-name=chrono_repro
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
# Calibration gate D1 (Chronocell erythroid reproduction). MODE=quick for a timing run.
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate chrono
cd ~/tack/code
MODE=${MODE:-full}
if [ "$MODE" = "quick" ]; then
  python chrono_repro.py --out ~/tack/out/repro_quick --quick --procs 8
else
  python chrono_repro.py --out ~/tack/out/repro --procs 8 --n-random 100
fi
