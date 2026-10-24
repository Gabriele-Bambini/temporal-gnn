#!/bin/bash
#SBATCH --job-name=precis
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate chrono
cd ~/tack/code
MODE=${MODE:-full}
echo "MODE=$MODE start=$(date -Is)"
if [ "$MODE" = "smoke" ]; then python precision_study.py --quick || exit 1; else python precision_study.py || exit 1; fi
echo "end=$(date -Is)"
