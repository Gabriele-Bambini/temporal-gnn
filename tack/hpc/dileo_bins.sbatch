#!/bin/bash
#SBATCH --job-name=dileobin
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate chrono
cd ~/tack/code
echo "start=$(date -Is)"
python ${SCRIPT:-dileo_bins_rebuild.py} || exit 1
echo "end=$(date -Is)"
