#!/bin/bash
#SBATCH --job-name=harness
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
# Our Task-2 forecasters inside their ORIGINAL loop (dileo_harness.py imports ~/tgl-grn-orig untouched).
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd ~/tack/code
echo "start=$(date -Is)"
python dileo_harness.py --models ${MODELS:-level4,level6,obslevel4,obslevel6} --names ${NAMES:-pancreas-human,gastrulation-human} --seeds ${SEEDS:-1234,42,7,0,1} --out ${OUT:-$HOME/tack/out/dileo_task2_harness} --target ${TARGET:-orig} || exit 1
echo "end=$(date -Is)"
