#!/bin/bash
#SBATCH --job-name=anchor
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
# Their ORIGINAL code (anonymous.4open.science/r/tgl-grn-1CCD, untouched in ~/tgl-grn-orig): run one of their
# Task-2 models to check that the harness reproduces the paper's Table 2 row before any comparison.
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd ~/tgl-grn-orig
MODEL=${MODEL:-linear}
SEED=${SEED:-1234}
for NAME in ${NAMES:-pancreas-human gastrulation-human}; do
  echo "== $MODEL $NAME seed=$SEED start=$(date -Is)"
  python train_gene.py --model "$MODEL" --dataset grn --name "$NAME" --gene_features --seed "$SEED" || echo "FAILED $NAME"
done
echo "end=$(date -Is)"
