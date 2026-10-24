#!/bin/bash
#SBATCH --job-name=corr
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/home/3393519/tack/logs/%x_%j.out
# Their own Task-2 models (original repo, untouched) under their protocol or a corrected one (dileo_corrected.py).
set +u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd ~/tack/code
echo "start=$(date -Is) protocol=${PROTOCOL:-orig} target=${TARGET:-orig}"
nvidia-smi -L || true
for P in ${PROTOCOLS:-${PROTOCOL:-orig}}; do for T in ${TARGETS:-${TARGET:-orig}}; do
  python dileo_corrected.py --protocol $P --target $T --models ${MODELS:-linear,mlp,chebnet,gcrngru,gcn,gat,evolvegcn} --names ${NAMES:-pancreas-human,gastrulation-human} --seeds ${SEEDS:-1234,42,7,0,1} || echo "FAILED $P $T"
done; done
echo "end=$(date -Is)"
