#!/bin/bash
#SBATCH --job-name=rafm_follow
#SBATCH --output=logs/rafm_follow_%j.out
#SBATCH --error=logs/rafm_follow_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Follow-up measurements after the version-0.1.10 jobs (668101 embryoid body, 668102 gastrulation):
#   1. embryoid body, PCA-50 robustness control, with the gene filter and the configuration chosen on the
#      validation folds (outputs/eb_select/choice.json). The first PCA-50 run (no filter, version 0.1.9) tied with
#      mean interpolation; this checks whether that still holds.
#   2. gastrulation, version-0.1.9 configuration with the gene filter, unless the choice was base (then 668102's
#      measurement already is it). Runs after 668102 because the QoS allows one running job per user.
# Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_followup.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
echo "[$(date)] follow-up on $(hostname)"

python -u scripts_dev/eb_benchmark.py --cache "$DATA/processed/moon2019_eb.npz" --prior "$DATA/priors/collectri_human.tsv" \
    --choice outputs/eb_select/choice.json --pca 50 --seeds 5 --n-hvg 1000 --out outputs/eb_v0110_pca50
echo "[$(date)] embryoid-body PCA-50 done"

CHOICE=outputs/argelaguet_select/choice.json
if [ -f "$CHOICE" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$CHOICE'))['winner'] == 'base' else 1)"; then
    echo "gastrulation choice is base: nothing to add"
else
    python -u scripts_dev/argelaguet_benchmark.py --cache "$DATA/processed/argelaguet2022_rna.npz" \
        --prior "$DATA/priors/collectri_mouse.tsv" --seeds 5 --n-hvg 1000 --max-cells 3000 --out outputs/argelaguet_filter_base
    echo "[$(date)] gastrulation base-configuration measurement done"
fi
echo "[$(date)] all done"
