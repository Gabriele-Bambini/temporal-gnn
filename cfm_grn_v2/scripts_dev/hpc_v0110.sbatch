#!/bin/bash
#SBATCH --job-name=rafm_v0110
#SBATCH --output=logs/rafm_v0110_%j.out
#SBATCH --error=logs/rafm_v0110_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Version 0.1.10 on one real data set, with the gene detection filter (min_detect 0.01, the scripts' default).
# The stud QoS allows 8 CPUs per user and one running job, so the two data sets are two jobs.
#   DATASET=eb | argelaguet
#   STEPS   any of: smoke select chosen base   (default "select chosen base"; order is always the one below)
#     smoke   selection with a tiny budget, fails fast if anything is wrong with the data set
#     select  selection on validation folds, test stage REMOVED from the data -> outputs/<ds>_select/choice.json
#     chosen  the single measurement on the test stage with the chosen configuration, every control, 5 seeds
#     base    the same measurement with the version-0.1.9 configuration (skipped if the choice is base), so the
#             effect of the gene filter and of the version-0.1.10 changes can be told apart
# Submit from ~/cfm_grn_v2:
#   sbatch --export=ALL,DATASET=eb scripts_dev/hpc_v0110.sbatch
#   sbatch --export=ALL,DATASET=argelaguet,STEPS="smoke select chosen" scripts_dev/hpc_v0110.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
DATASET="${DATASET:?set DATASET=eb or DATASET=argelaguet}"
STEPS=" ${STEPS:-select chosen base} "
has() { [[ "$STEPS" == *" $1 "* ]]; }
is_base() { python -c "import json,sys; sys.exit(0 if json.load(open('$1'))['winner'] == 'base' else 1)"; }

if [ "$DATASET" = eb ]; then
    C="$DATA/processed/moon2019_eb.npz"; P="$DATA/priors/collectri_human.tsv"
    bench() { python -u scripts_dev/eb_benchmark.py --cache "$C" --prior "$P" --seeds 5 --n-hvg 1000 "$@"; }
else
    C="$DATA/processed/argelaguet2022_rna.npz"; P="$DATA/priors/collectri_mouse.tsv"
    bench() { python -u scripts_dev/argelaguet_benchmark.py --cache "$C" --prior "$P" --seeds 5 --n-hvg 1000 \
                  --max-cells 3000 "$@"; }
fi
CHOICE="outputs/${DATASET}_select/choice.json"
echo "[$(date)] $DATASET on $(hostname), steps:$STEPS"

if has smoke; then
    python -u scripts_dev/select_config.py --dataset "$DATASET" --cache "$C" --prior "$P" \
        --out "outputs/${DATASET}_select_smoke" --smoke
    echo "[$(date)] smoke done"
fi
if has select; then
    python -u scripts_dev/select_config.py --dataset "$DATASET" --cache "$C" --prior "$P" --out "outputs/${DATASET}_select"
    echo "[$(date)] selection done"
fi
if has chosen; then
    bench --choice "$CHOICE" --out "outputs/${DATASET}_v0110"
    echo "[$(date)] chosen-configuration measurement done"
fi
if has base; then
    if is_base "$CHOICE"; then
        echo "choice is base: the chosen-configuration run already is the base measurement"
    else
        bench --out "outputs/${DATASET}_filter_base"
        echo "[$(date)] base-configuration measurement done"
    fi
fi
echo "[$(date)] all steps done"
