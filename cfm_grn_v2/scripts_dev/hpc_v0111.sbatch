#!/bin/bash
#SBATCH --job-name=rafm_v0111
#SBATCH --output=logs/rafm_v0111_%j.out
#SBATCH --error=logs/rafm_v0111_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Version 0.1.11 candidates on both real data sets (arm set v0111 of select_config.py: base = the linear centre
# chosen in 0.1.10; "time" = the field sees the biological time; "noft" = no marginal fine-tune stage).
# Per data set: selection on validation folds with the test stage REMOVED -> outputs/<ds>_select_v0111/choice.json;
# if an arm beats base by the pre-registered rule, the single measurement on the test stage with it, same seeds
# as the 0.1.10 measurement (outputs/<ds>_v0110), so the two can be compared seed by seed.
# Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_v0111.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1
DATA="$HOME/rafm_data"
is_base() { python -c "import json,sys; sys.exit(0 if json.load(open('$1'))['winner'] == 'base' else 1)"; }
echo "[$(date)] v0111 on $(hostname)"

for DS in eb argelaguet; do
    if [ "$DS" = eb ]; then
        C="$DATA/processed/moon2019_eb.npz"; P="$DATA/priors/collectri_human.tsv"
        BENCH=(python -u scripts_dev/eb_benchmark.py --cache "$C" --prior "$P" --seeds 5 --n-hvg 1000)
    else
        C="$DATA/processed/argelaguet2022_rna.npz"; P="$DATA/priors/collectri_mouse.tsv"
        BENCH=(python -u scripts_dev/argelaguet_benchmark.py --cache "$C" --prior "$P" --seeds 5 --n-hvg 1000 --max-cells 3000)
    fi
    python -u scripts_dev/select_config.py --dataset "$DS" --arms v0111 --cache "$C" --prior "$P" \
        --out "outputs/${DS}_select_v0111"
    echo "[$(date)] $DS selection done"
    if is_base "outputs/${DS}_select_v0111/choice.json"; then
        echo "$DS: no arm beats the version-0.1.10 configuration on the validation folds; nothing to measure"
    else
        "${BENCH[@]}" --choice "outputs/${DS}_select_v0111/choice.json" --out "outputs/${DS}_v0111"
        echo "[$(date)] $DS chosen-configuration measurement done"
    fi
done
echo "[$(date)] all done"
