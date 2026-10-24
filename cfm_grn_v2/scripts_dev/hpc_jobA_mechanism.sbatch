#!/bin/bash
#SBATCH --job-name=rafm_A_mech
#SBATCH --output=logs/rafm_A_mech_%j.out
#SBATCH --error=logs/rafm_A_mech_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Pre-registration v0.2, Step 2 (Job A): the mechanism runs on the old validation folds of both data sets, test stage
# removed (scripts_dev/mechanism_v02.py): recipe R-a (version 0.1.10, arms base / noft / gated / transfer read from
# ONE training through its stage checkpoints, plus the H5 / C2 / C3 diagnostics), R-b (field sees the biological
# time, with and without the second stage) and R-c (spline centre, no second stage); 3 seeds. Then --combine applies,
# by code, the reproducibility gate against the v0.1.11 selection (seeds 0 and 1), Rule B and the admission to Step 6
# -> outputs/mech_v02_decision.json. Nothing here touches a test stage.
# Needs the instrumented code (stage checkpoints, var_ratio) in ~/cfm_grn_v2 itself: the scripts put the cwd-relative
# src/ and scripts_dev/ first on sys.path, so a separate code copy is NOT used (job 669574 failed that way).
# Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_jobA_mechanism.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
CODE="$HOME/cfm_grn_v2"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1
python -c "import inspect, cfm_grn_v2.training.trainer as t; assert 'keep_snapshots' in inspect.signature(t.train_flow).parameters"
DATA="$HOME/rafm_data"
echo "[$(date)] Job A on $(hostname)"
for DS in eb argelaguet; do
    if [ "$DS" = eb ]; then
        C="$DATA/processed/moon2019_eb.npz"; P="$DATA/priors/collectri_human.tsv"
    else
        C="$DATA/processed/argelaguet2022_rna.npz"; P="$DATA/priors/collectri_mouse.tsv"
    fi
    python -u "$CODE/scripts_dev/mechanism_v02.py" --dataset "$DS" --cache "$C" --prior "$P" --seeds 3 \
        --reference "outputs/${DS}_select_v0111/results.csv" --out "outputs/mech_$DS"
    echo "[$(date)] $DS done"
done
python -u "$CODE/scripts_dev/mechanism_v02.py" --combine outputs/mech_eb outputs/mech_argelaguet \
    --decision outputs/mech_v02_decision.json
echo "[$(date)] Job A done"
