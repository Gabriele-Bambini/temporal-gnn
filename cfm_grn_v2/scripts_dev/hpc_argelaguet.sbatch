#!/bin/bash
#SBATCH --job-name=rafm_argel
#SBATCH --output=logs/rafm_argel_%j.out
#SBATCH --error=logs/rafm_argel_%j.err
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00

# Held-out-stage benchmark on the Argelaguet et al. (2022) gastrulation time course, end to end:
#   1. wait for the GEO download to be complete and size-verified (geo_download.sh writes "DONE ok=33 bad=0")
#   2. build the sparse RNA cache once (scripts_dev/argelaguet_prepare.py)
#   3. a quick run (1 seed, small budget) that fails fast if anything is wrong
#   4. the full run (5 seeds, every control) -> outputs/argelaguet_interp
# The model is CPU-only (small MLP field, Hungarian couplings), so no GPU is requested.
# Submit from ~/cfm_grn_v2:  sbatch scripts_dev/hpc_argelaguet.sbatch

set -euo pipefail
source /software/miniconda3/etc/profile.d/conda.sh
conda activate sheaf
cd "$HOME/cfm_grn_v2"
mkdir -p logs outputs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src PYTHONUNBUFFERED=1

DATA="$HOME/rafm_data"
LOG="$DATA/argelaguet2022/geo_download.log"
for i in $(seq 1 120); do
    if grep -q "^DONE ok=33 bad=0" "$LOG" 2>/dev/null; then break; fi
    if grep -q "^DONE" "$LOG" 2>/dev/null; then echo "download finished with failures:"; tail -n 5 "$LOG"; exit 1; fi
    sleep 60
done
grep -q "^DONE ok=33 bad=0" "$LOG" || { echo "download not complete after 2 h"; exit 1; }
echo "[$(date)] download verified: 33 files, sizes match GEO"

CACHE="$DATA/processed/argelaguet2022_rna.npz"
if [ ! -f "$CACHE" ]; then
    python -u scripts_dev/argelaguet_prepare.py --geo "$DATA/argelaguet2022/geo" \
        --meta "$DATA/argelaguet2022/meta/metadata.txt.gz" --out "$CACHE"
fi
echo "[$(date)] cache ready: $CACHE"

python -u scripts_dev/argelaguet_benchmark.py --cache "$CACHE" --prior "$DATA/priors/collectri_mouse.tsv" \
    --quick --seeds 1 --out outputs/argelaguet_quick
echo "[$(date)] quick run done"

python -u scripts_dev/argelaguet_benchmark.py --cache "$CACHE" --prior "$DATA/priors/collectri_mouse.tsv" \
    --seeds 5 --n-hvg 1000 --max-cells 3000 --out outputs/argelaguet_interp
echo "[$(date)] full run done"
