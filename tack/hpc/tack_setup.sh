#!/bin/bash
# TACK step 0 on the Bocconi HPC login node: clone baselines, fetch erythroid data, build the 'chrono' env.
# Run with: nohup bash ~/tack/tack_setup.sh > ~/tack/logs/setup.log 2>&1 &
set -u
T=~/tack
mkdir -p $T/ext $T/data/erythroid $T/logs $T/code $T/out
cd $T/ext
[ -d Chronocell ] || git clone https://github.com/pachterlab/Chronocell.git
[ -d FGP_2024 ] || git clone --depth 1 https://github.com/pachterlab/FGP_2024.git
echo "Chronocell $(cd Chronocell && git rev-parse HEAD)" > $T/logs/baseline_shas.txt
echo "FGP_2024 $(cd FGP_2024 && git rev-parse HEAD)" >> $T/logs/baseline_shas.txt
cat $T/logs/baseline_shas.txt

# erythroid lineage (same URL as scvelo.datasets.gastrulation_erythroid)
F=$T/data/erythroid/erythroid_lineage.h5ad
if [ ! -s "$F" ]; then
  curl -L -A "Mozilla/5.0 (X11; Linux x86_64)" --retry 3 -o "$F" https://ndownloader.figshare.com/files/27686871
fi
ls -l "$F"
sha256sum "$F" > $T/logs/erythroid_sha256.txt

# separate env so the frozen cfm_grn_v2 'sheaf' env is untouched
source /software/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^chrono "; then
  conda create -y -n chrono python=3.11
fi
conda activate chrono
pip install --quiet numpy scipy pandas matplotlib scikit-learn anndata h5py sympy tqdm numba
python -c "import numpy, scipy, anndata, sympy; print('env ok', numpy.__version__, scipy.__version__, anndata.__version__)"
grep -rhoE "^(import|from) [a-zA-Z_]+" $T/ext/Chronocell/Chronocell/*.py $T/ext/Chronocell/Chronocell/models/*.py | sort | uniq -c
echo SETUP_DONE
