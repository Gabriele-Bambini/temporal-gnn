"""Judge 2: LatentSB with the one non-causal scalar removed. lsb_model.LatentSB.__init__ sets nmed = median cell count
over ALL bins (bins > t included) and factor() caps the SVD cell weights with it. Here nmed is recomputed from bins
<= t at every origin t (median of nc[:t+1]); everything else is the unmodified judge2 copy of run_lsb.py.
Output: judge2/lsb_causal/results_<dataset>.json.  Usage: python run_lsb_causal.py <dataset>"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lsb_common as C
import lsb_model
import run_lsb

_factor = lsb_model.LatentSB.factor


def factor_causal(self, t, k, basis):
    if (t, k, basis) not in self._fac:
        self.nmed = float(np.median(self.d.nc[:t + 1]))
    return _factor(self, t, k, basis)


lsb_model.LatentSB.factor = factor_causal
C.OUT = os.path.join(HERE, "lsb_causal")
os.makedirs(C.OUT, exist_ok=True)

if __name__ == "__main__":
    run_lsb.main(sys.argv[1])
