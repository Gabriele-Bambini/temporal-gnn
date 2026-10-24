"""Re-run NASS 'full' (l1) from nass.py pieces, check the anchor on NASS's own masks, score on the CONTEXT masks."""
import json
import os

import numpy as np

import common as C

out = {}
for name in C.NAMES:
    e = C.Ext(name)
    P, z = C.nass_full_l1(e, cache=False)
    Y = e.Y
    mk_n = C.D.masks(e.G, e.T)
    S = C.D.score(Y, P, mk_n)
    tab_n = C.D.summarize(S, list(range(2, e.T)))
    mk = C.masks3(e.G, e.T)
    tab = C.pcc_table(Y, P, mk)
    out[name] = {"nass_masks_i>=2": tab_n,
                 "context_masks_i>=2": np.round(C.mean_from(tab, 2), 4).tolist(),
                 "context_masks_i>=1": np.round(C.mean_from(tab, 1), 4).tolist(),
                 "per_transition_pcc_context": np.round(np.nanmean(tab[:, :, 0], 0), 4).tolist(),
                 "per_transition_pcc_nassmasks": np.round(np.nanmean(S[:, :, 0], 0), 4).tolist()}
    print(name, out[name]["nass_masks_i>=2"], out[name]["context_masks_i>=2"], flush=True)
json.dump(out, open(os.path.join(C.OUT, "run_nass.json"), "w"), indent=1)
