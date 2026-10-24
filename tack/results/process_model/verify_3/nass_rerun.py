"""Rerun NASS 'full' on l1 (nass.py functions imported, never main(); graph and OT not needed for 'full'), check that
it reproduces results.json on nass's own 20 masks, save the strict predictions, and score them on the
1000 s + i masks (C6 comparison). Output: nass_l1_preds_<name>.npz, nass_rerun.json."""
import json
import os
import time

import numpy as np

import common as C
import nass as N

D = C.D


def main():
    out = {}
    ref = json.load(open(os.path.join(C.SA, "results.json")))
    for name in C.NAMES:
        t0 = time.time()
        d = D.Data(name)
        Y = d.Y("l1")
        F, cls = N.build_features(d, "l1", Y, None, None)
        fb = N.fallback(d, "l1", Y, F)
        G_ = N.GROUPS
        full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"]
        st = N.Stacker(F, cls, Y, full)
        pred, chosen = N.run_learned(st, [full], Y, d, fb)
        S = D.score(Y, pred, D.masks(d.G, d.T))
        own = D.summarize(S, list(range(2, d.T)))
        refv = ref[name]["targets"]["l1"]["table"]["full"]["i>=2"]
        P = np.stack([np.asarray(p, np.float64) for p in pred])
        np.savez_compressed(os.path.join(C.HERE, f"nass_l1_preds_{name}.npz"), pred=P)
        rows = C.score(Y, list(P), C.masks(d.G, d.T))
        np.save(os.path.join(C.HERE, f"nass_l1_rows_{name}.npy"), rows)
        out[name] = {"reproduce_on_nass_masks_i>=2": {"rerun": [own["pcc"], own["pta_up"], own["pta_down"]],
                                                      "results.json": [refv["pcc"], refv["pta_up"], refv["pta_down"]]},
                     "on_1000s+i_masks_i>=2": C.summ(rows, range(2, d.T)),
                     "on_1000s+i_masks_i>=1": C.summ(rows, range(1, d.T)),
                     "seconds": round(time.time() - t0)}
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(os.path.join(C.HERE, "nass_rerun.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
