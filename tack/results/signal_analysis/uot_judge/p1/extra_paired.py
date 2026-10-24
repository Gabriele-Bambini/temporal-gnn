"""Post-hoc paired comparisons from the saved per-transition scores of uot_sink.py (no refit).
Writes extra_paired.json: sink / KL-UOT / balanced / lin1 against the denoised-level baseline (persist) per target."""
import json
import os

import numpy as np

import uot_sink as U

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for name in ("gastrulation-human", "pancreas-human"):
    z = np.load(os.path.join(HERE, f"uot_sink_{name}_scores.npz"))
    out[name] = {}
    for t in ("l1", "linear", "orig"):
        S = {k[len(f"S_{t}_"):]: z[k] for k in z.files if k.startswith(f"S_{t}_")}
        res = {}
        for a in ("sink_norm_Kany", "sink_abs_Kany", "kluot_abs_Kany", "balanced_Kany", "lin1_Kany", "full_Kany"):
            for b in ("persist_Kany", "persist_L3_Kany"):
                res[f"{a} - {b}"] = U.paired(S[a], S[b])
        res["sink_norm_Kany - sink_abs_Kany"] = U.paired(S["sink_norm_Kany"], S["sink_abs_Kany"])
        out[name][t] = res
        for k, v in res.items():
            print(name, t, f"{k:40s}", "dPCC", v["dPCC"], "dJ", v["dJ"])
with open(os.path.join(HERE, "extra_paired.json"), "w") as fh:
    json.dump(out, fh, indent=1)
