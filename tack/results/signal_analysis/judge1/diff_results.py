"""Judge 1: compare original prototype outputs with my reruns (max |delta| per design, and selection agreement)."""
import json
import os
import sys

import numpy as np

B = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN\tack\results\signal_analysis"
J = os.path.join(B, "judge1")
out = {}


def walk(a, b, path, acc):
    """Collect |a-b| for every numeric leaf present in both; count mismatching non-numeric leaves."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                walk(a[k], b[k], path + [str(k)], acc)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [str(i)], acc)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a is None or b is None:
            return
        if np.isfinite(a) and np.isfinite(b):
            acc["num"].append((abs(a - b), "/".join(path)))
    elif a != b:
        acc["other"].append("/".join(path))


def cmp(tag, fa, fb, skip=("seconds", "sec", "nmf_and_law_fit_seconds", "protocol", "model")):
    A, Bb = json.load(open(fa)), json.load(open(fb))
    acc = {"num": [], "other": []}
    walk(A, Bb, [], acc)
    num = [x for x in acc["num"] if not any(s in x[1].split("/") for s in skip)]
    oth = [x for x in acc["other"] if not any(s in x.split("/") for s in skip)]
    d = np.array([x[0] for x in num]) if num else np.zeros(1)
    worst = sorted(num, reverse=True)[:5]
    out[tag] = {"n_numeric": len(num), "max_abs_delta": float(d.max()), "n_gt_0.005": int((d > 0.005).sum()),
                "n_gt_0.02": int((d > 0.02).sum()), "worst": [[round(w[0], 5), w[1]] for w in worst],
                "n_nonnumeric_mismatch": len(oth), "nonnumeric_examples": oth[:5]}
    print(tag, json.dumps(out[tag], indent=1))


if __name__ == "__main__":
    which = sys.argv[1:] or ["nass", "lsb", "popot"]
    if "nass" in which:
        for nm, sub in (("pancreas-human", "rerun_nass_p"), ("gastrulation-human", "rerun_nass_g")):
            A = json.load(open(os.path.join(B, "design_best-signal", "results.json")))[nm]
            Bb = json.load(open(os.path.join(J, sub, "results.json")))[nm]
            json.dump(A, open(os.path.join(J, f"_tmpA_{nm}.json"), "w"))
            json.dump(Bb, open(os.path.join(J, f"_tmpB_{nm}.json"), "w"))
            cmp(f"nass_{nm}", os.path.join(J, f"_tmpA_{nm}.json"), os.path.join(J, f"_tmpB_{nm}.json"))
            os.remove(os.path.join(J, f"_tmpA_{nm}.json"))
            os.remove(os.path.join(J, f"_tmpB_{nm}.json"))
    if "lsb" in which:
        for nm in ("pancreas-human", "gastrulation-human"):
            fb = os.path.join(J, "rerun_lsb", f"results_{nm}.json")
            if os.path.exists(fb):
                cmp(f"lsb_{nm}", os.path.join(B, "design_latent-sb", f"results_{nm}.json"), fb)
    if "popot" in which:
        for nm in ("pancreas-human", "gastrulation-human"):
            fb = os.path.join(J, "rerun_popot", f"popot_{nm}.json")
            if os.path.exists(fb):
                cmp(f"popot_{nm}", os.path.join(B, "design_population-ot", f"popot_{nm}.json"), fb)
    prev = {}
    fo = os.path.join(J, "diff_results.json")
    if os.path.exists(fo):
        prev = json.load(open(fo))
    prev.update(out)
    json.dump(prev, open(fo, "w"), indent=1)
