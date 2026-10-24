"""Offline scoring of saved Task-2 predictions (theirs from dileo_corrected.py, ours from dileo_harness.py).

Targets are rebuilt from their node files with the same formulas (dileo_protocol.TARGET_NP). For every saved
(model, seed) prediction set and every transition it computes, on the saved test genes:
  their metrics (mae, pearson, p_at_200_up / down with np.argsort, as _delta_metrics) as a cross-check;
  corrections from the protocol audit: pearson_nan0 (a constant prediction scores 0 instead of being dropped),
  spearman, tie-aware expected P@200 (p_ta_up / p_ta_down: credit split uniformly among tied genes at the cut, on
  both the truth and the prediction side), its ceiling p_max and the chance level k / n_test.
Paired comparison: our (strict, mask-independent) predictions are scored on THEIR test genes of the same seed, so
every difference is paired gene by gene. Averages are over transitions i >= 1 (i = 0 has no past) and also all i.
Output: <out>/rescore_<dataset>_<target>.json and a printed table.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import dileo_protocol

SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
TRAIN_SPLIT = {"gastrulation-human": 29, "pancreas-human": 14}
K = 200


def node_means(grn, name):
    node = pd.read_csv(os.path.join(grn, f"{name}_node.csv"))
    n = len(pd.read_csv(os.path.join(grn, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    return node[[f"t{b:03d}_s00_mean" for b in SNAPS[name]]].to_numpy(np.float32)[:n]


def targets(means, name, target):
    f = dileo_protocol.TARGET_NP[target]
    return np.stack([f(means[:, i], means[:, i + 1]) for i in range(TRAIN_SPLIT[name])])


def _incl(v, k):
    thr = np.sort(v)[::-1][k - 1]
    above, eq = v > thr, v == thr
    p = above.astype(float)
    p[eq] = (k - above.sum()) / eq.sum()
    return p


def metrics(y, p):
    y = np.asarray(y, np.float32)
    p = np.asarray(p, np.float32)
    k = min(K, len(y))
    out = {"mae": float(np.mean(np.abs(y - p)))}
    const = np.all(y == y[0]) or np.all(p == p[0])
    if const:
        out["pearson"] = float("nan")
        out["pearson_nan0"] = 0.0
        out["spearman"] = 0.0
    else:
        r = float(np.corrcoef(y, p)[0, 1])
        out["pearson"] = r
        out["pearson_nan0"] = r
        out["spearman"] = float(stats.spearmanr(y, p).correlation)
    out["p_at_200_up"] = len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k
    out["p_at_200_down"] = len(set(np.argsort(y)[:k]) & set(np.argsort(p)[:k])) / k
    for side, s in (("up", 1), ("down", -1)):
        ty, tp = _incl(s * y, k), _incl(s * p, k)
        out[f"p_ta_{side}"] = float((ty * tp).sum() / k)
        out[f"p_max_{side}"] = float((ty ** 2).sum() / k)
    out["chance"] = k / len(y)
    return out


def score_run(Y, pred, test_idx):
    rows = []
    for i in range(min(len(Y), len(pred))):
        t = test_idx[i]
        rows.append(metrics(Y[i][t], pred[i][t]))
    return rows


def mean_rows(rows, start):
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows[start:]])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grn", required=True)
    ap.add_argument("--theirs", required=True, help="root of dileo_corrected.py outputs")
    ap.add_argument("--ours", nargs="*", default=[], help="roots of dileo_harness.py outputs (one per target)")
    ap.add_argument("--target", default="orig")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for name in SNAPS:
        Y = targets(node_means(args.grn, name), name, args.target)
        res = {"theirs": {}, "ours_on_their_masks": {}}
        ours = {}
        for root in args.ours:
            for f in glob.glob(os.path.join(root, name, "*", "seed1234.npz")):
                m = os.path.basename(os.path.dirname(f))
                ours[m] = np.load(f)["pred"]
        for f in sorted(glob.glob(os.path.join(args.theirs, name, f"*_{args.target}", "*", "seed*.npz"))):
            proto = os.path.basename(os.path.dirname(os.path.dirname(f))).rsplit("_", 1)[0]
            model = os.path.basename(os.path.dirname(f))
            seed = os.path.basename(f)[4:-4]
            z = np.load(f)
            for part in ("pre", "post"):
                key = f"pred_{part}"
                if key not in z:
                    continue
                rows = score_run(Y, z[key], z["test_idx"])
                res["theirs"].setdefault(f"{proto}/{part}/{model}", {})[seed] = {"i>=1": mean_rows(rows, 1), "all": mean_rows(rows, 0)}
            for om, P in ours.items():
                rows = score_run(Y, P, z["test_idx"])
                res["ours_on_their_masks"].setdefault(f"{om}@{proto}/{model}", {})[seed] = {"i>=1": mean_rows(rows, 1), "all": mean_rows(rows, 0)}
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"rescore_{name}_{args.target}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"== {name} target={args.target} (i >= 1, mean over seeds)")
        for group in ("theirs", "ours_on_their_masks"):
            for key, by_seed in sorted(res[group].items()):
                agg = {k: np.mean([v["i>=1"][k] for v in by_seed.values()]) for k in ("pearson", "pearson_nan0", "spearman", "p_at_200_up", "p_ta_up", "p_at_200_down", "p_ta_down", "mae")}
                print(f"  {group[:5]} {key:45s} n_seeds={len(by_seed)} " + " ".join(f"{k}={v:.3f}" for k, v in agg.items()))


if __name__ == "__main__":
    main()
