"""Run our Task-2 forecasters inside Dileo & Sottoriva's ORIGINAL loop (anonymous.4open.science/r/tgl-grn-1CCD,
kept untouched in ~/tgl-grn-orig). Their code builds the dataset, the targets and the random 25 % test genes
(build_snapshot_split_future_node_delta_ranking, RandomNodeSplit) and scores the predictions (_delta_metrics);
ours only supplies the prediction vector. Seeding copies train_gene.py; every forecaster gets the same masks
for a given seed (re-seeded before each run), and forecasters never touch torch's RNG.

Output: <out>/<dataset>/<forecaster>/seed<s>.json = list of their per-snapshot metric dicts (logs-gene format)
and <out>/<dataset>/summary.json = nanmean over transitions per seed, then mean / sd over seeds.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

PAPER = {
    "gastrulation-human": {"pearson": 0.400, "p_at_200_up": 0.110, "p_at_200_down": 0.387},
    "pancreas-human": {"pearson": 0.434, "p_at_200_up": 0.178, "p_at_200_down": 0.354},
}
KEYS = ("mae", "pearson", "p_at_200_up", "p_at_200_down")


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def run(tm, dataset, snapshots, name, fc, seed):
    seed_all(seed)
    hist_all = [s.x.numpy().astype(np.float64) for s in snapshots]
    ts_all = [int(s.timestamp) for s in snapshots]
    out, preds, tests = [], [], []
    for i in range(len(snapshots)):
        snap, _, _, _ = tm.build_snapshot_split_future_node_delta_ranking(
            snapshots=snapshots, dataset=dataset, dataset_str="grn", dataset_name=name, i=i, mean_feat_idx=0,
            train_ratio=0.75, val_ratio=0.0, keep_target_2d=True, log_eps=1e-8, normalize_x=True,
            target_clip_value=5.0)
        y = snap.x_future_delta
        train_idx = torch.nonzero(snap.train_mask).view(-1).numpy()
        ctx = {} if getattr(fc, "strict", False) else {
            "train_idx": train_idx, "y_train": y[snap.train_mask].view(-1).numpy(),
            "edge_index": snap.edge_index, "edge_attr": snap.edge_attr}
        pred = np.asarray(fc.predict(hist_all[:i + 1], ts_all[:i + 1], ctx), np.float32)
        assert pred.shape == (y.shape[0],), (pred.shape, y.shape)
        pred_t = torch.from_numpy(pred).view(-1, 1)
        m = tm._delta_metrics(y[snap.test_mask], pred_t[snap.test_mask], ks=(200,))
        m["snapshot_idx"] = i
        yt = y[snap.test_mask].view(-1)
        m["n_test_at_plus5"] = int((yt >= 5 - 1e-6).sum())
        m["n_test_at_minus5"] = int((yt <= -5 + 1e-6).sum())
        out.append(m)
        preds.append(pred)
        tests.append(torch.nonzero(snap.test_mask).view(-1).numpy().astype(np.int32))
    run.last_arrays = {"pred": np.stack(preds), "test_idx": np.stack(tests)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=os.path.expanduser("~/tgl-grn-orig"))
    ap.add_argument("--names", default="pancreas-human,gastrulation-human")
    ap.add_argument("--models", default="level4,level6,obslevel4,obslevel6")
    ap.add_argument("--seeds", default="1234,42,7,0,1")
    ap.add_argument("--out", default=os.path.expanduser("~/tack/out/dileo_task2_harness"))
    ap.add_argument("--target", default="orig", help="orig | log1p | linear (see dileo_protocol.py)")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import task2_forecasters as tf
    sys.path.insert(0, args.orig)
    os.chdir(args.orig)  # their loader reads GRN/<name>.csv relative to the repo
    import dataset as their_dataset
    import train_models_gene as tm
    import dileo_protocol
    dileo_protocol.patch(tm, args.target)
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in args.names.split(","):
        t0 = time.time()
        dataset = their_dataset.load_dataset("grn", name, True)
        snapshots, _, _ = tm.train_val_test_split(dataset, "grn", name)
        print(f"[{name}] loaded {len(dataset)} snapshots, {len(snapshots)} transitions in {time.time() - t0:.0f}s", flush=True)
        summ = {"paper_best": PAPER.get(name, {}), "models": {}}
        for mname in args.models.split(","):
            fc = tf.REGISTRY[mname]()
            assert getattr(fc, "target", "orig") == args.target, f"{mname} predicts {getattr(fc, 'target', 'orig')}, run is {args.target}"
            per_seed = []
            for s in seeds:
                res = run(tm, dataset, snapshots, name, fc, s)
                d = os.path.join(args.out, name, mname)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"seed{s}.json"), "w") as fh:
                    json.dump(res, fh, indent=1)
                np.savez_compressed(os.path.join(d, f"seed{s}.npz"), **run.last_arrays)
                per_seed.append({k: float(np.nanmean([r[k] for r in res])) for k in KEYS})
                print(f"[{name}] {mname} seed {s}: " + " ".join(f"{k}={per_seed[-1][k]:.4f}" for k in KEYS), flush=True)
            summ["models"][mname] = {
                "strict": bool(getattr(fc, "strict", False)),
                "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in KEYS},
                "sd": {k: float(np.std([p[k] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0 for k in KEYS},
                "per_seed": per_seed}
            if getattr(fc, "selected", None):
                summ["models"][mname]["selected"] = {str(i): list(c) for (_, i), c in sorted(fc.selected.items(), key=lambda kv: kv[0][1])}
            with open(os.path.join(args.out, name, "summary.json"), "w") as fh:
                json.dump(summ, fh, indent=1)
        del dataset, snapshots


if __name__ == "__main__":
    main()
