"""Dileo & Sottoriva's own Task-2 models (imported from the untouched original repo) under corrected protocols.

--protocol orig  : their published wrapper (train_<family>_future_delta_ranking) as train_gene.py runs it: at each
                   transition fit 100 epochs on the 75 % training genes of THAT transition, then score the other
                   25 %. With --target orig it reproduces the paper's Linear row (job 713005).
--protocol preqA : evaluate-then-refine, the order the paper's text describes ("first evaluated on the next temporal
                   snapshot and subsequently refined"). At each transition: PRE = predict before any label of this
                   transition is used (the corrected score); then their training block unchanged (100 epochs, MSE,
                   75 % training genes); then POST = their evaluation block (the original-protocol score). PRE and
                   POST come from one run on identical test genes (eval forwards draw no random numbers).
--protocol preqB : as preqA but the refinement uses all genes of the transition (everything is known once t+1 is
                   observed); only PRE is reported. Equals the earlier 'strict' runner.
--target orig | l1 | linear | log1p   (dileo_protocol.patch; their masks, RNG use and metrics are unchanged)
Models: linear, mlp, gcn, gat, chebnet, evolvegcn, gcrngru (their live loop), roland (their ROLAND helpers).
A failing (model, seed) is recorded and skipped (their EvolveGCN cannot backprop twice with torch 2.5); finished
(model, seed) files are never recomputed. Runs in a work directory with GRN -> their data.
Output per (dataset, protocol, target, model, seed): seed<s>.json = their per-transition metric dicts
({"pre": [...], "post": [...]} for preq), seed<s>.npz = test indices and predictions over all genes
(for offline metric corrections; not available for ROLAND or the orig wrapper).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

KEYS = ("mae", "pearson", "p_at_200_up", "p_at_200_down")
WRAPPER = {"linear": "train_linear_future_delta_ranking", "mlp": "train_linear_future_delta_ranking",
           "gcn": "train_gnca_future_delta_ranking", "gat": "train_gnca_future_delta_ranking",
           "chebnet": "train_gnca_future_delta_ranking", "evolvegcn": "train_evolvegcn_future_delta_ranking",
           "gcrngru": "train_gcrngru_future_delta_ranking", "roland": "train_roland_future_delta_ranking",
           "tgnnmem": "train_tgnn_mem_future_delta_ranking"}
LIVE = ("linear", "mlp", "gcn", "gat", "chebnet", "evolvegcn", "gcrngru")


def seed_all(seed):
    """Exactly train_gene.main's seeding."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build(tm, snapshots, dataset, name, i):
    return tm.build_snapshot_split_future_node_delta_ranking(
        snapshots=snapshots, dataset=dataset, dataset_str="grn", dataset_name=name, i=i, mean_feat_idx=0,
        train_ratio=0.75, val_ratio=0.0, keep_target_2d=True, log_eps=1e-8, normalize_x=True, target_clip_value=5.0)


def _score(tm, model, snap):
    model.eval()
    with torch.no_grad():
        pred = model(snap.x, snap.edge_index, getattr(snap, "edge_attr", None), **tm._build_fm_forward_kwargs(snap))
        y_test = snap.x_future_delta[snap.test_mask]
        m = tm._delta_metrics(y_test, tm._reshape_prediction_like_target(pred[snap.test_mask], y_test), ks=(200,))
    return m, pred.view(-1).float().cpu().numpy()


def preq_live(tm, mg, dataset, snapshots, name, model_str, device, refine):
    """Copy of their train_live_model_future_delta_ranking (same RNG order, same training block) with a PRE score
    before the training block; refine = 'train' (their 75 % genes) or 'all'."""
    model = mg.load_model_node(model_str, "grn")
    optimizer = torch.optim.Adam(model.parameters(), lr=tm.lrs["grn"])
    model.reset_parameters()
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    pre, post, P_pre, P_post, tests = [], [], [], [], []
    for i in range(len(snapshots)):
        snap, _, _, _ = _build(tm, snapshots, dataset, name, i)
        snap = snap.to(device)
        m, p = _score(tm, model, snap)
        m["snapshot_idx"] = i
        pre.append(m)
        P_pre.append(p)
        tests.append(torch.nonzero(snap.test_mask).view(-1).cpu().numpy().astype(np.int32))
        fit = snap.train_mask if refine == "train" else torch.ones_like(snap.train_mask)
        ea = getattr(snap, "edge_attr", None)
        for _ in range(tm.epochs["grn"]):
            model.train()
            optimizer.zero_grad()
            pred = model(snap.x, snap.edge_index, ea, **tm._build_fm_forward_kwargs(snap))
            y_train = snap.x_future_delta[fit]
            loss = loss_fn(tm._reshape_prediction_like_target(pred[fit], y_train), y_train)
            loss.backward()
            optimizer.step()
        m, p = _score(tm, model, snap)
        m["snapshot_idx"] = i
        post.append(m)
        P_post.append(p)
    return {"pre": pre, "post": post}, {"test_idx": np.stack(tests), "pred_pre": np.stack(P_pre), "pred_post": np.stack(P_post)}


def preq_roland(tm, mg, dataset, snapshots, name, device, refine):
    """Their ROLAND driver in evaluate-then-refine order, over every transition (theirs stops one short)."""
    roland = mg.load_model_node("roland", "grn")
    roland = roland.to(device)
    rolopt = torch.optim.Adam(params=roland.parameters(), lr=tm.lrs["grn"])
    roland.reset_parameters()
    n, hd = snapshots[0].num_nodes, mg.dataset_dim["grn"]["hidden_dim"]
    last = [torch.zeros((n, hd)), torch.zeros((n, hd))]
    pre, post = [], []
    for i in range(len(snapshots)):
        snap, tr, va, te = _build(tm, snapshots, dataset, name, i)
        m = tm.roland_test_live_delta_ranking(roland, tr, te, snap, i, last, device=device)
        m = dict(m) if isinstance(m, dict) else {"raw": str(m)}
        m["snapshot_idx"] = i
        pre.append(m)
        if refine == "all":
            tr.train_mask[:] = True
        roland, rolopt, ev, last = tm.roland_train_single_snapshot_delta_ranking(
            model=roland, data=snap, dataset_str="grn", train_data=tr, val_data=va, test_data=te, isnap=i,
            last_embeddings=last, optimizer=rolopt, device=device, verbose=False, ranking_ks=(200,))
        ev = dict(ev)
        ev["snapshot_idx"] = i
        post.append(ev)
    return {"pre": pre, "post": post}, None


def _clean(rows):
    return [{k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else
                 (float(v) if torch.is_tensor(v) and v.numel() == 1 else v)) for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=os.path.expanduser("~/tgl-grn-orig"))
    ap.add_argument("--work", default=os.path.expanduser("~/tack/work/dileo_runs"))
    ap.add_argument("--names", default="pancreas-human,gastrulation-human")
    ap.add_argument("--models", default="linear,mlp,chebnet,gcrngru,gcn,gat,roland,evolvegcn")
    ap.add_argument("--seeds", default="1234,42,7,0,1")
    ap.add_argument("--protocol", default="preqA", choices=("orig", "preqA", "preqB", "strict"))
    ap.add_argument("--target", default="orig", choices=("orig", "l1", "linear", "log1p"))
    ap.add_argument("--out", default=os.path.expanduser("~/tack/out/dileo_corrected2"))
    args = ap.parse_args()
    protocol = "preqB" if args.protocol == "strict" else args.protocol
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, args.orig)
    os.makedirs(args.work, exist_ok=True)
    if not os.path.exists(os.path.join(args.work, "GRN")):
        os.symlink(os.path.join(args.orig, "GRN"), os.path.join(args.work, "GRN"))
    os.chdir(args.work)
    import dataset as their_dataset
    import models_gene as mg
    import train_models_gene as tm
    import dileo_protocol
    dileo_protocol.patch(tm, args.target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} protocol={protocol} target={args.target}", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in args.names.split(","):
        dataset = snapshots = None
        if protocol != "orig":
            t0 = time.time()
            dataset = their_dataset.load_dataset("grn", name, True)
            snapshots, _, _ = tm.train_val_test_split(dataset, "grn", name)
            print(f"[{name}] loaded in {time.time() - t0:.0f}s", flush=True)
        root = os.path.join(args.out, name, f"{protocol}_{args.target}")
        summ_path = os.path.join(root, "summary.json")
        summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}
        for model_str in args.models.split(","):
            if protocol != "orig" and model_str not in LIVE + ("roland",):
                print(f"skip {model_str}: no prequential variant", flush=True)
                continue
            rows, errors = {"pre": [], "post": []}, {}
            for s in seeds:
                t0 = time.time()
                d = os.path.join(root, model_str)
                path = os.path.join(d, f"seed{s}.json")
                if os.path.exists(path):  # resume: a finished (model, seed) is never recomputed
                    res = json.load(open(path))
                else:
                    seed_all(s)
                    arrays = None
                    try:
                        if protocol == "orig":
                            _, r = getattr(tm, WRAPPER[model_str])(model_str, "grn", name, True, device=device)
                            res = {"post": _clean(r)}
                        elif model_str == "roland":
                            res, arrays = preq_roland(tm, mg, dataset, snapshots, name, device,
                                                      "train" if protocol == "preqA" else "all")
                            res = {k: _clean(v) for k, v in res.items()}
                        else:
                            res, arrays = preq_live(tm, mg, dataset, snapshots, name, model_str, device,
                                                    "train" if protocol == "preqA" else "all")
                            res = {k: _clean(v) for k, v in res.items()}
                    except Exception as e:  # noqa: BLE001  (e.g. their EvolveGCN cannot backprop twice)
                        errors[str(s)] = f"{type(e).__name__}: {e}"[:500]
                        print(f"[{name}] {protocol}/{args.target} {model_str} seed {s} FAILED: {errors[str(s)]}", flush=True)
                        continue
                    if protocol == "preqB":
                        res.pop("post", None)  # refined on the test genes themselves: never reported
                    os.makedirs(d, exist_ok=True)
                    with open(path, "w") as fh:
                        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
                    if arrays is not None:
                        if protocol == "preqB":
                            arrays.pop("pred_post")
                        np.savez_compressed(os.path.join(d, f"seed{s}.npz"), **arrays)
                line = []
                for part in ("pre", "post"):
                    if part in res:
                        ps = {k: float(np.nanmean([r[k] for r in res[part]])) for k in KEYS}
                        ps1 = {k: float(np.nanmean([r[k] for r in res[part][1:]])) for k in KEYS}
                        rows[part].append({"seed": s, "all": ps, "without_i0": ps1})
                        line.append(f"{part}: " + " ".join(f"{k}={ps[k]:.4f}" for k in KEYS))
                print(f"[{name}] {protocol}/{args.target} {model_str} seed {s} ({time.time() - t0:.0f}s) " + " | ".join(line), flush=True)
            entry = {"errors": errors}
            for part, ps in rows.items():
                if ps:
                    entry[part] = {
                        "mean": {k: float(np.mean([p["all"][k] for p in ps])) for k in KEYS},
                        "sd_over_seeds": {k: float(np.std([p["all"][k] for p in ps], ddof=1)) if len(ps) > 1 else 0.0 for k in KEYS},
                        "mean_without_i0": {k: float(np.mean([p["without_i0"][k] for p in ps])) for k in KEYS},
                        "per_seed": ps}
            summ[model_str] = entry
            os.makedirs(root, exist_ok=True)
            with open(summ_path, "w") as fh:
                json.dump(summ, fh, indent=1)


if __name__ == "__main__":
    main()
