"""
Benchmark v3: one run -> one JSON. Same data, folds and seeds as v2 (constants and
scenarios are imported from run_benchmark_v2), so every v3 run pairs with v2 runs.

family "cfm"   snapshot-only perturbation models (as v2), now optionally with a SIGNED
               prior, and with edge diagnostics: do the learned edge weights tell true
               prior edges from spurious ones?
    models: static_graph, static_learned, graph_cfm

family "mech"  mechanistic route: learn the dynamics dx/dt = f(x) from velocity-like
               observations of cells relaxing out of equilibrium (control + training
               knockouts only), then predict an unseen knockout by clamping the gene
               to zero and integrating f. Uses MORE data than "cfm" (velocities):
               this compares data regimes as much as models.
    models: mech_true (integrates the true simulator drift: ceiling),
            mech_mlp (no graph), mech_graph (prior graph),
            mech_graph_learned (prior graph + learned static edge weights)
    --vnoise: velocity noise as a multiple of each gene's velocity std (0 = clean).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from torch.optim import AdamW

from cfm.flow_matcher import OptimalTransportCFM
from cfm.solver import ODESolver
from data.dataset import create_dataloaders
from data.synthetic_grn import SyntheticGRNSimulator
from evaluate import compute_evaluation_metrics
from models.baselines import LearnedStaticEdgeVectorField, MLPVectorField, StaticGraphVectorField
from models.graph_vector_field import GraphVectorField
from run_benchmark_v2 import CELLS, DENSITY, NUM_FOLDS, NUM_GENES, NUM_PERTURBATIONS, SCENARIOS
from train import train_epoch

CFM_MODELS = ["static_graph", "static_learned", "graph_cfm"]
MECH_MODELS = ["mech_true", "mech_mlp", "mech_graph", "mech_graph_learned"]
HIDDEN = 48


def setup(scenario, seed, fold, sign_error=0.0):
    """Identical to run_benchmark_v2: same simulator, dataset, fold split, prior."""
    sc = SCENARIOS[scenario]
    sim = SyntheticGRNSimulator(
        num_genes=NUM_GENES, density=DENSITY, seed=1000 + seed,
        repression_frac=sc["repression_frac"], context_frac=sc["context_frac"],
    )
    data = sim.generate_benchmark_dataset(cells_per_condition=CELLS, num_perturbations=NUM_PERTURBATIONS)
    genes = list(data["perturbed_genes"])
    order = np.random.RandomState(seed).permutation(len(genes))
    folds = np.array_split(np.array(genes)[order], NUM_FOLDS)
    held_out = [int(g) for g in folds[fold]]
    train_genes = [int(g) for g in genes if int(g) not in held_out]
    prior, sign = sim.signed_prior(0.5, 0.2, seed, noisy=sc["noisy_prior"], sign_error=sign_error)
    return sim, data, held_out, train_genes, prior, sign


def edge_diagnostics(model, prior, true_set, ctrl):
    """AUROC of learned edge weights for true vs spurious prior edges."""
    is_true = np.array([(int(s), int(d)) in true_set for s, d in zip(prior[0], prior[1])])
    if is_true.all() or not is_true.any():
        return {}
    with torch.no_grad():
        if hasattr(model, "edge_logits"):
            w = torch.sigmoid(model.edge_logits)
        elif getattr(model, "use_mask", False):
            zeros = torch.zeros(ctrl.shape[0], ctrl.shape[1], device=ctrl.device)
            t = torch.full((ctrl.shape[0], 1), 0.5, device=ctrl.device)
            w = model.mask_module(ctrl, t, zeros).mean(0)
        else:
            return {}
    w = w.cpu().numpy()
    wt, ws = w[is_true], w[~is_true]
    u = mannwhitneyu(wt, ws, alternative="two-sided").statistic
    return {
        "edge_auc_true_vs_spurious": float(u / (len(wt) * len(ws))),
        "edge_w_true_mean": float(wt.mean()),
        "edge_w_spurious_mean": float(ws.mean()),
        "n_prior_true": int(is_true.sum()),
        "n_prior_spurious": int((~is_true).sum()),
    }


def build_cfm(name, prior, sign):
    if name == "static_graph":
        return StaticGraphVectorField(NUM_GENES, prior, NUM_GENES, HIDDEN, 2, edge_sign=sign)
    if name == "static_learned":
        return LearnedStaticEdgeVectorField(NUM_GENES, prior, NUM_GENES, HIDDEN, 2, edge_sign=sign)
    if name == "graph_cfm":
        return GraphVectorField(NUM_GENES, prior, NUM_GENES, HIDDEN, 2, True, "per_edge", edge_sign=sign)
    raise ValueError(name)


def build_mech(name, prior, sign):
    if name == "mech_mlp":
        return MLPVectorField(num_genes=NUM_GENES, perturbation_dim=0, hidden_dim=128, num_layers=3)
    if name == "mech_graph":
        return StaticGraphVectorField(NUM_GENES, prior, 0, HIDDEN, 2, edge_sign=sign)
    if name == "mech_graph_learned":
        return LearnedStaticEdgeVectorField(NUM_GENES, prior, 0, HIDDEN, 2, edge_sign=sign)
    raise ValueError(name)


def run_cfm(args, sim, data, held_out, prior, sign, device):
    model = build_cfm(args.model, prior.to(device), sign.to(device) if args.signed else None).to(device)
    loader, _ = create_dataloaders(data, batch_size=128, held_out_genes=held_out)
    cfm = OptimalTransportCFM(sigma_min=1e-3, per_condition=True)
    opt = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss = None
    for _ in range(args.epochs):
        loss = train_epoch(model, loader, cfm, opt, device)
    ctrl = data["control"].to(device)
    solver = ODESolver(method="rk4", num_steps=20)
    preds = {}
    with torch.no_grad():
        for g in held_out:
            cond = torch.zeros(ctrl.shape[0], NUM_GENES, device=device)
            cond[:, g] = 1.0
            preds[g], _ = solver.sample(model, ctrl, condition=cond)
    return model, preds, loss


def run_mech(args, sim, data, held_out, train_genes, prior, sign, device):
    ctrl = data["control"].to(device)
    dt, n_steps = 0.05, 100

    model, loss = None, None
    if args.model != "mech_true":
        X, V, M = [], [], []
        for k in [None] + train_genes:
            s, d = sim.simulate_transients(CELLS, knockout_gene=k)
            m = torch.ones_like(d)
            if k is not None:
                m[:, k] = 0.0  # the clamped gene follows the intervention, not the dynamics
            X.append(s); V.append(d); M.append(m)
        X, V, M = torch.cat(X), torch.cat(V), torch.cat(M)
        if args.vnoise > 0:
            v_std = V[: CELLS * 5].std(0, keepdim=True)  # control block: natural dynamics only
            V = V + args.vnoise * v_std * torch.randn_like(V)
        X, V, M = X.to(device), V.to(device), M.to(device)

        model = build_mech(args.model, prior.to(device), sign.to(device) if args.signed else None).to(device)
        opt = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        n = X.shape[0]
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            tot = 0.0
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                t0 = torch.zeros(len(idx), 1, device=device)
                pred = model(X[idx], t0, None)
                l = (M[idx] * (pred - V[idx]) ** 2).sum() / M[idx].sum()
                opt.zero_grad(); l.backward(); opt.step()
                tot += l.item() * len(idx)
            loss = tot / n
        model.eval()

    preds = {}
    with torch.no_grad():
        for g in held_out:
            x = ctrl.clone()
            x[:, g] = 0.0
            for _ in range(n_steps):
                if model is None:
                    v = sim.drift(x.cpu(), knockout_gene=g).to(device)
                else:
                    v = model(x, torch.zeros(x.shape[0], 1, device=device), None)
                v[:, g] = 0.0
                x = x + dt * v + sim.noise_level * np.sqrt(dt) * torch.randn_like(x)
                x = torch.clamp(x, min=0.0)
                x[:, g] = 0.0
            preds[g] = x
    return model, preds, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["cfm", "mech"], required=True)
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--model", choices=CFM_MODELS + MECH_MODELS, required=True)
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--vnoise", type=float, default=0.0)
    ap.add_argument("--sign_error", type=float, default=0.0,
                    help="fraction of true prior edges with flipped sign (only with --signed)")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results_v3")
    args = ap.parse_args()
    if args.epochs is None:
        args.epochs = 80 if args.family == "cfm" else 40

    os.makedirs(args.out, exist_ok=True)
    tag = (f"{args.family}__{args.scenario}__{args.model}__{'signed' if args.signed else 'unsigned'}"
           f"__v{args.vnoise:g}" + (f"__se{args.sign_error:g}" if args.sign_error > 0 else "")
           + f"__f{args.fold}__s{args.seed}")
    out_path = os.path.join(args.out, tag + ".json")
    if os.path.exists(out_path):
        print(f"skip (exists): {out_path}")
        return

    torch.set_num_threads(1)
    device = torch.device(args.device)
    sim, data, held_out, train_genes, prior, sign = setup(args.scenario, args.seed, args.fold,
                                                          sign_error=args.sign_error)
    true_set = set(map(tuple, sim.edge_index.t().tolist()))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    if args.family == "cfm":
        model, preds, loss = run_cfm(args, sim, data, held_out, prior, sign, device)
    else:
        model, preds, loss = run_mech(args, sim, data, held_out, train_genes, prior, sign, device)

    ctrl = data["control"].to(device)
    per_gene = {}
    for g in held_out:
        m = compute_evaluation_metrics(preds[g], data["perturbed"][g].to(device), target_gene_idx=g, x_ctrl=ctrl)
        per_gene[str(g)] = {k: float(v) for k, v in m.items()}
    keys = next(iter(per_gene.values())).keys()
    mean = {k: float(np.nanmean([per_gene[g][k] for g in per_gene])) for k in keys}
    diag = edge_diagnostics(model, prior, true_set, ctrl) if model is not None else {}

    record = {
        "family": args.family, "scenario": args.scenario, "model": args.model,
        "signed": args.signed, "vnoise": args.vnoise, "sign_error": args.sign_error,
        "fold": args.fold, "seed": args.seed,
        "held_out": held_out, "epochs": args.epochs, "final_train_loss": loss,
        "train_seconds": time.time() - t0, "mean": mean, "per_gene": per_gene, "edge_diag": diag,
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1)
    print(f"done {tag} in {record['train_seconds']:.0f}s | downstream r "
          f"{mean.get('pearson_delta_downstream', float('nan')):.3f} | mse_dd "
          f"{mean.get('mse_delta_downstream', float('nan')):.3f}"
          + (f" | edge AUC {diag['edge_auc_true_vs_spurious']:.2f}" if diag else ""))


if __name__ == "__main__":
    main()
