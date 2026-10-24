"""
Benchmark v2: one (scenario, model, OT mode, fold, seed) run -> one JSON file.

Designed to test the project's hypotheses with ablations that can actually separate them:

  scenarios
    oracle         prior = exact true GRN, activation-only, no gating
    noisy          30% repressive edges; prior drops 20% true edges, adds 50% spurious
    noisy_context  as noisy + 30% of edges gated by a third gene (state-dependent activity)

  models
    persistence      predict no change (null model)
    mlp              no graph
    static_graph     prior graph, all edge weights = 1
    static_learned   prior graph, one learned weight per edge (not state-dependent)
    graph_cfm        prior graph, contextual mask, perturbation fed per edge (fixed)
    graph_cfm_dense  prior graph, contextual mask, perturbation fed densely (original)

  folds: the top-15 out-degree regulators are knocked out; 5 folds hold out 3 each.
  ot:    per_condition (fixed) or global (original, mixes labels).

Reported metrics are effect-based (see evaluate.compute_delta_metrics); the original
absolute-level metrics are kept alongside for comparison.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.optim import AdamW

from cfm.flow_matcher import OptimalTransportCFM
from cfm.solver import ODESolver
from data.dataset import create_dataloaders
from data.synthetic_grn import SyntheticGRNSimulator
from evaluate import compute_evaluation_metrics
from models.baselines import LearnedStaticEdgeVectorField, MLPVectorField, StaticGraphVectorField
from models.graph_vector_field import GraphVectorField
from train import train_epoch

SCENARIOS = {
    "oracle": dict(repression_frac=0.0, context_frac=0.0, noisy_prior=False),
    "noisy": dict(repression_frac=0.3, context_frac=0.0, noisy_prior=True),
    "noisy_context": dict(repression_frac=0.3, context_frac=0.3, noisy_prior=True),
}
MODELS = ["persistence", "mlp", "static_graph", "static_learned", "graph_cfm", "graph_cfm_dense"]

NUM_GENES = 30
DENSITY = 0.15
NUM_PERTURBATIONS = 15
NUM_FOLDS = 5
CELLS = 300


def build_model(name, num_genes, prior_edges, hidden=48):
    if name == "mlp":
        return MLPVectorField(num_genes=num_genes, perturbation_dim=num_genes, hidden_dim=128, num_layers=3)
    if name == "static_graph":
        return StaticGraphVectorField(num_genes, prior_edges, num_genes, hidden, 2)
    if name == "static_learned":
        return LearnedStaticEdgeVectorField(num_genes, prior_edges, num_genes, hidden, 2)
    if name == "graph_cfm":
        return GraphVectorField(num_genes, prior_edges, num_genes, hidden, 2, True, "per_edge")
    if name == "graph_cfm_dense":
        return GraphVectorField(num_genes, prior_edges, num_genes, hidden, 2, True, "dense")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--ot", choices=["per_condition", "global"], default="per_condition")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results_v2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.scenario}__{args.model}__{args.ot}__f{args.fold}__s{args.seed}"
    out_path = os.path.join(args.out, tag + ".json")
    if os.path.exists(out_path):
        print(f"skip (exists): {out_path}")
        return

    torch.set_num_threads(1)
    device = torch.device(args.device)
    sc = SCENARIOS[args.scenario]

    # Data depends on the seed (not the fold), so folds of one seed share one dataset
    # and CIs over seeds include data variability, not only initialisation.
    sim = SyntheticGRNSimulator(
        num_genes=NUM_GENES, density=DENSITY, seed=1000 + args.seed,
        repression_frac=sc["repression_frac"], context_frac=sc["context_frac"],
    )
    data = sim.generate_benchmark_dataset(cells_per_condition=CELLS, num_perturbations=NUM_PERTURBATIONS)
    genes = list(data["perturbed_genes"])
    order = np.random.RandomState(args.seed).permutation(len(genes))
    folds = np.array_split(np.array(genes)[order], NUM_FOLDS)
    held_out = [int(g) for g in folds[args.fold]]

    prior = sim.noisy_prior(fp_rate=0.5, fn_rate=0.2, seed=args.seed) if sc["noisy_prior"] else sim.edge_index
    true_set = set(map(tuple, sim.edge_index.t().tolist()))
    prior_set = set(map(tuple, prior.t().tolist()))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ctrl = data["control"].to(device)
    t0 = time.time()
    final_loss = None

    if args.model == "persistence":
        model = None
    else:
        model = build_model(args.model, NUM_GENES, prior.to(device)).to(device)
        train_loader, _ = create_dataloaders(data, batch_size=args.batch, held_out_genes=held_out)
        cfm = OptimalTransportCFM(sigma_min=1e-3, per_condition=(args.ot == "per_condition"))
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for _ in range(args.epochs):
            final_loss = train_epoch(model, train_loader, cfm, opt, device)

    solver = ODESolver(method="rk4", num_steps=20)
    per_gene = {}
    with torch.no_grad():
        for g in held_out:
            x_true = data["perturbed"][g].to(device)
            if model is None:
                x_pred = ctrl.clone()
            else:
                cond = torch.zeros(ctrl.shape[0], NUM_GENES, device=device)
                cond[:, g] = 1.0
                x_pred, _ = solver.sample(model, ctrl, condition=cond)
            m = compute_evaluation_metrics(x_pred, x_true, target_gene_idx=g, x_ctrl=ctrl)
            per_gene[str(g)] = {k: float(v) for k, v in m.items()}

    keys = next(iter(per_gene.values())).keys()
    mean = {k: float(np.nanmean([per_gene[g][k] for g in per_gene])) for k in keys}

    record = {
        "scenario": args.scenario, "model": args.model, "ot": args.ot,
        "fold": args.fold, "seed": args.seed, "held_out": held_out,
        "epochs": args.epochs, "final_train_loss": final_loss,
        "train_seconds": time.time() - t0,
        "n_true_edges": len(true_set), "n_prior_edges": len(prior_set),
        "n_prior_true": len(prior_set & true_set),
        "mean": mean, "per_gene": per_gene,
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=1)
    print(f"done {tag} in {record['train_seconds']:.0f}s | pearson_delta_downstream "
          f"{mean.get('pearson_delta_downstream', float('nan')):.3f} | knockdown {mean['knockdown_fraction']:.2f}")


if __name__ == "__main__":
    main()
