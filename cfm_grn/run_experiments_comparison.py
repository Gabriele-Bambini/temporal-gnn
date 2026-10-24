"""
Script comparativo scientifico: esegue l'addestramento e il benchmark
tra il modello proposto (CFM con maschera GRN), la baseline a Grafo Statico
e la baseline MLP non strutturata.

Genera:
1. Tabella comparativa con Energy Distance, correlazione di Pearson e accuratezza di knockdown.
2. Figure diagnostiche salvate in output/:
   - Traiettorie continue PCA (trajectory_flow_pca.png)
   - Heatmap della maschera di attivita' regolatoria (regulatory_mask_heatmap.png)
"""

import os
import torch
import pandas as pd
from torch.optim import AdamW

from cfm.flow_matcher import OptimalTransportCFM
from cfm.solver import ODESolver
from models.graph_vector_field import GraphVectorField
from models.baselines import MLPVectorField, StaticGraphVectorField, PersistenceBaseline
from data.synthetic_grn import SyntheticGRNSimulator
from data.dataset import create_dataloaders
from evaluate import compute_evaluation_metrics
from visualize import plot_flow_trajectories_pca, plot_regulatory_mask_heatmap
from train import train_epoch


def run_full_comparison(epochs: int = 30, num_genes: int = 20):
    os.makedirs("output", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f" BENCHMARK SCIENTIFICO: Confronto Modelli Generativi CFM su Gene Held-Out")
    print(f"{'='*70}")

    # 1. Dati
    print("\n[Fase 1] Generazione del benchmark biologico controllato...")
    sim = SyntheticGRNSimulator(num_genes=num_genes, density=0.25, seed=42)
    benchmark_data = sim.generate_benchmark_dataset(cells_per_condition=350)

    perturbed_genes = benchmark_data["perturbed_genes"]
    held_out_gene = perturbed_genes[0]
    print(f"  Geni: {num_genes} | Archi: {sim.num_edges}")
    print(f"  Geni di Training: {perturbed_genes[1:]} | Gene di Test HELD-OUT: {held_out_gene}")

    train_loader, _ = create_dataloaders(
        benchmark_data, batch_size=64, held_out_genes=[held_out_gene]
    )

    ctrl_cells = benchmark_data["control"]
    test_pert_cells = benchmark_data["perturbed"][held_out_gene]
    N_eval = min(len(ctrl_cells), len(test_pert_cells))
    x_0_eval = ctrl_cells[:N_eval].to(device)
    x_true_eval = test_pert_cells[:N_eval].to(device)

    pert_vec_eval = torch.zeros(N_eval, num_genes, device=device)
    pert_vec_eval[:, held_out_gene] = 1.0

    edge_index = sim.edge_index.to(device)

    # 2. Definizione dei tre modelli
    models = {
        "Graph_CFM (Proposto)": GraphVectorField(
            num_genes=num_genes,
            edge_index=edge_index,
            perturbation_dim=num_genes,
            hidden_dim=48,
            num_layers=2,
            use_mask=True,
        ).to(device),
        "Static_Graph_CFM": StaticGraphVectorField(
            num_genes=num_genes,
            edge_index=edge_index,
            perturbation_dim=num_genes,
            hidden_dim=48,
            num_layers=2,
        ).to(device),
        "MLP_CFM (Senza Grafo)": MLPVectorField(
            num_genes=num_genes,
            perturbation_dim=num_genes,
            hidden_dim=128,
            num_layers=3,
        ).to(device),
    }

    cfm = OptimalTransportCFM(sigma_min=1e-3)
    solver = ODESolver(method="rk4", num_steps=20)
    results = []

    # 3. Addestramento e Valutazione di ciascun modello
    saved_trajectory = None
    saved_mask = None

    for model_name, model in models.items():
        print(f"\n[Fase 2] Addestramento: {model_name}...")
        optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, cfm, optimizer, device)
            if epoch % 10 == 0 or epoch == epochs:
                print(f"  Epoca {epoch:02d}/{epochs:02d} | CFM Loss: {loss:.4f}")

        # Valutazione sul gene HELD-OUT
        model.eval()
        with torch.no_grad():
            if model_name == "Graph_CFM (Proposto)":
                x_pred, trajectory = solver.sample(
                    model, x_0_eval, condition=pert_vec_eval, return_trajectory=True
                )
                saved_trajectory = trajectory
                # Calcola la maschera su uno stato medio
                t_mid = torch.full((N_eval, 1), 0.5, device=device)
                saved_mask = model.get_latest_mask(x_0_eval, t_mid, pert_vec_eval)
            else:
                x_pred, _ = solver.sample(model, x_0_eval, condition=pert_vec_eval)

            metrics = compute_evaluation_metrics(x_pred, x_true_eval, target_gene_idx=held_out_gene)
            metrics["Modello"] = model_name
            results.append(metrics)

    # 4. Controllo di Persistenza
    pers = PersistenceBaseline()
    x_pers = pers.predict(x_0_eval)
    pers_metrics = compute_evaluation_metrics(x_pers, x_true_eval, target_gene_idx=held_out_gene)
    pers_metrics["Modello"] = "Controllo di Persistenza"
    results.append(pers_metrics)

    # 5. Generazione Figure
    print("\n[Fase 3] Generazione figure e visualizzazioni...")
    if saved_trajectory is not None:
        plot_flow_trajectories_pca(
            trajectory=saved_trajectory,
            x_true=x_true_eval,
            target_gene_idx=held_out_gene,
            save_path=os.path.join("output", "trajectory_flow_pca.png"),
        )

    if saved_mask is not None:
        plot_regulatory_mask_heatmap(
            mask_weights=saved_mask,
            edge_index=edge_index,
            num_genes=num_genes,
            target_gene_idx=held_out_gene,
            save_path=os.path.join("output", "regulatory_mask_heatmap.png"),
        )

    # 6. Tabella finale dei risultati
    df_results = pd.DataFrame(results)[[
        "Modello", "energy_distance", "pearson_mean_corr", "target_gene_error", "mean_w1_distance"
    ]]
    df_results.columns = [
        "Modello", "Energy Distance", "Corr. Pearson (Mean)", "Errore Target Gene", "Wasserstein-1"
    ]

    print(f"\n{'='*70}")
    print(" TABELLA COMPARATIVA DELLE METRICHE SUL GENE HELD-OUT (Non Visto nel Training)")
    print(f"{'='*70}")
    print(df_results.to_string(index=False))
    print(f"{'='*70}")

    csv_path = os.path.join("output", "benchmark_held_out_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Risultati salvati in: {csv_path}\n")


if __name__ == "__main__":
    run_full_comparison(epochs=25, num_genes=20)
