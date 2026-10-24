"""
Script di Addestramento e Confronto Scientifico per CFM + GRN.

Esegue:
1. Generazione di dati benchmark sintetici con ground truth biologica nota.
2. Split tra perturbazioni di addestramento e condizione HELD-OUT (gene mai visto).
3. Addestramento con Optimal Transport Conditional Flow Matching (OT-CFM).
4. Valutazione rigorosa al test-time tramite risolutore ODE (RK4) confrontando:
   - Modello proposto (GraphVectorField con maschera appresa M_theta)
   - Baseline Grafo Fisso (StaticGraphVectorField)
   - Baseline Non Strutturata (MLPVectorField)
   - Baseline di Persistenza (controllo nullo)
"""

import argparse
import time
import torch
from torch.optim import AdamW

from cfm.flow_matcher import OptimalTransportCFM
from cfm.solver import ODESolver
from models.graph_vector_field import GraphVectorField
from models.baselines import MLPVectorField, StaticGraphVectorField, PersistenceBaseline
from data.synthetic_grn import SyntheticGRNSimulator
from data.dataset import create_dataloaders
from evaluate import compute_evaluation_metrics


def train_epoch(model, dataloader, cfm, optimizer, device, l1_reg: float = 1e-3):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_0, x_1, pert_vec, _ in dataloader:
        x_0 = x_0.to(device)
        x_1 = x_1.to(device)
        pert_vec = pert_vec.to(device)
        B = x_0.shape[0]

        optimizer.zero_grad()

        # Loss di regressione CFM su target analitico u_t
        loss_cfm, _ = cfm.compute_loss(model, x_0, x_1, condition=pert_vec)

        # Regolarizzazione L1 sulla maschera se presente nel modello
        reg_loss = 0.0
        if hasattr(model, "mask_module") and model.mask_module is not None:
            # Calcola maschera su campione a t=0.5
            t_mid = torch.full((B, 1), 0.5, device=device)
            mask = model.mask_module(x_0, t_mid, pert_vec)
            reg_loss = model.mask_module.compute_regularization_loss(mask, l1_weight=l1_reg)

        loss = loss_cfm + reg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

    return total_loss / total_samples


@torch.no_grad()
def evaluate_model(model, ctrl_cells, test_pert_cells, held_out_gene: int, num_genes: int, device):
    """Genera cellule perturbate integrando l'ODE a partire dai controlli e calcola le metriche."""
    model.eval()
    solver = ODESolver(method="rk4", num_steps=20)

    # Campiona un sottoinsieme di controlli per il test
    N_test = min(len(ctrl_cells), len(test_pert_cells))
    x_0 = ctrl_cells[:N_test].to(device)
    x_true = test_pert_cells[:N_test].to(device)

    # Vettore di perturbazione per il gene held-out
    pert_vec = torch.zeros(N_test, num_genes, device=device)
    pert_vec[:, held_out_gene] = 1.0

    # Integrazione numerica RK4 da t=0 a t=1
    x_pred, _ = solver.sample(model, x_0, condition=pert_vec)

    metrics = compute_evaluation_metrics(x_pred, x_true, target_gene_idx=held_out_gene)
    return metrics


def run_experiment(
    model_type: str = "graph_cfm",
    epochs: int = 50,
    lr: float = 2e-3,
    num_genes: int = 20,
    batch_size: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Avvio Esperimento: Modello = {model_type.upper()} | Device = {device}")
    print(f"{'='*60}")

    # 1. Creazione Simulatore e Dataset Sintetico
    print("[1/4] Simulazione dinamica cellulare biochimica (Hill Kinetics)...")
    sim = SyntheticGRNSimulator(num_genes=num_genes, density=0.25, seed=42)
    benchmark_data = sim.generate_benchmark_dataset(cells_per_condition=400)

    perturbed_genes = benchmark_data["perturbed_genes"]
    # Il primo gene perturbato e' tenuto fuori (HELD-OUT) per testare la generalizzazione
    held_out_gene = perturbed_genes[0]
    train_genes = perturbed_genes[1:]
    print(f"  Geni totali: {num_genes} | Archi biologici: {sim.num_edges}")
    print(f"  Geni di Training: {train_genes} | Gene HELD-OUT (Test): {held_out_gene}")

    train_loader, _ = create_dataloaders(
        benchmark_data, batch_size=batch_size, held_out_genes=[held_out_gene]
    )

    # 2. Inizializzazione Modello
    print(f"[2/4] Inizializzazione architettura: {model_type}...")
    edge_index = sim.edge_index.to(device)

    if model_type == "graph_cfm":
        model = GraphVectorField(
            num_genes=num_genes,
            edge_index=edge_index,
            perturbation_dim=num_genes,
            hidden_dim=48,
            num_layers=2,
            use_mask=True,
        ).to(device)
    elif model_type == "static_graph":
        model = StaticGraphVectorField(
            num_genes=num_genes,
            edge_index=edge_index,
            perturbation_dim=num_genes,
            hidden_dim=48,
            num_layers=2,
        ).to(device)
    elif model_type == "mlp":
        model = MLPVectorField(
            num_genes=num_genes,
            perturbation_dim=num_genes,
            hidden_dim=128,
            num_layers=3,
        ).to(device)
    else:
        raise ValueError(f"Modello sconosciuto: {model_type}")

    # 3. Addestramento con OT-CFM
    print(f"[3/4] Addestramento con OT-CFM per {epochs} epoche...")
    cfm = OptimalTransportCFM(sigma_min=1e-3)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, cfm, optimizer, device)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoca {epoch:02d}/{epochs:02d} | CFM Loss: {loss:.4f}")

    train_time = time.time() - start_time
    print(f"  Addestramento completato in {train_time:.1f} secondi.")

    # 4. Valutazione sul gene HELD-OUT mai visto
    print(f"[4/4] Valutazione sul gene HELD-OUT ({held_out_gene}) tramite integrazione RK4...")
    ctrl_cells = benchmark_data["control"]
    test_pert_cells = benchmark_data["perturbed"][held_out_gene]

    metrics = evaluate_model(model, ctrl_cells, test_pert_cells, held_out_gene, num_genes, device)

    # Calcolo baseline di persistenza per confronto
    pers = PersistenceBaseline()
    x_pers = pers.predict(ctrl_cells[:len(test_pert_cells)])
    pers_metrics = compute_evaluation_metrics(x_pers, test_pert_cells, target_gene_idx=held_out_gene)

    print("\n--- RISULTATI SCIENTIFICI SUL GENE HELD-OUT ---")
    print(f"  Energy Distance (piu' basso e' meglio): {metrics['energy_distance']:.4f} (Persistenza: {pers_metrics['energy_distance']:.4f})")
    print(f"  Pearson Correlation sul profilo medio: {metrics['pearson_mean_corr']:.4f} (Persistenza: {pers_metrics['pearson_mean_corr']:.4f})")
    print(f"  Errore knockdown gene target:          {metrics['target_gene_error']:.4f} (Persistenza: {pers_metrics['target_gene_error']:.4f})")
    print(f"  Wasserstein-1 medio sui geni:         {metrics['mean_w1_distance']:.4f} (Persistenza: {pers_metrics['mean_w1_distance']:.4f})")
    print(f"{'='*60}\n")

    return metrics, pers_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="graph_cfm", choices=["graph_cfm", "static_graph", "mlp"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--genes", type=int, default=20)
    args = parser.parse_args()

    run_experiment(model_type=args.model, epochs=args.epochs, lr=args.lr, num_genes=args.genes)
