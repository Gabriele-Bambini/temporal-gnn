"""
Script di Visualizzazione e Diagnostica Biologica per CFM-GRN.

Genera figure per:
1. Traiettorie cellulari continue nello spazio di espressione (da controllo t=0 a perturbato t=1).
2. Heatmap della Maschera di Attivita' Regolatoria M_theta (quali circuiti biologici si attivano/spengono).
3. Confronto a barre delle metriche scientifiche tra i tre modelli (GNN con maschera, Grafo statico, MLP).
"""

import os
from typing import List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_flow_trajectories_pca(
    trajectory: List[torch.Tensor],
    x_true: torch.Tensor,
    target_gene_idx: int,
    save_path: str = "trajectory_flow_pca.png",
    num_cells_to_plot: int = 30,
):
    """
    Traccia le traiettorie continue dei singoli profili cellulari proiettati sui primi 2 componenti principali.
    
    Args:
        trajectory: Lista di tensori [B, G] per ciascun time-step dell'integratore ODE.
        x_true: Cellule reali osservate al tempo finale t=1 [N, G].
        target_gene_idx: Indice del gene perturbato.
        save_path: Percorso dove salvare l'immagine PNG.
        num_cells_to_plot: Numero di singole cellule di cui tracciare la linea di flusso.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))

    # Converte i tensori in array numpy
    x_0 = trajectory[0].cpu().numpy()
    x_final = trajectory[-1].cpu().numpy()
    x_true_np = x_true.cpu().numpy()

    # Fit PCA congiunto sullo stato iniziale, finale e ground truth
    all_points = np.vstack([x_0, x_final, x_true_np])
    pca = PCA(n_components=2)
    pca.fit(all_points)

    # Proietta tutti i passi della traiettoria: shape [steps, B, 2]
    traj_pca = np.array([pca.transform(step.cpu().numpy()) for step in trajectory])
    true_pca = pca.transform(x_true_np)

    # 1. Plot della popolazione di controllo (t=0)
    ax.scatter(
        traj_pca[0, :, 0], traj_pca[0, :, 1],
        c="royalblue", alpha=0.5, s=25, label="Controllo (t=0, iniziale)"
    )

    # 2. Plot della popolazione generata al tempo t=1
    ax.scatter(
        traj_pca[-1, :, 0], traj_pca[-1, :, 1],
        c="crimson", alpha=0.6, s=30, label="Predizione generata (t=1)"
    )

    # 3. Plot della ground truth reale osservata
    ax.scatter(
        true_pca[:, 0], true_pca[:, 1],
        c="forestgreen", alpha=0.3, s=25, marker="^", label="Reale osservato (Target ground truth)"
    )

    # 4. Traccia le linee di flusso continue per un sottoinsieme di cellule
    n_plot = min(num_cells_to_plot, traj_pca.shape[1])
    for i in range(n_plot):
        line = traj_pca[:, i, :]
        ax.plot(line[:, 0], line[:, 1], color="dimgray", alpha=0.35, linewidth=1.0)
        # Freccia direzionale a meta' percorso
        mid = len(trajectory) // 2
        ax.annotate(
            "",
            xy=(line[mid + 1, 0], line[mid + 1, 1]),
            xytext=(line[mid, 0], line[mid, 1]),
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.2, mutation_scale=10),
        )

    ax.set_title(
        f"Linee di Flusso CFM continue verso il Knockout del Gene {target_gene_idx}\n"
        f"Varianza spiegata PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [Figura salvata]: {save_path}")


def plot_regulatory_mask_heatmap(
    mask_weights: torch.Tensor,
    edge_index: torch.Tensor,
    num_genes: int,
    target_gene_idx: int,
    save_path: str = "regulatory_mask_heatmap.png",
):
    """
    Visualizza la matrice di adiacenza efficace A_eff = A_prior * M_theta come heatmap,
    evidenziando i pesi degli archi regolatori attivi e repressi.
    """
    # Media dei pesi della maschera sul batch
    avg_mask = mask_weights.mean(dim=0).cpu().numpy()  # [E]
    edges = edge_index.cpu().numpy()

    adj_matrix = np.zeros((num_genes, num_genes))
    for e in range(edges.shape[1]):
        src = edges[0, e]
        dst = edges[1, e]
        adj_matrix[src, dst] = avg_mask[e]

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        adj_matrix,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Attivita' Regolatoria Appresa M_theta in [0, 1]"},
        linewidths=0.5,
        linecolor="whitesmoke",
    )
    plt.title(
        f"Matrice di Adiacenza Efficace A_eff (Perturbazione: Knockout Gene {target_gene_idx})\n"
        f"Pesi medi degli archi regolatori (TF -> Target)",
        fontsize=11, fontweight="bold"
    )
    plt.xlabel("Gene Bersaglio (Target, dst)", fontsize=10)
    plt.ylabel("Fattore di Trascrizione Regolatore (TF, src)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  [Figura salvata]: {save_path}")
