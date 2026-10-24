"""
Metriche di Valutazione per Previsioni di Perturbazione a Singola Cellula.

Metriche implementate:
1. Energy Distance (E-distance): Distanza statistica tra distribuzioni multivariate di cellule.
2. Pearson Correlation & Mean Squared Error (MSE) sul profilo medio di espressione.
3. Accuratezza specifica sul gene perturbato (target gene knockdown).
4. Marginale Wasserstein-1 media sui geni.
"""

from typing import Dict
import torch
import numpy as np
from scipy.stats import pearsonr, wasserstein_distance


def compute_energy_distance(x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    """
    Calcola l'Energy Distance (distanza di energia) tra la popolazione predetta e quella reale.
    
    Formula:
        D_E(P, Q) = 2 * E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
    
    P Proprieta':
        D_E(P, Q) >= 0, e D_E(P, Q) = 0 se e solo se le distribuzioni P e Q coincidono.
    """
    n_pred = x_pred.shape[0]
    n_true = x_true.shape[0]

    # Matrici delle distanze euclidee a coppie
    d_xy = torch.cdist(x_pred, x_true, p=2)
    d_xx = torch.cdist(x_pred, x_pred, p=2)
    d_yy = torch.cdist(x_true, x_true, p=2)

    term_xy = 2.0 * torch.mean(d_xy)
    term_xx = torch.mean(d_xx)
    term_yy = torch.mean(d_yy)

    e_dist = (term_xy - term_xx - term_yy).item()
    return max(0.0, e_dist)


def compute_evaluation_metrics(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    target_gene_idx: int,
) -> Dict[str, float]:
    """
    Calcola l'insieme completo di metriche diagnostiche per una condizione di perturbazione.
    
    Args:
        x_pred: Cellule generate [N_samples, G].
        x_true: Cellule reali osservate [N_true, G].
        target_gene_idx: Indice del gene bersaglio perturbato.
    """
    # 1. Energy distance distribuzionale
    e_dist = compute_energy_distance(x_pred, x_true)

    # 2. Medie dei profili di espressione sui geni
    mean_pred = torch.mean(x_pred, dim=0).cpu().numpy()
    mean_true = torch.mean(x_true, dim=0).cpu().numpy()

    # Correlazione di Pearson tra i profili medi di espressione
    corr, _ = pearsonr(mean_pred, mean_true)
    mse = np.mean((mean_pred - mean_true) ** 2)

    # 3. Errore specifico sul gene bersaglio
    target_pred_mean = mean_pred[target_gene_idx]
    target_true_mean = mean_true[target_gene_idx]
    target_abs_err = abs(target_pred_mean - target_true_mean)

    # 4. Media 1D Wasserstein distance per ogni gene
    w1_list = []
    x_pred_np = x_pred.cpu().numpy()
    x_true_np = x_true.cpu().numpy()
    for g in range(x_pred.shape[1]):
        w1 = wasserstein_distance(x_pred_np[:, g], x_true_np[:, g])
        w1_list.append(w1)
    mean_w1 = float(np.mean(w1_list))

    return {
        "energy_distance": float(e_dist),
        "pearson_mean_corr": float(corr),
        "mean_profile_mse": float(mse),
        "target_gene_error": float(target_abs_err),
        "mean_w1_distance": mean_w1,
    }
