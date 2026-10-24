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


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson that returns 0.0 when either side is constant (e.g. persistence predicts
    delta = 0 for every gene): a predictor that moves nothing carries no signal."""
    if a.size < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(pearsonr(a, b)[0])


def compute_delta_metrics(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    x_ctrl: torch.Tensor,
    target_gene_idx: int,
    de_threshold: float = 0.25,
) -> Dict[str, float]:
    """
    Metrics on the perturbation EFFECT (change vs control), not on absolute levels.

    Absolute-level Pearson is dominated by between-gene baseline differences: the
    'do nothing' predictor scores ~0.94 on it for free. Everything here is measured on
    deltas, and distributional distances are restricted to genes that actually move
    (|true delta| > de_threshold), so unchanged genes cannot hand a free win to
    persistence.

    'downstream' metrics exclude the knocked-out gene itself: hitting the target is
    easy when the perturbation flag is injected on that node; predicting the cascade
    through the regulatory network is the claim a GRN-informed model has to earn.
    """
    mean_pred = x_pred.mean(0).cpu().numpy()
    mean_true = x_true.mean(0).cpu().numpy()
    mean_ctrl = x_ctrl.mean(0).cpu().numpy()
    d_pred = mean_pred - mean_ctrl
    d_true = mean_true - mean_ctrl

    de = np.abs(d_true) > de_threshold
    de_idx = np.where(de)[0]
    down_idx = de_idx[de_idx != target_gene_idx]

    out = {
        "pearson_delta": _safe_pearson(d_pred, d_true),
        "mse_delta": float(np.mean((d_pred - d_true) ** 2)),
        "n_de_genes": int(de.sum()),
    }

    if down_idx.size >= 3:
        out["pearson_delta_downstream"] = _safe_pearson(d_pred[down_idx], d_true[down_idx])
        out["direction_acc_downstream"] = float(
            np.mean(np.sign(d_pred[down_idx]) == np.sign(d_true[down_idx]))
        )
        out["mse_delta_downstream"] = float(np.mean((d_pred[down_idx] - d_true[down_idx]) ** 2))
    else:
        out["pearson_delta_downstream"] = float("nan")
        out["direction_acc_downstream"] = float("nan")
        out["mse_delta_downstream"] = float("nan")

    if de_idx.size >= 1:
        cols = torch.as_tensor(de_idx, device=x_pred.device)
        out["energy_distance_de"] = compute_energy_distance(x_pred[:, cols], x_true[:, cols])
        xp, xt = x_pred.cpu().numpy(), x_true.cpu().numpy()
        out["w1_de"] = float(np.mean([wasserstein_distance(xp[:, g], xt[:, g]) for g in de_idx]))
    else:
        out["energy_distance_de"] = float("nan")
        out["w1_de"] = float("nan")

    need = mean_ctrl[target_gene_idx] - mean_true[target_gene_idx]
    got = mean_ctrl[target_gene_idx] - mean_pred[target_gene_idx]
    out["knockdown_fraction"] = float(got / need) if abs(need) > 1e-8 else float("nan")
    return out


def compute_evaluation_metrics(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    target_gene_idx: int,
    x_ctrl: torch.Tensor = None,
    de_threshold: float = 0.25,
) -> Dict[str, float]:
    """
    Calcola l'insieme completo di metriche diagnostiche per una condizione di perturbazione.

    Args:
        x_pred: Cellule generate [N_samples, G].
        x_true: Cellule reali osservate [N_true, G].
        target_gene_idx: Indice del gene bersaglio perturbato.
        x_ctrl: Cellule di controllo [N_ctrl, G]. When given, the effect-based metrics
            of compute_delta_metrics are added (these are the ones to report).
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

    metrics = {
        "energy_distance": float(e_dist),
        "pearson_mean_corr": float(corr),
        "mean_profile_mse": float(mse),
        "target_gene_error": float(target_abs_err),
        "mean_w1_distance": mean_w1,
    }
    if x_ctrl is not None:
        metrics.update(
            compute_delta_metrics(x_pred, x_true, x_ctrl, target_gene_idx, de_threshold)
        )
    return metrics
