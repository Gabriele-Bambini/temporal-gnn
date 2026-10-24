"""
Conditional Flow Matching (CFM) Engine.

Questo modulo implementa la matematica del Flow Matching continuo:
1. Campionamento del tempo t ~ Uniform(0, 1).
2. Costruzione del probability path condizionato p_t(x | x_0, x_1).
3. Calcolo del target analitico u_t(x | x_0, x_1) = x_1 - x_0 (OT path).
4. Mini-batch Optimal Transport (OT) coupling per raddrizzare le traiettorie.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class ConditionalFlowMatcher:
    """
    Motore base di Conditional Flow Matching (CFM).
    
    Teoria:
    Dati x_0 (controllo / stato basale) e x_1 (stato perturbato / target),
    il probability path gaussiano condizionato e' definito come:
        p_t(x | x_0, x_1) = N(x | mu_t(x_0, x_1), sigma_t^2 * I)
    
    Nel caso OT (Optimal Transport displacement interpolation):
        mu_t(x_0, x_1) = (1 - t) * x_0 + t * x_1
        sigma_t = sigma_min (costante piccola o nulla)
    
    Il campo vettoriale condizionato analitico (la derivata temporale di mu_t) e':
        u_t(x | x_0, x_1) = d/dt mu_t = x_1 - x_0
    
    La funzione di costo (CFM loss) e' una semplice regressione L2:
        L_CFM(theta) = E_{t, x_0, x_1, eps} [ || v_theta(x_t, t, c) - u_t(x | x_0, x_1) ||^2 ]
    """

    def __init__(self, sigma_min: float = 1e-4):
        """
        Args:
            sigma_min: Deviazione standard minima per regolarizzare il probability path.
                       Se sigma_min = 0, il path e' deterministico (lineare puro).
        """
        self.sigma_min = sigma_min

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Campiona t in modo uniforme tra 0 e 1 per ciascun elemento del batch.
        
        Args:
            batch_size: Numero di campioni nel batch.
            device: Device PyTorch (cpu o cuda).
            
        Returns:
            Tensor di forma [B, 1].
        """
        return torch.rand(batch_size, 1, device=device)

    def sample_location_and_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interpola tra x_0 e x_1 al tempo t e restituisce la posizione x_t
        e il campo vettoriale target u_t.
        
        Args:
            x_0: Stato iniziale [B, D] (es. cellule di controllo).
            x_1: Stato finale target [B, D] (es. cellule perturbate).
            t: Tempo opzionale [B, 1]. Se None, viene campionato uniformemente in [0, 1].
            
        Returns:
            t: Tempi campionati [B, 1].
            x_t: Punti nello spazio d'espressione al tempo t [B, D].
            u_t: Vettore velocita' target [B, D].
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        if t is None:
            t = self.sample_time(batch_size, device)

        # mu_t = (1 - t) * x_0 + t * x_1
        mu_t = (1.0 - t) * x_0 + t * x_1

        # Aggiunta di rumore epsilon ~ N(0, I) scalato da sigma_min
        if self.sigma_min > 0:
            noise = torch.randn_like(x_0)
            x_t = mu_t + self.sigma_min * noise
        else:
            x_t = mu_t

        # Target analitico: derivata temporale d(mu_t)/dt = x_1 - x_0
        u_t = x_1 - x_0

        return t, x_t, u_t

    def compute_loss(
        self,
        vector_field_model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calcola la loss di Flow Matching L2 per il batch corrente.
        
        Args:
            vector_field_model: Rete neurale v_theta(x, t, condition) che predice la velocita'.
            x_0: Profili di controllo [B, G].
            x_1: Profili perturbati [B, G].
            condition: Indicatore o embedding della perturbazione [B, C].
            
        Returns:
            loss: Scalare tensore da minimizzare con backpropagation.
            metrics: Dizionario con metriche diagnostiche (L2 loss, varianza target, ecc.).
        """
        t, x_t, u_t = self.sample_location_and_target(x_0, x_1)

        # La rete neurale predice la velocita' istantanea
        v_pred = vector_field_model(x_t, t, condition)

        # Regressione L2: || v_pred - u_t ||^2 mediata sul batch e sui geni
        loss = torch.mean((v_pred - u_t) ** 2)

        metrics = {
            "loss_cfm": loss.item(),
            "target_norm": torch.norm(u_t, dim=-1).mean().item(),
            "pred_norm": torch.norm(v_pred, dim=-1).mean().item(),
        }

        return loss, metrics


class OptimalTransportCFM(ConditionalFlowMatcher):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).
    
    Miglioramento rispetto a CFM standard:
    Invece di accoppiare casualmente x_0[i] con x_1[i], risolve un problema di
    assegnamento ottimo (Monge-Kantorovich) all'interno del mini-batch:
        min_pi sum_i || x_0[i] - x_1[pi(i)] ||^2
    
    Questo accoppiamento OT minimizza la lunghezza totale delle traiettorie,
    evita che le linee di flusso si incrocino e produce traiettorie quasi rettilinee.
    Risultato: l'integrazione numerica al test-time converge con pochissimi step (es. 10-20 step RK4).
    """

    def __init__(self, sigma_min: float = 1e-4):
        super().__init__(sigma_min=sigma_min)

    @torch.no_grad()
    def compute_ot_coupling(
        self, x_0: torch.Tensor, x_1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Risolve il problema di Optimal Transport nel mini-batch tramite l'algoritmo ungherese.
        
        Args:
            x_0: [B, D]
            x_1: [B, D]
            
        Returns:
            x_0: Tensore originale [B, D].
            x_1_matched: Tensore riordinato con la permutazione ottimale [B, D].
        """
        # Calcolo matrice di costo delle distanze euclidee al quadrato [B, B]
        cost_matrix = torch.cdist(x_0, x_1, p=2) ** 2
        cost_matrix_np = cost_matrix.detach().cpu().numpy()

        # Algoritmo Ungherese (Hungarian Algorithm) O(B^3)
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

        x_0_matched = x_0[row_ind]
        x_1_matched = x_1[col_ind]

        return x_0_matched, x_1_matched

    def sample_location_and_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Esegue il matching OT sul mini-batch prima di campionare la posizione e il target.
        """
        x_0_ot, x_1_ot = self.compute_ot_coupling(x_0, x_1)
        return super().sample_location_and_target(x_0_ot, x_1_ot, t=t)
