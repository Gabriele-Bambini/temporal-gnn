"""
Modelli di Controllo e Baseline per Ablazione Scientifica.

Per verificare rigorosamente le tre ipotesi scientifiche del PhD:
1. MLPVectorField: Modello non strutturato (nessun grafo). Verifica se il prior di rete e' necessario.
2. StaticGraphVectorField: Modello con grafo fisso A_prior (pesi = 1, nessuna maschera M_theta).
   Verifica se la modulazione contestuale appresa apporta un reale vantaggio predittivo.
3. PersistenceBaseline: Controllo di persistenza pura x_1 = x_0 (velocita' nulla).
"""

from typing import Optional
import torch
import torch.nn as nn
from .graph_vector_field import GraphVectorField


class MLPVectorField(nn.Module):
    """
    Baseline non strutturata: Perceptron Multistrato denso.
    Ignora completamente la topologia della rete di regolazione genica.
    Predice v_theta(x, t, c) attraverso connessioni dense globali tra tutti i geni.
    """

    def __init__(
        self,
        num_genes: int,
        perturbation_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_genes = num_genes

        # Input: x [G] + t [1] + perturbation [C]
        in_dim = num_genes + 1 + perturbation_dim

        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])

        # Output: velocita' istantanea per ciascun gene [G]
        layers.append(nn.Linear(hidden_dim, num_genes))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs = [x, t]
        if perturbation is not None:
            inputs.append(perturbation)
        cat_in = torch.cat(inputs, dim=-1)
        return self.net(cat_in)


class StaticGraphVectorField(GraphVectorField):
    """
    Baseline con Grafo Statico:
    Usa la stessa architettura MPNN orientata di GraphVectorField,
    ma fissa tutti i pesi della maschera ad 1.0 (nessuna modulazione M_theta appresa).
    """

    def __init__(
        self,
        num_genes: int,
        edge_index: torch.Tensor,
        perturbation_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__(
            num_genes=num_genes,
            edge_index=edge_index,
            perturbation_dim=perturbation_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_mask=False,  # Disabilita la maschera appresa
        )


class PersistenceBaseline:
    """
    Controllo di Persistenza:
    Assume che la cellula non risponda alla perturbazione (v = 0, x_1 = x_0).
    Fondamentale come benchmark nullo nei dataset di perturbazione.
    """

    def predict(self, x_0: torch.Tensor) -> torch.Tensor:
        return x_0.clone()
