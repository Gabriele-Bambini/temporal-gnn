"""
Simulatore Sintetico di Dinamiche di Regolazione Genica (Hill Kinetics & ODE biologiche).

Questo modulo genera dataset sintetici controllati dove:
1. La topologia della rete di regolazione A_true e' nota a priori (ground truth).
2. L'espressione e' regolata da equazioni cinetiche non-lineari (Hill kinetics):
       dx_i / dt = alpha_i + sum_{j in Act(i)} w_{ji} * (x_j^n / (K^n + x_j^n)) - gamma_i * x_i
3. Le perturbazioni (es. knockout del gene k) silenziano l'espressione del target (CRISPRi).
4. Fornisce popolazioni di cellule di controllo e cellule perturbate campionando stati stazionari
   con rumore intrinseco biologico (Langevin SDE).
"""

from typing import Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
import numpy as np


class SyntheticGRNSimulator:
    """
    Simulatore biochimico basato su cinetiche di Hill ed integrazione Runge-Kutta / Eulero-Maruyama.
    """

    def __init__(
        self,
        num_genes: int = 20,
        density: float = 0.2,
        hill_coefficient: float = 2.0,
        decay_rate: float = 1.0,
        basal_rate: float = 0.5,
        noise_level: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            num_genes: Numero di geni G.
            density: Densita' degli archi orientati (frazione di connessioni attive).
            hill_coefficient: Coefficiente n nella funzione di Hill (cooperativita').
            decay_rate: Costante di degradazione gamma_i.
            basal_rate: Tasso di trascrizione basale alpha_i.
            noise_level: Livello di rumore stocastico nella trascrizione.
        """
        self.num_genes = num_genes
        self.n = hill_coefficient
        self.decay_rate = decay_rate
        self.basal_rate = basal_rate
        self.noise_level = noise_level
        self.K = 1.0  # Costante di attivazione a meta' saturazione

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Genera matrice di adiacenza orientata casuale (DAG o grafo ciclico sparso)
        rand_matrix = np.random.rand(num_genes, num_genes)
        # Rimuove auto-anelli
        np.fill_diagonal(rand_matrix, 0.0)
        adj_binary = (rand_matrix < density).astype(np.float32)

        # Pesi di regolazione positivi (attivazione) tra 0.5 e 2.0
        weights = adj_binary * (0.5 + 1.5 * np.random.rand(num_genes, num_genes))

        self.W = torch.tensor(weights, dtype=torch.float32)
        self.edge_index = torch.nonzero(self.W > 0, as_tuple=False).t().contiguous()
        self.num_edges = self.edge_index.shape[1]

    def drift(self, x: torch.Tensor, knockout_gene: Optional[int] = None) -> torch.Tensor:
        """
        Calcola il campo vettoriale biologico dx/dt (derivate temporali delle concentrazioni).
        
        Args:
            x: Espressioni geniche [B, G].
            knockout_gene: Indice del gene perturbato (se None, cellula di controllo).
        """
        # Funzione di Hill per l'attivazione: f(x) = x^n / (K^n + x^n)
        x_clamped = torch.clamp(x, min=0.0)
        hill_act = (x_clamped ** self.n) / (self.K ** self.n + x_clamped ** self.n + 1e-8)

        # Trascrizione regolata: W^T * hill_act -> [B, G]
        transcription = torch.matmul(hill_act, self.W)

        # Sintesi totale = basale + trascrizione
        synthesis = self.basal_rate + transcription

        # Se il gene k e' in knockout (CRISPRi), la sua sintesi e' azzerata
        if knockout_gene is not None:
            synthesis[:, knockout_gene] = 0.0

        # Degradazione lineare: - gamma * x
        degradation = self.decay_rate * x_clamped

        dxdt = synthesis - degradation

        if knockout_gene is not None:
            # Forza la derivata a spingere il gene verso zero rapidamente
            dxdt[:, knockout_gene] = - 5.0 * x_clamped[:, knockout_gene]

        return dxdt

    def simulate_steady_state(
        self,
        num_cells: int = 500,
        knockout_gene: Optional[int] = None,
        num_steps: int = 100,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """
        Simula una popolazione di cellule all'equilibrio stazionario tramite Eulero-Maruyama SDE.
        """
        # Stato iniziale casuale
        x = torch.rand(num_cells, self.num_genes) * 2.0

        for _ in range(num_steps):
            drift_term = self.drift(x, knockout_gene=knockout_gene)
            diffusion = self.noise_level * np.sqrt(dt) * torch.randn_like(x)
            x = x + dt * drift_term + diffusion
            x = torch.clamp(x, min=0.0)  # Le concentrazioni biologiche sono >= 0

        return x.detach()

    def generate_benchmark_dataset(
        self,
        cells_per_condition: int = 300,
        perturbed_genes: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Genera un dataset completo con cellule di controllo e perturbazioni su geni selezionati.
        """
        if perturbed_genes is None:
            # Sceglie come target i primi 3 geni con grado uscente piu' alto (TF principali)
            out_degrees = torch.sum(self.W > 0, dim=1).numpy()
            perturbed_genes = list(np.argsort(-out_degrees)[:3])

        data_dict = {}

        # 1. Popolazione di controllo (non perturbata)
        ctrl_cells = self.simulate_steady_state(num_cells=cells_per_condition, knockout_gene=None)
        data_dict["control"] = ctrl_cells
        data_dict["perturbed"] = {}

        # 2. Popolazioni perturbate per ciascun gene target
        for gene_idx in perturbed_genes:
            pert_cells = self.simulate_steady_state(
                num_cells=cells_per_condition,
                knockout_gene=gene_idx,
            )
            data_dict["perturbed"][gene_idx] = pert_cells

        data_dict["edge_index"] = self.edge_index
        data_dict["W_true"] = self.W
        data_dict["num_genes"] = self.num_genes
        data_dict["perturbed_genes"] = perturbed_genes

        return data_dict


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset per coppie (controllo x_0, perturbato x_1, indicatore di perturbazione c).
    """

    def __init__(self, benchmark_data: Dict[str, torch.Tensor]):
        super().__init__()
        self.ctrl_cells = benchmark_data["control"]
        self.num_genes = benchmark_data["num_genes"]
        self.samples = []

        num_ctrl = self.ctrl_cells.shape[0]

        for gene_idx, pert_cells in benchmark_data["perturbed"].items():
            num_pert = pert_cells.shape[0]
            # Indicatore one-hot della perturbazione [G]
            pert_vec = torch.zeros(self.num_genes)
            pert_vec[gene_idx] = 1.0

            # Crea coppie appaiando o campionando
            for i in range(num_pert):
                ctrl_idx = i % num_ctrl
                x_0 = self.ctrl_cells[ctrl_idx]
                x_1 = pert_cells[i]
                self.samples.append((x_0, x_1, pert_vec, gene_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.samples[idx]
