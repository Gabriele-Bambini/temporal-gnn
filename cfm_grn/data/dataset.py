"""
Dataset e DataLoaders per profili di singola cellula sotto perturbazione.

Supporta:
1. Split controllato tra perturbazioni viste (Train) e perturbazioni ESCLUSE (Held-Out Test),
   fondamentale per valutare la generalizzazione a nuovi target genetici.
2. DataLoader con campionamento di coppie (controllo x_0, perturbato x_1).
"""

from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class PerturbationPairDataset(Dataset):
    """
    Dataset PyTorch per accoppiare cellule non perturbate (controllo) e cellule perturbate.
    """

    def __init__(
        self,
        ctrl_expression: torch.Tensor,
        pert_expression_dict: Dict[int, torch.Tensor],
        num_genes: int,
        held_out_genes: Optional[List[int]] = None,
        is_train: bool = True,
    ):
        """
        Args:
            ctrl_expression: Matrice cellule di controllo [N_ctrl, G].
            pert_expression_dict: Dizionario {gene_id: tensore_cellule [N_pert, G]}.
            num_genes: Numero totale di geni G.
            held_out_genes: Lista di geni esclusi dall'addestramento (held-out test set).
            is_train: Se True include solo geni di addestramento; se False solo held-out.
        """
        self.ctrl_expression = ctrl_expression
        self.num_genes = num_genes
        held_out_set = set(held_out_genes or [])

        self.samples = []
        num_ctrl = ctrl_expression.shape[0]

        for gene_id, pert_cells in pert_expression_dict.items():
            is_held_out = gene_id in held_out_set

            # Filtra in base al tipo di split
            if is_train and is_held_out:
                continue
            if (not is_train) and (not is_held_out):
                continue

            num_pert = pert_cells.shape[0]
            pert_vec = torch.zeros(num_genes, dtype=torch.float32)
            pert_vec[gene_id] = 1.0

            for i in range(num_pert):
                ctrl_idx = i % num_ctrl
                x_0 = ctrl_expression[ctrl_idx]
                x_1 = pert_cells[i]
                self.samples.append((x_0, x_1, pert_vec, gene_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.samples[idx]


def create_dataloaders(
    benchmark_data: Dict[str, torch.Tensor],
    batch_size: int = 64,
    held_out_genes: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Crea i DataLoader di Train e Held-Out Test."""
    ctrl = benchmark_data["control"]
    perts = benchmark_data["perturbed"]
    num_genes = benchmark_data["num_genes"]

    train_ds = PerturbationPairDataset(
        ctrl_expression=ctrl,
        pert_expression_dict=perts,
        num_genes=num_genes,
        held_out_genes=held_out_genes,
        is_train=True,
    )

    test_ds = PerturbationPairDataset(
        ctrl_expression=ctrl,
        pert_expression_dict=perts,
        num_genes=num_genes,
        held_out_genes=held_out_genes,
        is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
