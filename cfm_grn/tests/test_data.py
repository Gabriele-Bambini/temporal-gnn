"""
Test unitari per i moduli data (synthetic_grn e dataset).
Verifica:
1. Generazione corretta del grafo e simulazione delle concentrazioni cellulari stazionarie.
2. Knockout del gene target (la concentrazione del target deve crollare a zero).
3. Suddivisione corretta tra condizioni di train e held-out test.
"""

import pytest
import torch
from data.synthetic_grn import SyntheticGRNSimulator
from data.dataset import create_dataloaders


def test_synthetic_grn_simulation_and_knockout():
    """Verifica che la dinamica di Hill simuli correttamente il controllo e il knockdown."""
    sim = SyntheticGRNSimulator(num_genes=10, density=0.3, seed=123)

    # Simula controlli
    ctrl = sim.simulate_steady_state(num_cells=50, knockout_gene=None)
    assert ctrl.shape == (50, 10)
    assert (ctrl >= 0).all(), "Le concentrazioni devono essere non negative"

    # Simula knockout sul gene 2
    pert = sim.simulate_steady_state(num_cells=50, knockout_gene=2)
    assert pert.shape == (50, 10)

    # L'espressione del gene 2 perturbato deve essere marcatamente inferiore a quella di controllo
    ctrl_mean = ctrl[:, 2].mean().item()
    pert_mean = pert[:, 2].mean().item()
    assert pert_mean < ctrl_mean * 0.2, (
        f"L'espressione nel knockout ({pert_mean:.3f}) deve essere < 20% del controllo ({ctrl_mean:.3f})"
    )


def test_dataset_held_out_split():
    """Verifica che i dataloader separino rigorosamente le condizioni di train e test."""
    sim = SyntheticGRNSimulator(num_genes=10, density=0.3, seed=123)
    benchmark_data = sim.generate_benchmark_dataset(cells_per_condition=60)

    perturbed_genes = benchmark_data["perturbed_genes"]
    held_out = perturbed_genes[0]

    train_loader, test_loader = create_dataloaders(
        benchmark_data, batch_size=16, held_out_genes=[held_out]
    )

    # Verifica che nel train_loader non compaia mai il gene held-out
    for _, _, _, gene_id in train_loader.dataset:
        assert gene_id != held_out, f"Trovato gene held-out {held_out} nel training set!"

    # Verifica che nel test_loader compaia SOLO il gene held-out
    for _, _, _, gene_id in test_loader.dataset:
        assert gene_id == held_out, f"Trovato gene {gene_id} nel test set diverso da {held_out}!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
