"""
Test unitari per i modelli di rete neurale guidati da grafo (models).
Verifica:
1. Dimensioni dei tensori e bound [0, 1] della maschera regolatoria M_theta.
2. Propagazione dei messaggi e calcolo del campo vettoriale con GraphVectorField.
3. Calcolo e propagazione corretta dei gradienti all'indietro (backpropagation).
4. Corretto funzionamento delle baseline di ablazione (MLPVectorField, StaticGraphVectorField).
"""

import pytest
import torch
from models.regulatory_mask import RegulatoryActivityMask, SparseAdjacencyPrior
from models.graph_vector_field import GraphVectorField
from models.baselines import MLPVectorField, StaticGraphVectorField


@pytest.fixture
def dummy_grn():
    """Crea una piccola GRN giocattolo con G=10 geni ed E=15 archi orientati."""
    num_genes = 10
    # Archi diretti (src -> dst)
    edges = [
        [0, 1], [0, 2], [1, 3], [2, 3], [3, 4],
        [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
        [1, 5], [2, 6], [0, 9], [3, 7], [4, 8]
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return num_genes, edge_index


def test_regulatory_mask_bounds_and_gradients(dummy_grn):
    """Verifica che la maschera M_theta produca valori in [0, 1] e gradienti validi."""
    num_genes, edge_index = dummy_grn
    B = 8
    pert_dim = num_genes

    mask_net = RegulatoryActivityMask(
        num_genes=num_genes,
        edge_index=edge_index,
        perturbation_dim=pert_dim,
        hidden_dim=32,
    )

    x = torch.randn(B, num_genes)
    t = torch.rand(B, 1)
    pert = torch.zeros(B, pert_dim)
    pert[:, 0] = 1.0  # Knockout del gene 0

    mask = mask_net(x, t, pert)

    # Verifica forma: [B, E]
    assert mask.shape == (B, edge_index.shape[1]), f"Atteso {(B, edge_index.shape[1])}, ottenuto {mask.shape}"

    # Verifica bound [0, 1]
    assert (mask >= 0.0).all() and (mask <= 1.0).all(), "I valori della maschera devono essere in [0, 1]"

    # Verifica backward pass
    loss = mask_net.compute_regularization_loss(mask)
    loss.backward()

    # Controlla che i gradienti esistano e non siano NaN
    for p in mask_net.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_graph_vector_field_forward_backward(dummy_grn):
    """Verifica forward e backward pass dell'architettura GraphVectorField."""
    num_genes, edge_index = dummy_grn
    B = 16

    model = GraphVectorField(
        num_genes=num_genes,
        edge_index=edge_index,
        perturbation_dim=num_genes,
        hidden_dim=32,
        num_layers=2,
        use_mask=True,
    )

    x = torch.randn(B, num_genes)
    t = torch.rand(B, 1)
    pert = torch.zeros(B, num_genes)
    pert[:, 2] = 1.0

    v_pred = model(x, t, pert)

    # Forma attesa: [B, G]
    assert v_pred.shape == (B, num_genes), f"Atteso {(B, num_genes)}, ottenuto {v_pred.shape}"

    # Backward pass
    target = torch.randn(B, num_genes)
    loss = torch.mean((v_pred - target) ** 2)
    loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


def test_baselines(dummy_grn):
    """Verifica che le baseline MLP e Static Graph producano le stesse forme."""
    num_genes, edge_index = dummy_grn
    B = 8

    mlp = MLPVectorField(num_genes=num_genes, perturbation_dim=num_genes, hidden_dim=32)
    static_gnn = StaticGraphVectorField(num_genes=num_genes, edge_index=edge_index, perturbation_dim=num_genes, hidden_dim=32)

    x = torch.randn(B, num_genes)
    t = torch.rand(B, 1)
    pert = torch.randn(B, num_genes)

    v_mlp = mlp(x, t, pert)
    v_static = static_gnn(x, t, pert)

    assert v_mlp.shape == (B, num_genes)
    assert v_static.shape == (B, num_genes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
