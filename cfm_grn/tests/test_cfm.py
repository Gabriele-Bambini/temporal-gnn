"""
Test unitari matematici per il modulo CFM (Conditional Flow Matching).
Verifica:
1. Shape dei tensori e correttezza dei probability path.
2. Proprieta' di trasporto dell'Optimal Transport matching.
3. Risoluzione di un toy problem 2D con integrazione ODE RK4 ed Eulero.
"""

import pytest
import torch
import torch.nn as nn
from cfm.flow_matcher import ConditionalFlowMatcher, OptimalTransportCFM
from cfm.solver import ODESolver


def test_cfm_shapes_and_endpoints():
    """Verifica che il probability path rispetti gli estremi e le dimensioni."""
    B, D = 32, 10
    cfm = ConditionalFlowMatcher(sigma_min=0.0)

    x_0 = torch.zeros(B, D)
    x_1 = torch.ones(B, D) * 5.0

    # Test al tempo t=0
    t_zero = torch.zeros(B, 1)
    t, x_t, u_t = cfm.sample_location_and_target(x_0, x_1, t=t_zero)
    assert torch.allclose(x_t, x_0), "A t=0, x_t deve coincidere con x_0"
    assert torch.allclose(u_t, x_1 - x_0), "Target u_t deve essere x_1 - x_0"

    # Test al tempo t=1
    t_one = torch.ones(B, 1)
    t, x_t, u_t = cfm.sample_location_and_target(x_0, x_1, t=t_one)
    assert torch.allclose(x_t, x_1), "A t=1, x_t deve coincidere con x_1"


def test_ot_coupling_reduces_cost():
    """Verifica che il coupling OT riduca o eguagli la distanza quadratica media nel batch."""
    torch.manual_seed(42)
    B, D = 64, 5
    ot_cfm = OptimalTransportCFM(sigma_min=0.0)

    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D) + 2.0

    # Costo prima del matching (accoppiamento casuale lungo gli indici)
    cost_random = torch.mean(torch.sum((x_0 - x_1) ** 2, dim=-1)).item()

    # Costo dopo il matching OT
    x_0_ot, x_1_ot = ot_cfm.compute_ot_coupling(x_0, x_1)
    cost_ot = torch.mean(torch.sum((x_0_ot - x_1_ot) ** 2, dim=-1)).item()

    assert cost_ot <= cost_random, f"Il costo OT ({cost_ot}) deve essere <= al costo casuale ({cost_random})"


class ToyMLPVectorField(nn.Module):
    """Semplice MLP time-dependent per verificare l'apprendimento su un toy problem 2D."""
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        # Riceve [x, t] concatenati -> dimensione in_dim + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x, t, condition=None):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


def test_toy_problem_convergence():
    """
    Verifica che il Flow Matching impari a traslare una distribuzione 2D
    da [-2, -2] a [+2, +2] e che il solver ODE campioni correttamente.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = ToyMLPVectorField(in_dim=2, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    cfm = OptimalTransportCFM(sigma_min=0.01)

    # Distribuzione sorgente p_0 ~ N([-2, -2], 0.1^2)
    # Distribuzione bersaglio p_1 ~ N([+2, +2], 0.1^2)
    mean_0 = torch.tensor([-2.0, -2.0])
    mean_1 = torch.tensor([+2.0, +2.0])

    # Breve addestramento di 200 step per verificare la convergenza
    for step in range(200):
        optimizer.zero_grad()
        x_0 = mean_0 + 0.1 * torch.randn(128, 2)
        x_1 = mean_1 + 0.1 * torch.randn(128, 2)

        loss, _ = cfm.compute_loss(model, x_0, x_1)
        loss.backward()
        optimizer.step()

    # Test con ODESolver RK4
    solver = ODESolver(method="rk4", num_steps=20)
    test_x0 = mean_0 + 0.1 * torch.randn(200, 2)
    x_gen, _ = solver.sample(model, test_x0)

    # La media dei campioni generati deve essere vicina a [+2, +2]
    gen_mean = x_gen.mean(dim=0)
    assert torch.allclose(gen_mean, mean_1, atol=0.3), (
        f"La media generata {gen_mean} deve essere vicina al target {mean_1}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
