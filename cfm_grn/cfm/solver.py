"""
ODE Solvers per l'inferenza e la generazione con Flow Matching.

Questo modulo permette di integrare il campo vettoriale appreso:
    dx_t / dt = v_theta(x_t, t, condition)
da t=0 (cellula di controllo) a t=1 (cellula perturbata predetta).

Supporta:
- Metodo di Eulero (veloce, 1-step o pochi step)
- Metodo Runge-Kutta 4 (RK4, accurato a 4° ordine)
- Tracciamento della traiettoria completa per visualizzare l'evoluzione temporale continua.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn


class ODESolver:
    """
    Risolutore di Equazioni Differenziali Ordinarie (ODE) per generare campioni.
    """

    def __init__(self, method: str = "rk4", num_steps: int = 20):
        """
        Args:
            method: Algoritmo di integrazione numerica ('euler' o 'rk4').
            num_steps: Numero di intervalli di discretizzazione temporale tra t=0 e t=1.
        """
        assert method in ["euler", "rk4"], f"Metodo {method} non supportato. Usa 'euler' o 'rk4'."
        self.method = method
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        vector_field_model: nn.Module,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Integra l'ODE da t=0 a t=1 a partire dallo stato iniziale x_0.
        
        Args:
            vector_field_model: Modello neurale v_theta(x, t, condition).
            x_0: Stato iniziale [B, G] (es. profili di cellule di controllo).
            condition: Indicatore di perturbazione [B, C].
            return_trajectory: Se True, restituisce la lista di tutti gli stati intermedi x_t.
            
        Returns:
            x_final: Stato al tempo t=1 (profilo perturbato predetto) [B, G].
            trajectory: Lista dei tensori x_t per ogni step temporale se return_trajectory=True.
        """
        vector_field_model.eval()
        device = x_0.device
        batch_size = x_0.shape[0]

        dt = 1.0 / self.num_steps
        x_t = x_0.clone()

        trajectory = [x_t.clone()] if return_trajectory else None

        for step in range(self.num_steps):
            t_val = step * dt
            # Crea il tensore temporale [B, 1] costante per tutto il batch allo step corrente
            t_tensor = torch.full((batch_size, 1), t_val, device=device, dtype=x_0.dtype)

            if self.method == "euler":
                # Metodo di Eulero: x_{t+dt} = x_t + dt * v_theta(x_t, t)
                v = vector_field_model(x_t, t_tensor, condition)
                x_t = x_t + dt * v

            elif self.method == "rk4":
                # Runge-Kutta 4th order:
                # k1 = v(x, t)
                # k2 = v(x + dt/2 * k1, t + dt/2)
                # k3 = v(x + dt/2 * k2, t + dt/2)
                # k4 = v(x + dt * k3, t + dt)
                # x_{t+dt} = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
                
                half_dt = dt / 2.0
                t_mid = torch.full((batch_size, 1), t_val + half_dt, device=device, dtype=x_0.dtype)
                t_end = torch.full((batch_size, 1), t_val + dt, device=device, dtype=x_0.dtype)

                k1 = vector_field_model(x_t, t_tensor, condition)
                k2 = vector_field_model(x_t + half_dt * k1, t_mid, condition)
                k3 = vector_field_model(x_t + half_dt * k2, t_mid, condition)
                k4 = vector_field_model(x_t + dt * k3, t_end, condition)

                x_t = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if return_trajectory:
                trajectory.append(x_t.clone())

        return x_t, trajectory
