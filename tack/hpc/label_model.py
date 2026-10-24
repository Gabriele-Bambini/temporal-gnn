"""Labelled-RNA model for the sci-FATE2 clock probe (pre-registration v2, section 5).

Given unspliced/spliced means U(t), S(t) of a linear two-species model (dU/dt = alpha(t) - beta U,
dS/dt = beta U - gamma S), the RNA made in the last h process units is total minus what survives from t0 = t - h:
  old_u = U(t0) exp(-beta h)
  old_s = S(t0) exp(-gamma h) + beta U(t0) phi(h),  phi(h) = (exp(-gamma h) - exp(-beta h)) / (beta - gamma)
This identity is exact for any alpha(t), because old molecules evolve without production.
"""
import numpy as np
from scipy.special import exprel


def phi(beta, gamma, h):
    """(exp(-gamma h) - exp(-beta h)) / (beta - gamma), numerically stable, broadcasting."""
    beta, gamma = np.broadcast_arrays(np.asarray(beta, float), np.asarray(gamma, float))
    x = (beta - gamma) * h
    out = np.empty_like(x)
    small = x > -50  # exprel(-x) with -x < 50 does not overflow
    out[small] = h * np.exp(-gamma[small] * h) * exprel(-x[small])
    big = ~small  # gamma >> beta: both exponentials are tiny, use the direct difference
    out[big] = (np.exp(-beta[big] * h) - np.exp(-gamma[big] * h)) / (gamma[big] - beta[big])
    return out


def new_fraction(U, S, U0, S0, beta, gamma, h, amb_share, beta_flux=None):
    """Fraction of reads synthesised in the last h process units.

    U, S: means at t, shape (..., p); U0, S0: means at t0 = max(t - h, 0), same shape; beta, gamma: (p,);
    amb_share: (p,) ambiguous read share a_g, ambiguous reads counted with spliced kinetics.
    beta_flux: converts the given U units into the splicing flux (beta_flux * U0 molecules per unit time enter S);
    differs from beta only when U is observed through a capture factor. Default: beta.
    """
    if beta_flux is None:
        beta_flux = beta
    old_u = U0 * np.exp(-beta * h)
    old_s = S0 * np.exp(-gamma * h) + beta_flux * U0 * phi(beta, gamma, h)
    new_u = np.clip(U - old_u, 0.0, U)
    new_s = np.clip(S - old_s, 0.0, S)
    f_s = np.where(S > 0, new_s / np.maximum(S, 1e-300), 0.0)
    A = amb_share / (1.0 - amb_share) * (U + S)
    denom = U + S + A
    return np.where(denom > 0, (new_u + f_s * (S + A)) / np.maximum(denom, 1e-300), 0.0)


def _selftest():
    """Compare the closed form with a fine numerical integration of the labelled species."""
    for beta, gamma in [(3.0, 0.2), (0.5, 0.5000001), (0.2, 4.0), (10.0, 9.99)]:
        dt = 1e-4
        ts = np.arange(0, 6, dt)
        alpha = np.where(ts < 2, 1.0, np.where(ts < 3.5, 4.0, 0.5))  # switches, one inside the window
        u = np.zeros_like(ts)
        s = np.zeros_like(ts)
        u[0], s[0] = 1.0 / beta, 1.0 / gamma  # steady state of alpha = 1
        for i in range(1, len(ts)):
            u[i] = u[i - 1] + dt * (alpha[i - 1] - beta * u[i - 1])
            s[i] = s[i - 1] + dt * (beta * u[i - 1] - gamma * s[i - 1])
        t, h = 3.8, 0.7
        i_t, i_0 = int(round(t / dt)), int(round((t - h) / dt))
        un = np.zeros(i_t - i_0 + 1)
        sn = np.zeros(i_t - i_0 + 1)
        for k in range(1, len(un)):
            a = alpha[i_0 + k - 1]
            un[k] = un[k - 1] + dt * (a - beta * un[k - 1])
            sn[k] = sn[k - 1] + dt * (beta * un[k - 1] - gamma * sn[k - 1])
        U, S, U0, S0 = u[i_t], s[i_t], u[i_0], s[i_0]
        f = new_fraction(np.array([U]), np.array([S]), np.array([U0]), np.array([S0]),
                         np.array([beta]), np.array([gamma]), h, np.array([0.0]))[0]
        exact = (un[-1] + sn[-1]) / (U + S)
        assert abs(f - exact) < 2e-3, (beta, gamma, f, exact)
        print(f"beta={beta} gamma={gamma}: closed form {f:.5f} vs numerical {exact:.5f}")
    print("label_model selftest OK")


if __name__ == "__main__":
    _selftest()
