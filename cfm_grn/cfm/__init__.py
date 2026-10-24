"""
Conditional Flow Matching (CFM) engine.
Contiene l'implementazione del probability path, del calcolo del campo vettoriale condizionato (OT-CFM),
e degli integratori ODE per il campionamento (Eulero, RK4).
"""

from .flow_matcher import ConditionalFlowMatcher, OptimalTransportCFM
from .solver import ODESolver

__all__ = ["ConditionalFlowMatcher", "OptimalTransportCFM", "ODESolver"]
