"""
Data package for cfm_grn.
Include simulatori sintetici di dinamica con GRN nota (ground-truth Hill kinetics)
e dataset loader per profili cellulari a coppie di perturbazione.
"""

from .synthetic_grn import SyntheticGRNSimulator, SyntheticDataset
from .dataset import PerturbationPairDataset

__all__ = ["SyntheticGRNSimulator", "SyntheticDataset", "PerturbationPairDataset"]
