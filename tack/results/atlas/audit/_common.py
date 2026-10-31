"""Shared paths and helpers for the adversarial audit of the atlas confirmation (EXPLORATORY, read-only on frozen code).

Imports the frozen modules read-only (tack/hpc). Never writes outside tack/results/atlas/audit/.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
HPC = os.path.join(ROOT, "tack", "hpc")
AUDIT = os.path.dirname(os.path.abspath(__file__))
SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
ATLAS = os.path.join(SCR, "atlas")
ERY = os.path.join(SCR, "ery")
CONFIRM = os.path.join(ROOT, "tack", "results", "atlas", "confirm.json")
PARITY = os.path.join(ROOT, "tack", "results", "atlas", "parity_erythroid.json")
sys.path.insert(0, HPC)

import interp_bench_frozen as IB  # noqa: E402
import atlas_confirm as AC  # noqa: E402
from ery_forecast import kernel_regress  # noqa: E402,F401


def load(folder, name):
    return np.load(os.path.join(folder, name), allow_pickle=True)


def confirm():
    with open(CONFIRM) as fh:
        return json.load(fh)


def parity():
    with open(PARITY) as fh:
        return json.load(fh)


def dump(name, obj):
    path = os.path.join(AUDIT, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", path, flush=True)
