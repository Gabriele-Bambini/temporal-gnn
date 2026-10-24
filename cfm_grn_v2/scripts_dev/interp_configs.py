"""Test-side comparison of every quick GRN configuration on the interpolation fold t=3 (diagnostic only:
the selection for the report stays nested on the validation fold).

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/interp_configs.py
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from cfm_grn_v2.evaluation.metrics import EvalProtocol
from cfm_grn_v2.experiments.run_synthetic import GRN_TIMES, QUICK_GRN_GRID, grn_setup
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import run_interpolation_benchmark

pd.set_option("display.width", 250)
s, *_ = grn_setup(150, 0, 20, 6)
base = TrainConfig(n_iters=1500, eval_every=150, patience=4, batch_size=128, depth=2)
proto = EvalProtocol(n_samples=300, seed=0, n_perm=20)
rows = []
for over in QUICK_GRN_GRID:
    cfg = base.replace(**over)
    df = run_interpolation_benchmark(s, GRN_TIMES, 3.0, cfg, seeds=[0, 1], protocol=proto, dataset_name="interp_t3",
                                     mlp_kwargs={"hidden": 64, "n_iters": 300}, shuffled_control=False)
    ed = df.groupby("model")["energy_distance"].mean()
    bi = df[df.model == "flow_matching"]["best_iter"].mean()
    rows.append({**over, "fm": ed["flow_matching"], "ot_interp": ed["ot_interp"], "mean_interp": ed["mean_interp"],
                 "linear": ed["linear"], "best_iter": bi})
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)
