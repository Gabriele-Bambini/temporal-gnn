"""Measure seconds per training iteration at full Phase-1 scale, to set the full-run budget.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/time_full_scale.py
"""
import sys
import time

sys.path.insert(0, "src")
from cfm_grn_v2.evaluation.metrics import EvalProtocol
from cfm_grn_v2.experiments.run_synthetic import grn_setup, toy_setup
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import train_flow

proto = EvalProtocol(n_samples=1000, seed=0, n_perm=50)
s, fit, fit_times, val, test, truth = grn_setup(500, 0, 40, 10)
for clock, pair_mode in (("flow", "consecutive"), ("duration", "all")):
    cfg = TrainConfig(hidden=256, depth=3, n_iters=500, batch_size=256, eval_every=250, patience=8, seed=0,
                      clock=clock, pair_mode=pair_mode)
    t0 = time.time()
    model, hist = train_flow(s, fit, fit_times, cfg, val_task=val, protocol=proto)
    dt = time.time() - t0
    print(f"GRN full scale {clock}/{pair_mode}: {dt:.1f}s for 500 iters + 2 evals -> {dt / 500 * 1000:.0f} ms/iter, "
          f"val ED {hist[-1]['val_energy_distance']:.3f}", flush=True)
s, fit, fit_times, val, test = toy_setup("bimodal", 1500, 0)
cfg = TrainConfig(hidden=256, depth=3, n_iters=500, batch_size=256, eval_every=250, patience=8, seed=0)
t0 = time.time()
model, hist = train_flow(s, fit, fit_times, cfg, val_task=val, protocol=proto)
dt = time.time() - t0
print(f"toy bimodal full scale: {dt:.1f}s for 500 iters + 2 evals -> {dt / 500 * 1000:.0f} ms/iter", flush=True)
