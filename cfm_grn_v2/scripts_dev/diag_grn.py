"""Diagnostics for the synthetic-GRN forecast (why does validation ED rise while training?).

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/diag_grn.py
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.experiments.run_synthetic import grn_setup
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import build_transitions, task_groups, time_task, train_flow

s, fit, fit_times, val, test, truth = grn_setup(150, 0, 20, 6)
print("times", s.timepoints().tolist(), "fit_times", fit_times, "val", val.name, "test", test.name)
st = Standardizer().fit(s.x[fit])
xz = st.transform(s.x)
print("\nTraining transitions (standardised on fit cells):")
for tr in build_transitions(s.subset(fit), fit_times, xz[fit]):
    disp = np.linalg.norm(tr.x1.mean(0) - tr.x0.mean(0))
    print(f"  {tr.replicate} {tr.bio_t0:.0f}->{tr.bio_t1:.0f}: |mean disp|={disp:.2f} ED(persist)={energy_distance(tr.x0, tr.x1):.3f}"
          f" |x0|max={np.abs(tr.x0).max():.1f}")
for cond, rep, m0, m1 in task_groups(s, val):
    print(f"  val {rep} {val.bio_t0:.0f}->{val.bio_t1:.0f}: |mean disp|={np.linalg.norm(xz[m1].mean(0) - xz[m0].mean(0)):.2f}"
          f" ED(persist)={energy_distance(xz[m0], xz[m1]):.3f} |x0|max={np.abs(xz[m0]).max():.1f}")

proto = EvalProtocol(n_samples=300, seed=0, n_perm=5)
ins = time_task(s, fit_times[-2], fit_times[-1])
for tc, lr, steps in (("delta", 1e-3, 20), ("delta", 3e-4, 20), ("delta", 1e-3, 100), ("absolute", 1e-3, 20)):
    cfg = TrainConfig(hidden=128, depth=2, lr=lr, n_iters=1500, batch_size=128, eval_every=100, patience=100, seed=0,
                      time_conditioning=tc, n_ode_steps=steps)
    model, hist = train_flow(s, fit, fit_times, cfg, val_task=val, protocol=proto)
    print(f"\n[{tc} lr={lr} ode_steps={steps}] iter:loss/valED  " +
          " ".join(f"{h['iter']}:{h['loss']:.3f}/{h['val_energy_distance']:.3f}" for h in hist))
    vals = []
    for cond, rep, m0, m1 in task_groups(s, ins):
        pred = model.predict_standardized(xz[m0], cond, ins.bio_t0, ins.bio_t1)
        vals.append((energy_distance(pred, xz[m1]), energy_distance(xz[m0], xz[m1]), np.abs(pred).max()))
    print("   in-sample %.0f->%.0f: ED(fm)=%.3f ED(persist)=%.3f max|pred|=%.1f" % ((ins.bio_t0, ins.bio_t1) + tuple(np.mean(vals, 0))))
    vals = []
    for cond, rep, m0, m1 in task_groups(s, val):
        pred = model.predict_standardized(xz[m0], cond, val.bio_t0, val.bio_t1)
        vals.append((energy_distance(pred, xz[m1]), np.abs(pred).max(), np.linalg.norm(pred.mean(0) - xz[m0].mean(0))))
    print("   val (final weights): ED(fm)=%.3f max|pred|=%.1f |pred mean - x0 mean|=%.2f" % tuple(np.mean(vals, 0)))
