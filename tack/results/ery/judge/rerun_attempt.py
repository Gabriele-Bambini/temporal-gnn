"""Judge: verbatim re-run of attempt_forecast.run_grid for one origin, outputs redirected into judge/attempt_rerun.
Imports attempt_forecast without editing it (module global HERE patched to the judge folder; no bytecode written)."""
import argparse
import os
import sys

sys.dont_write_bytecode = True
JUDGE = os.path.dirname(os.path.abspath(__file__))
ERY = os.path.dirname(JUDGE)
sys.path.insert(0, os.path.join(ERY, "attempt_forecast"))
import attempt_forecast as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--origins", type=int, nargs="+", required=True)
a = ap.parse_args()
out = os.path.join(JUDGE, "attempt_rerun")
os.makedirs(out, exist_ok=True)
A.HERE = out
for s in a.origins:
    A.run_grid(a, s)
