"""Why does the flow model lose to mean interpolation in PCA-50 on the embryoid body? A diagnosis without training.

On the simulator the trained field reproduced its training path to within 0.005 (scripts_dev/diag_ceiling.py), and
the mean of the spline path equals the cubic spline through the snapshot means for any coupling rule. So two
questions can be answered here without a network:

  location  where does each construction put the centre of the held-out window?  (L2 error of the predicted mean)
            - linear interpolation of the two neighbouring window means  (what mean interpolation does)
            - natural cubic spline through the observed window means     (what the spline path's mean is)
  shape     how much of the energy distance is left once the centre is corrected?
            each prediction is shifted rigidly onto the true mean and re-scored

and one question about the ceiling: the spline path family itself, evaluated at the held-out window, against the
trained model's score. Everything is computed in the benchmark's own evaluation space (standardised on the
observed cells) and in PCA-50, fold t = 2 only, i.e. the fold already reported: this is a diagnosis of a measured
result, not a selection step, and nothing chosen here may be used to re-score that fold.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/eb_diagnose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.metrics import energy_distance
from cfm_grn_v2.flows.coupling import chain_ot
from cfm_grn_v2.flows.path import SplinePath
from cfm_grn_v2.training.search import interpolation_design
from eb_benchmark import CACHE, PRIOR, load_cache, prior_regulators

OUT = Path("outputs/eb_diagnose")
OUT.mkdir(parents=True, exist_ok=True)


def ed(a, b, rng, n=1000):
    return float(energy_distance(a[rng.choice(len(a), min(n, len(a)), replace=False)],
                                 b[rng.choice(len(b), min(n, len(b)), replace=False)]))


def diagnose(Z, t, d, label, seeds=(0, 1, 2)):
    """Z: cells in the evaluation space; t: their window index."""
    obs_t = d["obs_times"]
    true = Z[t == d["test_time"]]
    mu = {w: Z[t == w].mean(0) for w in sorted(set(t))}
    m_true = mu[d["test_time"]]
    lin = mu[d["t_prev"]] + d["frac"] * (mu[d["t_next"]] - mu[d["t_prev"]])
    spl = CubicSpline(obs_t, np.stack([mu[w] for w in obs_t]), bc_type="natural")(d["test_time"])
    spl_fit = CubicSpline(d["fit_times"], np.stack([mu[w] for w in d["fit_times"]]), bc_type="natural")(d["test_time"])
    rows = [{"space": label, "what": "mean error, linear between neighbours", "value": float(np.linalg.norm(lin - m_true))},
            {"space": label, "what": "mean error, cubic spline through all observed windows", "value": float(np.linalg.norm(spl - m_true))},
            {"space": label, "what": "mean error, cubic spline through the fit windows only", "value": float(np.linalg.norm(spl_fit - m_true))},
            {"space": label, "what": "distance between the two neighbouring means", "value": float(np.linalg.norm(mu[d['t_next']] - mu[d['t_prev']]))}]
    src = Z[t == d["t_prev"]]
    mean_interp = src + (lin - mu[d["t_prev"]])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        traj = chain_ot([Z[t == w] for w in obs_t], rng, 1000, mode="ot")
        path = SplinePath(obs_t, traj, bc_type="natural")
        x_path, _ = path.evaluate(np.full(path.n, d["test_time"]), np.arange(path.n))
        lin_path = traj[obs_t.index(d["t_prev"])] + d["frac"] * (traj[obs_t.index(d["t_next"])] - traj[obs_t.index(d["t_prev"])])
        for name, pred in (("spline path (the model's training target)", x_path),
                           ("mean interpolation (control)", mean_interp),
                           ("McCann / OT interpolant (straight chained pairs)", lin_path)):
            er = np.random.default_rng(100 + seed)
            rows.append({"space": label, "what": f"ED, {name}", "seed": seed, "value": ed(pred, true, er)})
            shifted = pred - pred.mean(0) + m_true
            rows.append({"space": label, "what": f"ED after moving the centre onto the truth, {name}", "seed": seed,
                         "value": ed(shifted, true, np.random.default_rng(100 + seed))})
        half = np.random.default_rng(seed).permutation(len(true))
        rows.append({"space": label, "what": "ED, noise floor (two halves of the true window)", "seed": seed,
                     "value": ed(true[half[: len(true) // 2]], true[half[len(true) // 2:]], np.random.default_rng(100 + seed))})
    return rows


def main() -> None:
    X, obs, symbols, _ = load_cache(CACHE)
    times = sorted(float(v) for v in np.unique(obs["time"]))
    d = interpolation_design(times, 2.0)
    Xn = normalize_log1p(X, keep_sparse=True)
    fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
    sel = select_genes(Xn[fit_cells], symbols, n_hvg=1000, must_keep=prior_regulators(PRIOR))
    s = snapshot_set_from_arrays(Xn, obs, symbols, time_key="time", replicate_key="replicate", condition_key="condition",
                                 gene_idx=sel, max_cells_per_group=2000, min_cells=200, rng=np.random.default_rng(0))
    t = s.biological_time if hasattr(s, "biological_time") else s.t
    t = np.asarray(t, dtype=float)
    observed = np.isin(t, d["obs_times"])
    Zg = Standardizer().fit(s.x[observed]).transform(s.x)
    rows = diagnose(Zg, t, d, "genes (1272)")

    from sklearn.decomposition import PCA
    fit = np.isin(t, d["fit_times"])
    P = PCA(n_components=50, svd_solver="randomized", random_state=0).fit(s.x[fit]).transform(s.x).astype(np.float32)
    Zp = Standardizer().fit(P[observed]).transform(P)
    rows += diagnose(Zp, t, d, "PCA-50")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    g = df.groupby(["space", "what"], sort=False)["value"].agg(["mean", "std"]).round(4)
    print(g.to_string())
    bench = {}
    for name, folder in (("genes (1272)", "outputs/eb_interp"), ("PCA-50", "outputs/eb_interp_pca50")):
        r = pd.read_csv(Path(folder) / "results.csv")
        bench[name] = r.groupby("model")[["energy_distance", "mean_error"]].mean().round(4).to_dict(orient="index")
    (OUT / "summary.json").write_text(json.dumps({"diagnosis": g.reset_index().to_dict(orient="records"),
                                                  "benchmark_for_reference": bench}, indent=1), encoding="utf-8")
    print("\nbenchmark (trained model) for reference:")
    for k, v in bench.items():
        print(k, {m: v[m] for m in ("flow_matching", "mean_interp", "ot_interp") if m in v})


if __name__ == "__main__":
    main()
