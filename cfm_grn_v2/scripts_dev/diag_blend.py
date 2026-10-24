"""Why does mean interpolation beat the flow model on the gastrulation series? A diagnosis without training.

Version 0.1.10 on Argelaguet (E8.0 from E7.75): flow 2.03, McCann/OT interpolant 1.99, mean interpolation 1.51.
The flow model and the OT interpolant tie, and both build the held-out population the same way: each predicted
cell is a convex combination of a real cell at one stage and its OT partner at the next. Mean interpolation
instead moves REAL cells rigidly. Hypothesis ("blending"): in a sparse, heterogeneous gene space a convex
combination of two different cells is not a cell. It has fewer zeros, less variance and sits off the data
manifold, so a cloud of blends has the wrong shape even when its centre is right.

Measured here, per fold, for every construction that needs no network:
  persistence, mean_interp, ot_interp, path_linear / path_spline (the spline path family the field is trained
  on, linear or spline centre), and ot_interp / path_linear with Brownian-bridge noise of scale sigma added
  (variance sigma^2 * f * (1 - f) at fraction f of the gap, zero at the observed stages: the bridge that an
  entropic / Schrodinger-bridge coupling would put there);
statistics against the held-out population, in the benchmark's evaluation space:
  ED            energy distance (U-statistic, 1000 cells a side)
  mean_error    L2 error of the predicted mean
  var_ratio     total variance predicted / held-out (1 = right spread; < 1 = contracted)
  nn_ratio      median distance from a predicted cell to its nearest held-out cell, divided by the same for one
                half of the held-out cells against the other half (1 = as close to the data as real cells)

Folds. "validation": the test stage is REMOVED from the data and the inner remaining stages are the folds, the
same folds select_config.py uses; anything chosen from this script (sigma) is chosen here. "test": the reported
fold, full data, labelled as a diagnosis of a measured result, never used to choose anything.

  PYTHONPATH=src python scripts_dev/diag_blend.py --dataset eb --cache <npz> --prior <tsv> --out outputs/diag_blend_eb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.metrics import energy_distance
from cfm_grn_v2.flows.coupling import chain_ot, ot_coupling
from cfm_grn_v2.flows.path import SplinePath
from cfm_grn_v2.training.search import interpolation_design
from select_config import load_dataset

SIGMAS = (0.25, 0.5, 1.0)


def sub(x, n, rng):
    return x[rng.choice(len(x), min(n, len(x)), replace=False)]


def stats(pred, truth, seed):
    # the energy distance uses exactly the draws of EvalProtocol(n_samples=1000, seed=0).evaluate(pred, truth,
    # seed_offset=seed), so a deterministic construction (persistence, mean_interp) reproduces the benchmark's number
    r = np.random.default_rng(seed)
    ed = float(energy_distance(sub(pred, 1000, r), sub(truth, 1000, r)))
    r = np.random.default_rng(100 + seed)
    p, t = sub(pred, 1000, r), sub(truth, 2000, r)
    ta, tb = t[: len(t) // 2], t[len(t) // 2:]
    nn = NearestNeighbors(n_neighbors=1).fit(tb)
    d_pred = np.median(nn.kneighbors(p[: len(ta)])[0])
    d_real = np.median(nn.kneighbors(ta)[0])
    return {"ED": ed,
            "mean_error": float(np.linalg.norm(pred.mean(0) - truth.mean(0))),
            "var_ratio": float(pred.var(0).sum() / truth.var(0).sum()),
            "nn_ratio": float(d_pred / d_real)}


def fold_rows(Xn, obs, symbols, regs, times, fold, max_cells, min_detect, n_hvg, label, seeds):
    d = interpolation_design(times, fold)
    fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
    sel = select_genes(Xn[fit_cells], symbols, n_hvg=n_hvg, must_keep=regs, min_detect=min_detect)
    s = snapshot_set_from_arrays(Xn, obs, symbols, time_key="time", replicate_key="replicate",
                                 condition_key="condition", gene_idx=sel, max_cells_per_group=max_cells,
                                 min_cells=200, rng=np.random.default_rng(0), name="diag_blend")
    tt = s.biological_time
    observed = np.isin(tt, d["obs_times"])
    std = Standardizer().fit(s.x[observed])
    x = std.transform(s.x)
    src, nxt, truth = x[tt == d["t_prev"]], x[tt == d["t_next"]], x[tt == d["test_time"]]
    f = d["frac"]
    rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        preds = {"persistence": src, "mean_interp": src + f * (nxt.mean(0) - src.mean(0))}
        i0, i1 = ot_coupling(src, nxt, rng, max_size=len(src))
        preds["ot_interp"] = src[i0] + f * (nxt[i1] - src[i0])
        traj = chain_ot([x[tt == w] for w in d["obs_times"]], rng, 1000, mode="ot")
        for center in ("linear", "spline"):
            path = SplinePath(d["obs_times"], traj, center=center)
            preds[f"path_{center}"], _ = path.evaluate(np.full(path.n, d["test_time"]), np.arange(path.n))
        # second-moment correction: keep each construction's centre and shape, rescale every gene's spread to the
        # straight-line interpolation of the per-gene standard deviation between the bracketing stages
        s_target = (1.0 - f) * src.std(0) + f * nxt.std(0)
        for base in ("ot_interp", "path_linear"):
            pr = preds[base]
            m = pr.mean(0)
            preds[f"{base}+scale"] = m + (pr - m) * (s_target / np.maximum(pr.std(0), 1e-6))
        for sg in SIGMAS:
            noise_sd = sg * np.sqrt(f * (1.0 - f))
            for base in ("ot_interp", "path_linear"):
                preds[f"{base}+bridge{sg}"] = preds[base] + noise_sd * rng.standard_normal(preds[base].shape
                                                                                           ).astype(np.float32)
        for name, pr in preds.items():
            rows.append({"split": label, "fold": fold, "seed": seed, "construction": name, "n_genes": len(sel),
                         **stats(np.asarray(pr, dtype=np.float32), truth, seed)})
        print(f"  {label} fold {fold} seed {seed}: " + ", ".join(
            f"{r['construction']} {r['ED']:.3f}" for r in rows if r["seed"] == seed and r["fold"] == fold
            and r["split"] == label and "bridge" not in r["construction"]), flush=True)
        print(f"  {label} fold {fold} seed {seed}: t_prev/t_next spread ratio "
              f"{float(nxt.var(0).sum() / src.var(0).sum()):.3f}, held-out/t_prev {float(truth.var(0).sum() / src.var(0).sum()):.3f}",
              flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("eb", "argelaguet"), required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--min-detect", type=float, default=0.01)
    p.add_argument("--n-hvg", type=int, default=1000)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, obs, symbols, regs, test_time, max_cells, _ = load_dataset(a.dataset, a.cache, a.prior)
    Xn_all = normalize_log1p(X, keep_sparse=True)
    times_all = sorted(float(v) for v in np.unique(obs["time"]))

    keep = (obs["time"] != test_time).to_numpy()
    obs_v = obs.loc[keep].reset_index(drop=True)
    times_v = sorted(float(v) for v in np.unique(obs_v["time"]))
    rows = []
    for fold in times_v[1:-1]:
        rows += fold_rows(Xn_all[keep], obs_v, symbols, regs, times_v, fold, max_cells, a.min_detect, a.n_hvg,
                          "validation", a.seeds)
    rows += fold_rows(Xn_all, obs, symbols, regs, times_all, test_time, max_cells, a.min_detect, a.n_hvg,
                      "test (diagnosis only)", a.seeds)
    df = pd.DataFrame(rows)
    df.to_csv(out / "blend.csv", index=False)
    tab = df.groupby(["split", "fold", "construction"])[["ED", "mean_error", "var_ratio", "nn_ratio"]
                                                         ].mean().round(4)
    print()
    print(tab.to_string())
    val = df[df.split == "validation"]
    best = {}
    for base in ("ot_interp", "path_linear"):
        cand = val[val.construction.str.fullmatch(base + r"(\+bridge.*|\+scale)?")]
        m = cand.groupby("construction").ED.mean()
        best[base] = {"validation_ED": m.round(4).to_dict(), "best": m.idxmin()}
    (out / "summary.json").write_text(json.dumps({"dataset": a.dataset, "sigmas": SIGMAS, "best_on_validation": best,
                                                  "table": tab.reset_index().to_dict(orient="records")},
                                                 indent=1, default=float), encoding="utf-8")
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
