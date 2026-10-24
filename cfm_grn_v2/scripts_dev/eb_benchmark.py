"""Held-out-timepoint benchmark on the Moon et al. (2019) embryoid-body time course (real data).

Same protocol as the synthetic gate G1, same controls, same metrics: the middle time window is removed,
predicted from the previous one over the observed horizon, and scored against persistence, mean
interpolation, the McCann/OT interpolant, linear and MLP maps at the same horizon, a velocity replay and
the same model trained with shuffled time labels. Nested early stopping uses a second inner window
(`interpolation_design`'s own rule, not a choice made here); the final model is refit on every observed
window, as selected on the synthetic validation folds for version 0.1.9.

Genes are chosen on the FIT cells only: highly variable genes union the transcription factors of the
CollecTRI prior that are expressed, which keeps the regulatory phase open. Normalisation is per cell, so
it cannot carry information between cells or between windows. Nothing in the pipeline sees the held-out
window except as the target it is scored against.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/eb_benchmark.py [--quick] [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "src")
from cfm_grn_v2.data.real import load_collectri, normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.metrics import EvalProtocol
from cfm_grn_v2.evaluation.report import paired_difference, per_seed
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import interpolation_design, run_interpolation_benchmark

CACHE = r"C:\Users\Gabriele\data\rafm\processed\moon2019_eb.npz"
PRIOR = r"C:\Users\Gabriele\data\rafm\priors\collectri_human.tsv"
WINDOWS = {0.0: "days 0-3", 1.0: "days 6-9", 2.0: "days 12-15", 3.0: "days 18-21", 4.0: "days 24-27"}
BASE = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot")
FT = dict(marginal_iters=1000, marginal_batch=256, marginal_steps=8, marginal_eval_every=25, marginal_patience=10,
          marginal_prior=1.0, marginal_lr=3e-4, marginal_window=1.0)


def load_cache(path: str):
    z = np.load(path, allow_pickle=False)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    obs = pd.DataFrame({"time": z["time"], "sample": z["sample"].astype(str),
                        "replicate": "eb1", "condition": "EB"})
    return X, obs, z["gene_symbols"].astype(str).tolist(), json.loads(str(z["qc_json"]))


def prior_regulators(path: str) -> set:
    df = pd.read_csv(path, sep="\t")
    return set(df["source_genesymbol"].astype(str))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="small gene space and budget, to size the run")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--test-time", type=float, default=2.0)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--max-cells", type=int, default=2000)
    p.add_argument("--pca", type=int, default=0, help="robustness control: run in this many principal components, fitted on the fit cells only (0 = gene space, the primary task)")
    p.add_argument("--out", default="outputs/eb_interp")
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--prior", default=PRIOR)
    p.add_argument("--choice", default=None, help="choice.json written by select_config.py: its overrides are applied")
    p.add_argument("--min-detect", type=float, default=0.01,
                   help="a gene must be detected in this fraction of the fit cells (0 = the first real-data runs)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, obs, symbols, qc = load_cache(str(Path(a.cache).expanduser()))
    print(f"loaded {X.shape[0]} cells x {X.shape[1]} genes ({time.time() - t0:.0f}s)", flush=True)

    times = sorted(float(t) for t in np.unique(obs["time"]))
    d = interpolation_design(times, float(a.test_time))
    print(f"design: test {d['test_time']} ({WINDOWS[d['test_time']]}) from {d['t_prev']}, horizon {d['horizon']}, "
          f"bracketing {d['t_prev']}->{d['t_next']} observed; validation {d['val_time']} from {d['v_prev']}; "
          f"fit {d['fit_times']}", flush=True)

    Xn = normalize_log1p(X, keep_sparse=True)
    fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
    n_hvg = 300 if a.quick else a.n_hvg
    regs = prior_regulators(str(Path(a.prior).expanduser()))
    sel = select_genes(Xn[fit_cells], symbols, n_hvg=n_hvg, must_keep=regs, min_detect=a.min_detect)
    names = [symbols[i] for i in sel]
    n_tf = sum(1 for g in names if g in regs)
    print(f"genes: {len(sel)} selected on the {int(fit_cells.sum())} fit cells "
          f"({n_hvg} HVG union the expressed prior regulators), of which {n_tf} are prior regulators "
          f"({time.time() - t0:.0f}s)", flush=True)

    space = Xn
    space_names = symbols
    gene_idx = sel
    if a.pca > 0:
        from sklearn.decomposition import PCA
        dense = np.asarray(Xn[:, sel].todense(), dtype=np.float32)
        pca = PCA(n_components=int(a.pca), svd_solver="randomized", random_state=0)
        pca.fit(dense[fit_cells])                        # fitted on the fit cells only, like every other transform
        space = pca.transform(dense).astype(np.float32)
        space_names = [f"PC{i + 1}" for i in range(int(a.pca))]
        gene_idx = None
        print(f"PCA control: {a.pca} components, {pca.explained_variance_ratio_.sum() * 100:.1f} % of the variance "
              f"of the selected genes ({time.time() - t0:.0f}s)", flush=True)

    s = snapshot_set_from_arrays(space, obs, space_names, time_key="time", replicate_key="replicate",
                                 condition_key="condition", gene_idx=gene_idx, max_cells_per_group=a.max_cells,
                                 min_cells=200, rng=np.random.default_rng(0), name="moon2019_eb")
    print(f"snapshots: {s.n_cells} cells, {s.n_genes} genes, timepoints {s.timepoints().tolist()} "
          f"({time.time() - t0:.0f}s)", flush=True)

    try:
        if a.pca > 0:
            raise ValueError("prior coverage is reported for the gene space only")
        prior = load_collectri(Path(a.prior).expanduser(), names)
        cover = {"prior_edges_within_selected_genes": prior.n_edges, "prior_regulators_present": int(len(prior.regulators))}
    except ValueError as exc:
        cover = {"prior_error": str(exc)}
    print("prior coverage:", cover, flush=True)

    cfg = BASE.replace(**FT)
    overrides = {}
    if a.choice:
        overrides = json.loads(Path(a.choice).read_text(encoding="utf-8"))["overrides"]
        cfg = cfg.replace(**overrides)
        print(f"configuration chosen on the validation folds: {overrides or 'base (version 0.1.9)'}", flush=True)
    if a.quick:
        cfg = cfg.replace(n_iters=1200, eval_every=150, patience=4, marginal_iters=200, marginal_eval_every=25)
    proto = EvalProtocol(n_samples=1000, seed=0)
    seeds = list(range(a.seeds))
    print(f"space: {s.n_genes} dimensions ({'PCA control' if a.pca else 'gene space, primary task'})", flush=True)
    df = run_interpolation_benchmark(s, times, float(a.test_time), cfg, seeds, proto, "moon2019_eb",
                                     refit_on_all=True, shuffled_control=True,
                                     mlp_kwargs={"hidden": 128, "n_iters": 600 if a.quick else 2000}, verbose=True)
    df.to_csv(out / "results.csv", index=False)

    summary = per_seed(df).groupby("model")["energy_distance"].agg(["mean", "std", "count"]).sort_values("mean")
    print()
    print(summary.round(4).to_string(), flush=True)
    pairs = {}
    for ctrl in ("mean_interp", "ot_interp", "linear", "mlp", "persistence", "freefm_replay",
                 "flow_matching_shuffled_time"):
        if ctrl in set(df["model"]):
            pairs[f"flow_vs_{ctrl}"] = paired_difference(df, "moon2019_eb", "flow_matching", ctrl)
    print()
    for k, v in pairs.items():
        print(f"{k:36s} mean {v['mean']:+.4f}  CI95 [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  n={v['n']}", flush=True)
    null_q95 = float(per_seed(df).get("null_q95", pd.Series([np.nan])).mean())
    (out / "summary.json").write_text(json.dumps(
        {"summary": summary.reset_index().to_dict(orient="records"), "paired": pairs, "design": d,
         "genes": {"n_selected": int(len(sel)), "n_hvg_requested": n_hvg, "n_prior_regulators": n_tf, **cover},
         "qc": qc, "config": cfg.to_dict(), "null_q95": null_q95, "quick": bool(a.quick), "pca": int(a.pca), "min_detect": a.min_detect,
         "overrides": overrides,
         "max_cells_per_group": a.max_cells, "seconds": round(time.time() - t0, 1)},
        indent=1, default=float), encoding="utf-8")
    summary.round(4).to_csv(out / "summary.csv")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
