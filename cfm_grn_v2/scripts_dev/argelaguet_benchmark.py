"""Held-out-stage benchmark on the Argelaguet et al. (2022) mouse gastrulation time course (real data, HPC).

The wild-type time course E7.5, E7.75, E8.0, E8.5, E8.75 is used; the Brachyury CRISPR samples at E8.5 (knockout
and its matched wild type) are a perturbation experiment and are kept out of this task. Biological time is the
embryonic day in units of a quarter day (E7.5 -> 0, E7.75 -> 1, E8.0 -> 2, E8.5 -> 4, E8.75 -> 5), so the
unequal spacing of the stages is kept, which the duration clock handles natively.

Same protocol as the synthetic gate and the embryoid-body run, same controls: E8.0 is removed and predicted from
E7.75; the bracketing transition E7.75 -> E8.5 is observed; early stopping uses the inner stage chosen by the
design rule; the final model is refit on every observed stage. Genes are chosen on the fit cells only (most
variable genes union the expressed regulators of the mouse CollecTRI prior).

Replicates. Stages E7.5, E8.0, E8.5 and E8.75 have two biological replicates, E7.75 has one. Replicate labels do
not link embryos across stages, so the replicates of a stage are pooled into one population. What replicates DO
give is a biological yardstick: the energy distance between the two E8.0 replicates, in the same evaluation space,
is how far one embryo pool is from another at the very stage we predict. A prediction at that distance is as good
as a second biological sample; the technical noise floor (two halves of the same pooled target) is lower still.

Run on the HPC (from ~/cfm_grn_v2):  PYTHONPATH=src python scripts_dev/argelaguet_benchmark.py --seeds 5
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
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import load_collectri, normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.evaluation.report import paired_difference, per_seed
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.search import interpolation_design, run_interpolation_benchmark

BASE = TrainConfig(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250,
                   patience=8, clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot")
FT = dict(marginal_iters=1000, marginal_batch=256, marginal_steps=8, marginal_eval_every=25, marginal_patience=10,
          marginal_prior=1.0, marginal_lr=3e-4, marginal_window=1.0)
DAY0, UNIT = 7.5, 0.25


def load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    obs = pd.DataFrame({"sample": z["sample"].astype(str), "stage": z["stage"].astype(str),
                        "genotype": z["genotype"].astype(str), "e_day": z["e_day"]})
    return X, obs, z["gene_symbols"].astype(str).tolist(), json.loads(str(z["qc_json"]))


def wild_type_course(X, obs):
    """The wild-type time course (CRISPR samples dropped), with biological time in quarter days from E7.5 and the
    replicates of each stage pooled."""
    course = (~obs["sample"].str.contains("CRISPR")) & (obs["genotype"] == "WT")
    X, obs = X[course.to_numpy()], obs.loc[course].reset_index(drop=True)
    obs["time"] = (obs["e_day"] - DAY0) / UNIT
    obs["replicate"], obs["condition"] = "pool", "WT"
    return X, obs


def replicate_distances(Xn, obs, sel, fit_mask_std, stages, n=1000, seed=0) -> dict:
    """Energy distance between the two biological replicates of each stage that has two, in the evaluation space
    (standardised on the observed cells, as the benchmark does). This is the biological yardstick."""
    dense = np.asarray(Xn[:, sel].todense(), dtype=np.float32)
    std = Standardizer().fit(dense[fit_mask_std])
    Z = std.transform(dense)
    out = {}
    rng = np.random.default_rng(seed)
    for st in stages:
        reps = sorted(obs.loc[obs["stage"] == st, "sample"].unique())
        if len(reps) != 2:
            continue
        a = np.flatnonzero((obs["sample"] == reps[0]).to_numpy())
        b = np.flatnonzero((obs["sample"] == reps[1]).to_numpy())
        a = rng.choice(a, min(n, len(a)), replace=False)
        b = rng.choice(b, min(n, len(b)), replace=False)
        out[st] = {"replicates": reps, "energy_distance": float(energy_distance(Z[a], Z[b]))}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="~/rafm_data/processed/argelaguet2022_rna.npz")
    p.add_argument("--prior", default="~/rafm_data/priors/collectri_mouse.tsv")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--test-stage", default="E8.0")
    p.add_argument("--n-hvg", type=int, default=1000)
    p.add_argument("--max-cells", type=int, default=3000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="outputs/argelaguet_interp")
    p.add_argument("--choice", default=None, help="choice.json written by select_config.py: its overrides are applied")
    p.add_argument("--min-detect", type=float, default=0.01,
                   help="a gene must be detected in this fraction of the fit cells (0 = the first real-data run)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, obs, symbols, qc = load_cache(Path(a.cache).expanduser())
    X, obs = wild_type_course(X, obs)
    print(f"wild-type time course: {X.shape[0]} cells, stages "
          f"{obs.groupby('stage').size().to_dict()} ({time.time() - t0:.0f}s)", flush=True)

    times = sorted(float(t) for t in np.unique(obs["time"]))
    test_time = (dict(zip(obs["stage"], obs["time"])))[a.test_stage]
    d = interpolation_design(times, float(test_time))
    stage_of = dict(zip(obs["time"], obs["stage"]))
    print(f"design: test {stage_of[d['test_time']]} from {stage_of[d['t_prev']]} (horizon {d['horizon'] * UNIT:.2f} d); "
          f"bracketing {stage_of[d['t_prev']]} -> {stage_of[d['t_next']]} observed; validation "
          f"{stage_of[d['val_time']]} from {stage_of[d['v_prev']]}; fit {[stage_of[t] for t in d['fit_times']]}", flush=True)

    Xn = normalize_log1p(X, keep_sparse=True)
    fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
    obs_cells = obs["time"].isin(d["obs_times"]).to_numpy()
    regs = set(pd.read_csv(Path(a.prior).expanduser(), sep="\t")["source_genesymbol"].astype(str))
    n_hvg = 300 if a.quick else a.n_hvg
    sel = select_genes(Xn[fit_cells], symbols, n_hvg=n_hvg, must_keep=regs, min_detect=a.min_detect)
    names = [symbols[i] for i in sel]
    n_tf = sum(1 for g in names if g in regs)
    print(f"genes: {len(sel)} selected on the {int(fit_cells.sum())} fit cells ({n_hvg} HVG union the expressed "
          f"prior regulators), of which {n_tf} are prior regulators ({time.time() - t0:.0f}s)", flush=True)
    try:
        prior = load_collectri(Path(a.prior).expanduser(), names)
        cover = {"prior_edges_within_selected_genes": prior.n_edges, "prior_regulators_present": int(len(prior.regulators))}
    except ValueError as exc:
        cover = {"prior_error": str(exc)}
    print("prior coverage:", cover, flush=True)

    reps = replicate_distances(Xn, obs, sel, obs_cells, sorted(obs["stage"].unique()))
    print("replicate yardstick (energy distance between the two biological replicates):",
          {k: round(v["energy_distance"], 4) for k, v in reps.items()}, flush=True)

    s = snapshot_set_from_arrays(Xn, obs, symbols, time_key="time", replicate_key="replicate",
                                 condition_key="condition", gene_idx=sel, max_cells_per_group=a.max_cells,
                                 min_cells=200, rng=np.random.default_rng(0), name="argelaguet2022")
    print(f"snapshots: {s.n_cells} cells, {s.n_genes} genes, times {s.timepoints().tolist()} ({time.time() - t0:.0f}s)",
          flush=True)

    cfg = BASE.replace(**FT)
    overrides = {}
    if a.choice:
        overrides = json.loads(Path(a.choice).read_text(encoding="utf-8"))["overrides"]
        cfg = cfg.replace(**overrides)
        print(f"configuration chosen on the validation folds: {overrides or 'base (version 0.1.9)'}", flush=True)
    if a.quick:
        cfg = cfg.replace(n_iters=1200, eval_every=150, patience=4, marginal_iters=200)
    proto = EvalProtocol(n_samples=1000, seed=0)
    df = run_interpolation_benchmark(s, times, float(test_time), cfg, list(range(a.seeds)), proto, "argelaguet2022",
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
            pairs[f"flow_vs_{ctrl}"] = paired_difference(df, "argelaguet2022", "flow_matching", ctrl)
    print()
    for k, v in pairs.items():
        print(f"{k:36s} mean {v['mean']:+.4f}  CI95 [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  n={v['n']}", flush=True)
    null_q95 = float(per_seed(df).loc[lambda x: x["model"] == "flow_matching", "null_q95"].mean())
    (out / "summary.json").write_text(json.dumps(
        {"summary": summary.reset_index().to_dict(orient="records"), "paired": pairs, "design": d,
         "stage_of_time": {str(k): v for k, v in stage_of.items()}, "replicate_yardstick": reps,
         "genes": {"n_selected": int(len(sel)), "n_hvg_requested": n_hvg, "n_prior_regulators": n_tf, **cover},
         "qc": qc, "config": cfg.to_dict(), "null_q95": null_q95, "quick": bool(a.quick),
         "overrides": overrides, "min_detect": a.min_detect, "max_cells_per_group": a.max_cells, "seconds": round(time.time() - t0, 1)},
        indent=1, default=float), encoding="utf-8")
    summary.round(4).to_csv(out / "summary.csv")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
