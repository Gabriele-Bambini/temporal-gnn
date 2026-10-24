"""Version 0.1.10 selection on a real data set, with the TEST stage REMOVED from the data.

Why. The diagnosis of the embryoid-body result (scripts_dev/eb_diagnose.py, fold t = 2) found two separate failures:
  - centre: the spline path's mean is a cubic spline through the window means, and it lands further from the
    held-out window than a straight line between its neighbours (3.02 vs 2.69 in genes);
  - fit:    on real data the network does not reproduce its training target (target 0.160, model 0.278 in
    genes), while on the 40-gene simulator it did to 0.005; a 128-unit field for ~1 300 genes is a suspect.
Both corrections are chosen HERE, on validation folds of a data set from which the test stage has been deleted,
and then measured once on it. The diagnosis looked at the test fold; the choice must not.

Data sets.  eb          embryoid body (Moon et al. 2019), test window 2 (days 12-15); remaining windows 0, 1, 3, 4
            argelaguet  mouse gastrulation (Argelaguet et al. 2022), test E8.0 (time 2); remaining times 0, 1, 4, 5
Folds are the inner remaining times (1 and 3, or 1 and 4). Each fold has its own nested design, gene selection on
its own fit cells (same rule as the benchmark, detection filter included), early stopping, and refit on its
observed stages, exactly as the benchmark does.

Arm sets (--arms).
  v0110  base         version 0.1.9 as published (hidden 128, spline centre)
         center       linear centre (SplinePath center="linear")
         wide         hidden 512
         wide_center  both
  v0111  base         version 0.1.10 as chosen on both data sets (linear centre)
         time         + the field sees the biological time (field_time="bio")
         noft         + no marginal fine-tune stage (marginal_iters=0), a diagnostic of the second stage
         Why: on the embryoid body the path family alone reaches 0.126 at the held-out window and the trained
         model 0.267 (scripts_dev/diag_blend.py); an autonomous field can only average the path velocities
         where the population passes the same state at different times.

Decision rule, fixed before the run. For every arm other than base, compute the paired change of the energy
distance against base over the (fold, seed) pairs. An arm qualifies if it improves on base in at least 3 of the 4
pairs and on average. The winner is the qualifying arm with the most negative mean change; if no arm qualifies,
base is kept. The rule is applied by code and written to choice.json, which the final measurement reads.

Also recorded, without a network: the ED of the path family itself (spline vs linear centre) at each fold's held-
out stage, so a failure of the field can be told apart from a failure of the construction.

Run from cfm_grn_v2/:  PYTHONPATH=src python scripts_dev/select_config.py --dataset eb --cache <npz> --prior <tsv>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts_dev")
from cfm_grn_v2.data.preprocessing import Standardizer
from cfm_grn_v2.data.real import normalize_log1p, select_genes, snapshot_set_from_arrays
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance, median_bandwidth
from cfm_grn_v2.flows.coupling import chain_ot
from cfm_grn_v2.flows.path import SplinePath
from cfm_grn_v2.training.search import interpolation_design
from cfm_grn_v2.training.trainer import task_groups, time_task, train_flow

ARM_SETS = {
    "v0110": {"base": {}, "center": {"path_center": "linear"}, "wide": {"hidden": 512},
              "wide_center": {"hidden": 512, "path_center": "linear"}},
    "v0111": {"base": {"path_center": "linear"}, "time": {"path_center": "linear", "field_time": "bio"},
              "noft": {"path_center": "linear", "marginal_iters": 0}},
}


def load_dataset(name: str, cache: str, prior: str):
    """(X, obs with time/replicate/condition, gene symbols, prior regulators, test time, cells per group, config)"""
    cache, prior = str(Path(cache).expanduser()), str(Path(prior).expanduser())
    if name == "eb":
        import eb_benchmark as B
        X, obs, symbols, _ = B.load_cache(cache)
        return X, obs, symbols, B.prior_regulators(prior), 2.0, 2000, B.BASE.replace(**B.FT)
    import argelaguet_benchmark as B
    X, obs, symbols, _ = B.load_cache(Path(cache))
    X, obs = B.wild_type_course(X, obs)
    regs = set(pd.read_csv(prior, sep="\t")["source_genesymbol"].astype(str))
    return X, obs, symbols, regs, 2.0, 3000, B.BASE.replace(**B.FT)


def decide(df: pd.DataFrame, arms: dict) -> dict:
    base = df[df.arm == "base"].set_index(["fold", "seed"]).energy_distance
    out = {}
    for arm in [a for a in arms if a != "base"]:
        sub = df[df.arm == arm].set_index(["fold", "seed"]).energy_distance
        dlt = (sub - base.reindex(sub.index)).dropna()
        out[arm] = {"mean_change": float(dlt.mean()), "n_better": int((dlt < 0).sum()), "n": int(len(dlt)),
                    "changes": [float(v) for v in dlt]}
    qualifying = {a: v for a, v in out.items() if v["n"] >= 4 and v["n_better"] >= 3 and v["mean_change"] < 0}
    winner = min(qualifying, key=lambda a: qualifying[a]["mean_change"]) if qualifying else "base"
    return {"rule": "improve on base in >= 3 of 4 (fold, seed) pairs and on average; most negative mean change; "
                    "else base", "per_arm": out, "winner": winner, "overrides": arms[winner]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("eb", "argelaguet"), required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--n-hvg", type=int, default=1000)
    p.add_argument("--min-detect", type=float, default=0.01)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true", help="tiny budget to check the script runs end to end")
    p.add_argument("--arms", choices=sorted(ARM_SETS), default="v0110")
    a = p.parse_args()
    arms = ARM_SETS[a.arms]
    X, obs, symbols, regs, test_time, max_cells, base_cfg = load_dataset(a.dataset, a.cache, a.prior)
    smoke = dict(n_iters=300, eval_every=100, patience=2, marginal_iters=40, marginal_eval_every=20) if a.smoke else {}
    if a.smoke:
        a.n_hvg, a.seeds, max_cells = 150, 2, 400
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    keep = (obs["time"] != test_time).to_numpy()
    X, obs = X[keep], obs.loc[keep].reset_index(drop=True)
    times = sorted(float(v) for v in np.unique(obs["time"]))
    assert test_time not in times
    folds = times[1:-1]
    print(f"{a.dataset}: test time {test_time} removed: {X.shape[0]} cells, times {times}, folds {folds}", flush=True)
    Xn = normalize_log1p(X, keep_sparse=True)

    rows, ceil_rows = [], []
    for fold in folds:
        d = interpolation_design(times, fold)
        fit_cells = obs["time"].isin(d["fit_times"]).to_numpy()
        sel = select_genes(Xn[fit_cells], symbols, n_hvg=a.n_hvg, must_keep=regs, min_detect=a.min_detect)
        s = snapshot_set_from_arrays(Xn, obs, symbols, time_key="time", replicate_key="replicate",
                                     condition_key="condition", gene_idx=sel, max_cells_per_group=max_cells,
                                     min_cells=200, rng=np.random.default_rng(0), name=f"{a.dataset}_select")
        fit_mask, all_mask = s.mask(times=d["fit_times"]), s.mask(times=d["obs_times"])
        val_task, task = time_task(s, d["v_prev"], d["val_time"]), time_task(s, d["t_prev"], d["test_time"])
        eval_std = Standardizer().fit(s.x[all_mask])
        x_eval = eval_std.transform(s.x)
        proto = EvalProtocol(n_samples=1000, seed=0, bandwidth=median_bandwidth(x_eval[all_mask]))
        groups = task_groups(s, task)
        print(f"\n=== fold {fold}: predicted from {d['t_prev']}, bracketing {d['t_prev']}->{d['t_next']}; "
              f"validation {d['val_time']}; fit {d['fit_times']}; {len(sel)} genes ({time.time() - t0:.0f}s) ===",
              flush=True)

        tt = s.biological_time
        truth = x_eval[tt == d["test_time"]]
        for center in ("spline", "linear"):
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                traj = chain_ot([x_eval[tt == w] for w in d["obs_times"]], rng, 1000, mode="ot")
                path = SplinePath(d["obs_times"], traj, center=center)
                xp, _ = path.evaluate(np.full(path.n, d["test_time"]), np.arange(path.n))
                er = np.random.default_rng(100 + seed)
                e = float(energy_distance(xp[er.choice(len(xp), min(1000, len(xp)), replace=False)],
                                          truth[er.choice(len(truth), min(1000, len(truth)), replace=False)]))
                ceil_rows.append({"fold": fold, "center": center, "seed": seed, "path_energy_distance": e})
        print("  path family, no network:", pd.DataFrame([r for r in ceil_rows if r["fold"] == fold])
              .groupby("center").path_energy_distance.mean().round(4).to_dict(), flush=True)

        for seed in range(a.seeds):
            for arm, over in arms.items():
                cfg = base_cfg.replace(seed=seed, **{**smoke, **over})       # the arm wins over the smoke budget
                t1 = time.time()
                model, _ = train_flow(s, fit_mask, d["fit_times"], cfg, val_task, proto)
                n_iters = max(int(model.best_iter), cfg.eval_every)
                model, _ = train_flow(s, all_mask, d["obs_times"], cfg.replace(n_iters=n_iters), None, proto)
                res = []
                for cond, rep, m0, m1 in groups:
                    pred = eval_std.transform(model.predict(s.x[m0], cond, d["t_prev"], d["test_time"]))
                    res.append(proto.evaluate(pred, x_eval[m1], seed_offset=seed))
                r = {k: float(np.mean([x[k] for x in res])) for k in ("energy_distance", "mean_error")}
                rows.append({"fold": fold, "seed": seed, "arm": arm, **r, "best_iter": int(model.best_iter),
                             "n_genes": int(len(sel)), "seconds": round(time.time() - t1, 1)})
                print(f"  seed {seed}  {arm:12s} ED {r['energy_distance']:.4f}  mean_err {r['mean_error']:.3f}  "
                      f"({time.time() - t1:.0f}s, total {(time.time() - t0) / 60:.0f} min)", flush=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(ceil_rows).to_csv(out / "path_ceiling.csv", index=False)
    choice = {"dataset": a.dataset, "arm_set": a.arms, "test_time_removed": test_time, "folds": folds,
              "min_detect": a.min_detect, "smoke": bool(a.smoke), **decide(df, arms)}
    (out / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    print()
    print(df.pivot_table(index="arm", columns="fold", values="energy_distance", aggfunc="mean").round(4).to_string())
    print(json.dumps(choice, indent=1))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
