"""Strict rolling-origin evaluation of the gene-level sink / source OT forecaster (uot_core) on Dileo Task 2.

Transition i maps snapshot i -> i+1 (gastrulation-human bins 2..31, 29 transitions; pancreas-human bins 0..14, 14)
and reads snapshots 0..i only. Couplings are between snapshot i-Le and i with Le = min(L, i).
Families (each selected per transition by rolling origin on PAST transitions only: mean dist_bridge.criterion J =
(Pearson + tie-aware P@K up + down) / 3 on all genes, K = 200 N / n_test, over transitions max(1, i-8) .. i-1,
targets rebuilt from bins <= i; ties within 1e-3 go to the earlier config in the listed order, which puts larger
lam0 first; transition i = 1 has no past transition and uses the a-priori default):
    uot          lam0 in LAMS x L in {1, 3} x delta in DELTAS            default (1, 1, 1)
    uot_forced   lam0 = 128 only (sink / source carry only the forced |f_A - f_B|; same extrapolation rule)
    uot_lam<x>   one lam0, L and delta selected (lam0 profile)
    bal          balanced OT on normalised positive parts + log-linear nonzero-fraction trend (dist_bridge
                 coupling / sb_step, count scale), L x delta                default (1, 1)
    persist      delta = 0: today's law + the same observation head (no transport)
    trend        repeat last change y_{i-1}
Targets: l1 (expected value under the forecast law, n_i cells, today's library and the forecast library), linear
(forecast mean - m_i), orig (dist_bridge.expected_target finite-cell head), l1_plugin (plug-in log1p CP10k change).
Scoring: dileo_rescore.metrics on a fixed random 25 % of genes per transition (np.random.default_rng(1000 s + i)
.permutation(G)[:n_test], s in 0, 1, 2), mean over transitions (i >= 1 and i >= 2) then over the 3 mask seeds.
Usage: python run_uot.py <dataset> [workers]  ->  results_<dataset>.json, sel_preds_<dataset>.npz
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import concurrent.futures as cf  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uot_core as U  # noqa: E402

sys.path.insert(0, U.HPC)
import dileo_protocol  # noqa: E402
import dileo_rescore  # noqa: E402

LAMS = (128.0, 4.0, 1.0, 0.25, 0.0625)       # larger price first (tie preference)
LS = (1, 3)
DELTAS = (1.0, 0.5, 2.0)
TARGETS = ("l1", "linear", "orig", "l1_plugin")
DS_ID = {"gastrulation-human": 1, "pancreas-human": 2}
KIND_ID = {"uot": 1, "bal": 2, "persist": 3}
WINDOW, TIE = 8, 1e-3
SCRATCH = os.environ.get("UOT_SCRATCH", r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad")
SEEDS = (0, 1, 2)
CONTEXT = {  # numbers to compare with (task text; strict, i >= 1 unless noted), PCC / P@200 up / P@200 down
    "gastrulation-human": {"l1": {"trend": [-0.169, 0.102, 0.092], "DistBridge": [0.137, 0.221, 0.216],
                                  "NASS_i>=2": [0.335, 0.276, 0.290], "theirs_corrected_best": [0.104, 0.140, 0.147]},
                           "linear": {"trend": [0.177, 0.220, 0.195], "DistBridge": [0.299, 0.285, 0.243]}},
    "pancreas-human": {"l1": {"trend": [0.159, 0.223, 0.220], "DistBridge": [0.106, 0.177, 0.215],
                              "NASS_i>=2": [0.308, 0.221, 0.283], "theirs_corrected_best": [0.094, 0.088, 0.235]},
                       "linear": {"trend": [0.377, 0.291, 0.360], "DistBridge": [0.304, 0.267, 0.314]}},
}

_DATA = {}


def _get(name):
    if name not in _DATA:
        _DATA[name] = U.load(name)
    return _DATA[name]


def job(args):
    """One coupling (kind, transition i, lag Le, lam0) and the forecasts / heads for every delta."""
    name, kind, i, Le, lam0 = args
    warnings.filterwarnings("ignore")
    X, trip, n, m = _get(name)
    B = trip[i]
    t0 = time.time()
    diag, err, over, iters = {}, 0.0, 0.0, 0
    if kind == "persist":
        steps = {0.0: B.copy()}
    else:
        A = trip[i - Le]
        if kind == "uot":
            cp = U.uot_coupling(A, B, Le, lam0)
            diag = U.uot_diag(cp)
            steps = {d: U.uot_step(B, cp, Le, d) for d in DELTAS}
        else:
            cp = U.bal_coupling(A, B, Le)
            steps = {d: U.bal_step(B, cp, Le, d) for d in DELTAS}
        err, over, iters = cp["err"], cp.get("over", 0.0), cp["iters"]
    out = {}
    for d, H in steps.items():
        seed = [DS_ID[name], KIND_ID[kind], i, Le, int(round(lam0 * 1e4)), int(round(d * 100))]
        out[d] = U.heads(H, m[i], n[i], seed, TARGETS)
        out[d]["fhat_mean"] = float(H[:, 1].mean())
    if diag:
        diag["log_n_ratio"] = float(np.log(n[i] / n[i - Le]))
    return args, out, diag, float(err), float(over), int(iters), time.time() - t0


def masks(G, T):
    n_test = int(round(0.25 * G))
    return {s: [np.sort(np.random.default_rng(1000 * s + i).permutation(G)[:n_test]) for i in range(T)] for s in SEEDS}


def main():
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    X, trip, n, m = _get(name)
    T, G = len(trip) - 1, trip[0].shape[0]
    n_test = int(round(0.25 * G))
    q = 200.0 / n_test
    print(f"== {name}: G={G} T={T} n_test={n_test} cells/bin={np.round(n).astype(int).tolist()}", flush=True)
    Y = {tg: np.stack([dileo_protocol.TARGET_NP["l1" if tg == "l1_plugin" else tg](m[i], m[i + 1]).astype(np.float64)
                       for i in range(T)]) for tg in TARGETS}
    # ---- jobs (unique couplings; L = 3 at i < 3 shares the Le = i coupling)
    jobs = set()
    for i in range(1, T):
        jobs.add((name, "persist", i, 0, 0.0))
        for L in LS:
            Le = min(L, i)
            jobs.add((name, "bal", i, Le, 0.0))
            for lam in LAMS:
                jobs.add((name, "uot", i, Le, lam))
    jobs = sorted(jobs, key=lambda a: (-a[2], a[1], a[3], a[4]))
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    res = {}
    t0 = time.time()
    cache = os.path.join(SCRATCH, f"uot_res_{name}.pkl")
    if "--reuse" in sys.argv and os.path.exists(cache):
        with open(cache, "rb") as fh:
            res = pickle.load(fh)
        print(f"reused {cache}", flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for k, r in enumerate(ex.map(job, jobs, chunksize=1)):
                res[r[0]] = r[1:]
                if (k + 1) % 25 == 0 or k + 1 == len(jobs):
                    print(f"  {k + 1}/{len(jobs)} done, {time.time() - t0:.0f}s elapsed", flush=True)
        os.makedirs(SCRATCH, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(res, fh)
    # ---- prediction tables per config: P[cfg][tg] (T, G), row 0 = NaN (never scored, never used in selection)
    def blank():
        return {tg: np.full((T, G), np.nan, np.float32) for tg in TARGETS}
    P, cfgs = {}, {"uot": [], "bal": [], "persist": [("persist",)]}
    for lam in LAMS:
        for L in LS:
            for d in DELTAS:
                cfgs["uot"].append(("uot", lam, L, d))
    for L in LS:
        for d in DELTAS:
            cfgs["bal"].append(("bal", L, d))
    for c in cfgs["uot"] + cfgs["bal"] + cfgs["persist"]:
        P[c] = blank()
    for i in range(1, T):
        for tg in TARGETS:
            P[("persist",)][tg][i] = res[(name, "persist", i, 0, 0.0)][0][0.0][tg]
        for L in LS:
            Le = min(L, i)
            for d in DELTAS:
                for tg in TARGETS:
                    P[("bal", L, d)][tg][i] = res[(name, "bal", i, Le, 0.0)][0][d][tg]
                    for lam in LAMS:
                        P[("uot", lam, L, d)][tg][i] = res[(name, "uot", i, Le, lam)][0][d][tg]
    trend = blank()
    for tg in TARGETS:
        trend[tg][1:] = Y[tg][:-1]
    # ---- selection (strict rolling origin)
    Jall = {c: {tg: np.array([np.nan] + [U.DB.criterion(Y[tg][j], P[c][tg][j], q) for j in range(1, T)])
                for tg in TARGETS} for c in P}
    fams = {"uot": (cfgs["uot"], ("uot", 1.0, 1, 1.0)),
            "uot_forced": ([c for c in cfgs["uot"] if c[1] == 128.0], ("uot", 128.0, 1, 1.0)),
            "bal": (cfgs["bal"], ("bal", 1, 1.0))}
    for lam in LAMS:
        fams[f"uot_lam{lam:g}"] = ([c for c in cfgs["uot"] if c[1] == lam], ("uot", lam, 1, 1.0))
    sel_pred, chosen = {}, {}
    for fam, (cl, default) in fams.items():
        sel_pred[fam], chosen[fam] = blank(), {}
        for tg in TARGETS:
            chosen[fam][tg] = {}
            for i in range(1, T):
                if i == 1:
                    c = default
                else:
                    W = list(range(max(1, i - WINDOW), i))
                    sc = np.array([np.mean(Jall[cc][tg][W]) for cc in cl])
                    c = cl[int(np.where(sc >= sc.max() - TIE)[0][0])]
                sel_pred[fam][tg][i] = P[c][tg][i]
                chosen[fam][tg][i] = list(c)
    sel_pred["persist"] = P[("persist",)]
    sel_pred["trend"] = trend
    # ---- scoring on fixed masks
    mk = masks(G, T)
    keys = ("pearson", "p_ta_up", "p_ta_down", "p_at_200_up", "p_at_200_down", "spearman")

    def per_trans(pred, tg):
        """(seeds, T, keys) metrics; NaN at i = 0."""
        out = np.full((len(SEEDS), T, len(keys)), np.nan)
        for si, s in enumerate(SEEDS):
            for i in range(1, T):
                t = mk[s][i]
                r = dileo_rescore.metrics(Y[tg][i][t], pred[tg][i][t])
                out[si, i] = [r[k] for k in keys]
        return out

    def summ(S, start):
        per_seed = np.nanmean(S[:, start:], axis=1)
        return {k: round(float(per_seed[:, j].mean()), 4) for j, k in enumerate(keys)} | \
            {"pearson_sd_over_seeds": round(float(per_seed[:, 0].std()), 4)}

    scores, S_all = {}, {}
    for fam, pred in sel_pred.items():
        scores[fam], S_all[fam] = {}, {}
        for tg in TARGETS:
            S = per_trans(pred, tg)
            S_all[fam][tg] = S
            scores[fam][tg] = {"i>=1": summ(S, 1), "i>=2": summ(S, 2)}
            with np.errstate(invalid="ignore", divide="ignore"):
                ag = [np.corrcoef(Y[tg][i], pred[tg][i])[0, 1] if np.std(pred[tg][i]) > 0 else np.nan for i in range(1, T)]
            scores[fam][tg]["allgenes_pcc_i>=1"] = round(float(np.nanmean(ag)), 4)  # mask-free robustness check
    # every fixed config (profile; the best one in hindsight is NOT strict)
    cfg_scores = {}
    for c in P:
        cfg_scores["/".join(map(str, c))] = {}
        for tg in ("l1", "linear"):
            S = per_trans(P[c], tg)
            cfg_scores["/".join(map(str, c))][tg] = {"i>=1": summ(S, 1), "J_i>=1": round(float(np.nanmean(
                np.nanmean(S[:, 1:, 0] + S[:, 1:, 1] + S[:, 1:, 2], axis=1))), 4)}
    hindsight = {}
    for fam in ("uot", "bal"):
        for tg in ("l1", "linear"):
            best = max(cfgs[fam], key=lambda c: cfg_scores["/".join(map(str, c))][tg]["J_i>=1"])
            hindsight[f"{fam}/{tg}"] = {"config": list(best), **cfg_scores["/".join(map(str, best))][tg]}
    # ---- paired comparisons over transitions (mask-averaged per transition)
    def paired(fa, fb, tg, start=1):
        A, B = S_all[fa][tg][:, start:], S_all[fb][tg][:, start:]
        out = {}
        for nm, f in (("pcc", lambda S: S[:, :, 0]), ("J", lambda S: S[:, :, 0] + S[:, :, 1] + S[:, :, 2])):
            d = np.nanmean(f(A), 0) - np.nanmean(f(B), 0)
            d = d[np.isfinite(d)]
            p = float(stats.wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else float("nan")
            out[nm] = {"mean_diff": round(float(d.mean()), 4), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                       "wilcoxon_p": round(p, 4)}
        return out
    pairs = {}
    for tg in TARGETS:
        pairs[tg] = {f"{a} vs {b}": paired(a, b, tg) for a, b in
                     (("uot", "bal"), ("uot", "uot_forced"), ("uot_forced", "bal"), ("uot", "trend"),
                      ("uot", "persist"), ("bal", "trend"), ("persist", "trend"))}
    # ---- sink / source diagnostics per (lam0, L) over transitions i >= 1 (couplings with Le = L)
    diag = {}
    for lam in LAMS:
        for L in LS:
            rows = [res[(name, "uot", i, L, lam)][1] for i in range(L, T) if (name, "uot", i, L, lam) in res]
            rows = [r for r in rows if r]
            if not rows:
                continue
            dd = {}
            for k in rows[0]:
                v = np.array([r[k] for r in rows], float)
                dd[k] = np.round(np.nanmean(v, 0), 4).tolist()
            lr = np.array([r["log_n_ratio"] for r in rows])
            for k in ("sink_over_fA", "source_over_fB", "excess_death_over_fA", "excess_birth_over_fB"):
                v = np.array([r[k] for r in rows])
                dd[f"corr_{k}_vs_log_n_ratio"] = round(float(np.corrcoef(v, lr)[0, 1]), 3) if len(v) > 2 else None
            dd["per_transition_sink_over_fA"] = [round(r["sink_over_fA"], 4) for r in rows]
            dd["per_transition_excess_death"] = [round(r["excess_death_over_fA"], 4) for r in rows]
            dd["per_transition_log_n_ratio"] = [round(r["log_n_ratio"], 3) for r in rows]
            diag[f"lam{lam:g}_L{L}"] = dd
    conv = {"uot_max_potential_move_over_eps": max(v[2] for k, v in res.items() if k[1] == "uot"),
            "uot_max_marginal_overshoot": max(v[3] for k, v in res.items() if k[1] == "uot"),
            "bal_max_row_err": max(v[2] for k, v in res.items() if k[1] == "bal"),
            "seconds_per_job_mean": round(float(np.mean([v[5] for v in res.values()])), 1)}
    out = {"dataset": name, "n_genes": G, "n_transitions": T, "n_test": n_test, "cells_per_bin": n.round(1).tolist(),
           "grid": {"lam0": LAMS, "L": LS, "delta": DELTAS, "window": WINDOW},
           "scores": scores, "paired": pairs, "hindsight_best_fixed_config_NOT_STRICT": hindsight,
           "config_profile": cfg_scores, "selected": chosen, "sink_diagnostics": diag, "convergence": conv,
           "context_numbers": CONTEXT[name]}
    with open(os.path.join(HERE, f"results_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(os.path.join(HERE, f"sel_preds_{name}.npz"),
                        **{f"{fam}__{tg}": sel_pred[fam][tg] for fam in ("uot", "uot_forced", "bal", "persist")
                           for tg in ("l1", "linear")})
    # ---- headline
    print(f"\n== {name}: PCC / P@200 up / P@200 down (tie-aware; argsort in brackets), mean over 3 mask seeds", flush=True)
    for tg in TARGETS:
        print(f" target {tg}")
        for fam in ["trend", "persist", "bal", "uot_forced", "uot"] + [f"uot_lam{lam:g}" for lam in LAMS]:
            a, b = scores[fam][tg]["i>=1"], scores[fam][tg]["i>=2"]
            print(f"   {fam:14s} i>=1 {a['pearson']:+.3f} / {a['p_ta_up']:.3f} / {a['p_ta_down']:.3f} "
                  f"[{a['p_at_200_up']:.3f} / {a['p_at_200_down']:.3f}]   i>=2 {b['pearson']:+.3f} / {b['p_ta_up']:.3f} / {b['p_ta_down']:.3f}"
                  f"   all-gene PCC {scores[fam][tg]['allgenes_pcc_i>=1']:+.3f}")
        for k, v in pairs[tg].items():
            print(f"   paired {k:22s} dPCC {v['pcc']['mean_diff']:+.3f} ({v['pcc']['wins']}/{v['pcc']['losses']}, p={v['pcc']['wilcoxon_p']})"
                  f"  dJ {v['J']['mean_diff']:+.3f} ({v['J']['wins']}/{v['J']['losses']}, p={v['J']['wilcoxon_p']})")
    print(" hindsight (non-strict):", json.dumps(hindsight))
    print(" convergence:", json.dumps(conv))
    for k, v in diag.items():
        print(f" diag {k}: sink/fA {v['sink_over_fA']} forced {v['forced_death_over_fA']} excess {v['excess_death_over_fA']} "
              f"| source/fB {v['source_over_fB']} excess {v['excess_birth_over_fB']} | death lvl ratio {v['death_level_ratio_median']} "
              f"birth lvl ratio {v['birth_level_ratio_median']} | death c1 vs >=2 {v['death_rate_count1_vs_ge2']} "
              f"birth c1 vs >=2 {v['birth_rate_count1_vs_ge2']} | corr(sink, log n ratio) {v['corr_sink_over_fA_vs_log_n_ratio']}")
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
