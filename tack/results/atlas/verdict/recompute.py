"""Independent recomputation of the five pre-registered atlas verdicts + secondary metrics + exploratory checks.

Pre-registration: tack/prereg/2026-10-25-atlas-confirmation-prereg.md (rules quoted in RULE below).
Input (read-only): tack/results/atlas/confirm.json (frozen output of tack/hpc/atlas_confirm.py, HPC).
Output: recompute.json next to this script. Nothing here can change the pre-registered verdict; everything under
"exploratory" is labelled as such.

Sections
  A  reproduction of the per-unit values: interp per-unit ED_U = mean over the 3 reps (checked against the stored
     'mean' / 'pool_mean'); forecast per-unit = one deterministic run (no reps).
  B  the five verdicts recomputed from the per-unit ED_U ONLY (the 'verdicts' block of confirm.json is read only at the
     end, to compare). Two readings of "against every listed B": per baseline (wins / ties / losses vs each B
     separately, each B needing >= need wins and a lower mean) and joint (a unit counts only if A beats ALL B in it:
     the frozen beats()). Ties (exact float equality) are counted separately and are not wins (rule: strictly lower).
  C  secondary metrics exactly as the prereg lists them (reported, not used for the verdict): ED_U in gene space,
     ED_U per embryo pool (>= 20 cells), forecast gene-level PCC of the mean change, ESS of the reweighting.
  D  exploratory: H2 tie structure (lambda = 0 chosen), H1c meaningfulness (indep vs union margins), rep-to-rep spread
     of union vs ot_pairs (paired), sha256 of the local frozen files vs the prereg.
  E  exploratory, only with --data: re-run of the frozen run_interp / run_forecast on the local copy of the atlas
     products (bit-level reproduction check) and covariance traces of the predicted clouds vs the held stage.

Usage: PYTHONIOENCODING=utf-8 python recompute.py [--confirm ../confirm.json] [--data <dir with interp*.npz,
       origin*.npz>] [--out recompute.json]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True   # never write __pycache__ into the frozen tack/hpc folder

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
HPC = os.path.join(ROOT, "tack", "hpc")

RULE = ('"A beats B" = A has a strictly lower ED_U in at least ceil(2/3 of the units) (5 of 7; 4 of 6) AND a lower '
        'mean ED_U over the units, against every listed B.')
HYPS = {   # name: (family, A, [B...])  -- exactly the prereg list
    "H1_ot_pairs": ("interp", "ot_pairs", ["prev", "next", "union"]),
    "H1b_uot": ("interp", "uot", ["prev", "next", "union"]),
    "H1c_pairing": ("interp", "ot_pairs", ["indep"]),
    "H2_uot_rw": ("forecast", "uot_rw", ["persist", "meanshift"]),
    "H2b_ot_needed": ("forecast", "uot_rw", ["tilt_rw"]),
}
PREREG_SHA = {
    "atlas_prep.py": "305b47f8afa41afc51391a94e5e32ec494906a0c1eea46948bbab375256db3a0",
    "atlas_confirm.py": "0594c9d11f9d0d6a64e86e1557e9c30b945494bd41627b2a67550301e80e503f",
    "interp_bench_frozen.py": "5231d262a19ebc40f56a4f65fad55aadf5c1c952cf0febb335fdd4968efa6e16",
    "ery_forecast.py": "64351c57f109b3f86213d7cd06ecc526108a169a5e5fb92674f1b1512ce94a9d",
}


def need_of(n):
    return int(math.ceil(2 * n / 3 - 1e-12))   # 7 -> 5, 6 -> 4 (exact rational ceil)


def clean(x):
    """JSON-safe: NaN / inf -> None, numpy -> python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not math.isfinite(x) else x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def compare(vals, a, bs, units=None):
    """vals[u][m] -> per-baseline and joint comparison of a against every b in bs (strict <, ties separate)."""
    units = sorted(vals, key=lambda u: (int(u) if str(u).isdigit() else u)) if units is None else units
    n, need = len(units), need_of(len(units))
    per_b = {}
    for b in bs:
        w = [u for u in units if vals[u][a] < vals[u][b]]
        t = [u for u in units if vals[u][a] == vals[u][b]]
        lo = [u for u in units if vals[u][a] > vals[u][b]]
        ma, mb = float(np.mean([vals[u][a] for u in units])), float(np.mean([vals[u][b] for u in units]))
        per_b[b] = {"wins": len(w), "ties": len(t), "losses": len(lo), "win_units": w, "tie_units": t,
                    "loss_units": lo, "mean_A": ma, "mean_B": mb, "mean_ok": bool(ma < mb),
                    "pass_vs_this_B": bool(len(w) >= need and ma < mb)}
    joint = [u for u in units if all(vals[u][a] < vals[u][b] for b in bs)]
    mean_ok = all(per_b[b]["mean_ok"] for b in bs)
    return {"A": a, "B": bs, "units": units, "of": n, "need": need, "per_baseline": per_b,
            "joint_wins": len(joint), "joint_win_units": joint, "mean_ok_all": bool(mean_ok),
            "pass_joint_reading": bool(len(joint) >= need and mean_ok),
            "pass_per_baseline_reading": bool(all(per_b[b]["pass_vs_this_B"] for b in bs))}


def sha_lf(path):
    b = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- A: per-unit values

def per_unit(C):
    ki = sorted(C["interp"], key=int)
    fo = sorted(C["forecast"], key=int)
    imeth = list(C["interp"][ki[0]]["mean"])
    fmeth = list(C["forecast"][fo[0]])
    chk = {"interp_mean_is_mean_over_reps": {}, "max_abs_diff": {}}
    worst = {"ED_U": 0.0, "ED_U_gene": 0.0, "pool_ED_U": 0.0}
    nreps = {}
    I = {"ED_U": {}, "ED_U_gene": {}, "pool": {}, "reps_ED_U": {}}
    for k in ki:
        r = C["interp"][k]
        nreps[k] = len(r["reps"])
        I["ED_U"][k], I["ED_U_gene"][k], I["pool"][k], I["reps_ED_U"][k] = {}, {}, {}, {}
        for m in imeth:
            for key in ("ED_U", "ED_U_gene"):
                mine = float(np.mean([rp[m][key] for rp in r["reps"]]))
                worst[key] = max(worst[key], abs(mine - r["mean"][m][key]))
                I[key][k][m] = mine
            I["reps_ED_U"][k][m] = [float(rp[m]["ED_U"]) for rp in r["reps"]]
            I["pool"][k][m] = {}
            for p in r["pools"]:
                mine = float(np.mean([rp[m]["pool_ED_U"][p] for rp in r["reps"]]))
                worst["pool_ED_U"] = max(worst["pool_ED_U"], abs(mine - r["pool_mean"][m][p]))
                I["pool"][k][m][p] = mine
    chk["max_abs_diff"] = worst
    chk["interp_mean_is_mean_over_reps"] = bool(all(v < 1e-12 for v in worst.values()))
    chk["reps_per_unit"] = nreps
    F = {key: {s: {m: C["forecast"][s][m][key] for m in fmeth} for s in fo}
         for key in ("ED_U", "ED_U_gene", "pcc", "ess", "backtest_ED_U")}
    F["hp"] = {s: {m: C["forecast"][s][m]["hp"] for m in fmeth} for s in fo}
    F["pool"] = {s: {m: C["forecast"][s][m]["pool_ED_U"] for m in fmeth} for s in fo}
    chk["forecast_per_unit"] = "single deterministic run per origin (no reps in confirm.json)"
    return I, F, chk, ki, fo


# ---------------------------------------------------------------- B: verdicts

def verdicts(I, F, C):
    out = {}
    for h, (fam, a, bs) in HYPS.items():
        vals = I["ED_U"] if fam == "interp" else F["ED_U"]
        res = compare(vals, a, bs)
        code = C["verdicts"][h]
        res["confirm_json_verdict"] = code
        res["matches_confirm_json"] = bool(code["wins"] == res["joint_wins"] and code["need"] == res["need"]
                                           and code["of"] == res["of"] and code["mean_ok"] == res["mean_ok_all"]
                                           and code["pass"] == res["pass_joint_reading"])
        res["readings_agree"] = bool(res["pass_joint_reading"] == res["pass_per_baseline_reading"])
        res["VERDICT"] = "PASS" if (res["pass_joint_reading"] and res["pass_per_baseline_reading"]) else (
            "FAIL" if not (res["pass_joint_reading"] or res["pass_per_baseline_reading"]) else "READINGS DISAGREE")
        out[h] = res
    return out


# ---------------------------------------------------------------- C: secondary metrics

def pool_compare(P, a, bs):
    """P[u][m][pool] -> wins of a vs each b over all (unit, pool) cells, per-unit breakdown, every-pool consistency."""
    per_b = {}
    for b in bs:
        cells, per_u = [], {}
        for u in sorted(P, key=int):
            pools = sorted(P[u][a])
            w = [p for p in pools if P[u][a][p] < P[u][b][p]]
            t = [p for p in pools if P[u][a][p] == P[u][b][p]]
            per_u[u] = {"pools": len(pools), "wins": len(w), "ties": len(t), "losses": len(pools) - len(w) - len(t),
                        "every_pool": bool(len(w) == len(pools)), "win_pools": w,
                        "diff_A_minus_B": {p: P[u][a][p] - P[u][b][p] for p in pools}}
            cells += [(u, p) for p in pools]
        nw = sum(v["wins"] for v in per_u.values())
        nt = sum(v["ties"] for v in per_u.values())
        per_b[b] = {"cells": len(cells), "wins": nw, "ties": nt, "losses": len(cells) - nw - nt,
                    "win_fraction": nw / len(cells), "units_won_in_every_pool": sum(v["every_pool"] for v in per_u.values()),
                    "units_with_majority_of_pools_won": sum(v["wins"] > v["pools"] / 2 for v in per_u.values()),
                    "per_unit": per_u}
    joint = 0
    tot = 0
    for u in P:
        for p in P[u][a]:
            tot += 1
            joint += all(P[u][a][p] < P[u][b][p] for b in bs)
    return {"A": a, "B": bs, "per_baseline": per_b, "joint_cells_won": joint, "cells": tot}


def secondary(I, F, fo):
    sec = {"note": "Secondary metrics as listed in the prereg ('reported, not used for the verdict'). The 5/7, 4/6 rule "
                   "is applied to them ONLY descriptively, to see whether they tell the same story as the primary."}
    sec["gene_ED_U"] = {h: compare(I["ED_U_gene"] if fam == "interp" else F["ED_U_gene"], a, bs)
                        for h, (fam, a, bs) in HYPS.items()}
    for v in sec["gene_ED_U"].values():
        v["would_pass_joint_reading"] = v.pop("pass_joint_reading")
        v["would_pass_per_baseline_reading"] = v.pop("pass_per_baseline_reading")
    sec["pool_ED_U"] = {h: pool_compare(I["pool"] if fam == "interp" else F["pool"], a, bs)
                        for h, (fam, a, bs) in HYPS.items()}
    # forecast pcc and ess
    pcc = {s: F["pcc"][s] for s in fo}
    defined = {m: [s for s in fo if F["pcc"][s][m] is not None and math.isfinite(F["pcc"][s][m])]
               for m in F["pcc"][fo[0]]}
    both = [s for s in fo if s in defined["uot_rw"] and s in defined["meanshift"]]
    both_t = [s for s in fo if s in defined["uot_rw"] and s in defined["tilt_rw"]]
    sec["forecast_pcc"] = {
        "values": pcc, "defined_at": defined,
        "note": "pcc = PCC over genes of predicted vs true mean change from stage s (higher is better). persist is NaN "
                "by construction (zero predicted change); uot_rw / tilt_rw are NaN where rolling origin chose lambda 0 "
                "(the forecast IS persist). So pcc cannot rank uot_rw vs persist at all, and vs meanshift / tilt_rw "
                "only where both are defined.",
        "uot_rw_vs_meanshift": {"units": both, "uot_rw_higher": [s for s in both if pcc[s]["uot_rw"] > pcc[s]["meanshift"]],
                                "uot_rw_mean": float(np.mean([pcc[s]["uot_rw"] for s in both])) if both else None,
                                "meanshift_mean": float(np.mean([pcc[s]["meanshift"] for s in both])) if both else None},
        "uot_rw_vs_tilt_rw": {"units": both_t, "uot_rw_higher": [s for s in both_t if pcc[s]["uot_rw"] > pcc[s]["tilt_rw"]]},
        "meanshift_all_origins": {"values": {s: pcc[s]["meanshift"] for s in fo},
                                  "mean": float(np.mean([pcc[s]["meanshift"] for s in fo]))},
    }
    sec["forecast_ess"] = {"values": {s: F["ess"][s] for s in fo}, "n_points_scored": 1500,
                           "note": "ESS = 1 / sum w^2 of the scored (subsampled) weights; 1500 = unweighted. Not a "
                                   "competition metric: it measures how hard the reweighting is."}
    return sec


# ---------------------------------------------------------------- D: exploratory from confirm.json alone

def exploratory_json(I, F, C, ki, fo):
    ex = {}
    # D1: H2 tie structure
    ties = {}
    for s in fo:
        r = C["forecast"][s]
        same = all(r["uot_rw"][key] == r["persist"][key] for key in ("ED_U", "ED_U_gene", "ED_V", "backtest_ED_U"))
        ties[s] = {"uot_rw_hp": r["uot_rw"]["hp"], "tilt_rw_hp": r["tilt_rw"]["hp"],
                   "uot_rw_identical_to_persist": bool(same and r["uot_rw"]["hp"]["lam"] == 0.0),
                   "tilt_rw_identical_to_persist": bool(r["tilt_rw"]["ED_U"] == r["persist"]["ED_U"]
                                                        and r["tilt_rw"]["hp"]["lam"] == 0.0),
                   "uot_rw_backtest_best": r["uot_rw"]["backtest_ED_U"], "persist_backtest": r["persist"]["backtest_ED_U"]}
    n_id = sum(v["uot_rw_identical_to_persist"] for v in ties.values())
    ex["H2_tie_structure"] = {
        "per_origin": ties, "n_origins_uot_rw_equals_persist": n_id,
        "max_possible_wins_vs_persist": len(fo) - n_id, "need": need_of(len(fo)),
        "note": "Where rolling origin picked lambda 0 the uot_rw forecast is the persist forecast (identical arrays, "
                "exact float equality, a tie under 'strictly lower'). This is part of the frozen method (lambda 0 is in "
                "the grid), not a reason to relax the rule."}
    # D2: H1c meaningfulness
    E = I["ED_U"]
    rows = {}
    for k in ki:
        e = E[k]
        m_ind = e["indep"] - e["ot_pairs"]
        m_uni = e["ot_pairs"] - e["union"]
        rows[k] = {"union": e["union"], "ot_pairs": e["ot_pairs"], "indep": e["indep"],
                   "order_union_lt_ot_pairs_lt_indep": bool(e["union"] < e["ot_pairs"] < e["indep"]),
                   "margin_indep_minus_ot_pairs": m_ind, "margin_ot_pairs_minus_union": m_uni,
                   "ratio_margins": m_ind / m_uni if m_uni != 0 else None,
                   "indep_over_union": e["indep"] / e["union"], "ot_pairs_over_union": e["ot_pairs"] / e["union"]}
    ex["H1c_meaningfulness"] = {
        "per_k": rows,
        "indep_vs_union": compare(E, "indep", ["union"]),
        "indep_vs_prev_next_union": compare(E, "indep", ["prev", "next", "union"]),
        "sb_half_vs_prev_next_union": compare(E, "sb_half", ["prev", "next", "union"]),
        "n_k_order_union_lt_ot_pairs_lt_indep": sum(v["order_union_lt_ot_pairs_lt_indep"] for v in rows.values()),
        "mean_ED_U": {m: float(np.mean([E[k][m] for k in ki])) for m in E[ki[0]]},
        "construction": "indep = midpoint 0.5 x + 0.5 y of INDEPENDENT draws: covariance 0.25 S_A + 0.25 S_B (half of "
                        "the average within-stage covariance), mean 0.5 mu_A + 0.5 mu_B. Every balanced method (union, "
                        "indep, ot_pairs, sb_half) has the same mean (interp_bench_frozen header), so ot_pairs vs indep "
                        "differs only in shape / spread: H1c tests the quality of the pairing given displacement "
                        "interpolation, not whether interpolating beats the 50/50 mixture.",
    }
    # D3: rep-to-rep spread (paired)
    R = I["reps_ED_U"]
    spread = {}
    for k in ki:
        row = {}
        for m in ("union", "ot_pairs", "uot", "indep", "sb_half", "prev", "next"):
            v = np.array(R[k][m])
            row[m] = {"reps": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
        for a in ("ot_pairs", "uot"):
            d = np.array(R[k][a]) - np.array(R[k]["union"])
            sd = float(d.std(ddof=1))
            row[f"paired_{a}_minus_union"] = {
                "diffs": d.tolist(), "mean": float(d.mean()), "sd": sd,
                "n_reps_A_worse": int((d > 0).sum()), "n_reps": int(len(d)),
                "t_paired": float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else None,
                "mean_over_max_marginal_sd": float(d.mean() / max(row[a]["sd"], row["union"]["sd"]))}
        spread[k] = row
    po = [spread[k]["paired_ot_pairs_minus_union"] for k in ki]
    ex["rep_spread"] = {
        "per_k": spread,
        "summary_ot_pairs_minus_union": {
            "k_all_3_reps_worse": [k for k, v in zip(ki, po) if v["n_reps_A_worse"] == v["n_reps"]],
            "k_mean_gt_2sd_paired": [k for k, v in zip(ki, po) if v["mean"] > 2 * v["sd"]],
            "k_mean_gt_2_max_marginal_sd": [k for k, v in zip(ki, po) if v["mean_over_max_marginal_sd"] > 2],
            "median_abs_t": float(np.median([abs(v["t_paired"]) for v in po if v["t_paired"] is not None])),
            "pooled_mean_diff_over_k": float(np.mean([v["mean"] for v in po])),
            "sign_test_7_of_7_two_sided_p": float(2 * 0.5 ** len(ki)) if all(v["mean"] > 0 for v in po) else None},
        "caveat": "Reps are PAIRED: the held-stage target subsample is the same (seed 1) in every rep; only the fit-side "
                  "subsamples (seeds 1000+r, 2000+r) and the pair / mixture draws (rng 1000k+r) change. So the rep sd "
                  "measures fit-side sampling noise only, not target-side or embryo-level variation, and with n = 3 the "
                  "sd itself is very noisy. Per-pool differences (secondary.pool_ED_U) are the closer proxy for "
                  "embryo-level variation."}
    # D4: frozen file hashes
    ex["frozen_sha256_LF"] = {f: {"local": sha_lf(os.path.join(HPC, f)), "prereg": h,
                                  "match": sha_lf(os.path.join(HPC, f)) == h} for f, h in PREREG_SHA.items()}
    return ex


# ---------------------------------------------------------------- E: optional re-run on the local products

def rerun(data, C, ki, fo):
    sys.path.insert(0, HPC)
    import atlas_confirm as AC          # noqa: E402  (frozen, imported read-only)
    import interp_bench_frozen as IB    # noqa: E402
    t0 = time.time()
    rep = {"interp_max_abs_diff": {}, "forecast_max_abs_diff": {}, "traces": {}}
    for k in ki:
        kk = int(k)
        mine = AC.run_interp(data, kk)
        diff = 0.0
        for r, rp in enumerate(C["interp"][k]["reps"]):
            for m, v in rp.items():
                diff = max(diff, abs(mine["reps"][r][m]["ED_U"] - v["ED_U"]), abs(mine["reps"][r][m]["ED_U_gene"] - v["ED_U_gene"]))
                for p, x in v["pool_ED_U"].items():
                    diff = max(diff, abs(mine["reps"][r][m]["pool_ED_U"][p] - x))
        rep["interp_max_abs_diff"][k] = diff
        rep["traces"][k] = traces(data, kk, IB, AC)
        print("rerun interp", k, f"maxdiff {diff:.2e}", f"[{time.time() - t0:.0f}s]", flush=True)
    for s in fo:
        mine = AC.run_forecast(data, int(s))
        diff = 0.0
        for m, v in C["forecast"][s].items():
            for key in ("ED_U", "ED_U_gene", "ED_V", "backtest_ED_U", "ess"):
                diff = max(diff, abs(mine[m][key] - v[key]))
            if mine[m]["hp"] != v["hp"]:
                diff = float("inf")
        rep["forecast_max_abs_diff"][s] = diff
        print("rerun forecast", s, f"maxdiff {diff:.2e}", f"[{time.time() - t0:.0f}s]", flush=True)
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def wtrace(Z, w):
    w = w / w.sum()
    mu = w @ Z
    return float(w @ ((Z - mu) ** 2).sum(1)), mu


def traces(data, k, IB, AC):
    """Replica of run_interp's sampling (same seeds, same rng order) to get the scored point clouds; returns the
    weighted covariance trace of each method's cloud, its squared mean offset from the held stage, and the ED_U
    recomputed from the same cloud (must equal the frozen value)."""
    d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
    comps = d["pca_comps"].astype(np.float64)
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in set(IB.S_GENES + IB.G2M_GENES) for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    cc_fit = (Xcc - Xcc.mean(0)).mean(1)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], Xh[it], S, with_sd=False)
    tr_h, mu_h = wtrace(Zh[it], np.ones(len(it)))
    out = {"held": {"trace": tr_h}, "reps": []}
    for r in range(3):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        M.pop("_plan", None)
        M.pop("_uot", None)
        rec = {}
        for name, (Z, X, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
            if name not in AC.INTERP_METHODS:
                continue
            tr, mu = wtrace(Zs, ws)
            rec[name] = {"trace": tr, "trace_over_held": tr / tr_h, "mean_offset_sq": float(((mu - mu_h) ** 2).sum()),
                         "ED_U_replica": IB.ed_uv(Zs, ws, tgt)[0]}
        out["reps"].append(rec)
    out["mean"] = {m: {key: float(np.mean([rp[m][key] for rp in out["reps"]]))
                       for key in ("trace", "trace_over_held", "mean_offset_sq", "ED_U_replica")} for m in out["reps"][0]}
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default=os.path.join(HERE, "..", "confirm.json"))
    ap.add_argument("--data", default=None, help="local atlas products: enables section E (re-run + traces)")
    ap.add_argument("--out", default=os.path.join(HERE, "recompute.json"))
    args = ap.parse_args()
    C = json.load(open(args.confirm))
    I, F, chk, ki, fo = per_unit(C)
    res = {"prereg_rule": RULE, "confirm_json": os.path.abspath(args.confirm),
           "confirm_json_sha256": hashlib.sha256(open(args.confirm, "rb").read()).hexdigest(),
           "A_per_unit_check": chk,
           "per_unit_ED_U": {"interp": I["ED_U"], "forecast": F["ED_U"]},
           "forecast_hp": F["hp"],
           "B_verdicts": verdicts(I, F, C)}
    res["B_summary"] = {h: {"VERDICT": v["VERDICT"], "joint_wins": v["joint_wins"], "of": v["of"], "need": v["need"],
                            "mean_ok_all": v["mean_ok_all"],
                            "wins_per_baseline": {b: f'{x["wins"]}W/{x["ties"]}T/{x["losses"]}L'
                                                  for b, x in v["per_baseline"].items()},
                            "mean_A_vs_B": {b: [round(x["mean_A"], 4), round(x["mean_B"], 4)]
                                            for b, x in v["per_baseline"].items()},
                            "matches_confirm_json": v["matches_confirm_json"]}
                        for h, v in res["B_verdicts"].items()}
    res["C_secondary"] = secondary(I, F, fo)
    res["exploratory"] = {"note": "EXPLORATORY, not pre-registered. Nothing here alters the pre-registered verdict; "
                                  "only a real bug in the frozen pipeline would matter, and that would need a new "
                                  "pre-registered run.",
                          **exploratory_json(I, F, C, ki, fo)}
    if args.data:
        res["exploratory"]["E_rerun_local"] = rerun(args.data, C, ki, fo)
        res["exploratory"]["E_rerun_local"]["data"] = os.path.abspath(args.data)
    json.dump(clean(res), open(args.out, "w"), indent=1, allow_nan=False)
    print(json.dumps(clean(res["B_summary"]), indent=1))
    print("written", args.out)


if __name__ == "__main__":
    main()
