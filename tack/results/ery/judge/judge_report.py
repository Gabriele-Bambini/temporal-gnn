"""Judge: compare every re-run / re-implementation with the reported numbers, apply the rules, write summary.json.

Inputs (all inside judge/ except the audited reports, which are only READ):
  rescore_rerun/rescore.json      verbatim re-run of diag_metric/ery_forecast_fixed.py (--no-orig)
  attempt_rerun/scores_o<s>.json  verbatim re-run of attempt_forecast.run_grid (output redirected)
  interp_rerun/results.json       verbatim re-run of attempt_interp/interp_bench.py (1 rep, no extras)
  indep/forecast_o<s>.json        independent re-implementation (judge_forecast.py)
  indep/interp_k<k>.json          independent re-implementation (judge_interp.py)
  strict.json                     truncation / data-layer / metric-scale tests (judge_strict.py)
Rules: forecasting = decisions.md 2026-10-24 (strict <, >= 3 of origins 2-5 AND lower raw mean); interpolation =
interp_bench's builder-set rule (same form over k = 2..5).
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ERY = os.path.dirname(HERE)
ORIG = ["2", "3", "4", "5"]
OLD = ["persist", "meanshift", "ot", "uot_growth", "vel_ss", "vel_dyn", "ot_vel_ss", "ot_vel_dyn"]
TOL5 = 0.05
FINDINGS = [
    "interp ED_U union k=2: judge 2.078 vs reported 2.206 (5.8%). Not a bug: over 40 fresh draws of the 406 E7.75 "
    "cells the union ED_U is 2.154 +- 0.071; the audited 3 reps (2.205/2.206/2.206) and the judge's (2.068-2.083) "
    "each happen to cluster, so the reported rep sd understates the sampling sd of union at k=2. ot_pairs/uot margins "
    "(1.91/1.76 vs ~2.15) stay outside this noise.",
    "interp ED_U next k=4: judge 2.662 vs reported 2.802 (5.0%): the 1500-of-2811 E8.0 target subsample differs "
    "(seed 11 vs 1). Rankings unchanged.",
    "forecast alt protocol (other target subsample + S from 3000 cells): persist energy o3 2.096 vs 1.996 (5.05%). The "
    "target subsample alone moves scores by ~5%, i.e. 5-10x the family gaps at o2-o4 (0.01-0.07); verdicts are paired "
    "and unchanged, but origin 2-4 margins are below this noise.",
    "attempt_interp claim 'LOSO approximate at k=4 (3.2412 vs 3.2414)': pool 35 is 0.851 vs diag_batch 0.805 (5.8%), "
    "pool 33 1.424 vs 1.393 (2.3%).",
    "ot noise=1.0 (energy Mode B, o2/o3): judge's own draw 4.950/5.701 vs 4.931/5.764 (<=1.1%), known RNG difference.",
    "Not verifiable locally: gene filter / HVG selection / kNN graph / scVelo dynamical fit of the prep (HPC raw h5ad). "
    "Only the PCA, steady-state velocity and proliferation centring were checked (all past-only).",
    "Truncation tests cannot detect label leakage: comp_atlas / mk_atlas use atlas cell types assigned on all stages "
    "(already labelled LEAK by the attempt).",
    "Scope limits: the rescore re-run used --no-orig, so the flawed sd_orig was only spot-checked (floor rows and "
    "4 predictions per origin in strict.json, o2 persist 0.885 = reported); attempt_report.py was not re-executed, "
    "its roll selection and post-hoc rule were recomputed here from scores_o*.json (hp identical, values to 2e-7).",
    "uot_rw per target pool (energy_U, evaluation only): at o2 under the sd_fixed hp it loses pool 12 (+0.009) and "
    "ties pool 13 (-0.002); o3-o5 it beats persistence in every pool (-0.10 to -1.14).",
    "interp n_pred = min(n_held, 1500): the held-stage SIZE sets the number of predicted points (no distributional "
    "information).",
    "sd_fixed (pre-registered primary after the fix) prefers contracted clouds: halves of the true stage score best at "
    "15-20% contraction in every forecasting space (o2 0.080 -> 0.061) and lambda 0.8-0.85 in every interp space; "
    "energy V/U are minimal at no contraction. vel_ss's only extra win (vs persist, sd_fixed only) comes with a 4-10% "
    "smaller spread than persistence; contract/xfit 4/4 sd_fixed passes are the same artefact.",
]


def hk(hp):
    return json.dumps(hp, sort_keys=True)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def rule(v, cand, comps, keys=ORIG, tol=1e-9):
    lt = lambda a, b: a < b - tol * abs(b)  # noqa: E731
    wins = [all(lt(v[o][cand], v[o][c]) for c in comps) for o in keys]
    mc = float(np.mean([v[o][cand] for o in keys]))
    means = {c: float(np.mean([v[o][c] for o in keys])) for c in comps}
    mok = all(lt(mc, means[c]) for c in comps)
    need = 3 if len(keys) >= 4 else len(keys)
    return {"wins": int(sum(wins)), "win_at": [o for o, w in zip(keys, wins) if w], "mean": mc, "comp_means": means,
            "mean_ok": bool(mok), "pass": bool(sum(wins) >= need and mok)}


def prereg(v):
    ot = {f: rule(v, f, ["persist", "meanshift"]) for f in ("ot", "uot_growth", "ot_vel_ss", "ot_vel_dyn")}
    vel = {f: rule(v, f, ["ot"]) for f in ("vel_ss", "vel_dyn", "ot_vel_ss", "ot_vel_dyn")}
    vel_strict = {f: rule(v, f, ["ot", "persist", "meanshift"]) for f in ("vel_ss", "vel_dyn", "ot_vel_ss", "ot_vel_dyn")}
    sink = rule(v, "uot_growth", ["ot"])
    return {"OT_helps": any(r["pass"] for r in ot.values()), "velocity_helps_vs_ot": any(r["pass"] for r in vel.values()),
            "velocity_also_beats_persist_meanshift": any(r["pass"] for r in vel_strict.values()),
            "sink_helps": sink["pass"],
            "detail": {"OT": {f: f"{r['wins']}/4 mean_ok={r['mean_ok']}" for f, r in ot.items()},
                       "velocity_vs_ot": {f: f"{r['wins']}/4 mean_ok={r['mean_ok']}" for f, r in vel.items()},
                       "velocity_vs_ot+persist+meanshift": {f: f"{r['wins']}/4 mean_ok={r['mean_ok']}" for f, r in vel_strict.items()},
                       "sink": f"{sink['wins']}/4 mean_ok={sink['mean_ok']}"}}


def compare_rescore():
    a = json.load(open(os.path.join(ERY, "diag_metric", "rescore.json")))
    b = json.load(open(os.path.join(HERE, "rescore_rerun", "rescore.json")))
    worst, sel_mismatch, n = {}, [], 0
    for s in ORIG:
        for f in OLD:
            fa, fb = a["origins"][s]["families"][f], b["origins"][s]["families"][f]
            for m in ("energy", "sd_fixed", "mmd"):
                if hk(fa["sel"][m]) != hk(fb["sel"][m]):
                    sel_mismatch.append((s, f, m))
            for key, ra in fa["final"].items():
                if key not in fb["final"]:
                    continue
                for m in ("energy", "sd_fixed", "mmd", "pearson"):
                    x, y = fb["final"][key][m], ra[m]
                    if np.isfinite(x) and np.isfinite(y):
                        worst[m] = max(worst.get(m, 0.0), rel(x, y))
                        n += 1
        worst["S_fixed"] = max(worst.get("S_fixed", 0.0), rel(b["origins"][s]["S_fixed"], a["origins"][s]["S_fixed"]))
    return {"n_values": n, "max_rel_diff": worst, "selection_mismatches": sel_mismatch,
            "pass_5pct": all(v < TOL5 for v in worst.values()) and not sel_mismatch}


def compare_attempt():
    worst, n, missing = {}, 0, []
    for s in ["1", "2", "3", "4", "5"]:
        a = json.load(open(os.path.join(ERY, "attempt_forecast", f"scores_o{s}.json")))
        p = os.path.join(HERE, "attempt_rerun", f"scores_o{s}.json")
        if not os.path.exists(p):
            missing.append(s)
            continue
        b = json.load(open(p))
        for part in ("final", "same"):
            for fam, g in a[part].items():
                for key, ra in g.items():
                    rb = b[part][fam][key]
                    for m in ("energy", "sd_fixed", "mmd", "energy_gene", "pearson"):
                        if m in ra and np.isfinite(ra[m]) and np.isfinite(rb.get(m, np.nan)):
                            worst[f"{part}:{m}"] = max(worst.get(f"{part}:{m}", 0.0), rel(rb[m], ra[m]))
                            n += 1
        for fam, g in a["baselines"].items():
            for key, ra in g.items():
                for m in ("energy", "sd_fixed"):
                    worst[f"baselines:{m}"] = max(worst.get(f"baselines:{m}", 0.0), rel(b["baselines"][fam][key][m], ra[m]))
                    n += 1
    return {"n_values": n, "max_rel_diff": worst, "missing_origins": missing,
            "pass_5pct": (not missing) and all(v < TOL5 for v in worst.values())}


def compare_interp_rerun():
    a = json.load(open(os.path.join(ERY, "attempt_interp", "results.json")))
    b = json.load(open(os.path.join(HERE, "interp_rerun", "results.json")))
    worst, n = {}, 0
    for k in ("2", "3", "4", "5"):
        ma, mb = a["k"][k]["reps"][0]["m"], b["k"][k]["reps"][0]["m"]
        for name, ra in ma.items():
            for m in ("sd_fixed", "ED_U", "ED_V", "W1", "W2", "ED_U_gene"):
                worst[m] = max(worst.get(m, 0.0), abs(mb[name][m] - ra[m]) / max(abs(ra[m]), 0.05))
                n += 1
        if a["k"][k]["velocity_fit"] != b["k"][k]["velocity_fit"]:
            worst["velocity_fit_mismatch_k" + k] = 1.0
    return {"n_values": n, "max_rel_diff (denominator floored at 0.05)": worst,
            "pass_5pct": all(v < TOL5 for v in worst.values())}


def indep_forecast():
    rescore = json.load(open(os.path.join(ERY, "diag_metric", "rescore.json")))
    J = {s: json.load(open(os.path.join(HERE, "indep", f"forecast_o{s}.json"))) for s in ["1", "2", "3", "4", "5"]}
    att = {s: json.load(open(os.path.join(ERY, "attempt_forecast", f"scores_o{s}.json"))) for s in ["1", "2", "3", "4", "5"]}
    out = {"S": {}, "values_at_reported_hp": [], "hp_selection": [], "verdicts": {}, "post_hoc": {}}
    for s in ORIG:
        out["S"][s] = {"reported": rescore["origins"][s]["S_fixed"], "judge_proto": J[s]["S_proto"],
                       "judge_3000cells": J[s]["S_alt"], "rel_alt": J[s]["S_alt"] / rescore["origins"][s]["S_fixed"] - 1}
    # values of the first-run families at the reported Mode-B hp, judge protocol vs reported
    tabs = {}
    for scoring in ("proto", "alt"):
        for m in ("sd_fixed", "energy"):
            for hpsrc in ("reported", "judge"):
                v = {}
                for s in ORIG:
                    v[s] = {}
                    for f in OLD:
                        fin = J[s]["old"][f]["final"]
                        key = (hk(rescore["origins"][s]["families"][f]["sel"][m]) if hpsrc == "reported"
                               else J[s]["old"][f]["sel_judge"][m])
                        v[s][f] = fin[key][scoring][m]
                tabs[(scoring, m, hpsrc)] = v
    for s in ORIG:
        for f in OLD:
            for m in ("sd_fixed", "energy"):
                key = hk(rescore["origins"][s]["families"][f]["sel"][m])
                rep = rescore["origins"][s]["families"][f]["final"][key][m]
                mine = J[s]["old"][f]["final"][key]["proto"][m]
                alt = J[s]["old"][f]["final"][key]["alt"][m]
                noisy = json.loads(key).get("noise", 0) > 0
                out["values_at_reported_hp"].append({"origin": s, "family": f, "metric": m, "hp": key, "reported": rep,
                                                     "judge_proto": mine, "rel_proto": rel(mine, rep),
                                                     "judge_alt_protocol": alt, "rel_alt": rel(alt, rep),
                                                     "noise_draw_differs": noisy})
                out["hp_selection"].append({"origin": s, "family": f, "metric": m, "reported": key,
                                            "judge": J[s]["old"][f]["sel_judge"][m],
                                            "same": key == J[s]["old"][f]["sel_judge"][m]})
    for (scoring, m, hpsrc), v in tabs.items():
        out["verdicts"][f"{m}|{scoring}|hp_{hpsrc}"] = prereg(v)
    # recorded hp (Mode A)
    for m in ("sd_fixed", "energy"):
        v = {s: {f: J[s]["old"][f]["final"][hk(rescore["origins"][s]["families"][f]["sel"]["recorded"])]["proto"][m]
                 for f in OLD} for s in ORIG}
        out["verdicts"][f"{m}|proto|hp_recorded_ModeA"] = prereg(v)
    # energy_U (unbiased) at judge-selected hp
    v = {s: {f: J[s]["old"][f]["final"][J[s]["old"][f]["sel_judge"]["energy_U"]]["proto"]["energy_U"] for f in OLD}
         for s in ORIG}
    out["verdicts"]["energy_U|proto|hp_judge"] = prereg(v)
    out["tables"] = {f"{m}|{sc}|{h}": {s: {f: round(x, 4) for f, x in row.items()} for s, row in v.items()}
                     for (sc, m, h), v in tabs.items() if sc == "proto"}
    # ---- post-hoc families: rolling-origin selection (grid at origin s-1 scored vs stage s), judge numbers
    NEWF = list(J["2"]["new"])
    for m in ("sd_fixed", "energy", "energy_U"):
        v, hps, cmp_rows = {}, {}, []
        for s in ORIG:
            prev = J[str(int(s) - 1)]["new"]
            v[s] = {"persist": J[s]["old"]["persist"]["final"]["{}"]["proto"][m],
                    "meanshift": J[s]["old"]["meanshift"]["final"]["{}"]["proto"][m]}
            hps[s] = {}
            for f in NEWF:
                keys = list(prev[f])
                kbest = keys[int(np.argmin([prev[f][k][m] for k in keys]))]
                v[s][f] = J[s]["new"][f][kbest][m]
                hps[s][f] = kbest
                if m in ("sd_fixed", "energy"):
                    # the audited roll choice and value
                    pa = att[str(int(s) - 1)]["final"][f]
                    ka = min(pa, key=lambda k: pa[k][m])
                    cmp_rows.append({"origin": s, "family": f, "metric": m, "hp_judge": kbest, "hp_reported": ka,
                                     "value_judge": v[s][f], "value_reported_at_reported_hp": att[s]["final"][f][ka][m],
                                     "value_judge_at_reported_hp": J[s]["new"][f][ka][m],
                                     "rel_at_reported_hp": rel(J[s]["new"][f][ka][m], att[s]["final"][f][ka][m]),
                                     "ess_reported": att[s]["final"][f][ka].get("ess")})
        out["post_hoc"][m] = {"rules_vs_persist_meanshift": {f: rule(v, f, ["persist", "meanshift"]) for f in NEWF},
                              "uot_rw_vs_tilt_rw": rule(v, "uot_rw", ["tilt_rw"]),
                              "hp": hps, "values": {s: {f: round(x, 4) for f, x in row.items()} for s, row in v.items()},
                              "compare": cmp_rows}
    # uot_rw at the reported roll hp (sd_fixed and energy): ED_U margin vs persist
    marg = {}
    for m in ("sd_fixed", "energy"):
        marg[m] = {}
        for s in ORIG:
            pa = att[str(int(s) - 1)]["final"]["uot_rw"]
            ka = min(pa, key=lambda k: pa[k][m])
            r = J[s]["new"]["uot_rw"][ka]
            p0 = J[s]["old"]["persist"]["final"]["{}"]["proto"]
            marg[m][s] = {"hp": ka, "ess": r["ess"], "energy_U_uot_rw": r["energy_U"], "energy_U_persist": p0["energy_U"],
                          "energy_V_margin": r["energy"] - p0["energy"], "energy_U_margin": r["energy_U"] - p0["energy_U"],
                          "spread_ratio_vs_persist": r["spread"] / p0["spread"]}
    out["uot_rw_margins"] = marg
    return out


def indep_interp():
    tab = pd.read_csv(os.path.join(ERY, "attempt_interp", "table.csv"))
    rep = {met: tab.pivot(index="method", columns="k", values=met) for met in ("ED_U", "sd_fixed", "ED_U_gene")}
    J = {k: json.load(open(os.path.join(HERE, "indep", f"interp_k{k}.json"))) for k in ("2", "3", "4", "5")}
    names = list(J["2"]["reps"][0])
    mine = {met: {k: {n: float(np.mean([r[n][met] for r in J[k]["reps"]])) for n in names} for k in J}
            for met in ("ED_U", "sd_fixed", "ED_U_gene", "spread_ratio")}
    sds = {met: {k: {n: float(np.std([r[n][met] for r in J[k]["reps"]])) for n in names} for k in J}
           for met in ("ED_U", "sd_fixed")}
    rows = []
    mapn = {"vel_ss_sd_fixed": "vel_ss", "vel_ss_energy_U": "vel_ss_cED"}
    for met in ("ED_U", "sd_fixed", "ED_U_gene"):
        for k in J:
            for n in names:
                rn = mapn.get(n, n)
                if rn in rep[met].index:
                    r_ = float(rep[met].loc[rn, int(k)])
                    rows.append({"metric": met, "k": k, "method": n, "reported": r_, "judge": mine[met][k][n],
                                 "judge_rep_sd": sds.get(met, {}).get(k, {}).get(n),
                                 "rel": abs(mine[met][k][n] - r_) / max(abs(r_), 0.05)})
    ks = ["2", "3", "4", "5"]
    rules = {}
    for met in ("ED_U", "sd_fixed", "ED_U_gene"):
        v = {k: mine[met][k] for k in ks}
        base = ["prev", "next", "union"]
        rules[met] = {"R1_ot_pairs": rule(v, "ot_pairs", base, ks), "R1_sb": rule(v, "sb", base, ks),
                      "R1_sb_half": rule(v, "sb_half", base, ks), "R1_uot": rule(v, "uot", base, ks),
                      "R1_ot_bary": rule(v, "ot_bary", base, ks), "R1_ot_bary_sym": rule(v, "ot_bary_sym", base, ks),
                      "R2_pairing": rule(v, "ot_pairs", ["indep"], ks), "R3_sb_vs_ot": rule(v, "sb", ["ot_pairs"], ks),
                      "R3_sb_half_vs_ot": rule(v, "sb_half", ["ot_pairs"], ks),
                      "R4_uot_vs_ot": rule(v, "uot", ["ot_pairs"], ks),
                      "R4b_uot_beta0_vs_ot": rule(v, "uot_beta0", ["ot_pairs"], ks),
                      "R5_vel_ss": rule(v, "vel_ss_sd_fixed", ["prev", "union"], ks),
                      "R5_vel_ss_cED": rule(v, "vel_ss_energy_U", ["prev", "union"], ks),
                      "best_per_k": {k: min(v[k], key=v[k].get) for k in ks}}
    probe = {k: J[k]["probe"] for k in ks}
    return {"compare": rows, "max_rel": {met: max(r["rel"] for r in rows if r["metric"] == met) for met in ("ED_U", "sd_fixed", "ED_U_gene")},
            "rules_judge": rules, "means_judge": {met: {k: {n: round(x, 3) for n, x in v.items()} for k, v in mine[met].items()} for met in mine},
            "c_fit_judge": {k: J[k]["c_fit"] for k in ks}, "S": {k: {"judge_alt": J[k]["S_alt"], "judge_proto": J[k]["S_proto"]} for k in ks},
            "probe": probe}


def main():
    out = {"rescore_rerun": compare_rescore(), "attempt_rerun": compare_attempt(),
           "interp_rerun": compare_interp_rerun()}
    out["indep_forecast"] = indep_forecast()
    out["indep_interp"] = indep_interp()
    p = os.path.join(HERE, "strict.json")
    if os.path.exists(p):
        st = json.load(open(p))
        fl = st.get("forecast_layer", {})
        out["strict_summary"] = {
            "forecast_first_run_all_identical": all(r["identical"] == r["of"] for s in fl.values() for r in s["first_run"].values()),
            "forecast_attempt_all_identical": all(r["identical"] == r["of"] for s in fl.values() for r in s["attempt"].values()),
            "n_forecasts_checked": sum(r["of"] for s in fl.values() for part in ("first_run", "attempt") for r in s[part].values()),
            "failures": [(s, part, f, r) for s, v in fl.items() for part in ("first_run", "attempt")
                         for f, r in v[part].items() if r["identical"] != r["of"]],
            "interp_all_identical": all(all(v["methods_identical"].values()) and v["S_identical"] and v["cfit_identical"]
                                        for v in st.get("interp_layer", {}).values()),
            "data_layer": st.get("data_layer"), "metric_scale": st.get("metric_scale")}
    pe = os.path.join(HERE, "indep", "extra.json")
    if os.path.exists(pe):
        ex = json.load(open(pe))
        out["extra_pools_uot_rw_minus_persist_energy_U"] = {
            s: {m: {p: r[f"uot_rw[{m} hp]"]["energy_U"] - r["persist"]["energy_U"] for p, r in v["pools"].items()}
                for m in ("sd_fixed", "energy")} for s, v in ex["pools"].items()}
        out["extra_probe_forecast_spaces"] = ex["probe"]
    out["not_reproduced_or_flagged"] = FINDINGS
    with open(os.path.join(HERE, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    # console
    for k in ("rescore_rerun", "attempt_rerun", "interp_rerun"):
        print(k, json.dumps(out[k], default=float)[:800])
    IF = out["indep_forecast"]
    print("S", IF["S"])
    bad = [r for r in IF["values_at_reported_hp"] if r["rel_proto"] >= TOL5]
    print("indep forecast: values at reported hp, max rel (proto):",
          max(r["rel_proto"] for r in IF["values_at_reported_hp"]), " (alt):",
          max(r["rel_alt"] for r in IF["values_at_reported_hp"]), " n>=5%:", len(bad))
    for r in bad:
        print("   >=5%:", r)
    print("   max rel excluding noisy hp:", max(r["rel_proto"] for r in IF["values_at_reported_hp"] if not r["noise_draw_differs"]))
    print("hp selection mismatches:", [r for r in IF["hp_selection"] if not r["same"]])
    for k, v in IF["verdicts"].items():
        print("VERDICT", k, {x: v[x] for x in ("OT_helps", "velocity_helps_vs_ot", "velocity_also_beats_persist_meanshift", "sink_helps")}, v["detail"])
    for m, v in IF["post_hoc"].items():
        print("POSTHOC", m, {f: f"{r['wins']}/4 {r['pass']}" for f, r in v["rules_vs_persist_meanshift"].items()}, "hp", v["hp"])
        if v["compare"]:
            print("   max rel at reported hp:", max(r["rel_at_reported_hp"] for r in v["compare"]),
                  "hp mismatches:", [(r["origin"], r["family"], r["hp_judge"], r["hp_reported"]) for r in v["compare"] if r["hp_judge"] != r["hp_reported"]])
    print("uot_rw margins", json.dumps(IF["uot_rw_margins"], default=float))
    II = out["indep_interp"]
    print("interp max rel", II["max_rel"])
    for r in II["compare"]:
        if r["rel"] >= TOL5 and r["metric"] != "ED_U_gene":
            print("   interp >=5%:", r)
    for met, rr in II["rules_judge"].items():
        print("INTERP", met, {n: (f"{x['wins']}/4 {x['pass']}" if isinstance(x, dict) and "wins" in x else x) for n, x in rr.items()})
    print("probe", json.dumps(II["probe"], default=float)[:1500])
    if "strict_summary" in out:
        ss = out["strict_summary"]
        print("STRICT", {k: ss[k] for k in ("forecast_first_run_all_identical", "forecast_attempt_all_identical",
                                           "n_forecasts_checked", "failures", "interp_all_identical")})


if __name__ == "__main__":
    main()
