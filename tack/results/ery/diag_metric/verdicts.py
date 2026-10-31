"""Corrected tables and pre-registered verdicts from rescore.json (and spread_bias.json if present).

Pre-registered rule (docs/superpowers/decisions.md, row 2026-10-24), origins 2-5, lower distance = better:
  OT helps        some OT family F in {ot, uot_growth, ot_vel_ss, ot_vel_dyn} beats BOTH persist and meanshift
                  (strict <) in >= 3 of the 4 origins AND its raw mean over origins 2-5 is below both means;
  velocity helps  same rule for F in {vel_ss, vel_dyn, ot_vel_ss, ot_vel_dyn} against ot;
  sink helps      uot_growth against ot, same rule.
Extras (NOT pre-registered, labelled): the 'on average' part re-done with the mean log ratio (scale-free, the raw
mean is dominated by the origins with large distances); velocity families also against persist + meanshift.
Mode A = hp recorded in the first run (chosen by the flawed metric) re-scored; Mode B = hp re-selected on the same
backtest by the metric being judged (the metric-consistent verdict).
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAMS = ["persist", "meanshift", "ot", "uot_growth", "vel_ss", "vel_dyn", "ot_vel_ss", "ot_vel_dyn"]
OTF = ["ot", "uot_growth", "ot_vel_ss", "ot_vel_dyn"]
VELF = ["vel_ss", "vel_dyn", "ot_vel_ss", "ot_vel_dyn"]
DIST = ["sd_fixed", "energy", "mmd", "sd_orig"]
GENE = ["pearson", "p200_up", "p200_down"]
ORIG = ["2", "3", "4", "5"]


def hkey(hp):
    return json.dumps(hp, sort_keys=True)


def table(res, selector, metric):
    """value[origin][family] of `metric` for the hp chosen by `selector` ('recorded' = Mode A)."""
    out = {}
    for s, row in res["origins"].items():
        out[s] = {}
        for f in FAMS:
            fr = row["families"][f]
            r = fr["final"][hkey(fr["sel"][selector])]
            out[s][f] = r.get(metric, float("nan"))
    return out


def rule(v, cand, comps, lower=True):
    """Pre-registered rule for candidate family vs list of comparators on origins 2-5."""
    sign = 1.0 if lower else -1.0
    wins = [all(sign * v[o][cand] < sign * v[o][c] for c in comps) for o in ORIG]
    mean_c = float(np.mean([v[o][cand] for o in ORIG]))
    means = {c: float(np.mean([v[o][c] for o in ORIG])) for c in comps}
    mean_ok = all(sign * mean_c < sign * means[c] for c in comps)
    out = {"wins": int(sum(wins)), "win_origins": [o for o, w in zip(ORIG, wins) if w], "mean": mean_c,
           "comparator_means": means, "mean_ok": bool(mean_ok), "pass": bool(sum(wins) >= 3 and mean_ok)}
    if lower:  # scale-free extra: mean log ratio to each comparator < 0
        lr = {c: float(np.mean([np.log(v[o][cand] / v[o][c]) for o in ORIG])) for c in comps}
        out["extra_mean_log_ratio"] = lr
        out["extra_pass_logratio"] = bool(sum(wins) >= 3 and all(x < 0 for x in lr.values()))
    return out


def verdicts(v, lower=True):
    ot = {f: rule(v, f, ["persist", "meanshift"], lower) for f in OTF}
    vel = {f: rule(v, f, ["ot"], lower) for f in VELF}
    vel_strict = {f: rule(v, f, ["ot", "persist", "meanshift"], lower) for f in VELF}
    sink = rule(v, "uot_growth", ["ot"], lower)
    return {"OT_helps": any(r["pass"] for r in ot.values()), "OT_detail": ot,
            "velocity_helps": any(r["pass"] for r in vel.values()), "velocity_detail": vel,
            "extra_velocity_vs_ot_and_baselines": any(r["pass"] for r in vel_strict.values()),
            "extra_velocity_strict_detail": vel_strict,
            "sink_helps": sink["pass"], "sink_detail": sink,
            "extra_logratio": {"OT_helps": any(r.get("extra_pass_logratio", False) for r in ot.values()),
                               "velocity_helps": any(r.get("extra_pass_logratio", False) for r in vel.values()),
                               "sink_helps": sink.get("extra_pass_logratio", False)}}


def fmt(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:7.3f}"


def main():
    res = json.load(open(os.path.join(HERE, "rescore.json")))
    origins = list(res["origins"])
    rows, out = [], {"modes": {}, "floors": {}, "S_fixed": {}, "reproduction": {}, "selection_changes": {}}
    for s in origins:
        r = res["origins"][s]
        out["floors"][s] = {k: r["floor"][k] for k in DIST}
        out["S_fixed"][s] = r["S_fixed"]
        rep = {}
        for f in FAMS:
            fr = r["families"][f]
            rep[f] = {"argmin_sd_orig_equals_recorded": fr["recorded_reproduced"], **fr["check_vs_first_run"]}
            out["selection_changes"].setdefault(s, {})[f] = {m: fr["sel"][m] for m in fr["sel"]}
        out["reproduction"][s] = rep
    modes = [("A_recorded", "recorded")] + [(f"B_{m}", m) for m in ("sd_fixed", "energy", "mmd")]
    for mode, sel in modes:
        out["modes"][mode] = {}
        for metric in DIST + GENE + ["spread", "scale_orig"]:
            v = table(res, sel, metric)
            for s in origins:
                for f in FAMS:
                    fr = res["origins"][s]["families"][f]
                    rows.append({"mode": mode, "selector": sel, "origin": s, "family": f,
                                 "hp": hkey(fr["sel"][sel]), "metric": metric, "value": v[s][f]})
            if metric in ("sd_fixed", "energy", "mmd", "sd_orig") and all(o in v for o in ORIG):
                out["modes"][mode][metric] = verdicts(v, lower=True)
            if metric == "pearson" and all(o in v for o in ORIG):
                out["modes"][mode]["pearson_secondary"] = verdicts(v, lower=False)
    with open(os.path.join(HERE, "table_corrected.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "verdicts.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    # ---- console summary ----
    print("S_fixed per origin:", {s: round(x, 2) for s, x in out["S_fixed"].items()})
    print("floors:", {s: {k: round(x, 4) for k, x in fl.items()} for s, fl in out["floors"].items()})
    worst = {}
    for s, rep in out["reproduction"].items():
        for f, c in rep.items():
            for k, x in c.items():
                if k.startswith("argmin"):
                    if x is False:
                        print("  hp NOT reproduced:", s, f)
                    continue
                if x is not None:
                    worst[k] = max(worst.get(k, 0.0), x)
    print("reproduction of the first run (max abs diff over origins x families):", worst)
    for mode, sel in modes:
        for metric in (["sd_fixed", "energy", "mmd"] + (["sd_orig"] if mode == "A_recorded" else [])):
            if mode.startswith("B_") and metric != sel:
                continue
            v = table(res, sel, metric)
            print(f"\n[{mode}] metric {metric} (lower better)")
            print("origin " + " ".join(f"{f[:10]:>10}" for f in FAMS))
            for s in origins:
                print(f"{s:>6} " + " ".join(f"{fmt(v[s][f]):>10}" for f in FAMS))
            if metric in out["modes"][mode]:
                vd = out["modes"][mode][metric]
                print(f"  OT helps: {vd['OT_helps']}  ({', '.join(f'{f}: {d['wins']}/4 mean_ok={d['mean_ok']}' for f, d in vd['OT_detail'].items())})")
                print(f"  velocity helps: {vd['velocity_helps']}  ({', '.join(f'{f}: {d['wins']}/4 mean_ok={d['mean_ok']}' for f, d in vd['velocity_detail'].items())})")
                print(f"  sink helps: {vd['sink_helps']}  (uot vs ot {vd['sink_detail']['wins']}/4 mean_ok={vd['sink_detail']['mean_ok']})")
                print(f"  extras: velocity vs ot AND baselines {vd['extra_velocity_vs_ot_and_baselines']}; log-ratio means {vd['extra_logratio']}")
        v = table(res, sel, "pearson")
        print(f"[{mode}] gene PCC of mean change (higher better) under the {sel} hp:")
        for s in origins:
            print(f"{s:>6} " + " ".join(f"{fmt(v[s][f]):>10}" for f in FAMS))
    sp = os.path.join(HERE, "spread_bias.json")
    if os.path.exists(sp):
        sb = json.load(open(sp))
        print("\nDESCRIPTIVE spread-bias check (persist + N(0, sigma^2), vs true next stage):")
        for s, r in sb["origins"].items():
            for x in r["rows"]:
                print(f"  origin {s} sigma {x['sigma']:>4}: sd_orig {x['sd_orig']:.4f} (scale {x['scale_orig']:.1f})"
                      f"  sd_fixed {x['sd_fixed']:.4f}  energy {x['energy']:.4f}  mmd {x['mmd']:.4f}")


if __name__ == "__main__":
    main()
