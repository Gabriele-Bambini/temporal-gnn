"""Tables and verdicts for attempt_forecast (reads scores_o<s>.json, pools_o<s>.json; writes table.csv, verdicts.json).

Selection designs (fixed in attempt_forecast.py's docstring before scoring): 'roll' PRIMARY (rolling origin), 'same'
SENSITIVITY (first-run backtest), 'hind' DESCRIPTIVE (best hp in hindsight on the true target, not a forecast).
First-run families: diag_metric Mode-B hp for the metric being judged, re-run in attempt_forecast.py.
Rule (decisions.md 2026-10-24): beats = strictly lower (a relative tolerance of 1e-9 turns float noise into a tie) in
>= 3 of origins 2-5 AND lower raw mean over 2-5. Literal pre-registered verdicts use the first-run families only;
the same rule applied to the new families is POST HOC and labelled so.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ERY = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import attempt_forecast as A  # noqa: E402

ORIG = ["2", "3", "4", "5"]
OLD = A.OLD
NEW = list(A.FAMILIES)
OLD_OT = ["ot", "uot_growth", "ot_vel_ss", "ot_vel_dyn"]
OLD_VEL = ["vel_ss", "vel_dyn", "ot_vel_ss", "ot_vel_dyn"]
METRICS = ["sd_fixed", "energy"]
TOL = 1e-9
hkey = A.hkey


def load():
    sc = {str(s): A.load_scores(s) for s in range(1, 6)}
    pools = {}
    for s in ORIG:
        p = os.path.join(HERE, f"pools_o{s}.json")
        if os.path.exists(p):
            pools[s] = json.load(open(p))
    rescore = json.load(open(os.path.join(ERY, "diag_metric", "rescore.json")))
    return sc, pools, rescore


def sel_hp(sc, fam, s, m, des):
    cur, prev = sc[s], sc.get(str(int(s) - 1))
    return A.selected_hps(fam, int(s), m, cur, prev).get(des)


def value_table(sc, rescore, m, des, field=None):
    """v[origin][family] = score `field` (default: m) of the forecast selected by metric m under design des."""
    field = m if field is None else field
    v, hp = {}, {}
    for s in ORIG:
        v[s], hp[s] = {}, {}
        for f in OLD:
            k = hkey(rescore["origins"][s]["families"][f]["sel"][m])
            v[s][f] = sc[s]["baselines"][f][k].get(field, float("nan"))
            hp[s][f] = k
        for f in NEW:
            h = sel_hp(sc, f, s, m, des)
            v[s][f] = sc[s]["final"][f][hkey(h)].get(field, float("nan"))
            hp[s][f] = hkey(h)
    return v, hp


def lt(a, b):
    return a < b - TOL * abs(b)


def rule(v, cand, comps):
    wins = [all(lt(v[o][cand], v[o][c]) for c in comps) for o in ORIG]
    mean_c = float(np.mean([v[o][cand] for o in ORIG]))
    means = {c: float(np.mean([v[o][c] for o in ORIG])) for c in comps}
    mean_ok = all(lt(mean_c, means[c]) for c in comps)
    ties = [o for o in ORIG if any(abs(v[o][cand] - v[o][c]) <= TOL * abs(v[o][c]) for c in comps)]
    return {"wins": int(sum(wins)), "win_origins": [o for o, w in zip(ORIG, wins) if w], "ties_at": ties,
            "mean": mean_c, "comparator_means": means, "mean_ok": bool(mean_ok),
            "pass": bool(sum(wins) >= 3 and mean_ok)}


def verdict_block(v, tags):
    out = {"preregistered_literal": {
        "OT_helps": {f: rule(v, f, ["persist", "meanshift"]) for f in OLD_OT},
        "velocity_helps": {f: rule(v, f, ["ot"]) for f in OLD_VEL},
        "sink_helps": {"uot_growth": rule(v, "uot_growth", ["ot"])}}}
    lit = out["preregistered_literal"]
    out["preregistered_literal_summary"] = {k: any(r["pass"] for r in lit[k].values()) for k in lit}
    ot_new = [f for f in NEW if "OT" in tags[f]]
    vel_new = [f for f in NEW if "VEL" in tags[f]]
    ext = {"OT_helps_ext": {f: rule(v, f, ["persist", "meanshift"]) for f in ot_new},
           "velocity_helps_ext_vs_ot": {f: rule(v, f, ["ot"]) for f in vel_new},
           "velocity_ext_vs_ot_persist_meanshift": {f: rule(v, f, ["ot", "persist", "meanshift"]) for f in vel_new},
           "sink_helps_ext_vs_ot": {f: rule(v, f, ["ot"]) for f in ("uot_rw", "uot_damp", "c_uot")},
           "sink_ext_vs_ot_damp": {f: rule(v, f, ["ot_damp"]) for f in ("uot_rw", "uot_damp")},
           "all_new_vs_persist_meanshift": {f: rule(v, f, ["persist", "meanshift"]) for f in NEW},
           "ablation_vs_contract": {f: rule(v, f, ["contract"]) for f in
                                    ("c_trend", "c_tilt", "c_ot", "c_uot", "c_vel", "c_comp")},
           "contract_vs_persist": rule(v, "contract", ["persist"])}
    out["post_hoc_extension"] = ext
    out["post_hoc_summary"] = {
        "OT_helps_ext": sorted(f for f, r in ext["OT_helps_ext"].items() if r["pass"]),
        "velocity_helps_ext_vs_ot": sorted(f for f, r in ext["velocity_helps_ext_vs_ot"].items() if r["pass"]),
        "velocity_ext_strict": sorted(f for f, r in ext["velocity_ext_vs_ot_persist_meanshift"].items() if r["pass"]),
        "sink_ext_vs_ot": sorted(f for f, r in ext["sink_helps_ext_vs_ot"].items() if r["pass"]),
        "sink_ext_vs_ot_damp": sorted(f for f, r in ext["sink_ext_vs_ot_damp"].items() if r["pass"]),
        "new_families_passing_vs_persist_meanshift": sorted(
            f for f, r in ext["all_new_vs_persist_meanshift"].items() if r["pass"]),
        "combos_beating_contract_alone": sorted(f for f, r in ext["ablation_vs_contract"].items() if r["pass"])}
    return out


def pool_breakdown(pools, sc, rescore, m, des, fams):
    """per origin: pools where the family is strictly better than persistence (same metric m), mean difference."""
    out = {}
    for s, pr in pools.items():
        out[s] = {}
        k0 = hkey(rescore["origins"][s]["families"]["persist"]["sel"][m])
        base = pr["scores"]["OLD_persist"][k0]
        for f in fams:
            if f in OLD:
                k = hkey(rescore["origins"][s]["families"][f]["sel"][m])
                r = pr["scores"]["OLD_" + f][k]
            else:
                k = hkey(sel_hp(sc, f, s, m, des))
                r = pr["scores"][f][k]
            diffs = {p: r[p][m] - base[p][m] for p in base}
            out[s][f] = {"n_pools": len(diffs), "pools_better": int(sum(lt(r[p][m], base[p][m]) for p in base)),
                         "mean_diff": float(np.mean(list(diffs.values()))), "diffs": diffs,
                         "persist_by_pool": {p: base[p][m] for p in base}}
    return out


def floors(rescore):
    out = {s: {"halves_sd_fixed": rescore["origins"][s]["floor"]["sd_fixed"],
               "halves_energy": rescore["origins"][s]["floor"]["energy"]} for s in ORIG}
    p = os.path.join(ERY, "diag_batch", "floors.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if r["kind"] == "origin" and r["k"] in out and r["rep"] == "pca":
                out[r["k"]].update(loso_energy_V=float(r["loso_ED_V_fs_mean"]), oracle_energy_V=float(r["oracle_ED_V_fs"]))
    return out


def fmt(x, w=7):
    return " " * (w - 3) + "nan" if x is None or not np.isfinite(x) else f"{x:{w}.3f}"


def short_hp(k):
    h = json.loads(k)
    return ",".join(f"{a}={b:g}" for a, b in h.items() if not isinstance(b, str)) or "-"


def main():
    sc, pools, rescore = load()
    tags = {f: set(sc["2"]["families"][f]["tags"]) for f in NEW}
    fl = floors(rescore)
    out = {"floors": fl, "meta": {s: sc[s]["meta"] for s in sc if sc[s]}, "verdicts": {}, "pools": {},
           "selected_hp": {}, "hindsight": {}}
    rows = []
    print("Targets:", {s: sc[s]["meta"]["target"] for s in ORIG}, " rho_xfit:",
          {s: round(sc[s]["meta"]["rho_xfit"], 3) for s in sc if sc[s]})
    print("max |diff| of the re-run first-run families vs rescore.json (same noise draws):",
          {s: f"{sc[s]['meta']['max_abs_diff_vs_rescore_same_draws']:.1e}" for s in ORIG})
    for s in ORIG:
        for f, hs in sc[s]["check_vs_rescore"].items():
            for k, c in hs.items():
                if c["noise_draw_differs"]:
                    print(f"  origin {s} {f} {k}: different noise draw from rescore (recorded hp, global RNG): "
                          f"|d energy| {c['energy']:.3f} |d sd_fixed| {c['sd_fixed']:.3f}")
    print("floors (sampling halves; diag_batch leave-one-pool-out and oracle, energy V-stat):",
          {s: {k: round(x, 3) for k, x in v.items()} for s, v in fl.items()})
    for m in METRICS:
        for des in ("roll", "same", "hind"):
            v, hp = value_table(sc, rescore, m, des)
            vb = verdict_block(v, tags)
            out["verdicts"][f"{m}|{des}"] = vb
            out["selected_hp"][f"{m}|{des}"] = hp
            for s in ORIG:
                for f in OLD + NEW:
                    src = (sc[s]["baselines"][f][hp[s][f]] if f in OLD else sc[s]["final"][f][hp[s][f]])
                    rows.append({"selector": m, "design": des if f in NEW else "modeB_same", "origin": s,
                                 "target": sc[s]["meta"]["target"], "family": f,
                                 "tags": ";".join(sorted(tags.get(f, set()))) if f in NEW else "first_run",
                                 "hp": hp[s][f], **{k: src.get(k, float("nan")) for k in
                                                    ("sd_fixed", "energy", "mmd", "energy_gene", "pearson",
                                                     "p200_up", "p200_down", "spread", "ess")}})
            label = {"roll": "PRIMARY rolling-origin hp", "same": "SENSITIVITY first-run backtest hp",
                     "hind": "DESCRIPTIVE best hp in hindsight (not a forecast)"}[des]
            print(f"\n==== metric {m} ({'primary' if m == 'sd_fixed' else 'secondary'}), {label}; lower = better")
            print(f"{'family':>12} {'tags':>10} " + " ".join(f"{'o' + s:>7}" for s in ORIG) +
                  f" {'mean':>7}  wins_vs_P+M  hp(o2..o5)")
            for f in OLD + NEW:
                vals = [v[s][f] for s in ORIG]
                r = rule(v, f, ["persist", "meanshift"]) if f not in ("persist", "meanshift") else None
                tg = "first-run" if f in OLD else ",".join(sorted(tags[f]))
                print(f"{f:>12} {tg[:10]:>10} " + " ".join(fmt(x) for x in vals) + f" {fmt(np.mean(vals))}  " +
                      (f"{r['wins']}/4{' PASS' if r['pass'] else ''}{' mean_ok' if r['mean_ok'] else ''}"
                       if r else "   -   ") + "  " +
                      ("" if f in ("persist", "meanshift") else " | ".join(short_hp(hp[s][f]) for s in ORIG)))
            print("  pre-registered (literal, first-run families):", vb["preregistered_literal_summary"])
            print("  post hoc extension:", vb["post_hoc_summary"])
    # secondary gene-level metrics and gene-space energy for the primary selection
    for m in METRICS:
        print(f"\n==== secondary, hp by {m} under 'roll' (first-run families Mode B): gene PCC of the mean change, "
              "P@200 up, gene-space energy (robustness column)")
        vp, _ = value_table(sc, rescore, m, "roll", "pearson")
        vu, _ = value_table(sc, rescore, m, "roll", "p200_up")
        vg, _ = value_table(sc, rescore, m, "roll", "energy_gene")
        print(f"{'family':>12} " + " ".join(f"{'pcc' + s:>7}" for s in ORIG) + " " +
              " ".join(f"{'p@' + s:>6}" for s in ORIG) + " " + " ".join(f"{'Eg' + s:>7}" for s in ORIG))
        for f in OLD + NEW:
            print(f"{f:>12} " + " ".join(fmt(vp[s][f]) for s in ORIG) + " " +
                  " ".join(fmt(vu[s][f], 6) for s in ORIG) + " " + " ".join(fmt(vg[s][f]) for s in ORIG))
        out[f"gene_space_energy_rule|{m}|roll"] = {f: rule(vg, f, ["persist", "meanshift"])
                                                   for f in OLD + NEW if f not in ("persist", "meanshift")}
        out[f"pcc|{m}|roll"] = vp
    # check column: textbook Sinkhorn divergence of the forecasts selected by sd_fixed / energy
    if pools:
        for m in METRICS:
            for des in ("roll", "same"):
                vf = {}
                for s in ORIG:
                    vf[s] = {}
                    for f in OLD:
                        k = hkey(rescore["origins"][s]["families"][f]["sel"][m])
                        vf[s][f] = pools[s]["pooled_full"]["OLD_" + f][k]["sd_full"]
                    for f in NEW:
                        k = hkey(sel_hp(sc, f, s, m, des))
                        vf[s][f] = pools[s]["pooled_full"][f][k]["sd_full"]
                vb = verdict_block(vf, tags)
                out["verdicts"][f"sd_full_check|hp_by_{m}|{des}"] = vb
                if des == "roll":
                    print(f"\n==== CHECK sd_full (textbook Sinkhorn divergence) of the forecasts selected by {m}, "
                          f"'{des}' hp")
                    for f in OLD + NEW:
                        vals = [vf[s][f] for s in ORIG]
                        r = rule(vf, f, ["persist", "meanshift"]) if f not in ("persist", "meanshift") else None
                        print(f"{f:>12} " + " ".join(fmt(x) for x in vals) + f" {fmt(np.mean(vals))}  " +
                              (f"{r['wins']}/4{' PASS' if r['pass'] else ''}" if r else ""))
                print(f"  [{m}|{des}] literal:", vb["preregistered_literal_summary"], " post hoc:",
                      vb["post_hoc_summary"])
    mc = os.path.join(HERE, "metric_check.json")
    if os.path.exists(mc):
        mcd = json.load(open(mc))
        out["metric_check"] = mcd
        print("\n==== DESCRIPTIVE metric check: half A of the true stage contracted by a vs half B (argmin over a)")
        for s, v in mcd.items():
            for disp in (0.0, 1.0):
                rr = [r for r in v["rows"] if r["displaced"] == disp]
                print(f"  origin {s} {'displaced' if disp else 'same loc.'}: " + "  ".join(
                    f"{k} argmin a={min(rr, key=lambda r: r[k])['a']:.1f} "
                    f"({min(r[k] for r in rr) / rr[0][k]:.2f} of a=0)" for k in ("sd_fixed", "sd_full", "energy", "mmd")))
    # pool breakdown
    if pools:
        for m in METRICS:
            pb = pool_breakdown(pools, sc, rescore, m, "roll", [f for f in OLD + NEW if f != "persist"])
            out["pools"][f"{m}|roll"] = pb
            print(f"\n==== target-pool breakdown, metric {m}, 'roll' hp: pools where the family beats persistence "
                  "(of n) and mean difference to persistence over pools")
            print("pools:", {s: pools[s]["pools"] for s in pools}, "batches:", {s: pools[s]["batch"] for s in pools})
            for f in [x for x in OLD + NEW if x != "persist"]:
                print(f"{f:>12} " + "  ".join(f"o{s} {pb[s][f]['pools_better']}/{pb[s][f]['n_pools']} "
                                              f"{pb[s][f]['mean_diff']:+.3f}" for s in pools))
    with open(os.path.join(HERE, "verdicts.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else float(o))
    with open(os.path.join(HERE, "table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
