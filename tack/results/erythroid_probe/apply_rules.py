"""Apply the rules written in docs/superpowers/decisions.md (2026-10-02, before job 679104) to the erythroid probe.

R0 calibration: AP of random initialisations >= 0.6 and fraction of kept genes in [0.3, 0.6].
R2 K0: a fold counts only if the best control's regret against the oracle exceeds 3 SE.
R1 'age' POSITIVE if the paired 95% CI of (age - lmix) has lower bound > 0 in >= 4/5 interior folds;
   NEGATIVE if the upper bound < 0 in >= 3/5; otherwise TIE. Same criterion reported for 'gauss'.
R3 terminal fold descriptive only. R4 c reported with its profile, no clock claims.
Note: R1 counts out of 5 interior folds; folds failing K0 count as 'neither' (cannot support either verdict).
"""
import json
import pathlib

here = pathlib.Path(__file__).parent
probe = json.loads((here / "probe.json").read_text())
repro = json.loads((here / "repro.json").read_text())

out = {}
ri = repro["random_init"]
ws = repro["warm_start"]
frac_kept = ws["n_good"] / ws["n_genes"]
out["R0"] = dict(average_precision=ri["average_precision"], frac_correct=ri["frac_correct"],
                 frac_genes_kept=frac_kept, n_good=ws["n_good"], n_genes=ws["n_genes"],
                 passed=bool(ri["average_precision"] >= 0.6 and 0.3 <= frac_kept <= 0.6))

interior = ["E7.25", "E7.5", "E7.75", "E8.0", "E8.25"]
table = []
for h, f in probe["folds"].items():
    fr = f["frozen"]
    row = dict(fold=h, terminal=f["terminal"], K0_headroom_se=fr["K0"]["headroom_in_se"],
               K0_pass=bool(fr["K0"]["headroom_in_se"] is not None and fr["K0"]["headroom_in_se"] > 3),
               best_control=fr["K0"]["best_control"], ll=fr["ll"], ll_oracle=fr["ll_oracle"],
               age_minus_lmix=fr["paired"]["age-lmix"], gauss_minus_lmix=fr["paired"]["gauss-lmix"],
               age_minus_gauss=fr["paired"]["age-gauss"], age_minus_uniform=fr["paired"]["age-uniform"],
               gap_closed=fr["gap_closed"], c_hours_per_unit=f["fitted"]["age"]["c_hours_per_unit"],
               refit_age_minus_lmix=f["refit"]["paired"]["age-lmix"],
               cross_ageDecoder_agePrior_minus_uniformDecoder_lmix=f["cross"]["ageDecoder_agePrior_minus_uniformDecoder_lmix"],
               c_profile=f["c_profile"])
    table.append(row)
out["folds"] = table


def verdict(key):
    pos = neg = 0
    for r in table:
        if r["fold"] not in interior or not r["K0_pass"]:
            continue
        mean, lo, hi, _ = r[key]
        pos += lo > 0
        neg += hi < 0
    v = "POSITIVE" if pos >= 4 else ("NEGATIVE" if neg >= 3 else "TIE")
    return dict(n_ci_above_0=int(pos), n_ci_below_0=int(neg),
                n_counting_folds=int(sum(r["K0_pass"] for r in table if r["fold"] in interior)), verdict=v)


out["R1_age_vs_lmix_frozen"] = verdict("age_minus_lmix")
out["R1_gauss_vs_lmix_frozen"] = verdict("gauss_minus_lmix")
out["R1_age_vs_lmix_refit"] = verdict("refit_age_minus_lmix")
(here / "verdict.json").write_text(json.dumps(out, indent=1))

print("R0", json.dumps(out["R0"]))
for r in table:
    print(r["fold"], "K0", round(r["K0_headroom_se"] or 0, 1), r["K0_pass"], "best", r["best_control"],
          "| age-lmix", [round(x, 3) for x in r["age_minus_lmix"][:3]],
          "| gauss-lmix", [round(x, 3) for x in r["gauss_minus_lmix"][:3]],
          "| refit age-lmix", [round(x, 3) for x in r["refit_age_minus_lmix"][:3]],
          "| c", round(r["c_hours_per_unit"], 1))
for k in ("R1_age_vs_lmix_frozen", "R1_gauss_vs_lmix_frozen", "R1_age_vs_lmix_refit"):
    print(k, json.dumps(out[k]))
print("c profiles (hours per unit: relative train LL):")
for r in table:
    prof = r["c_profile"]
    best = max(v for _, v in prof)
    print(r["fold"], [(round(c, 1), round(v - best, 4)) for c, v in prof])
