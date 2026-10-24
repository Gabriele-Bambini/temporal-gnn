"""Judge 1: collect every measured number of this folder into verdict.json (numbers only, plus scores)."""
import json
import os

J = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(J, f))) if os.path.exists(os.path.join(J, f)) else None

v = {"diff_reruns_vs_originals": L("diff_results.json"),
     "strict_truncation_tests": {"nass": L("strict_nass.json"), "lsb": L("strict_lsb.json"), "popot": L("strict_popot.json")},
     "independent_reimplementation": L(os.path.join("indep", "indep_results.json")),
     "ot_increment_span": L("ot_span.json"), "ot_standalone": L("ot_standalone.json"), "noise_share_l1": L("noise_share.json")}

# LSB with a causal nmed (patched copy) vs the original nmed (all bins)
lsbd = {}
for nm in ("pancreas-human", "gastrulation-human"):
    a, b = L(os.path.join("rerun_lsb", f"results_{nm}.json")), L(os.path.join("rerun_lsb_causal", f"results_{nm}.json"))
    if a and b:
        lsbd[nm] = {tg: {s: {"orig_nmed_i>=1": [a["methods"][tg][s]["i>=1"][k] for k in ("pcc", "up", "down")],
                             "causal_nmed_i>=1": [b["methods"][tg][s]["i>=1"][k] for k in ("pcc", "up", "down")]}
                         for s in a["methods"][tg] if s in ("lsb_full", "lsb_obs", "k4_var_fixed", "lsb_fixed_k4ar", "lsb_stack", "regime+lsb")}
                    for tg in ("orig", "l1", "linear")}
v["lsb_nmed_leak_effect"] = lsbd

# held-out (library-normalised, non-integer) non-human builds, real NASS copy
ho = L(os.path.join("heldout_nass", "results.json"))
if ho:
    v["heldout_nonhuman_builds_nass"] = {nm: {tg: {m: ho[nm]["targets"][tg]["table"][m]["i>=2"] for m in ("trend", "regime", "full")}
                                              for tg in ("orig", "l1", "linear")} for nm in ho}
v["scoring_axes"] = {"A_beats_trend_and_DistBridge_where_it_matters(l1 both datasets; orig for comparability)": 5,
                     "B_genuinely_uses_OT_SB_and_it_contributes": 2, "C_interpretable": 1.5, "D_feasible_strict_reproducible": 1.5}
v["scores"] = {"NASS": {"A": 4.0, "B": 0.0, "C": 1.2, "D": 1.3, "total": 6.5},
               "PopOT": {"A": 1.0, "B": 1.0, "C": 1.1, "D": 1.1, "total": 4.2},
               "LatentSB": {"A": 0.8, "B": 0.5, "C": 0.8, "D": 0.5, "total": 2.6}}
v["key_adversarial_findings"] = {
    "LSB_nmed_leak": "self.nmed = median cell count over ALL bins caps the SVD column weights; with a causal median the "
                     "only claimed gain (regime+lsb vs regime, gastr l1 i>=3, +0.016 p=0.04) becomes +0.0002 (16/10, p=0.55)",
    "NASS_heldout_nonhuman_builds": "library-normalised non-integer builds (s^2/m ~0.03): NASS ties trend on l1",
    "OT_standalone": "per-gene OT push on top of the box-4 level: orig -0.018 gastr (1/26 better), -0.002 pancr; "
                     "l1 +0.06 gastr / +0.015 pancr but ends at 0.16 / -0.12, below the noise-aware stack"}
json.dump(v, open(os.path.join(J, "verdict.json"), "w"), indent=1)
print("written", os.path.join(J, "verdict.json"))
