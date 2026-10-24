"""Judge 2: collect the key measured numbers of this folder into judge2_verdict.json (no new computation)."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.dirname(HERE)
J = lambda *p: json.load(open(os.path.join(*p)))
out = {"what": "adversarial judge 2: reproduction, truncation strictness, paired DistBridge, independent re-implementation,"
               " robustness builds", "files": sorted(os.path.basename(f) for f in glob.glob(os.path.join(HERE, "*.py")))}

out["reproduction"] = {
    "NASS (nass.py copy, both builds)": "1728 + 1728 table cells and all paired deltas within 0.0025 of design results.json"
                                        " ('full', 'trend', 'level', 'regime' rows identical; only grn/ot variants differ <=0.002)",
    "LatentSB (run_lsb.py copy, both builds)": "1428 + 1938 method cells and 5334 + 5355 paired cells: max |diff| 0.0",
    "PopOT (popot.py copy, both builds)": "3139 + 4944 target leaves, composition TV and crossgene reproduction: max |diff| 0.0",
    "caches": "_node_*.npz (NASS) and _cache_*.npz (LatentSB) equal the node CSVs exactly; l1 target equals"
              " dileo_protocol.TARGET_NP to 1e-7"}

# truncation and OT spot-check
out["truncation"] = [json.loads(l) for l in open(os.path.join(HERE, "trunc_results.jsonl"))]
if os.path.exists(os.path.join(HERE, "ot_spot.jsonl")):
    out["ot_feature_spot_check"] = [json.loads(l) for l in open(os.path.join(HERE, "ot_spot.jsonl"))]

# independent re-implementation of the NASS core (l1)
out["independent_nass_core_l1"] = {n: {k: v for k, v in J(HERE, f"indep_{n}.json").items() if k != "dataset"}
                                   for n in ("pancreas-human", "gastrulation-human")
                                   if os.path.exists(os.path.join(HERE, f"indep_{n}.json"))}

# frozen NASS on other builds / masks
rows = ("trend", "full", "regime", "level(box4+obs)", "level(box4)", "full[h=0.5 fixed]")
ho = {}
for f in sorted(glob.glob(os.path.join(HERE, "heldout_*.json"))):
    r = json.load(open(f))
    key = os.path.basename(f)[8:-5]
    ho[key] = {"n_cells": r["n_cells"], "T": r["T"]}
    for tg, t in r["targets"].items():
        ho[key][tg] = {"i>=2 [pcc, up, down]": {k: [t["table"][k]["i>=2"][m] for m in ("pcc", "p200_up", "p200_down")]
                                               for k in rows if k in t["table"]},
                       "paired_i>=2": {k: v["pcc"] for k, v in t["paired_i>=2"].items()
                                       if k in ("full vs trend", "full vs regime", "full vs level(box4+obs)")}}
out["nass_frozen_other_builds"] = ho
out["nass_frozen_other_builds_note"] = (
    "'pancreas'/'gastrulation' (Dileo mouse builds) hold smoothed non-count values (non-integer sums, 98-99 % nonzero,"
    " sd/sqrt(m) ~0.15 vs ~1.0 on '-human'), so the count noise model and the finite-cell head are invalid there: NOT a"
    " valid held-out. gastr-merged2-off0/off1 = '-human' gastrulation with consecutive bins merged in pairs (same"
    " cells, half the sampling noise): a different noise regime, not a replicate. pancreas-human_seed99 = same build,"
    " different 25 % masks.")

# LatentSB causal-nmed
lc = {}
for n in ("pancreas-human", "gastrulation-human"):
    a = J(S, "design_latent-sb", f"results_{n}.json")
    b_f = os.path.join(HERE, "lsb_causal", f"results_{n}.json")
    if not os.path.exists(b_f):
        continue
    b = json.load(open(b_f))
    lc[n] = {}
    for tg in ("l1", "linear", "orig"):
        for m in ("lsb_full", "regime+lsb", "lsb_stack", "lsb_obs", "k4_var_fixed"):
            if m in a["methods"][tg]:
                lc[n][f"{tg}|{m}|i>=1 pcc"] = [a["methods"][tg][m]["i>=1"]["pcc"], b["methods"][tg][m]["i>=1"]["pcc"]]
    for k in ("regime+lsb_vs_regime_i>=3", "lsb_full_vs_trend_i>=1"):
        lc[n][f"l1|{k}"] = [a["paired"]["l1"][k]["pcc"], b["paired"]["l1"][k]["pcc"]]
out["latentsb_nmed_noncausal_vs_causal"] = lc

# paired DistBridge
if os.path.exists(os.path.join(HERE, "pair_db.json")):
    out["paired_vs_distbridge_same_masks"] = J(HERE, "pair_db.json")
for f in sorted(glob.glob(os.path.join(HERE, "db_*.json"))):
    out.setdefault("distbridge_on_nass_masks", {})[os.path.basename(f)] = {
        k: {"table": v["table"], "selected": v["selected"]} for k, v in json.load(open(f)).items()}

json.dump(out, open(os.path.join(HERE, "judge2_verdict.json"), "w"), indent=1)
print("written", len(json.dumps(out)), "bytes")
