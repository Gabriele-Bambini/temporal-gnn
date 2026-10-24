"""Collects the headline numbers of results_l1.json, results_linear.json, orig/results_orig.json and
truncation_<name>.json into summary.json (CONTEXT masks, i >= 1 and i >= 2; PCC / tie-aware P@200 up / down;
paired deltas vs NASS with Wilcoxon over transitions)."""
import json
import os

B = os.path.dirname(os.path.abspath(__file__))
NAMES = ("gastrulation-human", "pancreas-human")
SH = {"gastrulation-human": "gastr", "pancreas-human": "pancr"}


def tri(s):
    return [s["pcc"], s["up"], s["down"]]


def main():
    l1 = json.load(open(os.path.join(B, "results_l1.json")))
    lin = json.load(open(os.path.join(B, "results_linear.json")))
    org = json.load(open(os.path.join(B, "orig", "results_orig.json")))
    out = {"masks": "CONTEXT 1000 s + i, 3 seeds; metrics pcc / p_ta_up / p_ta_down", "targets": {}}
    spec = {"l1": (l1, "V0_full", ["V0_full", "V1_full+grn", "P_primary", "Pn_null", "V2_V1+grnbar", "V2b_full+grnbar",
                                   "V2n_V1+grnbarnull", "Vd_full+fgap+fmom", "Vm_V1+grnbar_mom", "Vauto_V1|V2|P",
                                   "trend", "level_obs(DistBridge nosb l1)"]),
            "linear": (lin, "V0_full", ["V0_full", "V1_full+grn", "trend", "level_obs(DistBridge nosb linear)"]),
            "orig": (org, "NASS_full_orig", ["NASS_full_orig", "O2_nass+C_head", "C_post_all[rolling]",
                                             "B_phat+post0[rolling]", "base[rolling]", "trend"])}
    for tg, (R, ref, models) in spec.items():
        o = {}
        for nm in NAMES:
            r = R[nm]
            rows = {}
            for m in models:
                t = r["table"][m]
                rec = {"i>=1": tri(t["ctx_i>=1"]), "i>=2": tri(t["ctx_i>=2"]),
                       "nass_masks_i>=2": [t["own_i>=2"]["pcc"], t["own_i>=2"]["pta_up"], t["own_i>=2"]["pta_down"]]}
                k = f"{m} - {ref}"
                if k in r["paired"]:
                    p = r["paired"][k]
                    rec["vs_NASS"] = {rng: {q: [p[rng][q]["mean_delta"], f"{p[rng][q]['better']}/{p[rng][q]['worse']}",
                                                p[rng][q]["wilcoxon_p"]] for q in ("pcc", "up", "down")}
                                      for rng in ("ctx_i>=1", "ctx_i>=2")}
                rows[m] = rec
            o[SH[nm]] = rows
        out["targets"][tg] = o
    out["l1_acceptance"] = l1.get("acceptance")
    out["orig_prediction_C_minus_base"] = {SH[nm]: org[nm]["prediction_C_minus_base"] for nm in NAMES}
    out["orig_reproduce_c12"] = {SH[nm]: org[nm]["reproduce_c12"] for nm in NAMES}
    tr = {}
    for nm in NAMES:
        f = os.path.join(B, f"truncation_{nm}.json")
        if os.path.exists(f):
            tr[SH[nm]] = {k: {"pass": v["pass"], **v["summary"]} for k, v in json.load(open(f)).items()}
    out["truncation"] = tr
    json.dump(out, open(os.path.join(B, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:20000])


if __name__ == "__main__":
    main()
