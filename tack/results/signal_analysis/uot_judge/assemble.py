"""Judge: merge every check of uot_judge into judge_results.json (no new computation).
Usage: python assemble.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(f):
    p = os.path.join(HERE, f)
    return json.load(open(p)) if os.path.exists(p) else "missing"


def main():
    out = {"what": "adversarial judge of uot_pop_sink (population sink, p1) and uot_gene_sink (gene sink, p2)",
           "reproduction_reruns_vs_prototype_outputs": {}, "strictness": {}, "independent_data_and_scores": load("indep_check.json"),
           "p1_closed_form_hybrid": {}, "p1_matched_config_profile": {}, "p1_claims_extracted": {}, "p1_free_turnover": {},
           "p2_solver_convergence": load("conv_p2.json"), "external_comparisons": load("external_compare.json")}
    cr = load("compare_rerun.json")
    if cr != "missing":
        for ds, o in cr.items():
            out["reproduction_reruns_vs_prototype_outputs"][ds] = {
                k: {kk: vv for kk, vv in v.items() if kk in ("max", "argmax", "n_num", "n_big", "big", "n_str_mismatch",
                                                            "missing", "max_abs_diff_over_arrays", "n_arrays_nonzero_diff")}
                for k, v in o.items()}
    for f in sorted(glob.glob(os.path.join(HERE, "strict_*.json"))):
        out["strictness"][os.path.basename(f)[:-5]] = json.load(open(f))
    for ds in ("pancreas-human", "gastrulation-human"):
        h = load(f"hybrid_p1_{ds}.json")
        if h != "missing":
            out["p1_closed_form_hybrid"][ds] = {k: h[k] for k in ("tv_sinknorm_lam0.1_R1_vs_hyb", "selected_sink_norm_lambda_counts",
                                                                 "scores", "paired")}
        out["p1_matched_config_profile"][ds] = load(f"profile_p1_{ds}.json")
        out["p1_claims_extracted"][ds] = load(f"claims_p1_{ds}.json")
        out["p1_free_turnover"][ds] = load(f"turnover_p1_{ds}.json")
    for f in ("posthoc_down_side.json", "level_check.json"):
        out[f"p2_rerun_{f[:-5]}"] = load(os.path.join("p2", f))
    out["p1_rerun_extra_paired"] = load(os.path.join("p1", "extra_paired.json"))
    with open(os.path.join(HERE, "judge_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("written", os.path.join(HERE, "judge_results.json"))


if __name__ == "__main__":
    main()
