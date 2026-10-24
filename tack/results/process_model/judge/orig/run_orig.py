"""Step 6: orig. Base (DistBridge no-SB, rolling pool: the level + observation-head baseline), A, B, C (primary),
D heads with the strict binomial detection filter (orig_lib, copied c12 module); NASS orig full and O2 (NASS orig
stacker with C's P0 and head output swapped in); trend. All recomputed here on the same masks.
Scoring: CONTEXT masks (1000 s + i, 3 seeds), dileo_rescore.metrics, i >= 1 and i >= 2; NASS's own 20 masks at i >= 2.
Paired: per-transition mask-averaged deltas, Wilcoxon over transitions (i >= 1 and i >= 2).
Reproduction targets (c12, CONTEXT i >= 1): base gastr 0.599/0.417/0.550, C 0.608/0.425/0.547;
base pancr 0.510/0.327/0.442, C 0.528/0.333/0.454.
Predictions: gastr C - base PCC +0.005..+0.012, up +0.005..+0.012, down within +-0.005; pancr not significant.
Outputs: results_orig.json, pertrans_orig_<name>.npz, run_orig.log (this folder)."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orig_lib as O  # noqa: E402

L, D, NS = O.L, O.D, O.NS
REPRO = {"gastrulation-human": {"base[rolling]": (0.599, 0.417, 0.550), "C_post_all[rolling]": (0.608, 0.425, 0.547)},
         "pancreas-human": {"base[rolling]": (0.510, 0.327, 0.442), "C_post_all[rolling]": (0.528, 0.333, 0.454)}}


def main():
    log = open(os.path.join(HERE, "run_orig.log"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    say(f"==== run_orig {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out = {}
    for name in L.NAMES:
        t0 = time.time()
        d = D.Data(name)
        T = d.T
        Y, preds, chosen, setting = O.orig_all(d)
        cmk = L.ctx_masks(d.G, T)
        nmk = D.masks(d.G, T)
        S = {k: L.score_ctx(Y, P, cmk) for k, P in preds.items()}
        SN = {k: D.score(Y, P, nmk) for k, P in preds.items()}
        tab = {k: {"ctx_i>=1": L.summ(S[k], 1), "ctx_i>=2": L.summ(S[k], 2),
                   "own_i>=2": D.summarize(SN[k], list(range(2, T))), "own_all": D.summarize(SN[k], list(range(T)))}
               for k in preds}
        say(f"== {name} orig ({time.time() - t0:.0f}s)")
        for k in preds:
            o = tab[k]["own_i>=2"]
            say(f"  {k:24s} ctx i>=1 {L.fmt(tab[k]['ctx_i>=1'])} | ctx i>=2 {L.fmt(tab[k]['ctx_i>=2'])} | "
                f"own i>=2 {o['pcc']:+.4f}/{o['pta_up']:.4f}/{o['pta_down']:.4f}")
        repro = {}
        for k, exp in REPRO[name].items():
            got = tab[k]["ctx_i>=1"]
            g3 = (got["pcc"], got["up"], got["down"])
            repro[k] = {"expected": exp, "got": g3, "pass": bool(all(abs(a - b) <= 0.0011 for a, b in zip(g3, exp)))}
            say(f"  reproduce c12 {k}: {'PASS' if repro[k]['pass'] else 'FAIL'} {g3} vs {exp}")
        pr = {}
        pairs = [("C_post_all[rolling]", "base[rolling]"), ("B_phat+post0[rolling]", "base[rolling]"),
                 ("A_phat[rolling]", "base[rolling]"), ("D_base+post0[rolling]", "base[rolling]"),
                 ("O2_nass+C_head", "NASS_full_orig"), ("C_post_all[rolling]", "O2_nass+C_head")]
        pairs += [(k, "NASS_full_orig") for k in preds if k != "NASS_full_orig" and (k, "NASS_full_orig") not in pairs]
        for a, b in pairs:
            pr[f"{a} - {b}"] = {"ctx_i>=1": L.paired_ctx(S[a], S[b], 1), "ctx_i>=2": L.paired_ctx(S[a], S[b], 2),
                                "nass_masks_i>=2": NS.paired(SN[a], SN[b], list(range(2, T)))}
            say(f"  paired {a + ' - ' + b:44s} ctx i>=1 {L.fmt_pair(pr[f'{a} - {b}']['ctx_i>=1'])}")
            say(f"  {'':51s} ctx i>=2 {L.fmt_pair(pr[f'{a} - {b}']['ctx_i>=2'])}")
        cb = pr["C_post_all[rolling] - base[rolling]"]["ctx_i>=1"]
        if name == "gastrulation-human":
            verdict = {"pcc in [+0.005, +0.012]": bool(0.005 <= cb["pcc"]["mean_delta"] <= 0.012),
                       "up in [+0.005, +0.012]": bool(0.005 <= cb["up"]["mean_delta"] <= 0.012),
                       "down within +-0.005": bool(abs(cb["down"]["mean_delta"]) <= 0.005)}
        else:
            verdict = {"pcc not significant (p > 0.05)": bool(cb["pcc"]["wilcoxon_p"] is None or cb["pcc"]["wilcoxon_p"] > 0.05)}
        say(f"  PREDICTION C - base (ctx i>=1): {verdict}")
        pt = {k: L.pertrans(S[k]) for k in preds}
        out[name] = {"table": tab, "paired": pr, "reproduce_c12": repro, "prediction_C_minus_base": verdict,
                     "chosen": chosen, "detection_setting": setting,
                     "per_transition_ctx_pcc_up_down": {k: np.round(v, 4).tolist() for k, v in pt.items()},
                     "seconds": round(time.time() - t0)}
        np.savez(os.path.join(HERE, f"pertrans_orig_{name}.npz"),
                 **{k.replace("[", "_").replace("]", "").replace("+", "p"): v for k, v in pt.items()})
        json.dump(out, open(os.path.join(HERE, "results_orig.json"), "w"), indent=1)
    log.close()


if __name__ == "__main__":
    main()
