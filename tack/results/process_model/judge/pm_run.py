"""Steps 3-5 and 7: process-model v2 forecaster on l1 (NASS stacker variants V0..Vauto, P = pre-registered primary)
and the linear record (NASS full, full+grn, trend), with every comparator recomputed here on the same masks:
NASS (V0 = NASS full, from the build copy of nass.py), trend (repeat last change), and the level + observation-head
baseline (DistBridge no-SB: rolling pool + finite-cell head, dist_bridge.REGISTRY *_nosb_l1 / *_nosb_linear).
Strict rolling origin everywhere (NASS learner: ridge on past transitions, (h, lambda, small-bin) by rolling J over
window 8, fallbacks at i = 0, 1; DistBridge: rolling pool by J over window 8, default kalman 0.2 at i <= 2).

Scoring: dileo_rescore.metrics on the CONTEXT masks (1000 s + i, 3 seeds): mean over i >= 1 and i >= 2, then over
seeds; NASS's own 20 masks (seed 20261022) via dsdata.score / summarize at i >= 2.
Paired: per-transition mask-averaged deltas, Wilcoxon over transitions (CONTEXT masks, i >= 2 and i >= 1) and
nass.paired (NASS masks, i >= 2).
Audits (always run): new-feature NaN strictness, lag1 row offsets, mask identity, V0 of the shared Stacker vs an
independent Stacker(full) (the anchor route). The full-pipeline truncation test is truncation.py.
Diagnostic (NOT strict, uses bin t+1): signal share of each transition from the unfloored delta-method R.
Outputs: results_l1.json, results_linear.json, pertrans_l1_<name>.npz, pertrans_linear_<name>.npz, pm_run.log."""
import json
import os
import time

import numpy as np
from scipy import stats

import pm_lib as L

D, NS = L.D, L.NS
PAIRS = [("P_primary", "V1_full+grn"), ("P_primary", "V0_full"), ("V2_V1+grnbar", "V1_full+grn"),
         ("V2b_full+grnbar", "V0_full"), ("V2n_V1+grnbarnull", "V1_full+grn"), ("Vd_full+fgap+fmom", "V0_full"),
         ("Pn_null", "V1_full+grn"), ("Vm_V1+grnbar_mom", "V1_full+grn"), ("Vauto_V1|V2|P", "V1_full+grn"),
         ("V1_full+grn", "V0_full"), ("P_primary", "Pn_null")]


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def main():
    log = open(os.path.join(L.B, "pm_run.log"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    say(f"==== pm_run {time.strftime('%Y-%m-%d %H:%M:%S')}")
    res_l1, res_lin = {}, {}
    for name in L.NAMES:
        t0 = time.time()
        sh = L.SHORT[name]
        d = D.Data(name)
        graph = NS.load_graph(d)
        cmk = L.ctx_masks(d.G, d.T)
        nmk = D.masks(d.G, d.T)
        T = d.T
        # ---------------------------------------------------------------- l1
        Y, F, cls, preds, chosen = L.l1_models(d, graph)
        preds["level_obs(DistBridge nosb l1)"], sel_lvl = L.level_obs(d, "l1")
        say(f"== {name} l1 models done ({time.time() - t0:.0f}s)")
        # audits
        aud = {"nan_strictness_max_diff": L.strict_nan_check(d, F, graph),
               "lag1_rows_equal_Y_prev": bool(all(np.array_equal(F["lag1"][i], Y[i - 1]) for i in range(1, T))),
               "ctx_masks_identical_on_regeneration": bool(np.array_equal(cmk, L.ctx_masks(d.G, T))),
               "nass_masks_identical_on_regeneration": bool(np.array_equal(nmk, D.masks(d.G, T)))}
        full, _ = L.subsets()
        fb = NS.fallback(d, "l1", Y, F)
        st0 = NS.Stacker(F, cls, Y, full)
        Pa, cha = NS.run_learned(st0, [full], Y, d, fb)
        del st0
        aud["V0_vs_independent_Stacker(full)_max_abs_diff"] = float(max(np.max(np.abs(np.asarray(a) - np.asarray(b)))
                                                                    for a, b in zip(Pa, preds["V0_full"])))
        aud["V0_chosen_equal"] = cha == chosen["V0_full"]
        say(f"  audits {aud}")
        S = {k: L.score_ctx(Y, P, cmk) for k, P in preds.items()}
        SN = {k: D.score(Y, P, nmk) for k, P in preds.items()}
        tab = {k: {"ctx_i>=1": L.summ(S[k], 1), "ctx_i>=2": L.summ(S[k], 2),
                   "own_i>=2": D.summarize(SN[k], list(range(2, T)))} for k in preds}
        for k in preds:
            o = tab[k]["own_i>=2"]
            say(f"  {k:32s} ctx i>=1 {L.fmt(tab[k]['ctx_i>=1'])} | ctx i>=2 {L.fmt(tab[k]['ctx_i>=2'])} | "
                f"own i>=2 {o['pcc']:+.4f}/{o['pta_up']:.4f}/{o['pta_down']:.4f}")
        pr = {}
        pairs = PAIRS + [(k, "V0_full") for k in preds if k != "V0_full" and (k, "V0_full") not in PAIRS]
        for a, b in pairs:
            pr[f"{a} - {b}"] = {"ctx_i>=2": L.paired_ctx(S[a], S[b], 2), "ctx_i>=1": L.paired_ctx(S[a], S[b], 1),
                                "nass_masks_i>=2": NS.paired(SN[a], SN[b], list(range(2, T)))}
            say(f"  paired {a + ' - ' + b:46s} ctx i>=2 {L.fmt_pair(pr[f'{a} - {b}']['ctx_i>=2'])} | "
                f"NASS masks dPCC {pr[f'{a} - {b}']['nass_masks_i>=2']['pcc']['mean_delta']:+.4f} "
                f"(p={pr[f'{a} - {b}']['nass_masks_i>=2']['pcc']['wilcoxon_p']})")
        # gains, breadth, location, relation to the signal share
        pt = {k: L.pertrans(S[k]) for k in preds}
        a_, b_, share = L.signal_share(d)
        gains = {}
        for a, b in (("P_primary", "V1_full+grn"), ("P_primary", "V0_full"), ("V2_V1+grnbar", "V1_full+grn"),
                     ("Vd_full+fgap+fmom", "V0_full")):
            g = pt[a][:, 0] - pt[b][:, 0]
            g2 = g[2:]
            rec = {"per_transition_i>=2": np.round(g2, 4).tolist(), "mean_i>=2": r4(g2.mean()),
                   "n_pos_i>=2": int((g2 > 1e-12).sum()), "n_neg_i>=2": int((g2 < -1e-12).sum()), "n_i>=2": len(g2),
                   "pearson_with_signal_share": r4(np.corrcoef(g2, share[2:])[0, 1]),
                   "spearman_with_signal_share": r4(stats.spearmanr(g2, share[2:]).correlation)}
            tot = g2.sum()
            at25 = g[2] + (g[5] if T > 5 else 0.0)
            rec["sum_i>=2"] = r4(tot)
            rec["share_of_net_gain_at_i_in_{2,5}"] = r4(at25 / tot) if tot > 0 else None
            pos = np.maximum(g2, 0)
            rec["share_of_positive_gain_at_i_in_{2,5}"] = r4((max(g[2], 0) + (max(g[5], 0) if T > 5 else 0)) /
                                                           pos.sum()) if pos.sum() > 0 else None
            gains[f"{a} - {b}"] = rec
            say(f"  gain {a} - {b}: mean i>=2 {rec['mean_i>=2']:+.4f}, {rec['n_pos_i>=2']}/{rec['n_i>=2']} positive, "
                f"net share at i in {{2,5}} {rec['share_of_net_gain_at_i_in_{2,5}']}, positive share "
                f"{rec['share_of_positive_gain_at_i_in_{2,5}']}, corr with signal share P {rec['pearson_with_signal_share']} "
                f"S {rec['spearman_with_signal_share']}")
        say(f"  signal share (diagnostic, unfloored R, uses bin t+1): mean noise_t/noise_t+1/signal "
            f"{a_.mean():.3f}/{b_.mean():.3f}/{share.mean():.3f}")
        res_l1[name] = {"table": tab, "paired": pr, "gains": gains, "audits": aud, "chosen": chosen,
                        "level_obs_selected": sel_lvl, "signal_share": np.round(share, 4).tolist(),
                        "per_transition_ctx_pcc_up_down": {k: np.round(v, 4).tolist() for k, v in pt.items()},
                        "seconds": round(time.time() - t0)}
        np.savez(os.path.join(L.B, f"pertrans_l1_{name}.npz"),
                 **{k.replace("|", "_").replace("(", "_").replace(")", "").replace(" ", "_"): v for k, v in pt.items()},
                 **{"nassmask_pcc__" + k.replace("|", "_").replace("(", "_").replace(")", "").replace(" ", "_"):
                    np.nanmean(SN[k][:, :, 0], 0) for k in SN},
                 signal_share=share)
        del F, preds, S, SN
        # ---------------------------------------------------------------- linear (record only)
        t1 = time.time()
        Yl, pl, chl = L.linear_models(d, graph)
        pl["level_obs(DistBridge nosb linear)"], sel_lin = L.level_obs(d, "linear")
        Sl = {k: L.score_ctx(Yl, P, cmk) for k, P in pl.items()}
        SNl = {k: D.score(Yl, P, nmk) for k, P in pl.items()}
        tabl = {k: {"ctx_i>=1": L.summ(Sl[k], 1), "ctx_i>=2": L.summ(Sl[k], 2),
                    "own_i>=2": D.summarize(SNl[k], list(range(2, T)))} for k in pl}
        prl = {}
        for k in pl:
            if k != "V0_full":
                prl[f"{k} - V0_full"] = {"ctx_i>=2": L.paired_ctx(Sl[k], Sl["V0_full"], 2),
                                         "ctx_i>=1": L.paired_ctx(Sl[k], Sl["V0_full"], 1)}
        prl["trend - V1_full+grn"] = {"ctx_i>=2": L.paired_ctx(Sl["trend"], Sl["V1_full+grn"], 2),
                                      "ctx_i>=1": L.paired_ctx(Sl["trend"], Sl["V1_full+grn"], 1)}
        say(f"== {name} linear ({time.time() - t1:.0f}s)")
        for k in pl:
            o = tabl[k]["own_i>=2"]
            say(f"  {k:36s} ctx i>=1 {L.fmt(tabl[k]['ctx_i>=1'])} | ctx i>=2 {L.fmt(tabl[k]['ctx_i>=2'])} | "
                f"own i>=2 {o['pcc']:+.4f}/{o['pta_up']:.4f}/{o['pta_down']:.4f}")
        for k, v in prl.items():
            say(f"  paired {k:46s} ctx i>=2 {L.fmt_pair(v['ctx_i>=2'])}")
        ptl = {k: L.pertrans(Sl[k]) for k in pl}
        res_lin[name] = {"table": tabl, "paired": prl, "chosen": chl, "level_obs_selected": sel_lin,
                         "per_transition_ctx_pcc_up_down": {k: np.round(v, 4).tolist() for k, v in ptl.items()}}
        np.savez(os.path.join(L.B, f"pertrans_linear_{name}.npz"),
                 **{k.replace("(", "_").replace(")", "").replace(" ", "_"): v for k, v in ptl.items()})
        del graph
        json.dump(res_l1, open(os.path.join(L.B, "results_l1.json"), "w"), indent=1)
        json.dump(res_lin, open(os.path.join(L.B, "results_linear.json"), "w"), indent=1)
    # ---------------------------------------------------------------- acceptance (step 5)
    acc = {}
    g, p = "gastrulation-human", "pancreas-human"
    pp = res_l1[p]["paired"]["P_primary - V1_full+grn"]["ctx_i>=2"]["pcc"]
    acc["A1 pancr P - V1 (ctx i>=2) in [0, +0.010] and >= 8 of 12 transitions positive (i=1 is the shared fallback)"] = {
        "mean": pp["mean_delta"], "n_pos": pp["better"], "n": pp["n"],
        "pass": bool(0 <= pp["mean_delta"] <= 0.010 and pp["better"] >= 8)}
    gp = res_l1[g]["paired"]["P_primary - V0_full"]["ctx_i>=2"]["pcc"]
    acc["A2 gastr P - V0 (ctx i>=2) in [-0.005, +0.008]"] = {"mean": gp["mean_delta"], "n_pos": gp["better"],
                                                             "n": gp["n"],
                                                             "pass": bool(-0.005 <= gp["mean_delta"] <= 0.008)}
    for nm in (g, p):
        for a in ("V2n_V1+grnbarnull", "Pn_null"):
            v = res_l1[nm]["paired"][f"{a} - V1_full+grn"]["ctx_i>=2"]["pcc"]["mean_delta"]
            acc[f"A3 {L.SHORT[nm]} {a} - V1 (ctx i>=2) <= +0.003"] = {"mean": v, "pass": bool(v <= 0.003)}
    sh = res_l1[g]["gains"]["P_primary - V0_full"]
    acc["A4 gastr: less than half of the P - V0 gain at i in {2, 5}"] = {
        "net_share": sh["share_of_net_gain_at_i_in_{2,5}"], "positive_share": sh["share_of_positive_gain_at_i_in_{2,5}"],
        "pass": (None if sh["share_of_net_gain_at_i_in_{2,5}"] is None else bool(sh["share_of_net_gain_at_i_in_{2,5}"] < 0.5))}
    big = {}
    for nm in (g, p):
        for k, v in res_l1[nm]["paired"].items():
            if not k.startswith(("trend", "level_obs")) and v["ctx_i>=2"]["pcc"]["mean_delta"] > 0.02:
                big[f"{L.SHORT[nm]} {k}"] = v["ctx_i>=2"]["pcc"]["mean_delta"]
    acc["A5 any l1 gain > 0.02 (leakage-audit trigger)"] = {
        "comparisons_over_0.02": big,
        "audit": {L.SHORT[nm]: res_l1[nm]["audits"] for nm in (g, p)},
        "note": "audits run unconditionally; the full-pipeline truncation test is truncation.py"}
    acc["upper_bound_caveat"] = ("feature choice (Wbar, fgap, fmom) rests on in-sample partial correlations over all "
                                 "transitions (revised model D); rolling origin protects coefficients and "
                                 "hyperparameters only, so any gain is an upper bound")
    for k, v in acc.items():
        say(f"  ACCEPT {k}: {v}")
    res_l1["acceptance"] = acc
    json.dump(res_l1, open(os.path.join(L.B, "results_l1.json"), "w"), indent=1)
    log.close()


if __name__ == "__main__":
    main()
