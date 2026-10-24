"""(2) Binning artefacts: uneven cell counts per equal-width pseudotime bin, library-size changes, tiny bins.

Per transition a -> b (all from their node files):
  n_a, n_b, log(n_b / n_a); lib = sum_genes(bin mean) = mean UMI per cell, log(lib_b / lib_a);
  gene classes zz (0 -> 0), on (0 -> +), off (+ -> 0), pp (+ -> +);
  per target: sd over genes, fraction at the +/-5 clip (orig), test-set ties at +5 / -5 vs k = 200 and the
  tie-aware P@200 ceiling (25 % masks, 20 draws);
  sampling-noise ceiling of Pearson: the next bin is n_b cells from the observed bin-b law (zero-inflated, positive
  part by moment matching, CLT for the sum, as task2_forecasters.expected_target); E_g = Monte Carlo variance of
  the target over 64 re-draws; ceiling = sqrt(1 - mean_g E_g / var_g(y)) (an oracle that knows the true next law;
  optimistic for genes unseen in bin b, whose re-draw variance is 0); plus the simulated oracle P@200 (expected law
  vs one re-draw, 25 % masks);
  strict reference forecasters per transition (orig: ObsLevel(4) with n = n_a, the value DistBridge also uses;
  the same with a strict extrapolated n_hat, and with the true n_b, NOT strict; l1 / linear: trend);
  linear target: corr(y, m_a) vs the sign of the library change; is the sign predictable from the last change?
Tiny bins: per-transition DistBridge strict (5 seeds, saved json) and their models' (preqA/preqB, saved json)
metrics; transition averages with and without the transitions that touch a bin of < 100 cells.
Output: artefacts.json in this folder.
"""
import glob
import json
import os

import numpy as np
from scipy import stats

import common
import task2_forecasters as tf

R_SIM = 64
DRAWS = 20
RES = os.path.join(common.ROOT, "tack", "results", "dileo_task2")


def hist_arrays(d, bins):
    return [np.stack([d["m"][:, b], d["med"][:, b], d["s"][:, b], d["f"][:, b], d["S"][:, b]], 1) for b in bins]


def obs_pred(d, hist_bins, n_next, k=4, seed=0):
    """task2_forecasters.ObsLevel(k) with an explicit next-bin cell count."""
    h = hist_arrays(d, hist_bins)[-k:]
    w = np.array([tf.n_cells(x) for x in h]); w = w / w.sum()
    f = sum(wi * x[:, 3] for wi, x in zip(w, h))
    m = sum(wi * x[:, 0] for wi, x in zip(w, h))
    ex2 = sum(wi * (x[:, 2] ** 2 + x[:, 0] ** 2) for wi, x in zip(w, h))
    return tf.expected_target(h[-1][:, 0], f, m, ex2, int(round(n_next)), 256,
                              np.random.default_rng(seed + len(hist_bins))).astype(np.float64)


def simulate_next(d, b, rng, R):
    """R re-draws of the bin-b mean of n_b cells from the observed bin-b law; (N, R)."""
    m, f, s = d["m"][:, b], np.clip(d["f"][:, b], 0, 1), d["s"][:, b]
    n = int(round(d["n"][b]))
    e = s ** 2 + m ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(f > 0, m / np.where(f > 0, f, 1), 0.0)
        var = np.where(f > 0, np.maximum(e / np.where(f > 0, f, 1) - mu ** 2, 0), 0.0)
    K = rng.binomial(n, f[:, None], size=(len(f), R))
    tot = np.maximum(K * mu[:, None] + np.sqrt(K * var[:, None]) * rng.standard_normal((len(f), R)), 0)
    return tot / max(n, 1)


def per_transition(d, name):
    snaps = common.SNAPS[name]
    trans = common.transitions(name)
    N = d["m"].shape[0]
    n = d["n"]
    lib = d["m"].sum(0)
    mk = common.masks(N, len(trans), DRAWS, seed=3)
    rng = np.random.default_rng(5)
    rows = []
    for ti, (a, b) in enumerate(trans):
        ma, mb = d["m"][:, a], d["m"][:, b]
        za, zb = ma == 0, mb == 0
        r = {"transition": f"{a}->{b}", "n_a": float(n[a]), "n_b": float(n[b]),
             "log_n_ratio": float(np.log(n[b] / n[a])), "lib_a": float(lib[a]), "lib_b": float(lib[b]),
             "log_lib_ratio": float(np.log(lib[b] / lib[a])),
             "frac_zz": float((za & zb).mean()), "frac_on": float((za & ~zb).mean()),
             "frac_off": float((~za & zb).mean()), "frac_pp": float((~za & ~zb).mean())}
        sims = simulate_next(d, b, rng, R_SIM)
        for T in common.TARGETS:
            y = common.target(T, ma, mb)
            r[f"{T}_sd"] = float(y.std())
            if T == "orig":
                r["orig_frac_plus5"] = float((y >= 5 - 1e-4).mean())
                r["orig_frac_minus5"] = float((y <= -5 + 1e-4).mean())
                r["orig_frac_exact0"] = float((y == 0).mean())
            ties_up, ties_dn, pmax_up, pmax_dn = [], [], [], []
            for draw in mk[:5]:
                yt = y[draw[ti]]
                ties_up.append(int((yt == yt.max()).sum())); ties_dn.append(int((yt == yt.min()).sum()))
                pt = common.p_tie_aware(yt, yt)
                pmax_up.append(pt["max_up"]); pmax_dn.append(pt["max_down"])
            r[f"{T}_test_ties_at_max"] = float(np.mean(ties_up)); r[f"{T}_test_ties_at_min"] = float(np.mean(ties_dn))
            r[f"{T}_p200_ceiling_ties_up"] = float(np.mean(pmax_up)); r[f"{T}_p200_ceiling_ties_down"] = float(np.mean(pmax_dn))
            # sampling-noise ceiling
            Ysim = np.stack([common.target(T, ma, sims[:, j]) for j in range(R_SIM)], 1)
            Eg = Ysim.var(1)
            r[f"{T}_noise_frac"] = float(Eg.mean() / max(y.var(), 1e-12))
            r[f"{T}_pcc_ceiling"] = float(np.sqrt(max(0.0, 1 - Eg.mean() / max(y.var(), 1e-12))))
            p_or = Ysim.mean(1)
            pc, up, dn = [], [], []
            for j, draw in enumerate(mk[:8]):
                idx = draw[ti]
                ys = Ysim[idx, j]
                pc.append(common.pearson(ys, p_or[idx])); u_, d_ = common.p_at_k(ys, p_or[idx]); up.append(u_); dn.append(d_)
            r[f"{T}_oracle_sim_pcc"] = float(np.mean(pc)); r[f"{T}_oracle_sim_p200_up"] = float(np.mean(up))
            r[f"{T}_oracle_sim_p200_down"] = float(np.mean(dn))
            r[f"{T}_corr_y_with_m_a"] = common.pearson(y, ma)
            r[f"{T}_corr_y_with_lvl_a"] = common.pearson(y, np.log1p(1e4 * ma / ma.sum()))
        rows.append(r)
    return rows


def reference_scores(d, name):
    """Strict reference forecasters per transition, plus the n_b oracle for ObsLevel (orig)."""
    snaps = common.SNAPS[name]
    trans = common.transitions(name)
    N = d["m"].shape[0]
    n = d["n"]
    mk = common.masks(N, len(trans), DRAWS, seed=3)
    Y = {T: [common.target(T, d["m"][:, a], d["m"][:, b]) for a, b in trans] for T in common.TARGETS}
    out = {}
    preds = {"orig/obs4_n_a": [], "orig/obs4_n_hat": [], "orig/obs4_n_b_ORACLE": [],
             "l1/trend": [], "linear/trend": [], "orig/trend": []}
    nhat_rows = []
    for ti, (a, b) in enumerate(trans):
        hist_bins = [s for s in snaps if s <= a]
        if len(hist_bins) < 2:
            for k in preds:
                preds[k].append(np.zeros(N))
            nhat_rows.append(None)
            continue
        p = hist_bins[-2]
        # strict n_hat: geometric extrapolation of the last two bins, shrunk halfway to persistence
        n_hat = float(n[a] * (n[a] / n[p]) ** 0.5)
        nhat_rows.append({"transition": f"{a}->{b}", "n_a": float(n[a]), "n_hat": n_hat, "n_b": float(n[b])})
        preds["orig/obs4_n_a"].append(obs_pred(d, hist_bins, n[a]))
        preds["orig/obs4_n_hat"].append(obs_pred(d, hist_bins, n_hat))
        preds["orig/obs4_n_b_ORACLE"].append(obs_pred(d, hist_bins, n[b]))
        for T in ("orig", "l1", "linear"):
            if f"{T}/trend" in preds:
                preds[f"{T}/trend"].append(common.target(T, d["m"][:, p], d["m"][:, a]))
    for k, P in preds.items():
        T = k.split("/")[0]
        out[k] = common.score_masked(Y[T][1:], P[1:], [dr[1:] for dr in mk])
    # n prediction errors (log scale)
    ok = [r for r in nhat_rows if r is not None]
    err = {"persistence_n_a": float(np.mean([abs(np.log(r["n_b"] / r["n_a"])) for r in ok])),
           "n_hat": float(np.mean([abs(np.log(r["n_b"] / r["n_hat"])) for r in ok]))}
    return out, err, ok


def library_sign(rows):
    llr = np.array([r["log_lib_ratio"] for r in rows])
    cy = np.array([r["linear_corr_y_with_m_a"] for r in rows])
    same_sign = float(np.mean(np.sign(llr) == np.sign(cy)))
    prev_pred = float(np.mean(np.sign(llr[1:]) == np.sign(llr[:-1])))
    return {"frac_transitions_sign_corr_linear_y_m_a_equals_sign_lib_change": same_sign,
            "spearman_corr_linear_y_m_a_vs_log_lib_ratio": float(stats.spearmanr(cy, llr).correlation),
            "lag1_autocorr_log_lib_ratio": float(np.corrcoef(llr[1:], llr[:-1])[0, 1]),
            "sign_of_lib_change_predicted_by_previous_sign": prev_pred,
            "mean_abs_corr_linear_y_m_a": float(np.mean(np.abs(cy)))}


def saved_per_transition(name):
    """Per-transition orig-target metrics already on disk: DistBridge strict (seed mean) and their models."""
    out = {}
    for f in ("distbridge_strict", "distbridge_strict_nosb", "obslevel4"):
        files = sorted(glob.glob(os.path.join(RES, "distbridge", name, f, "seed*.json")))
        if files:
            M = np.array([[[r["pearson"], r["p_at_200_up"], r["p_at_200_down"]] for r in json.load(open(x))] for x in files])
            out[f"ours/{f}"] = np.nanmean(M, 0)
    for proto in ("preqA_orig", "preqB_orig"):
        for mdir in sorted(glob.glob(os.path.join(RES, "corrected2", name, proto, "*"))):
            if not os.path.isdir(mdir):
                continue
            for part in ("pre", "post"):
                Ms = []
                for x in sorted(glob.glob(os.path.join(mdir, "seed*.json"))):
                    z = json.load(open(x))
                    if part in z and z[part]:
                        Ms.append([[r["pearson"], r["p_at_200_up"], r["p_at_200_down"]] for r in z[part]])
                if Ms:
                    out[f"theirs/{proto}/{part}/{os.path.basename(mdir)}"] = np.nanmean(np.array(Ms, float), 0)
    return out


def tiny_bin_weight(name, rows, saved, thr=100):
    trans = common.transitions(name)
    tiny = np.array([min(r["n_a"], r["n_b"]) < thr for r in rows])
    res = {"tiny_transitions": [r["transition"] for r, t in zip(rows, tiny) if t], "threshold_cells": thr}
    for k, M in saved.items():
        if M.shape[0] != len(trans):
            continue
        allm = np.nanmean(M, 0); rest = np.nanmean(M[~tiny], 0); tinym = np.nanmean(M[tiny], 0)
        share = np.nansum(M[tiny], 0) / np.nansum(M, 0)
        res[k] = {"all": allm.round(4).tolist(), "without_tiny": rest.round(4).tolist(), "tiny_only": tinym.round(4).tolist(),
                  "share_of_sum_from_tiny": share.round(4).tolist()}
    return res


def cross_transition_corr(rows, ref_per_trans, key):
    """Spearman over transitions of a per-transition reference metric with the binning covariates."""
    cov = {c: np.array([r[c] for r in rows]) for c in ("n_a", "n_b", "log_n_ratio", "log_lib_ratio", "frac_on", "frac_off")}
    cov["abs_log_n_ratio"] = np.abs(cov["log_n_ratio"]); cov["abs_log_lib_ratio"] = np.abs(cov["log_lib_ratio"])
    cov["min_n"] = np.minimum(cov["n_a"], cov["n_b"])
    out = {}
    for mi, metric in enumerate(("pcc", "p200_up", "p200_down")):
        v = np.array([x[mi] for x in ref_per_trans])
        out[metric] = {c: float(stats.spearmanr(v, cv[1:]).correlation) for c, cv in cov.items()}
    return out


def main():
    out = {}
    for name in common.SNAPS:
        d = common.load_node(name)
        rows = per_transition(d, name)
        refs, nerr, nrows = reference_scores(d, name)
        saved = saved_per_transition(name)
        res = {"per_transition": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
               "reference_strict": {k: {kk: vv for kk, vv in v.items()} for k, v in refs.items()},
               "n_prediction_abs_log_error": nerr, "n_hat_rows": nrows,
               "library": library_sign(rows),
               "tiny_bins": tiny_bin_weight(name, rows, saved),
               "cross_transition_spearman": {k: cross_transition_corr(rows, refs[k]["per_transition"], k)
                                             for k in ("orig/obs4_n_a", "l1/trend", "linear/trend")}}
        # summaries across transitions
        summ = {}
        for c in rows[0]:
            if c == "transition":
                continue
            summ[c] = float(np.mean([r[c] for r in rows]))
        res["mean_over_transitions"] = summ
        # metric vs ceiling per target
        res["ceiling_vs_reference"] = {
            T: {"mean_pcc_ceiling": summ[f"{T}_pcc_ceiling"], "mean_oracle_sim_pcc": summ[f"{T}_oracle_sim_pcc"],
                "mean_oracle_sim_p200_up": summ[f"{T}_oracle_sim_p200_up"],
                "mean_oracle_sim_p200_down": summ[f"{T}_oracle_sim_p200_down"],
                "mean_noise_frac": summ[f"{T}_noise_frac"]} for T in common.TARGETS}
        out[name] = res
        print("==", name)
        print(" n err", nerr)
        print(" library", res["library"])
        for k, v in refs.items():
            print(f" ref {k:24s} pcc={v['pcc']:.3f} up={v['p200_up']:.3f} down={v['p200_down']:.3f}")
        for T in common.TARGETS:
            print(" ceiling", T, {k: round(v, 3) for k, v in res["ceiling_vs_reference"][T].items()})
        print(" tiny", json.dumps(res["tiny_bins"])[:1500])
        for r in rows:
            print("  ", r["transition"], int(r["n_a"]), int(r["n_b"]), "lib %.0f->%.0f" % (r["lib_a"], r["lib_b"]),
                  "on %.3f off %.3f" % (r["frac_on"], r["frac_off"]),
                  "ceil o/l1/lin %.2f %.2f %.2f" % (r["orig_pcc_ceiling"], r["l1_pcc_ceiling"], r["linear_pcc_ceiling"]),
                  "ties+5 %.0f pmax_up %.2f" % (r["orig_test_ties_at_max"], r["orig_p200_ceiling_ties_up"]),
                  "linear corr(y,m) %.2f" % r["linear_corr_y_with_m_a"], flush=True)
    with open(os.path.join(common.OUT, "artefacts.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
