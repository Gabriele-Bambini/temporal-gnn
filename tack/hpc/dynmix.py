"""Dynamic mixture model (Fisher-Rao view: the cloud changes by reweighting fixed cell states, not by moving cells).
Pre-registration: tack/prereg/2026-10-29-dynmix-prereg.md.

States: k-means (S = 100, seed 0) on the PCA scores of the fit cells. Composition: multinomial logistic regression of
the state on time (tau = t - t_target in timepoint steps; features tau and tau^2 when >= 3 fit timepoints, tau when 2;
sklearn L2, C = 1, intercepts unpenalised), cells weighted so that every timepoint has equal total weight and every
embryo (biological unit) equal weight inside its timepoint. Prediction = real reference cells, state s carrying the
predicted share pi_s spread uniformly over its reference cells.
  interpolation of k: fit = timepoints k-2, k-1, k+1, k+2 that exist; reference cells = timepoints k-1 and k+1
                      (states without cells there: all fit cells of the state); pi = model at tau = 0.
                      union = 50/50 of k-1 and k+1 (cells uniform within side), in the same product.
                      dm_lin2 (secondary) = same model fitted on k-1 and k+1 only (linear in tau).
  forecast from s:    fit = timepoints s-2..s (origin<s>.npz space); reference = timepoint s (fallback s-1, then all);
                      log pi = log pi(0) + PHI (log pi(1) - log pi(0)), PHI = 0.5 (damped trend); persist = stage s.
Metric ED_U (interp_bench_frozen.ed_uv) in the PCA space of the product vs the target subsample (<= 1500, seed 1).
Usage: python dynmix.py --data zf|atlas [--out PATH]
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_bench_frozen as IB  # noqa: E402

N_STATES, C_REG, PHI, MIN_POOL = 100, 1.0, 0.5, 20


# ---------------------------------------------------------------- core

def cell_weights(t, unit):
    """Equal total weight per timepoint, equal weight per unit (embryo / pool) inside its timepoint."""
    w = np.zeros(len(t))
    tps = np.unique(t)
    for tp in tps:
        m = t == tp
        us = np.unique(unit[m])
        for u in us:
            mu = m & (unit == u)
            w[mu] = 1.0 / (len(tps) * len(us) * mu.sum())
    return w / w.mean()


def fit_composition(state, tau, w, n_states):
    """Multinomial logit of state on (tau, tau^2) or tau; returns a function tau -> share vector over all states."""
    quad = len(np.unique(tau)) >= 3
    F = np.c_[tau, tau ** 2] if quad else tau[:, None]
    clf = LogisticRegression(C=C_REG, max_iter=5000).fit(F, state, sample_weight=w)

    def share(x):
        f = np.array([[x, x ** 2]]) if quad else np.array([[x]])
        p = np.full(n_states, 0.0)
        p[clf.classes_] = clf.predict_proba(f)[0]
        return p
    return share


def mixture_from_shares(Z, state, ref_mask, fit_mask, pi):
    """Reference cells of each state carry pi_s uniformly; states absent from the reference use all their fit cells."""
    idx, w = [], []
    for s in np.nonzero(pi > 0)[0]:
        cells = np.where(ref_mask & (state == s))[0]
        if len(cells) == 0:
            cells = np.where(fit_mask & (state == s))[0]
        if len(cells) == 0:
            continue
        idx.append(cells)
        w.append(np.full(len(cells), pi[s] / len(cells)))
    idx, w = np.concatenate(idx), np.concatenate(w)
    return Z[idx], w / w.sum()


def states_of(Zfit, seed=0):
    km = KMeans(n_clusters=N_STATES, n_init=4, random_state=seed).fit(Zfit)
    return km


def score(P, w, tgt, ptg):
    return IB.ed_uv(P, w, tgt)[0], {p: IB.ed_uv(P, w, t)[0] for p, t in ptg.items()}


def targets(Zt, unit_t, S):
    tgt = IB.Target(Zt[IB.sub(len(Zt), seed=1)], None, S, with_sd=False)
    ptg = {u: IB.Target(Zt[unit_t == u], None, S, with_sd=False)
           for u in sorted(set(unit_t)) if (unit_t == u).sum() >= MIN_POOL}
    return tgt, ptg


def interp_unit(Zf, tf, uf, Zt, ut, k):
    """Zf, tf, uf: fit cells (PCA scores, timepoint index, unit); held timepoint k."""
    km = states_of(Zf)
    st = km.labels_
    tau = (tf - k).astype(float)
    w = cell_weights(tf, uf)
    S = IB.med_scale(Zf)
    tgt, ptg = targets(Zt, ut, S)
    near = (tf == k - 1) | (tf == k + 1)
    allf = np.ones(len(tf), bool)
    out = {}
    pi = fit_composition(st, tau, w, N_STATES)(0.0)
    out["dynmix"] = score(*mixture_from_shares(Zf, st, near, allf, pi), tgt, ptg)
    pi2 = fit_composition(st[near], tau[near], w[near], N_STATES)(0.0)
    out["dm_lin2"] = score(*mixture_from_shares(Zf, st, near, allf, pi2), tgt, ptg)
    A, B = np.where(tf == k - 1)[0], np.where(tf == k + 1)[0]
    Pu = np.vstack([Zf[A], Zf[B]])
    wu = np.r_[np.full(len(A), 0.5 / len(A)), np.full(len(B), 0.5 / len(B))]
    out["union"] = score(Pu, wu, tgt, ptg)
    return {m: {"ED_U": v[0], "pool_ED_U": v[1]} for m, v in out.items()}


def forecast_unit(Zp, tp, up, Zt, ut, s):
    """Zp, tp, up: past cells (origin<s> space); forecast timepoint s + 1."""
    win = tp >= s - 2
    km = states_of(Zp[win])
    st = np.full(len(tp), -1)
    st[win] = km.labels_
    tau = (tp[win] - s).astype(float)
    w = cell_weights(tp[win], up[win])
    S = IB.med_scale(Zp)
    tgt, ptg = targets(Zt, ut, S)
    share = fit_composition(st[win], tau, w, N_STATES)
    p0, p1 = np.maximum(share(0.0), 1e-12), np.maximum(share(1.0), 1e-12)
    pi = np.exp(np.log(p0) + PHI * (np.log(p1) - np.log(p0)))
    pi /= pi.sum()
    ref = tp == s
    out = {"dynmix": score(*mixture_from_shares(Zp, st, ref, win, pi), tgt, ptg)}
    out["persist"] = score(Zp[ref], np.full(ref.sum(), 1.0 / ref.sum()), tgt, ptg)
    return {m: {"ED_U": v[0], "pool_ED_U": v[1]} for m, v in out.items()}


# ---------------------------------------------------------------- data

def load(name):
    """Returns X (cells x genes counts), timepoint index, unit, genes, cc_genes, n_timepoints."""
    import anndata as ad
    import pandas as pd
    import atlas_prep as AP
    if name == "zf":
        import zf_prep as ZP
        X, obs, genes = ZP.load()
        t = np.array([ZP.TPS.index(x) for x in obs["timepoint"]])
        unit = obs["embryo"].astype(str).to_numpy()
        n_tp = len(ZP.TPS)
    else:
        X, obs, genes = AP.load()
        meta = pd.read_csv(AP.META, sep="\t", usecols=["cell", "pool"])
        obs = obs.merge(meta, on="cell", how="left")
        t = np.array([AP.STAGES.index(x) for x in obs["stage"]])
        unit = obs["pool"].astype(str).to_numpy()
        n_tp = len(AP.STAGES)
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc = set(str(x).capitalize() for x in list(s_g) + list(g2m))
    return X, t, unit, genes, cc, n_tp


def run(name):
    import atlas_prep as AP
    X, t, unit, genes, cc, n_tp = load(name)
    res = {"interp": {}, "forecast": {}}
    for k in range(1, n_tp - 1):
        t0 = time.time()
        fit = np.where((np.abs(t - k) <= 2) & (t != k))[0]
        held = np.where(t == k)[0]
        r = AP.transform(X, fit, held, genes, cc)
        res["interp"][k] = interp_unit(r["Z_fit"].astype(np.float64), t[fit], unit[fit],
                                       r["Z_other"].astype(np.float64), unit[held], k)
        print(name, "interp", k, {m: round(v["ED_U"], 4) for m, v in res["interp"][k].items()},
              f"[{time.time() - t0:.0f}s]", flush=True)
    for s in range(2, n_tp - 1):
        t0 = time.time()
        past, fut = np.where(t <= s)[0], np.where(t == s + 1)[0]
        r = AP.transform(X, past, fut, genes, cc)
        res["forecast"][s] = forecast_unit(r["Z_fit"].astype(np.float64), t[past], unit[past],
                                           r["Z_other"].astype(np.float64), unit[fut], s)
        print(name, "forecast", s, {m: round(v["ED_U"], 4) for m, v in res["forecast"][s].items()},
              f"[{time.time() - t0:.0f}s]", flush=True)
    return res


def beats(res, a, b):
    units = sorted(res)
    n = len(units)
    need = int(np.ceil(2 * n / 3))
    wins = sum(res[u][a]["ED_U"] < res[u][b]["ED_U"] for u in units)
    mean_ok = bool(np.mean([res[u][a]["ED_U"] for u in units]) < np.mean([res[u][b]["ED_U"] for u in units]))
    return {"wins": int(wins), "of": n, "need": need, "mean_ok": mean_ok, "pass": bool(wins >= need and mean_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["zf", "atlas"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.expanduser(f"~/tack/out/{'zf' if a.data == 'zf' else 'atlas'}/dynmix_{a.data}.json")
    t0 = time.time()
    res = run(a.data)
    res["verdicts"] = {"H1_dynmix_vs_union": beats(res["interp"], "dynmix", "union"),
                       "H2_dynmix_vs_persist": beats(res["forecast"], "dynmix", "persist"),
                       "sec_dmlin2_vs_union": beats(res["interp"], "dm_lin2", "union")}
    res["seconds"] = round(time.time() - t0, 1)
    print("VERDICTS", json.dumps(res["verdicts"]), flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
