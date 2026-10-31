"""Pre-registered Wasserstein-Fisher-Rao (WFR / Hellinger-Kantorovich) test (tack/prereg/2026-10-31-wfr-prereg.md).

Question: the 50/50 mix of the neighbouring samples beat every OT interpolant, while the OT coupling itself is good.
The WFR geodesic moves mass along pairs closer than pi * delta and lets farther mass fade out at the source and fade in
at the target; its midpoint is the normalised mix when nothing is transported (delta -> 0) and McCann's OT midpoint
when everything is (delta -> inf). Does an intermediate delta, chosen without the target, beat the mix (interpolation)
and persistence (forecast)?

Construction (own code; static logarithmic entropy-transport form of HK, Liero-Mielke-Savare 2018; Chizat et al. 2018):
  theta_ij = |x_i - y_j| / (2 delta); cost c_ij = -2 log cos(theta_ij) if theta_ij < pi / 2, else +inf (no transport)
  plan gamma: entropic unbalanced Sinkhorn (interp_bench_frozen.sinkhorn), KL marginal weight 1 (fixed by the theory),
       eps = EPS_REL * median of the finite costs; rows / columns with no finite cost (or zero plan mass) are faded
  semi-couplings a_ij = gamma_ij mu0_i / gamma0_i, b_ij = gamma_ij mu1_j / gamma1_j; per pair the straight cone line
       z(t) = (1 - t) sqrt(a) (1, 0) + t sqrt(b) (cos theta, sin theta): mass |z|^2 at x_i + (phi / theta)(y_j - x_i),
       phi = angle of z; faded source mass (1 - t)^2 mu0_i at x_i, faded target mass t^2 mu1_j at y_j
  interpolation t = 1/2 (held stage halfway); forecast t = 2 from (s-1, s), a construction, not a geodesic: point
       reflection z = 2 z1 - z0 (pairs with 2 sqrt(b) cos theta <= sqrt(a) are dropped), faded source mass 0, faded
       target mass 4 mu1_j; delta = 0 gives exactly the mix (t = 1/2) and persistence (t = 2)
  pair atoms: N_DRAW * n pairs drawn with probability proportional to their mass, duplicates merged into weights (no
       duplicated points enter the U-statistic); then the frozen subsample / scoring path.
delta = DELTA_REL * sqrt(median squared distance between the two samples) of each problem; grid DELTAS (0 included).
Interpolation: frozen atlas_confirm.run_interp loop (same seeds, so prev / next / union / indep / ot_pairs reproduce
confirm.json), WFR with its own rng on the same subsamples and target. delta for held stage k = the grid value with the
best mean rank of ED_U over the files k' with |k' - k| >= 2 (none of them contains stage k cells); ties -> smaller.
Forecast: frozen atlas_confirm.Problem; WFR from <= NSUB cells of stages s-1 and s; delta by rolling origin (argmin
ED_U forecasting s from origin s-1); rivals persist and meanshift as frozen.
Usage: python wfr_confirm.py --data <dir> --out <json> --ks ... --origins ... [--frozen confirm.json]
       python wfr_confirm.py --larry <dir with sca1pos/, sca1neg/> --out <json> [--frozen larry_confirm.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_confirm as AC  # noqa: E402
import interp_bench_frozen as IB  # noqa: E402

DELTAS = (0.0, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
EPS_REL = 0.05
N_DRAW = 20
FROZEN_METHODS = ("prev", "next", "union", "indep", "ot_pairs")


# ---------------------------------------------------------------- WFR construction

def wfr_plan(ZA, ZB, delta_rel, eps_rel=EPS_REL):
    """Unbalanced plan with the HK cost. Returns (P, theta, med); P is None when nothing can be transported."""
    D2 = IB.sqdist(ZA, ZB)
    med = float(np.median(D2))
    if delta_rel <= 0:
        return None, None, med
    theta = np.sqrt(D2) / (2 * delta_rel * np.sqrt(med))
    near = theta < np.pi / 2
    rows, cols = near.any(1), near.any(0)
    P = np.zeros(D2.shape)
    if not rows.any():
        return None, theta, med
    th = theta[np.ix_(rows, cols)]
    C = np.where(th < np.pi / 2, -2.0 * np.log(np.cos(np.minimum(th, np.pi / 2 - 1e-9))), np.inf)
    eps = eps_rel * max(float(np.median(C[np.isfinite(C)])), 1e-12)
    a, b = IB.unif(len(ZA))[rows], IB.unif(len(ZB))[cols]
    P[np.ix_(rows, cols)] = IB.sinkhorn(a, b, C, eps, tau=1.0)
    return P, theta, med


def cone_atoms(P, theta, nA, nB, t):
    """Pair masses / position fractions at time t and faded masses, from a plan (uniform mu0, mu1)."""
    mu0, mu1 = 1.0 / nA, 1.0 / nB
    if P is None:
        fa, fb = np.full(nA, mu0), np.full(nB, mu1)
        return None, None, fa * max(1 - t, 0.0) ** 2, fb * t ** 2
    r, c = P.sum(1), P.sum(0)
    live_r, live_c = r > 0, c > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(live_r[:, None], P * (mu0 / np.where(live_r, r, 1.0))[:, None], 0.0)
        b = np.where(live_c[None, :], P * (mu1 / np.where(live_c, c, 1.0))[None, :], 0.0)
    ra, rb = np.sqrt(a), np.sqrt(b)
    th = np.minimum(theta, np.pi / 2)
    zx = (1 - t) * ra + t * rb * np.cos(th)
    zy = t * rb * np.sin(th)
    mass = zx ** 2 + zy ** 2
    if t > 1:
        mass = np.where(zx > 0, mass, 0.0)
    phi = np.arctan2(zy, np.maximum(zx, 1e-300))
    frac = np.where(th > 1e-9, phi / np.maximum(th, 1e-9), t)
    mass = np.where(P > 0, mass, 0.0)
    fa = np.where(live_r, 0.0, mu0) * max(1 - t, 0.0) ** 2
    fb = np.where(live_c, 0.0, mu1) * t ** 2
    return mass, frac, fa, fb


def wfr_predict(ZA, XA, ZB, XB, delta_rel, t, n, rng):
    """Weighted atoms (Z, X, w) of the WFR construction at time t, plus the transported mass fraction."""
    P, theta, _ = wfr_plan(ZA, ZB, delta_rel)
    mass, frac, fa, fb = cone_atoms(P, theta, len(ZA), len(ZB), t)
    Zs, Xs, ws = [], [], []
    ia, ib = np.where(fa > 0)[0], np.where(fb > 0)[0]
    Zs += [ZA[ia], ZB[ib]]
    Xs += [XA[ia], XB[ib]]
    ws += [fa[ia], fb[ib]]
    pair_mass = 0.0 if mass is None else float(mass.sum())
    if pair_mass > 0:
        flat = rng.choice(mass.size, size=N_DRAW * n, replace=True, p=mass.ravel() / pair_mass)
        u, cnt = np.unique(flat, return_counts=True)
        i, j = np.unravel_index(u, mass.shape)
        f = frac[i, j][:, None]
        Zs.append(ZA[i] + f * (ZB[j] - ZA[i]))
        Xs.append(XA[i] + f * (XB[j] - XA[i]))
        ws.append(cnt / cnt.sum() * pair_mass)
    w = np.concatenate(ws)
    tot = float(w.sum())
    return np.vstack(Zs), np.vstack(Xs), w / tot, pair_mass / tot


# ---------------------------------------------------------------- interpolation

def run_interp_wfr(data, k, reps=3):
    """atlas_confirm.run_interp, unchanged up to the frozen scores, plus WFR for every delta on the same draws."""
    d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
    comps = d["pca_comps"].astype(np.float64)
    smp_held = d["sample_held"].astype(str)
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in set(IB.S_GENES + IB.G2M_GENES) for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    cc_fit = (Xcc - Xcc.mean(0)).mean(1)
    lo, hi = sorted(set(st.tolist()))
    A, B = np.where(st == lo)[0], np.where(st == hi)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], Xh[it], S, with_sd=False)
    pools = [p for p in sorted(set(smp_held)) if (smp_held == p).sum() >= AC.MIN_POOL]
    ptg = {p: IB.Target(Zh[smp_held == p], None, S, with_sd=False) for p in pools}
    out = {"pools": pools, "reps": []}
    for r in range(reps):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        M.pop("_plan", None)
        M.pop("_uot", None)
        rec = {}
        for name, (Z, X, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
            if name in FROZEN_METHODS:
                rec[name] = {"ED_U": IB.ed_uv(Zs, ws, tgt)[0], "ED_U_gene": IB.ed_uv(Xs, ws, tgt, gene=True)[0],
                             "pool_ED_U": {p: IB.ed_uv(Zs, ws, ptg[p])[0] for p in pools}}
        wr = np.random.default_rng(777000 + 1000 * k + r)
        for dl in DELTAS:
            Z, X, w, tf = wfr_predict(ZA, XA, ZB, XB, dl, 0.5, n_pred, wr)
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, wr)
            rec[f"wfr_{dl:g}"] = {"ED_U": IB.ed_uv(Zs, ws, tgt)[0], "ED_U_gene": IB.ed_uv(Xs, ws, tgt, gene=True)[0],
                                  "pool_ED_U": {p: IB.ed_uv(Zs, ws, ptg[p])[0] for p in pools}, "transported": tf}
        out["reps"].append(rec)
    names = list(out["reps"][0])
    out["mean"] = {m: {key: float(np.mean([rp[m][key] for rp in out["reps"]])) for key in ("ED_U", "ED_U_gene")}
                   for m in names}
    for m in names:
        if m.startswith("wfr_"):
            out["mean"][m]["transported"] = float(np.mean([rp[m]["transported"] for rp in out["reps"]]))
    out["pool_mean"] = {m: {p: float(np.mean([rp[m]["pool_ED_U"][p] for rp in out["reps"]])) for p in pools}
                        for m in names}
    return out


def mean_rank_choice(files):
    """files: list of {delta: ED_U}; the delta with the best (lowest) mean rank, ties -> smaller delta."""
    ranks = np.mean([np.argsort(np.argsort([f[dl] for dl in DELTAS], kind="stable"), kind="stable") for f in files], 0)
    return DELTAS[int(np.argmin(ranks))]


# ---------------------------------------------------------------- forecast

def forecast_wfr(prob, delta_rel, seed):
    iP, iN = IB.sub(len(prob.Zp), seed=21), IB.sub(len(prob.Zn), seed=22)
    rng = np.random.default_rng(seed)
    Z, X, w, tf = wfr_predict(prob.Zp[iP], prob.Xp[iP].astype(np.float64), prob.Zn[iN],
                              prob.Xn[iN].astype(np.float64), delta_rel, 2.0, len(iN), rng)
    return prob.score(Z, X, w, pools=True), tf


def run_forecast_wfr(data, s):
    final, back = AC.origin_problem(data, s), AC.origin_problem(data, s - 1)
    out = {f: final.score(*final.forecast(f, {}), pools=True) for f in ("persist", "meanshift")}
    bt = {dl: forecast_wfr(back, dl, 1000 * s + 1)[0]["ED_U"] for dl in DELTAS}
    dl = min(DELTAS, key=lambda x: (bt[x], x))
    allf = {d_: forecast_wfr(final, d_, 1000 * s + 2) for d_ in DELTAS}
    out["wfr"] = {"delta": dl, "backtest_ED_U": bt, "transported": allf[dl][1], **allf[dl][0]}
    out["wfr_all_deltas_oracle"] = {f"{d_:g}": v[0]["ED_U"] for d_, v in allf.items()}
    return out


# ---------------------------------------------------------------- verdicts

def beats(units, a, b):
    n = len(units)
    need = int(np.ceil(2 * n / 3))
    w = sum(u[a] < u[b] for u in units.values())
    mean_ok = bool(np.mean([u[a] for u in units.values()]) < np.mean([u[b] for u in units.values()]))
    return {"wins": int(w), "of": n, "need": need, "mean_ok": mean_ok, "pass": bool(w >= need and mean_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--larry")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=[])
    ap.add_argument("--origins", type=int, nargs="*", default=[])
    ap.add_argument("--frozen")
    args = ap.parse_args()
    t0 = time.time()
    res = {"config": vars(args), "deltas": DELTAS, "interp": {}, "forecast": {}}
    if args.larry:
        for pop in ("sca1pos", "sca1neg"):
            res["interp"][pop] = run_interp_wfr(os.path.join(args.larry, pop), 1)
            print(pop, {m: round(v["ED_U"], 4) for m, v in res["interp"][pop]["mean"].items()}, f"[{time.time() - t0:.0f}s]", flush=True)
        units = {}
        for pop, other in (("sca1pos", "sca1neg"), ("sca1neg", "sca1pos")):
            dl = mean_rank_choice([{d_: res["interp"][other]["mean"][f"wfr_{d_:g}"]["ED_U"] for d_ in DELTAS}])
            res["interp"][pop]["chosen_delta"] = dl
            pm = res["interp"][pop]["pool_mean"]
            for p in res["interp"][pop]["pools"]:
                units[f"{pop}_{p}"] = {"wfr": pm[f"wfr_{dl:g}"][p], **{m: pm[m][p] for m in FROZEN_METHODS}}
        res["units_interp"] = units
        res["verdicts"] = {"desc_wfr_vs_union": beats(units, "wfr", "union"),
                           "desc_wfr_vs_ot_pairs": beats(units, "wfr", "ot_pairs")}
    else:
        for k in args.ks:
            res["interp"][k] = run_interp_wfr(args.data, k)
            print("interp", k, {m: round(v["ED_U"], 3) for m, v in res["interp"][k]["mean"].items()}, f"[{time.time() - t0:.0f}s]", flush=True)
            json.dump(res, open(args.out, "w"), indent=1, default=float)
        units = {}
        for k in args.ks:
            val = [k2 for k2 in args.ks if abs(k2 - k) >= 2]
            dl = mean_rank_choice([{d_: res["interp"][k2]["mean"][f"wfr_{d_:g}"]["ED_U"] for d_ in DELTAS} for k2 in val])
            mk = res["interp"][k]["mean"]
            res["interp"][k]["chosen_delta"] = dl
            res["interp"][k]["validation_files"] = val
            units[k] = {"wfr": mk[f"wfr_{dl:g}"]["ED_U"], "transported": mk[f"wfr_{dl:g}"]["transported"],
                        **{m: mk[m]["ED_U"] for m in FROZEN_METHODS}}
        res["units_interp"] = units
        for s in args.origins:
            res["forecast"][s] = run_forecast_wfr(args.data, s)
            f = res["forecast"][s]
            print("forecast", s, {"persist": round(f["persist"]["ED_U"], 3), "meanshift": round(f["meanshift"]["ED_U"], 3),
                                  "wfr": (round(f["wfr"]["ED_U"], 3), f["wfr"]["delta"])}, f"[{time.time() - t0:.0f}s]", flush=True)
            json.dump(res, open(args.out, "w"), indent=1, default=float)
        fo = {s: {"wfr": f["wfr"]["ED_U"], "persist": f["persist"]["ED_U"], "meanshift": f["meanshift"]["ED_U"]}
              for s, f in res["forecast"].items()}
        res["verdicts"] = {"H1_wfr_vs_union": beats(units, "wfr", "union") if units else None,
                           "H2_wfr_vs_ot_pairs": beats(units, "wfr", "ot_pairs") if units else None,
                           "H3_wfr_vs_persist": beats(fo, "wfr", "persist") if fo else None,
                           "H3_wfr_vs_meanshift": beats(fo, "wfr", "meanshift") if fo else None}
        if fo:
            res["verdicts"]["H3_pass"] = bool(res["verdicts"]["H3_wfr_vs_persist"]["pass"]
                                              and res["verdicts"]["H3_wfr_vs_meanshift"]["pass"])
    if args.frozen:
        fz = json.load(open(args.frozen))
        diffs = []
        for key, r in res["interp"].items():
            for m in FROZEN_METHODS:
                diffs.append(abs(r["mean"][m]["ED_U"] - fz["interp"][str(key)]["mean"][m]["ED_U"]))
        for s, f in res["forecast"].items():
            for m in ("persist", "meanshift"):
                diffs.append(abs(f[m]["ED_U"] - fz["forecast"][str(s)][m]["ED_U"]))
        res["parity_max_abs_diff"] = float(max(diffs)) if diffs else None
        print("PARITY max |diff| vs frozen", res["parity_max_abs_diff"], flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    print("VERDICTS", json.dumps(res.get("verdicts")), flush=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
