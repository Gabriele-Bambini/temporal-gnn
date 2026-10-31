"""Judge: strictness and metric-scale tests ON THE AUDITED CODE (imported read-only, no bytecode written).

1. Data layer (the npz products, built on the HPC): is the PCA (mean, components) a fit of the PAST cells only?
   Are the steady-state velocities a function of the past moments only? Is the proliferation score centred on the past?
   Same PCA checks for the interp files (fit stages k-1, k+1 only).
2. Forecast layer, literal truncation test: every first-run family (ery_forecast_fixed.forecast) and every
   attempt_forecast family at EVERY grid point is built twice for each origin s = 1..5: once from the full origin
   file, once from a copy in which every *_fut array is DELETED (any access raises KeyError) and the target arrays
   passed to attempt_forecast.Problem are random garbage; obs.csv truncated to stages <= s for the sample labels.
   Predictions (Z, X, w) must be bit-identical.
3. Interp layer: interp_bench.methods() and its strict velocity-scale fit, rebuilt from the npz with the held-out
   stage replaced by garbage of the same shape, must give bit-identical predictions (rep-0 seeds of the audited run).
4. Metric scale: S_s does not move when the prediction or the future stage changes; sd_fixed / energy from the
   audited functions vs judge_lib on the same inputs; the old sinkhorn_div scale does move with the prediction.
Output: judge/strict.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ERY = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ERY, "attempt_interp"))
sys.path.insert(0, os.path.join(ERY, "attempt_forecast"))
sys.path.insert(0, os.path.join(ERY, "diag_metric"))
import judge_lib as J  # noqa: E402
import ery_forecast_fixed as F  # noqa: E402
import attempt_forecast as A  # noqa: E402
import interp_bench as IB  # noqa: E402

STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def subspace_angle_deg(U, V):
    """largest principal angle between row spaces of U (r x p) and V (r x p), orthonormalised."""
    Qu, _ = np.linalg.qr(U.T)
    Qv, _ = np.linalg.qr(V.T)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1, 1))))


def data_layer(data):
    out = {}
    for s in range(1, 6):
        d = np.load(os.path.join(data, f"origin{s}.npz"), allow_pickle=True)
        Xp = d["X_past"].astype(np.float64)
        Xf = d["X_fut"].astype(np.float64)
        mu, comps = d["pca_mean"].astype(np.float64), d["pca_comps"].astype(np.float64)
        r = comps.shape[0]
        Xall = np.vstack([Xp, Xf])
        _, _, Vp = np.linalg.svd(Xp - Xp.mean(0), full_matrices=False)
        _, _, Va = np.linalg.svd(Xall - Xall.mean(0), full_matrices=False)
        Zp_re = (Xp - mu) @ comps.T
        Zf_re = (Xf - mu) @ comps.T
        Zp, Zf = d["Z_past"].astype(np.float64), d["Z_fut"].astype(np.float64)
        Ms, Mu = d["Ms"].astype(np.float64), d["Mu"].astype(np.float64)
        sn, un = Ms / np.maximum(Ms.max(0), 1e-12), Mu / np.maximum(Mu.max(0), 1e-12)
        w = sn + un
        lo, hi = np.percentile(w, 5, axis=0), np.percentile(w, 95, axis=0)
        sel = (w <= lo) | (w >= hi)
        num, den = (Mu * Ms * sel).sum(0), (Ms * Ms * sel).sum(0)
        gam = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        v_re = Mu - gam * Ms
        v = d["v_ss"].astype(np.float64)
        out[str(s)] = {
            "n_past": int(len(Xp)), "n_fut": int(len(Xf)),
            "pca_mean_minus_past_mean_maxabs": float(np.abs(mu - Xp.mean(0)).max()),
            "pca_mean_minus_pastfut_mean_maxabs": float(np.abs(mu - Xall.mean(0)).max()),
            "angle_comps_vs_past_svd_deg": subspace_angle_deg(comps, Vp[:r]),
            "angle_comps_vs_past+fut_svd_deg": subspace_angle_deg(comps, Va[:r]),
            "Z_past_reconstruction_rel_err": float(np.linalg.norm(Zp - Zp_re) / np.linalg.norm(Zp)),
            "Z_fut_reconstruction_rel_err": float(np.linalg.norm(Zf - Zf_re) / np.linalg.norm(Zf)),
            "v_ss_recomputed_from_past_moments_rel_err": float(np.linalg.norm(v - v_re) / max(np.linalg.norm(v), 1e-12)),
            "gamma_recomputed_maxabs": float(np.abs(gam - d["gamma"]).max()),
            "cc_past_mean (0 if centred on past)": float(d["cc_past"].mean()),
            "cc_fut_mean": float(d["cc_fut"].mean()),
            "Ms_rows = n_past": bool(len(Ms) == len(Xp)), "v_dyn_rows = n_past": bool(len(d["v_dyn"]) == len(Xp)),
            "stage_past_max": int(d["stage_past"].max()), "stage_fut_unique": np.unique(d["stage_fut"]).tolist(),
        }
        print("data origin", s, out[str(s)], flush=True)
    for k in (2, 3, 4, 5):
        d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
        Xf_, Xh = d["X_fit"].astype(np.float64), d["X_held"].astype(np.float64)
        mu, comps = d["pca_mean"].astype(np.float64), d["pca_comps"].astype(np.float64)
        r = comps.shape[0]
        Xall = np.vstack([Xf_, Xh])
        _, _, Vp = np.linalg.svd(Xf_ - Xf_.mean(0), full_matrices=False)
        _, _, Va = np.linalg.svd(Xall - Xall.mean(0), full_matrices=False)
        out[f"interp{k}"] = {
            "fit_stages": np.unique(d["stage_fit"]).tolist(),
            "pca_mean_minus_fit_mean_maxabs": float(np.abs(mu - Xf_.mean(0)).max()),
            "pca_mean_minus_fit+held_mean_maxabs": float(np.abs(mu - Xall.mean(0)).max()),
            "angle_comps_vs_fit_svd_deg": subspace_angle_deg(comps, Vp[:r]),
            "angle_comps_vs_fit+held_svd_deg": subspace_angle_deg(comps, Va[:r]),
            "Z_held_reconstruction_rel_err": float(np.linalg.norm(d["Z_held"] - (Xh - mu) @ comps.T) /
                                                    np.linalg.norm(d["Z_held"])),
            "Ms_rows = n_fit": bool(len(d["Ms"]) == len(Xf_)),
        }
        print("data interp", k, out[f"interp{k}"], flush=True)
    return out


def truncated(d):
    return {k: v for k, v in d.items() if not k.endswith("_fut")}


def forecast_layer(data):
    obs_full = pd.read_csv(os.path.join(data, "obs.csv"))
    out = {}
    for s in range(1, 6):
        t0 = time.time()
        d_full = F.load(data, s)
        d_cut = truncated(d_full)
        res = {"first_run": {}, "attempt": {}}
        # ---- first-run families, every grid point (+ default), final problem (s, s-1)
        for fam, grid in F.GRIDS.items():
            hps = list(grid) + ([F.DEFAULT[fam]] if fam in F.DEFAULT else [])
            n_ok, n_all, err = 0, 0, None
            for hp in hps:
                F._DRIFT.clear()
                try:
                    a = F.forecast(fam, hp, d_full, s, (s, s - 1), rng=np.random.default_rng(5))
                    F._DRIFT.clear()
                    b = F.forecast(fam, hp, d_cut, s, (s, s - 1), rng=np.random.default_rng(5))
                    ok = all(same(x, y) for x, y in zip(a, b))
                except KeyError as e:
                    ok, err = False, f"KeyError {e}"
                n_ok += ok
                n_all += 1
            res["first_run"][fam] = {"identical": n_ok, "of": n_all, "error": err}
        F._DRIFT.clear()
        # ---- attempt_forecast families: Problem from the truncated file + garbage target, obs truncated
        codes_full = np.array([STAGES.index(x) for x in obs_full["stage"].astype(str)])
        obs_cut = obs_full[codes_full <= s].reset_index(drop=True)
        smp_full = obs_full["sample"].astype(str).to_numpy()[codes_full <= s]
        smp_cut = obs_cut["sample"].astype(str).to_numpy()
        assert (smp_full == smp_cut).all()
        km = A.KMeans(n_clusters=A.K_MEANS, n_init=10, random_state=0).fit_predict(d_cut["Z_past"].astype(np.float64))
        km = np.array([f"k{x}" for x in km])
        S = F.past_scale(d_cut["Z_past"])
        g = np.random.default_rng(3)
        Zt_bad = g.normal(0, 5, d_full["Z_fut"].shape).astype(np.float32)
        Xt_bad = g.normal(0, 1, d_full["X_fut"].shape).astype(np.float32)
        smp_bad = np.array(["x"] * len(Zt_bad))
        P_full = A.Problem(d_full, smp_full, s, S, d_full["Z_fut"], d_full["X_fut"].astype(np.float32),
                           obs_full["sample"].astype(str).to_numpy()[codes_full == s + 1], km, "full")
        P_cut = A.Problem(d_cut, smp_cut, s, S, Zt_bad, Xt_bad, smp_bad, km, "cut")
        for fam, (grid, fn, tags) in A.FAMILIES.items():
            n_ok, n_all, err = 0, 0, None
            for hp in grid:
                try:
                    a = fn(P_full, hp)
                    b = fn(P_cut, hp)
                    ok = all(same(x, y) for x, y in zip(a, b))
                except KeyError as e:
                    ok, err = False, f"KeyError {e}"
                n_ok += ok
                n_all += 1
            res["attempt"][fam] = {"identical": n_ok, "of": n_all, "error": err}
        res["S_from_truncated_equals_rescore"] = (abs(S - json.load(open(os.path.join(
            ERY, "diag_metric", "rescore.json")))["origins"][str(s)]["S_fixed"]) < 1e-9)
        res["seconds"] = round(time.time() - t0, 1)
        out[str(s)] = res
        print("forecast-layer origin", s, json.dumps(res), flush=True)
    return out


def interp_layer(data):
    out = {}
    for k in (2, 3, 4, 5):
        d = dict(np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True))
        g = np.random.default_rng(9)
        d_bad = dict(d)
        d_bad["Z_held"] = g.normal(0, 5, d["Z_held"].shape).astype(np.float32)
        d_bad["X_held"] = g.normal(0, 1, d["X_held"].shape).astype(np.float16)
        d_bad["celltype_held"] = np.array(["junk"] * len(d["Z_held"]))
        d_bad["sample_held"] = np.array(["0"] * len(d["Z_held"]))
        preds = []
        for dd in (d, d_bad):
            IB.STATS.update({"kernel_calls": 0})
            st = dd["stage_fit"]
            Zf, Xf = dd["Z_fit"].astype(np.float64), dd["X_fit"].astype(np.float32)
            comps = dd["pca_comps"].astype(np.float64)
            genes = [x.capitalize() for x in dd["genes"].astype(str)]
            ccm = np.array([x in set(IB.S_GENES + IB.G2M_GENES) for x in genes])
            Xcc = dd["X_fit"][:, ccm].astype(np.float64)
            cc_fit = (Xcc - Xcc.mean(0)).mean(1)
            A_, B_ = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
            S = IB.med_scale(Zf)
            n_pred = min(len(dd["Z_held"]), IB.NSUB)
            Vfull = {vk: dd[vk].astype(np.float64) / (1.0 + dd["Ms"].astype(np.float64)) for vk in ("v_ss", "v_dyn")}
            iA0, iB0 = IB.sub(len(A_), seed=1000), IB.sub(len(B_), seed=2000)
            tB_fit = IB.Target(Zf[B_[iB0]], None, S)
            tA_fit = IB.Target(Zf[A_[iA0]], None, S)
            cfit = {}
            for vk in ("v_ss", "v_dyn"):
                vzA, vzB = Vfull[vk][A_[iA0]] @ comps.T, Vfull[vk][B_[iB0]] @ comps.T
                cur = {"fwd": {"sd": [], "ED_U": []}, "back": {"sd": [], "ED_U": []}}
                for c_ in IB.C_GRID:
                    Pf = Zf[A_[iA0]] + c_ * vzA
                    cur["fwd"]["sd"].append(IB.sd_fixed(Pf, IB.unif(len(Pf)), tB_fit))
                    cur["fwd"]["ED_U"].append(IB.ed_uv(Pf, IB.unif(len(Pf)), tB_fit)[0])
                    Pb = Zf[B_[iB0]] - c_ * vzB
                    cur["back"]["sd"].append(IB.sd_fixed(Pb, IB.unif(len(Pb)), tA_fit))
                    cur["back"]["ED_U"].append(IB.ed_uv(Pb, IB.unif(len(Pb)), tA_fit)[0])
                cfit[vk] = {m_: {dr: IB.C_GRID[int(np.argmin(cur[dr][m_]))] for dr in ("fwd", "back")}
                            for m_ in ("sd", "ED_U")}
            rng = np.random.default_rng(1000 * k + 0)
            iA, iB = IB.sub(len(A_), seed=1000), IB.sub(len(B_), seed=2000)
            VA = {vk: (Vfull[vk][A_[iA]] @ comps.T, Vfull[vk][A_[iA]]) for vk in Vfull}
            VB = {vk: (Vfull[vk][B_[iB]] @ comps.T, Vfull[vk][B_[iB]]) for vk in Vfull}
            M = IB.methods(Zf[A_[iA]], Xf[A_[iA]].astype(np.float64), Zf[B_[iB]], Xf[B_[iB]].astype(np.float64),
                           cc_fit[A_[iA]], comps, n_pred, rng, VA=VA, VB=VB, cfit=cfit)
            M.pop("_plan")
            M.pop("_uot")
            preds.append((S, cfit, M))
        (S1, c1, M1), (S2, c2, M2) = preds
        ident = {name: all(same(x, y) for x, y in zip(M1[name], M2[name])) for name in M1}
        out[str(k)] = {"S_identical": S1 == S2, "cfit_identical": c1 == c2, "methods_identical": ident,
                       "n_pred_depends_on_n_held": "n_pred = min(n_held, 1500): only the NUMBER of predicted points"}
        print("interp-layer k", k, out[str(k)], flush=True)
    return out


def metric_scale(data):
    out = {}
    rescore = json.load(open(os.path.join(ERY, "diag_metric", "rescore.json")))
    for s in (2, 3, 4, 5):
        d = F.load(data, s)
        st = d["stage_past"]
        Zn = d["Z_past"][st == s]
        S = F.past_scale(d["Z_past"])
        S_alt = J.past_scale(d["Z_past"].astype(np.float64))
        S_alt2 = J.past_scale(d["Z_past"].astype(np.float64), seed=999)
        tgt = F.Target(d["Z_fut"], S)
        jt = J.Tgt(d["Z_fut"][F.sub(len(d["Z_fut"]), seed=1)], S)
        rows = []
        for lab, Zf in (("persist", Zn), ("persist+N(0,1)", Zn + np.random.default_rng(1).normal(0, 1, Zn.shape)),
                        ("persist+N(0,2)", Zn + np.random.default_rng(2).normal(0, 2, Zn.shape)),
                        ("contract a=0.3", Zn.mean(0) + 0.7 * (Zn - Zn.mean(0)))):
            i = F.sub(len(Zf))
            X = Zf[i]
            wx = F.unif(len(X))
            sd_theirs = float(F.sinkhorn_div_fixed(X, tgt.Y, wx, tgt.wy, tgt.S, tgt.yy_ot))
            sd_mine = J.sd_fixed(X.astype(np.float64), wx, jt)
            en_theirs = F.energy(X, tgt.Y)
            en_mine = J.energies(X.astype(np.float64), wx, jt)[0]
            sdo, sco = F.sinkhorn_div_orig(X, tgt.Y)
            rows.append({"pred": lab, "S_used_by_sd_fixed": tgt.S, "sd_fixed_audited": sd_theirs,
                         "sd_fixed_judge": sd_mine, "energy_audited": en_theirs, "energy_judge": en_mine,
                         "old_sinkhorn_div": sdo, "old_scale (moves with the prediction)": sco})
        # S does not see the future stage at all: recompute with Z_fut replaced by garbage
        out[str(s)] = {"S_audited": S, "S_reported": rescore["origins"][str(s)]["S_fixed"],
                       "S_judge_3000cells_seed123": S_alt, "S_judge_3000cells_seed999": S_alt2,
                       "S_rel_diff_judge_vs_audited": S_alt / S - 1, "rows": rows,
                       "past_scale_reads_only": "Z_past (signature past_scale(Z_past)); Target.S is passed in"}
        print("metric origin", s, json.dumps(out[str(s)], default=float)[:600], flush=True)
    # interp S: fit cells only
    for k in (2, 3, 4, 5):
        d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
        Zf = d["Z_fit"].astype(np.float64)
        Zall = np.vstack([Zf, d["Z_held"].astype(np.float64)])
        out[f"interp{k}"] = {"S_fit_audited_rule": IB.med_scale(Zf), "S_fit_judge": J.past_scale(Zf),
                             "S_if_held_were_included": J.past_scale(Zall)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = {"data_layer": data_layer(a.data)}
    res["metric_scale"] = metric_scale(a.data)
    res["interp_layer"] = interp_layer(a.data)
    res["forecast_layer"] = forecast_layer(a.data)
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "strict.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done", res["seconds"], flush=True)


if __name__ == "__main__":
    main()
