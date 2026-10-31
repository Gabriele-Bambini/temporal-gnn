"""EXPLORATORY (not pre-registered): why do ot_pairs / uot lose to the 50/50 union on the atlas (H1 / H1b FAIL)?

Per held stage k = 1..7 (interp<k>.npz, PCA space of the product, frozen IB scoring):
 composition   cell-type proportions at k-1, k, k+1; TV(held, prev / next / 50-50 mix / best alpha-mix); mass in
               one-sided types (< MIN_PRESENT cells at k-1 or at k+1); transient-peak excess at k.
 ed_identity   energy distance is an MMD, so ED(1/2 A + 1/2 B, H) = 1/2 ED(A,H) + 1/2 ED(B,H) - 1/4 ED(A,B) exactly:
               the union earns a 'convexity bonus' 1/4 ED(prev, next) that a displacement interpolant does not.
               Checked on the frozen rep subsamples (weighted full 50/50 cloud, deterministic) and against the
               recorded union ED_U (random 750+750 subsample).
 stage_vote    15-NN classifier prev-vs-next trained on fit cells NOT used by the rep-0 subsamples (balanced),
               applied to held cells and to rep-0 predictions: bimodal votes = coexisting populations (mixture
               regime), votes near 0.5 = intermediate cells (displacement regime). Per held pool too.
 off_manifold  NN distance of each prediction point to the held target subsample, vs held cells outside the
               target subsample (reference, its 95th pct = threshold): fraction of off-manifold points; coverage of
               the target by the prediction at equal size.
 pair_types    rep-0 entropic plan (eps_rel 0.05): plan mass in same-type pairs (vs independence) and the
               off-manifold fraction of same-type vs cross-type McCann midpoints; top flows.
 t_scan        ORACLE-labelled: McCann position t in 0..1 (ot_pairs at t = 0.5 is the frozen method) on the rep plans;
 alpha_scan    ORACLE-labelled: mixture weight alpha of next in 0..1 (union = 0.5), weighted full rep subsamples.
 t_spread      ORACLE-labelled: each OT pair at its own t = 0.5 + h u, u ~ U(-1, 1) ('uniform') or u = +-1
               ('two_point'), h in 0..0.5: is the held stage a spread of developmental times rather than one midpoint?
 pool_oracle   ORACLE-labelled: t-scan and alpha-scan against each held pool (>= 20 cells) separately: does OT at the
               pool's own best t beat the pool's best mixture (embryo-level asynchrony)?
 pools         from confirm.json pool_mean (no compute): per held pool, union vs ot_pairs vs uot.
Usage: python interp_mechanism.py [--ks 1 2 ..]   -> interp_mechanism.json
"""
import argparse
import os
import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

import common as C

IB = C.IB
MIN_PRESENT = 10
T_GRID = [round(x, 2) for x in np.linspace(0, 1, 11)]
A_GRID = [round(x, 2) for x in np.linspace(0, 1, 21)]
H_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def composition(ctf, st, cth, k):
    types = sorted(set(ctf) | set(cth))
    cnt = {"prev": np.array([((st == k - 1) & (ctf == t)).sum() for t in types]),
           "held": np.array([(cth == t).sum() for t in types]),
           "next": np.array([((st == k + 1) & (ctf == t)).sum() for t in types])}
    p = {s: v / v.sum() for s, v in cnt.items()}
    mix = 0.5 * (p["prev"] + p["next"])
    tva = [C.tv(p["held"], (1 - a) * p["prev"] + a * p["next"]) for a in np.linspace(0, 1, 101)]
    ia = int(np.argmin(tva))
    absent_prev, absent_next = cnt["prev"] < MIN_PRESENT, cnt["next"] < MIN_PRESENT
    one_sided = absent_prev ^ absent_next
    dpn = np.abs(p["next"] - p["prev"])
    peak = np.maximum(0, p["held"] - np.maximum(p["prev"], p["next"]))
    trough = np.maximum(0, np.minimum(p["prev"], p["next"]) - p["held"])
    table = {t: {"prev": int(cnt["prev"][i]), "held": int(cnt["held"][i]), "next": int(cnt["next"][i])}
             for i, t in enumerate(types)}
    return {
        "counts": table,
        "TV_held_prev": C.tv(p["held"], p["prev"]), "TV_held_next": C.tv(p["held"], p["next"]),
        "TV_held_mix5050": C.tv(p["held"], mix), "TV_prev_next": C.tv(p["prev"], p["next"]),
        "best_alpha_next": ia / 100, "TV_held_best_alpha_mix": float(tva[ia]),
        "held_mass_types_absent_at_prev": float(p["held"][absent_prev].sum()),
        "held_mass_types_absent_at_next": float(p["held"][absent_next].sum()),
        "next_mass_types_absent_at_prev(appearing)": float(p["next"][absent_prev].sum()),
        "prev_mass_types_absent_at_next(vanishing)": float(p["prev"][absent_next].sum()),
        "share_of_TV_prev_next_from_one_sided_types": float(dpn[one_sided].sum() / max(dpn.sum(), 1e-12)),
        "held_transient_peak_excess": float(peak.sum()), "held_transient_trough_deficit": float(trough.sum()),
        "held_mass_in_types_between_neighbours": float(p["held"][(p["held"] >= np.minimum(p["prev"], p["next"])) &
                                                               (p["held"] <= np.maximum(p["prev"], p["next"]))].sum()),
        "_p": p, "_types": types,
    }


def vote_stats(v):
    v = np.asarray(v)
    h, _ = np.histogram(v, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0001])
    return {"mean_P_next": float(v.mean()), "frac_prev_like(<0.2)": float((v < 0.2).mean()),
            "frac_next_like(>0.8)": float((v > 0.8).mean()), "frac_intermediate[0.2,0.8]": float(((v >= 0.2) & (v <= 0.8)).mean()),
            "hist5": (h / len(v)).round(4).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=list(range(1, 8)))
    args = ap.parse_args()
    ref = C.confirm()
    out = {"script": "interp_mechanism.py", "exploratory": True,
           "note": "EXPLORATORY, not pre-registered; does not change the pre-registered verdicts (H1/H1b FAIL).",
           "k": {}}
    t0 = time.time()
    for k in args.ks:
        d = C.load_interp(k)
        st = d["stage_fit"]
        ctf, cth = d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
        smp_h = d["sample_held"].astype(str)
        Zf, Zh = d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
        res = {"held": C.STAGES[k]}
        comp = composition(ctf, st, cth, k)
        p, types = comp.pop("_p"), comp.pop("_types")
        res["composition"] = comp

        # frozen-convention run (no masks) keeping rep-0 clouds
        run, r0 = C.run_interp_subset(d, k, keep_rep0=True)
        for m in C.METHODS:
            assert abs(run["mean"][m]["ED_U"] - ref["interp"][str(k)]["mean"][m]["ED_U"]) < 1e-9
        tgt = r0["tgt"]
        A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]

        # ---- ED identity on the three rep subsamples ----
        S = IB.med_scale(Zf)
        ptg = {pp: IB.Target(Zh[smp_h == pp], None, S, with_sd=False) for pp in sorted(set(smp_h))
               if (smp_h == pp).sum() >= C.AC.MIN_POOL}
        idr = []
        for r in range(3):
            iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
            ZA, ZB = Zf[A[iA]], Zf[B[iB]]
            eAH = IB.ed_uv(ZA, IB.unif(len(ZA)), tgt)[0]
            eBH = IB.ed_uv(ZB, IB.unif(len(ZB)), tgt)[0]
            eAB = IB.ed_uv(ZA, IB.unif(len(ZA)), IB.Target(ZB, None, S, with_sd=False))[0]
            w = np.r_[np.full(len(ZA), 0.5 / len(ZA)), np.full(len(ZB), 0.5 / len(ZB))]
            emix = IB.ed_uv(np.vstack([ZA, ZB]), w, tgt)[0]
            # time-t McCann (oracle t-scan) on this rep's plan, pairs drawn once
            P, _ = IB.plan(ZA, ZB, IB.DEFAULT["eps_rel"])
            n = min(len(tgt.Z), IB.NSUB)
            i, j = IB.draw_pairs(P, n, np.random.default_rng(50_000 + 100 * k + r))
            ts = [IB.ed_uv((1 - t) * ZA[i] + t * ZB[j], IB.unif(n), tgt)[0] for t in T_GRID]
            wa = lambda a: np.r_[np.full(len(ZA), (1 - a) / len(ZA)), np.full(len(ZB), a / len(ZB))]  # noqa: E731
            al = [IB.ed_uv(np.vstack([ZA, ZB]), wa(a), tgt)[0] for a in A_GRID]
            # time-spread OT (ORACLE scan): each pair at its own t = 0.5 + h u, u ~ U(-1, 1) (uniform) or u = +-1
            # (two-point; h = 0.5 two-point = the endpoints of the pairs = a union with OT-plan marginals)
            u = np.random.default_rng(60_000 + 100 * k + r).uniform(-1, 1, n)
            spr = {}
            for kind, uu in (("uniform", u), ("two_point", np.sign(u))):
                for h in H_GRID:
                    tt = (0.5 + h * uu)[:, None]
                    spr[f"{kind}_h{h}"] = IB.ed_uv((1 - tt) * ZA[i] + tt * ZB[j], IB.unif(n), tgt)[0]
            # per held pool (ORACLE): t-scan and alpha-scan against each pool separately
            pts, pal = {}, {}
            for pp, ptgt in ptg.items():
                pts[pp] = [IB.ed_uv((1 - t) * ZA[i] + t * ZB[j], IB.unif(n), ptgt)[0] for t in T_GRID]
                pal[pp] = [IB.ed_uv(np.vstack([ZA, ZB]), wa(a), ptgt)[0] for a in A_GRID]
            idr.append({"ED_prev_held": eAH, "ED_next_held": eBH, "ED_prev_next": eAB, "ED_full_mix_held": emix,
                        "identity_pred": 0.5 * eAH + 0.5 * eBH - 0.25 * eAB, "t_scan": ts, "alpha_scan": al,
                        "spread": spr, "pool_t": pts, "pool_a": pal})
        mean = lambda key: float(np.mean([x[key] for x in idr]))  # noqa: E731
        ot = run["mean"]["ot_pairs"]["ED_U"]
        res["ed_identity"] = {
            "ED_prev_held": mean("ED_prev_held"), "ED_next_held": mean("ED_next_held"), "ED_prev_next": mean("ED_prev_next"),
            "avg_endpoints": 0.5 * (mean("ED_prev_held") + mean("ED_next_held")),
            "convexity_bonus_quarter_ED_prev_next": 0.25 * mean("ED_prev_next"),
            "identity_pred_union": mean("identity_pred"), "full_weighted_mix_ED_U": mean("ED_full_mix_held"),
            "recorded_union_ED_U": run["mean"]["union"]["ED_U"], "recorded_ot_pairs_ED_U": ot,
            "recorded_uot_ED_U": run["mean"]["uot"]["ED_U"],
            "ot_pairs_gain_vs_avg_endpoints": 0.5 * (mean("ED_prev_held") + mean("ED_next_held")) - ot,
            "union_gain_vs_avg_endpoints": 0.5 * (mean("ED_prev_held") + mean("ED_next_held")) - run["mean"]["union"]["ED_U"],
        }
        tsm = np.mean([x["t_scan"] for x in idr], 0)
        alm = np.mean([x["alpha_scan"] for x in idr], 0)
        res["t_scan_ORACLE"] = {"t": T_GRID, "ED_U": tsm.round(5).tolist(), "best_t": T_GRID[int(np.argmin(tsm))],
                                "best_ED_U": float(tsm.min()), "at_t0.5": float(tsm[5])}
        res["alpha_scan_ORACLE"] = {"alpha_next": A_GRID, "ED_U": alm.round(5).tolist(),
                                    "best_alpha": A_GRID[int(np.argmin(alm))], "best_ED_U": float(alm.min()),
                                    "at_alpha0.5": float(alm[10])}
        res["best_t_OT_vs_best_alpha_mix"] = {"best_OT": float(tsm.min()), "best_mix": float(alm.min()),
                                              "mix_better": bool(alm.min() < tsm.min())}
        spm = {key: float(np.mean([x["spread"][key] for x in idr])) for key in idr[0]["spread"]}
        res["t_spread_scan_ORACLE"] = {"ED_U": spm, "best": min(spm, key=spm.get), "best_ED_U": float(min(spm.values()))}
        pool_or = {}
        for pp in ptg:
            pt_ = np.mean([x["pool_t"][pp] for x in idr], 0)
            pa_ = np.mean([x["pool_a"][pp] for x in idr], 0)
            pool_or[pp] = {"n": int((smp_h == pp).sum()), "best_t": T_GRID[int(np.argmin(pt_))], "best_t_ED_U": float(pt_.min()),
                           "t0.5_ED_U": float(pt_[5]), "best_alpha": A_GRID[int(np.argmin(pa_))],
                           "best_alpha_ED_U": float(pa_.min()), "alpha0.5_ED_U": float(pa_[10]),
                           "best_t_OT_beats_best_mix": bool(pt_.min() < pa_.min())}
        res["pool_oracle_scans"] = pool_or

        # ---- stage vote (classifier trained on fit cells outside the rep-0 subsamples, balanced) ----
        cA = np.setdiff1d(A, r0["A_idx"])
        cB = np.setdiff1d(B, r0["B_idx"])
        m = min(len(cA), len(cB))
        rg = np.random.default_rng(7)
        trA, trB = rg.choice(cA, m, replace=False), rg.choice(cB, m, replace=False)
        knn = KNeighborsClassifier(n_neighbors=15).fit(Zf[np.r_[trA, trB]], np.r_[np.zeros(m), np.ones(m)])
        pn = lambda Z: knn.predict_proba(Z)[:, 1]  # noqa: E731
        vh = pn(Zh)
        sv = {"n_train_each": int(m), "held_all": vote_stats(vh),
              "acc_on_prev_rep0": float((pn(Zf[r0["A_idx"]]) < 0.5).mean()),
              "acc_on_next_rep0": float((pn(Zf[r0["B_idx"]]) >= 0.5).mean())}
        for mth in ("union", "ot_pairs", "uot", "indep", "sb_half"):
            sv[mth] = vote_stats(pn(r0["clouds"][mth][0]))
        sv["held_per_pool_mean_P_next"] = {pp: float(vh[smp_h == pp].mean()) for pp in sorted(set(smp_h))
                                           if (smp_h == pp).sum() >= C.AC.MIN_POOL}
        sv["held_per_pool_n"] = {pp: int((smp_h == pp).sum()) for pp in sorted(set(smp_h))}
        # per held cell type: mean vote (which neighbour each type resembles)
        sv["held_per_type_mean_P_next"] = {t: float(vh[cth == t].mean()) for t in types if (cth == t).sum() >= 50}
        res["stage_vote"] = sv

        # ---- off-manifold / coverage ----
        others = np.setdiff1d(np.arange(len(Zh)), r0["held_sub_idx"])
        nnT = NearestNeighbors(n_neighbors=1).fit(tgt.Z)
        d_ref = nnT.kneighbors(Zh[others])[0][:, 0]
        thr = float(np.quantile(d_ref, 0.95))
        mcov = len(others)
        nnO = NearestNeighbors(n_neighbors=1).fit(Zh[others])
        d_cov_ref = nnO.kneighbors(tgt.Z)[0][:, 0]
        thr_cov = float(np.quantile(d_cov_ref, 0.95))
        om = {"n_target": int(len(tgt.Z)), "n_reference_held_others": int(mcov), "thr_q95": thr,
              "reference_median_nn": float(np.median(d_ref)), "methods": {}}
        for mth in C.METHODS:
            Zs = r0["clouds"][mth][0]
            dd = nnT.kneighbors(Zs)[0][:, 0]
            sel = np.random.default_rng(11).choice(len(Zs), min(mcov, len(Zs)), replace=False)
            dc = NearestNeighbors(n_neighbors=1).fit(Zs[sel]).kneighbors(tgt.Z)[0][:, 0]
            om["methods"][mth] = {"frac_off_manifold": float((dd > thr).mean()),
                                  "median_nn_ratio": float(np.median(dd) / np.median(d_ref)),
                                  "frac_target_uncovered": float((dc > thr_cov).mean())}
        om["reference_frac_target_uncovered"] = float((d_cov_ref > thr_cov).mean())
        res["off_manifold"] = om

        # ---- pair types (rep-0 plan) ----
        ZA, ZB = Zf[r0["A_idx"]], Zf[r0["B_idx"]]
        tA, tB = ctf[r0["A_idx"]], ctf[r0["B_idx"]]
        P, _ = IB.plan(ZA, ZB, IB.DEFAULT["eps_rel"])
        P = P / P.sum()
        same = tA[:, None] == tB[None, :]
        pa = np.array([(tA == t).mean() for t in types])
        pb = np.array([(tB == t).mean() for t in types])
        n = min(len(tgt.Z), IB.NSUB)
        i, j = IB.draw_pairs(P, n, np.random.default_rng(5))
        mid = 0.5 * ZA[i] + 0.5 * ZB[j]
        dm = nnT.kneighbors(mid)[0][:, 0]
        sp = tA[i] == tB[j]
        flows = {}
        for t1 in sorted(set(tA)):
            for t2 in sorted(set(tB)):
                v = float(P[np.ix_(tA == t1, tB == t2)].sum())
                if v > 0.01:
                    flows[f"{t1} -> {t2}"] = round(v, 4)
        res["pair_types"] = {
            "plan_mass_same_type": float(P[same].sum()), "independent_same_type": float(pa @ pb),
            "frac_pairs_same_type_drawn": float(sp.mean()),
            "off_manifold_same_type_midpoints": float((dm[sp] > thr).mean()) if sp.any() else None,
            "off_manifold_cross_type_midpoints": float((dm[~sp] > thr).mean()) if (~sp).any() else None,
            "top_flows(>1%)": dict(sorted(flows.items(), key=lambda x: -x[1])[:12]),
        }

        # ---- pools (confirm.json, no compute) ----
        pm = ref["interp"][str(k)]["pool_mean"]
        res["pools_confirm"] = {pp: {mth: round(pm[mth][pp], 4) for mth in ("prev", "next", "union", "ot_pairs", "uot")}
                                for pp in ref["interp"][str(k)]["pools"]}
        res["pools_union_beats_ot_pairs"] = {pp: bool(pm["union"][pp] < pm["ot_pairs"][pp]) for pp in ref["interp"][str(k)]["pools"]}
        out["k"][k] = res
        print(f"k={k} done [{time.time() - t0:.0f}s]", {"bonus": round(res['ed_identity']['convexity_bonus_quarter_ED_prev_next'], 3),
              "pred": round(res['ed_identity']['identity_pred_union'], 3), "union": round(res['ed_identity']['recorded_union_ED_U'], 3),
              "best_t": res["t_scan_ORACLE"]["best_t"], "bestOT": round(res["t_scan_ORACLE"]["best_ED_U"], 3),
              "best_a": res["alpha_scan_ORACLE"]["best_alpha"], "bestmix": round(res["alpha_scan_ORACLE"]["best_ED_U"], 3)}, flush=True)
        C.dump(os.path.join(C.HERE, "interp_mechanism.json"), out)
    out["seconds"] = round(time.time() - t0, 1)
    C.dump(os.path.join(C.HERE, "interp_mechanism.json"), out)


if __name__ == "__main__":
    main()
