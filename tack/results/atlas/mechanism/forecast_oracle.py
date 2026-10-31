"""EXPLORATORY (not pre-registered): why does uot_rw fail to forecast on the atlas (H2 / H2b FAIL)?
Selection (rolling origin picks badly) or model class (no reweighting of stage-s cells beats persist)?

Per origin s = 2..7, with the frozen atlas_confirm.Problem / origin_problem / UOT_GRID / TILT_GRID (read-only):
 repro         the rolling-origin choice and its ED_U must equal confirm.json (else: bug flag)
 grid          every grid point scored on the FINAL target (ORACLE, best in hindsight) and on the backtest (origin
               s-1 -> stage s, what the frozen rule saw); Spearman(backtest, final) over the grid; oracle gain over
               persist (lam = 0 is persist exactly, so only lam > 0 can beat it)
 extended      ORACLE, outside the frozen grid: uot_rw lam in {3, 4, 6} x (tau, beta) of the grid, tilt_rw lam {3, 4}
 truth_bounds  ORACLE, they USE THE FUTURE: (a) kNN density ratio fut / now at the stage-s cells (k 30, same
               smoothing and clip as Problem.ratio) ^ lam; (b) cell-type composition weights p_fut(type)/p_now(type).
               Upper bounds for any reweighting of stage-s cells with no displacement. PRIMARY rows are estimated on
               the future cells OUTSIDE the scoring subsample (out-of-sample); truth_bounds_ORACLE_insample uses all
               future cells (incl. the scored ones) and is optimistic by construction. 'best' only among rows with
               ESS >= ESS_MIN: the weighted ED_U (self pairs excluded, / (1 - sum w^2)) goes far below the floor
               when weights concentrate (first run: -0.331 at s = 2, lam 3), so low-ESS rows are reported, not ranked;
               ED_V and a persist subsample of ESS_MIN cells (noise scale) are reported alongside.
 support       fraction of stage s+1 cells whose NN distance to the stage-s cells exceeds the 95th pct of the stage-s
               leave-one-out NN distance; stage s+1 mass in types with < MIN_PRESENT cells at stage s
 floor         ED_U of the stage s+1 cells NOT in the scoring subsample against the scoring target (a perfect forecast
               up to sampling and pool differences)
Usage: python forecast_oracle.py   -> forecast_oracle.json
"""
import os
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

import common as C

AC, IB = C.AC, C.IB
MIN_PRESENT = 10
EXT_UOT = [{"lam": x, "tau": t, "beta": b} for t in (0.1, 1.0) for b in (0.0, 2.0) for x in (3.0, 4.0, 6.0)]
EXT_TILT = [{"lam": x} for x in (3.0, 4.0)]
TRUTH_LAM = [0.5, 1.0, 1.5, 2.0, 3.0]
ESS_MIN = 300  # weighted ED_U is unreliable (and can go far below the floor) when the weights concentrate


def truth_ratio(Zn, Zt, k=AC.K_NN):
    Zall = np.vstack([Zn, Zt])
    lab = np.r_[np.zeros(len(Zn)), np.ones(len(Zt))]
    k = min(k, len(Zall) - 1)
    _, ind = NearestNeighbors(n_neighbors=k + 1).fit(Zall).kneighbors(Zn)
    q = (lab[ind[:, 1:]].sum(1) + 1) / (k + 2)
    return np.clip(q / (1 - q) * len(Zn) / len(Zt), 0.05, 20.0)


def main():
    ref = C.confirm()
    t0 = time.time()
    out = {"script": "forecast_oracle.py", "exploratory": True,
           "note": "EXPLORATORY, not pre-registered; ORACLE rows use the forecast target; nothing here changes the "
                   "pre-registered verdicts (H2 / H2b FAIL).", "s": {}}
    for s in range(2, 8):
        final, back = AC.origin_problem(C.DATA, s), AC.origin_problem(C.DATA, s - 1)
        d = np.load(os.path.join(C.DATA, f"origin{s}.npz"), allow_pickle=True)
        persist = final.score(*final.forecast("persist", {}))
        res = {"forecast_stage": C.STAGES[s + 1], "persist_ED_U": persist["ED_U"], "persist_ED_V": persist["ED_V"], "families": {}}
        ok = abs(persist["ED_U"] - ref["forecast"][str(s)]["persist"]["ED_U"]) < 1e-9
        for fam, grid, ext in (("uot_rw", AC.UOT_GRID, EXT_UOT), ("tilt_rw", AC.TILT_GRID, EXT_TILT)):
            bt = [back.score(*back.forecast(fam, hp))["ED_U"] for hp in grid]
            fin = [final.score(*final.forecast(fam, hp)) for hp in grid]
            fe = np.array([f["ED_U"] for f in fin])
            ic, io = int(np.argmin(bt)), int(np.argmin(fe))
            ok = ok and grid[ic] == ref["forecast"][str(s)][fam]["hp"] and abs(fe[ic] - ref["forecast"][str(s)][fam]["ED_U"]) < 1e-9
            pos = [i for i, hp in enumerate(grid) if hp["lam"] > 0]
            bpos = min(pos, key=lambda i: fe[i])
            sp = spearmanr(bt, fe)
            exts = [final.score(*final.forecast(fam, hp)) for hp in ext]
            ie = int(np.argmin([x["ED_U"] for x in exts]))
            res["families"][fam] = {
                "rolling_choice": grid[ic], "rolling_ED_U": float(fe[ic]), "rolling_rank_in_final(1=best)": int((fe < fe[ic]).sum() + 1),
                "oracle_hp": grid[io], "oracle_ED_U": float(fe[io]), "oracle_ess": float(fin[io]["ess"]),
                "best_lam_pos_hp": grid[bpos], "best_lam_pos_ED_U": float(fe[bpos]), "best_lam_pos_ED_V": float(fin[bpos]["ED_V"]),
                "best_lam_pos_ess": float(fin[bpos]["ess"]), "min_ess_lam_pos": float(min(fin[i]["ess"] for i in pos)),
                "any_lam_pos_beats_persist_ED_V": bool(min(fin[i]["ED_V"] for i in pos) < persist["ED_V"]),
                "any_lam_pos_beats_persist": bool(fe[bpos] < persist["ED_U"]),
                "oracle_gain_vs_persist": float(persist["ED_U"] - fe[bpos]),
                "oracle_rel_gain_vs_persist": float((persist["ED_U"] - fe[bpos]) / persist["ED_U"]),
                "n_grid_lam_pos_beating_persist": int(sum(fe[i] < persist["ED_U"] for i in pos)), "n_grid_lam_pos": len(pos),
                "spearman_backtest_vs_final": float(sp.correlation) if np.isfinite(sp.correlation) else None,
                "spearman_lam_pos_only": float(spearmanr([bt[i] for i in pos], fe[pos]).correlation),
                "backtest_gain_of_choice_vs_backtest_persist": float(bt[0] - bt[ic]) if grid[0]["lam"] == 0 else None,
                "grid": [{"hp": hp, "backtest_ED_U": float(b), "final_ED_U": float(f["ED_U"]), "final_ED_V": float(f["ED_V"]), "final_ess": float(f["ess"]),
                          "final_pcc": f["pcc"]} for hp, b, f in zip(grid, bt, fin)],
                "extended_ORACLE": {"best_hp": ext[ie], "best_ED_U": float(exts[ie]["ED_U"]), "best_ess": float(exts[ie]["ess"]),
                                    "best_ED_V": float(exts[ie]["ED_V"]),
                                    "all": [{"hp": hp, "ED_U": float(x["ED_U"]), "ED_V": float(x["ED_V"]), "ess": float(x["ess"])} for hp, x in zip(ext, exts)]},
            }
        # ---- truth-using bounds (ORACLE) ----
        # out-of-sample (primary): ratio / composition estimated on the future cells OUTSIDE the scoring subsample
        # (final.tgt = Zt[IB.sub(len(Zt), seed=1)]); in-sample (all future cells, incl. the scored ones) kept, labelled.
        Zt = d["Z_fut"].astype(np.float64)
        it_t = IB.sub(len(Zt), seed=1)
        rest_t = np.setdiff1d(np.arange(len(Zt)), it_t)
        st_p, ct_p, ct_f = d["stage_past"], d["celltype_past"].astype(str), d["celltype_fut"].astype(str)
        ct_n = ct_p[st_p == s]
        types = sorted(set(ct_n) | set(ct_f))
        pn = {t: (ct_n == t).mean() for t in types}
        pf = {t: (ct_f == t).mean() for t in types}

        def bounds(fut_idx):
            tr = truth_ratio(final.Zn, Zt[fut_idx])
            tb = {}
            for lam in TRUTH_LAM:
                sc = final.score(final.Zn, final.Xn, tr ** lam)
                tb[f"knn_true_ratio_lam{lam}"] = {"ED_U": float(sc["ED_U"]), "ED_V": float(sc["ED_V"]), "ess": float(sc["ess"])}
            pf_ = {t: (ct_f[fut_idx] == t).mean() for t in types}
            sc = final.score(final.Zn, final.Xn, np.array([pf_[t] / pn[t] for t in ct_n]))
            tb["celltype_composition_weights"] = {"ED_U": float(sc["ED_U"]), "ED_V": float(sc["ED_V"]), "ess": float(sc["ess"])}
            okess = [x for x in tb if tb[x]["ess"] >= ESS_MIN]
            best_tb = min(okess, key=lambda x: tb[x]["ED_U"]) if okess else None
            return {**tb, "n_future_cells_used": int(len(fut_idx)), "ess_min_for_best": ESS_MIN, "best": best_tb,
                    "best_ED_U": tb[best_tb]["ED_U"] if best_tb else None,
                    "best_ED_V": tb[best_tb]["ED_V"] if best_tb else None,
                    "best_ess": tb[best_tb]["ess"] if best_tb else None,
                    "best_rel_gain_vs_persist": float((persist["ED_U"] - tb[best_tb]["ED_U"]) / persist["ED_U"]) if best_tb else None}

        res["truth_bounds_ORACLE"] = bounds(rest_t)
        res["truth_bounds_ORACLE_insample"] = bounds(np.arange(len(Zt)))
        best_tb = res["truth_bounds_ORACLE"]["best"]
        # where the rolling-origin uot_rw choice moved mass: mean normalised weight per stage-s cell type (>= 50 cells)
        hp_c = res["families"]["uot_rw"]["rolling_choice"]
        if hp_c["lam"] > 0:
            w = final.forecast("uot_rw", hp_c)[2]
            w = w / w.mean()
            tw = {t: float(w[ct_n == t].mean()) for t in types if (ct_n == t).sum() >= 50}
            res["uot_rw_rolling_weight_by_type"] = {"hp": hp_c, "mean_weight": dict(sorted(tw.items(), key=lambda x: -x[1])),
                                                    "future_over_now_type_ratio": {t: float(pf[t] / pn[t]) for t in tw},
                                                    "spearman_weight_vs_true_type_ratio": float(spearmanr([tw[t] for t in tw], [pf[t] / pn[t] for t in tw]).correlation)}
        # noise scale of ED_U at ESS_MIN effective points: random ESS_MIN-cell subsamples of stage s (unweighted persist)
        nz = []
        for q in range(10):
            ii = np.sort(np.random.default_rng(900 + q).choice(len(final.Zn), ESS_MIN, replace=False))
            nz.append(IB.ed_uv(final.Zn[ii], IB.unif(ESS_MIN), final.tgt)[0])
        res["persist_subsample_ESSMIN_ED_U"] = {"mean": float(np.mean(nz)), "sd": float(np.std(nz))}
        # ---- support and composition ----
        nnN = NearestNeighbors(n_neighbors=2).fit(final.Zn)
        d_self = nnN.kneighbors(final.Zn)[0][:, 1]
        thr = float(np.quantile(d_self, 0.95))
        d_fut = nnN.kneighbors(Zt, n_neighbors=1)[0][:, 0]
        cnt_n = {t: int((ct_n == t).sum()) for t in types}
        res["support"] = {"frac_future_off_support": float((d_fut > thr).mean()),
                          "median_nn_future_over_self": float(np.median(d_fut) / np.median(d_self)),
                          "future_mass_types_absent_now": float(np.mean([cnt_n[t] < MIN_PRESENT for t in ct_f])),
                          "TV_now_future_types": C.tv([pn[t] for t in types], [pf[t] for t in types])}
        # ---- floor: the future cells outside the scoring subsample, scored like a forecast ----
        it = IB.sub(len(Zt), seed=1)
        rest = np.setdiff1d(np.arange(len(Zt)), it)
        res["floor_ED_U_future_rest"] = float(IB.ed_uv(Zt[rest], IB.unif(len(rest)), final.tgt)[0]) if len(rest) > 1 else None
        res["repro_ok"] = bool(ok)
        out["s"][s] = res
        f = res["families"]
        print(f"s={s} persist {persist['ED_U']:.3f} | uot oracle {f['uot_rw']['best_lam_pos_ED_U']:.3f} {f['uot_rw']['best_lam_pos_hp']} "
              f"rho {f['uot_rw']['spearman_backtest_vs_final']} | tilt oracle {f['tilt_rw']['best_lam_pos_ED_U']:.3f} | "
              f"truth {res['truth_bounds_ORACLE']['best_ED_U']} ({best_tb}) | floor {res['floor_ED_U_future_rest']:.3f} | "
              f"offsup {res['support']['frac_future_off_support']:.3f} repro {ok} [{time.time() - t0:.0f}s]", flush=True)
        C.dump(os.path.join(C.HERE, "forecast_oracle.json"), out)
    fs = out["s"]
    out["summary"] = {
        fam: {"origins_any_lam_pos_beats_persist": [int(s) for s in fs if fs[s]["families"][fam]["any_lam_pos_beats_persist"]],
              "oracle_beats_strict_persist_count": int(sum(fs[s]["families"][fam]["any_lam_pos_beats_persist"] for s in fs)),
              "mean_oracle_ED_U": float(np.mean([fs[s]["families"][fam]["best_lam_pos_ED_U"] for s in fs])),
              "mean_rolling_ED_U": float(np.mean([fs[s]["families"][fam]["rolling_ED_U"] for s in fs])),
              "mean_persist_ED_U": float(np.mean([fs[s]["persist_ED_U"] for s in fs])),
              "median_spearman_backtest_vs_final": float(np.median([fs[s]["families"][fam]["spearman_backtest_vs_final"] for s in fs
                                                                     if fs[s]["families"][fam]["spearman_backtest_vs_final"] is not None]))}
        for fam in ("uot_rw", "tilt_rw")}
    out["summary"]["all_repro_ok"] = bool(all(fs[s]["repro_ok"] for s in fs))
    out["seconds"] = round(time.time() - t0, 1)
    C.dump(os.path.join(C.HERE, "forecast_oracle.json"), out)
    print(out["summary"])


if __name__ == "__main__":
    main()
