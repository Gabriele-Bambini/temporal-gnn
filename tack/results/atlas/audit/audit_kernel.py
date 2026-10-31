"""AUDIT 3 (forecast-only code path): ery_forecast.kernel_regress as used by atlas_confirm.Problem.growth (uot_rw).
Suspect: the Gaussian Nadaraya-Watson weights exp(-d^2 / 2h^2) are not shifted before exp, so a stage-s cell farther
than ~38.6 h from every stage-(s-1) cell gets an all-zero weight row; its regressed growth is then 0 and is clipped to
1e-6, i.e. uot_rw silently deletes it (a numerical defect: the NW limit is the nearest neighbour's value). Atlas-
specific trigger: new cell types at stage s far from every stage s-1 cell (multi-lineage); erythroid for reference.
Part A (check): per origin, h, the share of stage-s cells with an all-zero / subnormal weight row, and the share of
the uot_rw weight they lose. Part B (EXPLORATORY counterfactual, does not change any verdict): the frozen
run_forecast with ONLY kernel_regress replaced by the shift-stabilised version (identical wherever no underflow),
same grids, rolling-origin selection and scoring; the H2 / H2b rules recomputed. Part B runs only with --with-b and
was NOT run: Part A found zero underflow rows in every origin (largest row-minimum exponent 11.0, origin 2, vs the 745
threshold), so the stabilised regression equals the frozen one. Output: audit/kernel.json.
"""
import sys

import numpy as np

from _common import AC, ATLAS, ERY, IB, confirm, dump, kernel_regress, load

UFLOW = 745.0  # exp(-x) == 0 in float64 for x > ~745.13


def bandwidth(src, k=15):
    d_ss = np.sqrt(IB.sqdist(src[:min(len(src), 800)], src))
    return float(np.median(np.sort(d_ss, 1)[:, min(k, d_ss.shape[1] - 1)]) + 1e-9)


def kr_stable(src, vals, tgt, k=15):
    """same estimator as ery_forecast.kernel_regress, exponent shifted by its row minimum (no underflow)."""
    d_ss = np.sqrt(IB.sqdist(src[:min(len(src), 800)], src))
    h = np.median(np.sort(d_ss, 1)[:, min(k, d_ss.shape[1] - 1)]) + 1e-9
    E = IB.sqdist(tgt, src) / (2 * h * h)
    W = np.exp(-(E - E.min(1, keepdims=True)))
    W /= W.sum(1, keepdims=True)
    return W @ vals


def part_a(folder, s):
    d = load(folder, f"origin{s}.npz")
    st = d["stage_past"]
    Zn = d["Z_past"][st == s].astype(np.float64)
    Zp = d["Z_past"][st == s - 1].astype(np.float64)
    h = bandwidth(Zp)
    E = IB.sqdist(Zn, Zp) / (2 * h * h)
    emin = E.min(1)
    zero = emin > UFLOW
    # frozen output on a constant field: rows with zero weight come back 0 instead of 1
    one = kernel_regress(Zp, np.ones((len(Zp), 1)), Zn)[:, 0]
    out = {"n_now": len(Zn), "n_prev": len(Zp), "h": h, "h_first800_pools": None,
           "share_zero_rows_by_exponent": float(zero.mean()), "share_rows_regress_const_not_1": float((np.abs(one - 1) > 1e-6).mean()),
           "share_rows_emin_gt_700": float((emin > 700).mean()), "emin_quantiles": np.quantile(emin, [0.5, 0.9, 0.99, 1.0]).tolist()}
    if "sample_past" in d.files:
        smp = d["sample_past"].astype(str)[st == s - 1]
        out["h_first800_pools"] = sorted(set(smp[:800].tolist()))
        out["n_pools_prev"] = len(set(smp.tolist()))
    if zero.any() and "celltype_past" in d.files:
        ct = d["celltype_past"].astype(str)[st == s]
        u, c = np.unique(ct[zero], return_counts=True)
        out["zero_row_celltypes"] = dict(zip(u.tolist(), c.tolist()))
    return out


def lost_weight(s):
    """share of the frozen uot_rw (selected hp) weight lost on zero-row cells, final problem of origin s."""
    cj = confirm()["forecast"][str(s)]["uot_rw"]["hp"]
    if cj["lam"] == 0:
        return None
    P = AC.origin_problem(ATLAS, s)
    w = P.growth(cj["tau"], cj["beta"]) ** cj["lam"]
    floor = w <= 1e-6 ** cj["lam"] * 1.0000001
    return {"hp": cj, "share_cells_at_floor": float(floor.mean()), "w_max_share": float(w.max() / w.sum())}


def part_b():
    AC.kernel_regress = kr_stable  # the ONLY change; every other line is the frozen run_forecast
    fo = {}
    for s in range(2, 8):
        r = AC.run_forecast(ATLAS, s)
        fo[s] = {f: {"ED_U": v["ED_U"], "hp": v["hp"], "backtest_ED_U": v["backtest_ED_U"]} for f, v in r.items()}
        print("B stable-kr forecast", s, {f: (round(v["ED_U"], 3), v["hp"]) for f, v in r.items()}, flush=True)
    tab = {s: {f: v["ED_U"] for f, v in r.items()} for s, r in fo.items()}
    return {"forecast": fo, "H2_uot_rw": AC.beats(tab, "uot_rw", ["persist", "meanshift"], 4),
            "H2b_ot_needed": AC.beats(tab, "uot_rw", ["tilt_rw"], 4)}


def main():
    out = {"A_atlas": {}, "A_ery": {}, "lost_weight_atlas": {}}
    for s in range(1, 8):
        out["A_atlas"][s] = part_a(ATLAS, s)
        print("A atlas origin", s, out["A_atlas"][s], flush=True)
    for s in range(1, 6):
        out["A_ery"][s] = part_a(ERY, s)
        print("A ery origin", s, out["A_ery"][s], flush=True)
    for s in range(2, 8):
        out["lost_weight_atlas"][s] = lost_weight(s)
        print("lost weight", s, out["lost_weight_atlas"][s], flush=True)
    if "--with-b" in sys.argv:
        out["B_exploratory_stable_kr"] = part_b()
        print("B verdicts", out["B_exploratory_stable_kr"]["H2_uot_rw"], out["B_exploratory_stable_kr"]["H2b_ot_needed"])
    dump("kernel.json", out)


if __name__ == "__main__":
    main()
