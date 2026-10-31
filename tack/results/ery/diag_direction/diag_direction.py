"""Direction vs real time on the erythroid lineage. DESCRIPTIVE diagnostic: it looks at the TRUE next stage s+1,
so nothing here is a forecast and nothing here may be used to pick a forecaster's hyperparameters.

Input: origin<s>.npz from tack/hpc/ery_prep.py (s = 1..5; gene filter, HVG, PCA, moments, velocities fitted on
stages <= s; stage s+1 only transformed). obs.csv for samples / sequencing batches. Origin 1 is degenerate
(108 past cells, 105 genes): computed and reported, not used for conclusions.

Per origin s:
 A. OT displacement of the stage-s cells toward the true stage s+1, origin's 30-d PCA space: own balanced
    log-domain Sinkhorn (ery_forecast.ot_coupling: cost / median cost, eps_rel = 0.05 as in the 'ot' family),
    barycentric map T(z_i) - z_i. Robustness: eps_rel 0.01 / 0.2, subsample seeds 0 / 1 / 2, and the exact
    assignment (scipy linear_sum_assignment) on equal-size subsamples (no entropic blur).
 B. RNA velocity of the same cells on the log scale, V = v / (1 + Ms), projected into PCA: V @ comps.T; v_ss (own
    steady-state) and v_dyn (scVelo dynamical), with and without MURK genes (MURK columns of V set to 0 first).
 C. Per cell type of the source cell: median / mean cosine and fraction of cells with cosine > 0, against
      disp        the full displacement
      resid_disp  displacement minus its population mean (velocity uncentred)
      resid_both  displacement minus its mean vs velocity minus its mean (per-cell content of both)
      meanshift   the population mean shift mu_{s+1} - mu_s (same vector for every cell)
    Nulls (200 permutations): velocity rows permuted within the stage, and within the cell type. Both keep the
    population (type) level direction of the velocities and break only the cell-to-cell match. Random directions in
    30-d give cosines with sd ~ 1/sqrt(30) = 0.18.
 D. Mean-shift consistency: cosine of mu_s - mu_{s-1} vs mu_{s+1} - mu_s (PCA and gene space), norm ratio; the whole
    chain of consecutive shifts in origin 5's space (stages 0..6); within-cell-type vs composition split of the shift.
 E. Movement vs spread: |shift| / RMS radius of stage s, / sd along the shift axis (both stages), AUC of the
    projection on the shift axis, share of the exact W2^2 explained by the translation, per-sample means located
    along the shift axis (are samples of one stage further apart than the two stages?).
Secondary, STRICT (no stage s+1): the same cosine comparison on the previous transition s-1 -> s (velocity of the
stage s-1 cells, which is what the ot_vel_* families combine with the OT drift).
Gene-space secondary: cosine of the gene-space barycentric displacement with V (MURK zeroed in both when dropped;
for v_dyn also restricted to genes with a nonzero dynamical velocity).

Usage: python diag_direction.py [DATA_DIR]   (writes results.json, cosines.csv, samples.csv, percell.npz here)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "hpc")))
import ery_forecast as ef  # noqa: E402  (own Sinkhorn; imports numpy only)

DATA = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ERY_DATA", "")
STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
NSUB, NPERM = 1500, 200
EPS_MAIN, EPS_ALT, SEEDS = 0.05, (0.01, 0.2), (0, 1, 2)
MIN_N = 10            # cell-type groups below this are flagged
MIN_SAMPLE = 20       # samples below this are not used for sample-mean distances
VELS, MURKS = ("v_ss", "v_dyn"), ("with", "without")
TARGETS = ("disp", "resid_disp", "resid_both", "meanshift")
ROWS = []             # flat cosine table
KEY = {"s->s+1": "fut", "s-1->s": "past"}


def sub(n, k=NSUB, seed=0):
    r = np.random.default_rng(seed)
    return np.sort(r.choice(n, min(n, k), replace=False))


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)


def cos_rows(A, B):
    return (unit(A) * unit(B)).sum(-1)


def cosv(a, b):
    return float(cos_rows(a[None], b[None])[0])


def target_pair(D, V, mshift, tgt):
    if tgt == "disp":
        return D, V
    if tgt == "resid_disp":
        return D - D.mean(0), V
    if tgt == "resid_both":
        return D - D.mean(0), V - V.mean(0)
    return np.broadcast_to(mshift, D.shape), V


def group_stats(c, groups):
    out = {}
    for g in ["all"] + sorted(set(groups)):
        m = np.ones(len(c), bool) if g == "all" else groups == g
        out[g] = {"n": int(m.sum()), "median": float(np.median(c[m])), "mean": float(c[m].mean()),
                  "frac_pos": float((c[m] > 0).mean()), "small_n": bool(m.sum() < MIN_N)}
    return out


def permuted(n, groups, rng, within_type):
    if not within_type:
        return rng.permutation(n)
    idx = np.arange(n)
    for g in set(groups):
        w = np.where(groups == g)[0]
        idx[w] = w[rng.permutation(len(w))]
    return idx


def cosine_block(D, V, mshift, groups, tgt, nulls=True, seed=0):
    A, B = target_pair(D, V, mshift, tgt)
    c = cos_rows(A, B)
    st = group_stats(c, groups)
    if nulls and tgt != "meanshift":
        rng = np.random.default_rng(1000 + seed)
        for kind, wt in (("stage", False), ("type", True)):
            med = {g: [] for g in st}
            fp = {g: [] for g in st}
            for _ in range(NPERM):
                p = permuted(len(V), groups, rng, wt)
                Ap, Bp = target_pair(D, V[p], mshift, tgt)
                cp = cos_rows(Ap, Bp)
                for g in st:
                    m = np.ones(len(cp), bool) if g == "all" else groups == g
                    med[g].append(np.median(cp[m]))
                    fp[g].append((cp[m] > 0).mean())
            for g in st:
                mm = np.array(med[g])
                st[g][f"null_{kind}_median_mean"] = float(mm.mean())
                st[g][f"null_{kind}_median_q95"] = float(np.quantile(mm, 0.95))
                st[g][f"null_{kind}_fracpos_mean"] = float(np.mean(fp[g]))
                st[g][f"p_{kind}"] = float((1 + (mm >= st[g]["median"]).sum()) / (1 + NPERM))
    return st, c


def record(origin, transition, method, vel, murk, tgt, st):
    for g, r in st.items():
        ROWS.append({"origin": origin, "transition": transition, "method": method, "velocity": vel, "murk": murk,
                     "target": tgt, "celltype": g, **r})


def velocity_pca(f, rows, comps, murk_mask):
    """Per variant: (PCA-projected log velocity, gene-space log velocity)."""
    Ms = f["Ms"][rows].astype(np.float64)
    out = {}
    for vk in VELS:
        V = f[vk][rows].astype(np.float64) / (1.0 + Ms)
        for mk in MURKS:
            Vm = V.copy()
            if mk == "without":
                Vm[:, murk_mask] = 0.0
            out[(vk, mk)] = (Vm @ comps.T, Vm)
    return out


def barycentric(Z0, Z1, eps, X0=None, X1=None):
    P, a = ef.ot_coupling(Z0, Z1, eps_rel=eps)
    row = P.sum(1)
    D = (P @ Z1) / row[:, None] - Z0
    marg = {"row_err": float(np.abs(row / a - 1).max()),
            "col_err": float(np.abs(P.sum(0) * len(Z1) - 1).max())}
    DX = None if X0 is None else (P @ X1) / row[:, None] - X0
    return D, DX, marg


def exact_assignment(Z0, Z1):
    C = ef.sqdist(Z0, Z1)
    r, c = linear_sum_assignment(C)
    return Z1[c] - Z0[r], float(C[r, c].mean()), r


def auc(a, b):
    """P(proj of a stage-(s+1) cell > proj of a stage-s cell)."""
    rk = rankdata(np.concatenate([a, b]))
    return float((rk[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))


def shift_decomposition(Z0, ct0, Z1, ct1):
    """mu1 - mu0 = sum_c pbar_c (m1_c - m0_c) [within] + sum_c (p1_c - p0_c) mbar_c [composition]; a type missing
    on one side takes the other side's mean (zero within-type change)."""
    types = sorted(set(ct0) | set(ct1))
    W = np.zeros(Z0.shape[1])
    Cm = np.zeros(Z0.shape[1])
    per = {}
    for t in types:
        p0, p1 = (ct0 == t).mean(), (ct1 == t).mean()
        m0 = Z0[ct0 == t].mean(0) if (ct0 == t).any() else Z1[ct1 == t].mean(0)
        m1 = Z1[ct1 == t].mean(0) if (ct1 == t).any() else m0
        W += 0.5 * (p0 + p1) * (m1 - m0)
        Cm += (p1 - p0) * 0.5 * (m0 + m1)
        per[t] = {"p_s": float(p0), "p_s1": float(p1), "n_s": int((ct0 == t).sum()), "n_s1": int((ct1 == t).sum()),
                  "within_shift_norm": float(np.linalg.norm(m1 - m0))}
    tot = Z1.mean(0) - Z0.mean(0)
    u = tot / np.linalg.norm(tot)
    return {"total_norm": float(np.linalg.norm(tot)), "within_norm": float(np.linalg.norm(W)),
            "composition_norm": float(np.linalg.norm(Cm)),
            "within_share_along_shift": float(W @ u / np.linalg.norm(tot)),
            "composition_share_along_shift": float(Cm @ u / np.linalg.norm(tot)),
            "cos_within_composition": cosv(W, Cm), "identity_err": float(np.linalg.norm(W + Cm - tot)),
            "per_type": per}


def movement_vs_spread(Z0, Z1, X0m, X1m, R0X):
    d = Z1.mean(0) - Z0.mean(0)
    u = d / np.linalg.norm(d)
    c0, c1 = Z0 - Z0.mean(0), Z1 - Z1.mean(0)
    rms0 = float(np.sqrt((c0 ** 2).sum(1).mean()))
    rms1 = float(np.sqrt((c1 ** 2).sum(1).mean()))
    sd0, sd1 = float((c0 @ u).std()), float((c1 @ u).std())
    dX = np.linalg.norm(X1m - X0m)
    return {"shift_norm_pca": float(np.linalg.norm(d)), "rms_radius_s": rms0, "rms_radius_s1": rms1,
            "shift_over_rms_s": float(np.linalg.norm(d) / rms0),
            "sd_along_shift_s": sd0, "sd_along_shift_s1": sd1,
            "shift_over_pooled_sd_along": float(np.linalg.norm(d) / np.sqrt(0.5 * (sd0 ** 2 + sd1 ** 2))),
            "auc_along_shift": auc(Z0 @ u, Z1 @ u),
            "shift_norm_gene": float(dX), "rms_radius_s_gene": R0X, "shift_over_rms_s_gene": float(dX / R0X)}


def sample_positions(Zs, samp_s, bat_s, Zs1, samp_s1, bat_s1, s):
    mu0, mu1 = Zs.mean(0), Zs1.mean(0)
    d = mu1 - mu0
    L = np.linalg.norm(d)
    u = d / L
    R = np.sqrt(((Zs - mu0) ** 2).sum(1).mean())
    rows, means = [], {}
    for lab, Z, sm, bt in ((s, Zs, samp_s, bat_s), (s + 1, Zs1, samp_s1, bat_s1)):
        for x in sorted(set(sm)):
            m = sm == x
            mean = Z[m].mean(0)
            off = mean - mu0
            rows.append({"origin": s, "stage": STAGES[lab], "stage_idx": lab, "sample": int(x),
                         "batch": int(bt[m][0]), "n": int(m.sum()), "t_along_shift": float(off @ u / L),
                         "perp_over_shift": float(np.linalg.norm(off - (off @ u) * u) / L),
                         "used_for_distances": bool(m.sum() >= MIN_SAMPLE)})
            if m.sum() >= MIN_SAMPLE:
                means[(lab, int(x))] = (mean, int(m.sum()), int(bt[m][0]))
    pair = []
    keys = sorted(means)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (a, na, ba), (b, nb, bb) = means[keys[i]], means[keys[j]]
            if keys[i][0] != keys[j][0]:
                continue
            dist = np.linalg.norm(a - b)
            pair.append({"stage_idx": keys[i][0], "samples": [keys[i][1], keys[j][1]], "same_batch": ba == bb,
                         "dist_over_shift": float(dist / L),
                         "dist_over_sampling_noise": float(dist / (R * np.sqrt(1 / na + 1 / nb)))})
    summ = {}
    for lab in (s, s + 1):
        pp = [p for p in pair if p["stage_idx"] == lab]
        if pp:
            summ[STAGES[lab]] = {"n_pairs": len(pp),
                                 "max_dist_over_shift": max(p["dist_over_shift"] for p in pp),
                                 "median_dist_over_shift": float(np.median([p["dist_over_shift"] for p in pp])),
                                 "median_dist_over_noise": float(np.median([p["dist_over_sampling_noise"] for p in pp]))}
    return rows, pair, summ


def run_transition(origin, label, Z0, Z1, X0, X1, ct0, vel, mshift, out, percell, gene=True):
    """Cosines of OT displacement (Sinkhorn variants + exact) with every velocity variant."""
    res = {}
    methods = [(f"sinkhorn_eps{EPS_MAIN}_seed{sd}", EPS_MAIN, sd) for sd in SEEDS]
    methods += [(f"sinkhorn_eps{e}_seed0", e, 0) for e in EPS_ALT]
    for name, eps, sd in methods:
        t0 = time.time()
        i, j = sub(len(Z0), seed=sd), sub(len(Z1), seed=sd + 1)
        main = name == f"sinkhorn_eps{EPS_MAIN}_seed0"
        D, DX, marg = barycentric(Z0[i], Z1[j], eps, X0[i] if (main and gene) else None,
                                  X1[j] if (main and gene) else None)
        r = {"n_src": len(i), "n_tgt": len(j), "marginals": marg,
             "median_disp_norm_over_meanshift": float(np.median(np.linalg.norm(D, axis=1)) / np.linalg.norm(mshift)),
             "median_resid_over_mean_disp": float(np.median(np.linalg.norm(D - D.mean(0), axis=1))
                                                   / np.linalg.norm(D.mean(0))),
             "cos_mean_disp_meanshift": cosv(D.mean(0), mshift), "cos": {}}
        for (vk, mk), (Vz, Vg) in vel.items():
            Vz_i = Vz[i]
            for tgt in TARGETS:
                st, c = cosine_block(D, Vz_i, mshift, ct0[i], tgt, nulls=(main or sd == 0), seed=sd)
                r["cos"][f"{vk}|{mk}|{tgt}"] = st
                record(origin, label, name, vk, mk, tgt, st)
                if main:
                    percell[f"o{origin}_{KEY[label]}_{vk}_{mk}_{tgt}"] = c.astype(np.float32)
            if main:
                r.setdefault("velocity", {})[f"{vk}|{mk}"] = {
                    "median_norm_ratio_disp_over_vel": float(np.median(np.linalg.norm(D, axis=1)
                                                                       / np.maximum(np.linalg.norm(Vz_i, axis=1), 1e-12))),
                    "meanshift_over_mean_vel_norm": float(np.linalg.norm(mshift) / np.linalg.norm(Vz_i.mean(0))),
                    "cos_mean_vel_meanshift": cosv(Vz_i.mean(0), mshift),
                    "median_frac_vel_norm_in_pca": float(np.median(np.linalg.norm(Vz_i, axis=1)
                                                                  / np.maximum(np.linalg.norm(Vg[i], axis=1), 1e-12))),
                    "n_zero_vel_rows": int((np.linalg.norm(Vg[i], axis=1) == 0).sum())}
        if main:
            percell[f"o{origin}_{KEY[label]}_celltype"] = ct0[i].astype(str)
            if gene and DX is not None:
                g = {"median_frac_disp_norm_in_pca": float(np.median(np.linalg.norm(D, axis=1)
                                                                     / np.linalg.norm(DX, axis=1)))}
                for (vk, mk), (Vz, Vg) in vel.items():
                    A, B = DX.copy(), Vg[i].copy()
                    if mk == "without":
                        A[:, out["murk_mask"]] = 0.0
                    g[f"{vk}|{mk}"] = group_stats(cos_rows(A, B), ct0[i])
                    if vk == "v_dyn":
                        nz = out["dyn_nonzero"] & (~out["murk_mask"] if mk == "without" else True)
                        g[f"{vk}|{mk}|dyn_genes_only"] = group_stats(cos_rows(A[:, nz], B[:, nz]), ct0[i])
                res["gene_space"] = g
        r["seconds"] = round(time.time() - t0, 1)
        res[name] = r
        print(f"  origin {origin} {label} {name} {r['seconds']}s marg {marg}", flush=True)
    # exact assignment on equal-size subsamples
    n = min(len(Z0), len(Z1), NSUB)
    i, j = sub(len(Z0), n, seed=0), sub(len(Z1), n, seed=1)
    Dex, w2, r_ = exact_assignment(Z0[i], Z1[j])
    ex = {"n": n, "w2sq": w2, "translation_share_of_w2sq": float(np.linalg.norm(mshift) ** 2 / w2), "cos": {}}
    for (vk, mk), (Vz, _) in vel.items():
        for tgt in TARGETS:
            st, _c = cosine_block(Dex, Vz[i][r_], mshift, ct0[i][r_], tgt, nulls=True)
            ex["cos"][f"{vk}|{mk}|{tgt}"] = st
            record(origin, label, "exact_assignment", vk, mk, tgt, st)
    res["exact_assignment"] = ex
    return res


def main():
    if not DATA or not os.path.isdir(DATA):
        sys.exit("give the folder with origin<s>.npz and obs.csv as first argument")
    T0 = time.time()
    obs = pd.read_csv(os.path.join(DATA, "obs.csv"))
    ostage = np.array([STAGES.index(x) for x in obs["stage"].astype(str)])
    results = {"note": "DESCRIPTIVE diagnostic: uses the true stage s+1; not a forecast, not for model selection",
               "data_dir": DATA, "eps_main": EPS_MAIN, "n_perm": NPERM, "nsub": NSUB,
               "random_direction_cos_sd_30d": float(1 / np.sqrt(30)), "origins": {}}
    percell = {}
    sample_rows, pair_rows = [], []
    for s in range(1, 6):
        t0 = time.time()
        f = np.load(os.path.join(DATA, f"origin{s}.npz"), allow_pickle=True)
        st_p = f["stage_past"]
        ct_p = f["celltype_past"].astype(str)
        ct_f = f["celltype_fut"].astype(str)
        Zp = f["Z_past"].astype(np.float64)
        Zf = f["Z_fut"].astype(np.float64)
        Xp = f["X_past"].astype(np.float64)
        Xf = f["X_fut"].astype(np.float64)
        comps = f["pca_comps"].astype(np.float64)
        murk = f["murk"].astype(bool)
        dyn_nonzero = np.abs(f["v_dyn"]).sum(0) > 0
        # alignment with obs.csv (past = stage <= s in original order, fut = stage == s+1)
        ob_p, ob_f = obs[ostage <= s], obs[ostage == s + 1]
        assert (ob_p["celltype"].astype(str).to_numpy() == ct_p).all() and (ob_f["celltype"].astype(str).to_numpy() == ct_f).all()
        assert (np.array([STAGES.index(x) for x in ob_p["stage"]]) == st_p).all()
        samp_p, bat_p = ob_p["sample"].to_numpy(), ob_p["sequencing.batch"].to_numpy()
        samp_f, bat_f = ob_f["sample"].to_numpy(), ob_f["sequencing.batch"].to_numpy()
        now, prev = np.where(st_p == s)[0], np.where(st_p == s - 1)[0]
        out = {"degenerate": s == 1, "n_now": len(now), "n_prev": len(prev), "n_fut": len(Zf),
               "n_genes": int(len(murk)), "n_murk_in_hvg": int(murk.sum()),
               "n_genes_nonzero_v_dyn": int(dyn_nonzero.sum()), "n_murk_nonzero_v_dyn": int((murk & dyn_nonzero).sum()),
               "celltypes_now": {k: int(v) for k, v in zip(*np.unique(ct_p[now], return_counts=True))},
               "celltypes_fut": {k: int(v) for k, v in zip(*np.unique(ct_f, return_counts=True))},
               "murk_mask": murk, "dyn_nonzero": dyn_nonzero}
        mu = {k: Zp[st_p == k].mean(0) for k in range(s + 1)}
        mu[s + 1] = Zf.mean(0)
        muX = {k: Xp[st_p == k].mean(0) for k in range(s + 1)}
        muX[s + 1] = Xf.mean(0)
        shift_next, shift_prev = mu[s + 1] - mu[s], mu[s] - mu[s - 1]
        out["mean_shift"] = {
            "cos_prev_next_pca": cosv(shift_prev, shift_next),
            "cos_prev_next_gene": cosv(muX[s] - muX[s - 1], muX[s + 1] - muX[s]),
            "norm_ratio_next_over_prev_pca": float(np.linalg.norm(shift_next) / np.linalg.norm(shift_prev)),
            "norm_ratio_next_over_prev_gene": float(np.linalg.norm(muX[s + 1] - muX[s]) / np.linalg.norm(muX[s] - muX[s - 1])),
            "pearson_prev_next_gene": float(np.corrcoef(muX[s] - muX[s - 1], muX[s + 1] - muX[s])[0, 1]),
            "meanshift_forecast_error_over_persist_error_pca": float(np.linalg.norm(shift_next - shift_prev)
                                                                     / np.linalg.norm(shift_next))}
        if s >= 2:
            out["mean_shift"]["cos_prevprev_next_pca"] = cosv(mu[s - 1] - mu[s - 2], shift_next)
        Zn = Zp[now]
        R0X = float(np.sqrt(((Xp[now] - muX[s]) ** 2).sum(1).mean()))
        out["movement"] = movement_vs_spread(Zn, Zf, muX[s], muX[s + 1], R0X)
        out["movement"]["prev_transition"] = movement_vs_spread(Zp[prev], Zn, muX[s - 1], muX[s],
                                                                float(np.sqrt(((Xp[prev] - muX[s - 1]) ** 2).sum(1).mean())))
        out["shift_decomposition"] = shift_decomposition(Zn, ct_p[now], Zf, ct_f)
        out["shift_decomposition_prev"] = shift_decomposition(Zp[prev], ct_p[prev], Zn, ct_p[now])
        rows, pairs, summ = sample_positions(Zn, samp_p[now], bat_p[now], Zf, samp_f, bat_f, s)
        sample_rows += rows
        pair_rows += [{"origin": s, **p} for p in pairs]
        out["samples"] = summ
        # floors for W2^2: two disjoint halves of stage s+1 and of stage s (exact assignment, same sizes)
        m = min(len(Zf) // 2, NSUB)
        h = np.random.default_rng(7).permutation(len(Zf))
        out["w2sq_floor_fut_halves"] = {"n": m, "w2sq": exact_assignment(Zf[h[:m]], Zf[h[m:2 * m]])[1]}
        m0 = min(len(Zn) // 2, NSUB)
        h0 = np.random.default_rng(8).permutation(len(Zn))
        out["w2sq_floor_now_halves"] = {"n": m0, "w2sq": exact_assignment(Zn[h0[:m0]], Zn[h0[m0:2 * m0]])[1]}
        # velocities
        vel_now = velocity_pca(f, now, comps, murk)
        vel_prev = velocity_pca(f, prev, comps, murk)
        agree = {}
        for mk in MURKS:
            agree[f"ss_vs_dyn|{mk}"] = group_stats(cos_rows(vel_now[("v_ss", mk)][0], vel_now[("v_dyn", mk)][0]), ct_p[now])
        for vk in VELS:
            agree[f"{vk}|with_vs_without_murk"] = group_stats(cos_rows(vel_now[(vk, "with")][0], vel_now[(vk, "without")][0]), ct_p[now])
        out["velocity_agreement"] = agree
        out["future"] = run_transition(s, "s->s+1", Zn, Zf, Xp[now], Xf, ct_p[now], vel_now, shift_next, out, percell)
        out["past_strict"] = run_transition(s, "s-1->s", Zp[prev], Zn, Xp[prev], Xp[now], ct_p[prev], vel_prev,
                                            shift_prev, out, percell, gene=False)
        if s == 5:  # whole chain of consecutive shifts in origin 5's space (stages 0..6)
            sh = [mu[k + 1] - mu[k] for k in range(6)]
            shX = [muX[k + 1] - muX[k] for k in range(6)]
            results["chain_origin5_space"] = {
                "transitions": [f"{STAGES[k]}->{STAGES[k + 1]}" for k in range(6)],
                "norm_pca": [float(np.linalg.norm(x)) for x in sh],
                "norm_gene": [float(np.linalg.norm(x)) for x in shX],
                "cos_consecutive_pca": [cosv(sh[k], sh[k + 1]) for k in range(5)],
                "cos_consecutive_gene": [cosv(shX[k], shX[k + 1]) for k in range(5)],
                "cos_matrix_pca": [[cosv(a, b) for b in sh] for a in sh]}
        del out["murk_mask"], out["dyn_nonzero"]
        out["seconds"] = round(time.time() - t0, 1)
        results["origins"][s] = out
        print(f"origin {s} done {out['seconds']}s", flush=True)
        with open(os.path.join(HERE, "results.json"), "w") as fh:
            json.dump(results, fh, indent=1, default=float)
        pd.DataFrame(ROWS).to_csv(os.path.join(HERE, "cosines.csv"), index=False)
    pd.DataFrame(sample_rows).to_csv(os.path.join(HERE, "samples.csv"), index=False)
    pd.DataFrame(pair_rows).to_csv(os.path.join(HERE, "sample_pairs.csv"), index=False)
    np.savez_compressed(os.path.join(HERE, "percell.npz"), **percell)
    results["seconds_total"] = round(time.time() - T0, 1)
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("total", results["seconds_total"], "s")


if __name__ == "__main__":
    main()
