"""C8: partial correlation of the l1 change y_t with candidate per-bin-summary features, given own history.

Controls ('full own-history'): NASS l1 features lag1..3, rev_box2/4/8, rev_kal1, rev_kal3, rev_eb, mom_eb, logse,
frac, lag1_x_se, rev4_x_se (14, NASS's own arrays; row t uses bins <= t) + intercept + NASS level-class dummies
('lin'); variant 'int': the 14 controls interacted with the 5 level classes (NASS's stacker design).
Candidates (bins <= t; CPx_b = 1e4 x_b / lib_b, lib_b = sum_g m_gb):
  f_mom  = log1p(CPlam_t) - log1p(CPlam_{t-1}),  lam_b = -log(1 - min(f, 1 - 1/(2 n_b)))  (Poisson zero-based mean)
  f_rev  = log1p(CPlam_t) - L_t                  (level gap: zero-based estimate minus mean-based level)
  med_mom = log1p(CPmed_t) - log1p(CPmed_{t-1});  med_rev = log1p(CPmed_t) - L_t   (s01_median)
  disp   = (s_t^2 - m_t) / m_t^2 - CV2_lib_t for m_t > 0, 0 for m_t = 0;  disp_rank = within-transition rank of disp
           (heavy tail: a gene seen in one cell has (s^2 - m)/m^2 = n - 1 - n/k)
Per transition t >= 1 and CONTEXT mask seed: residual-residual Pearson on the masked genes; mean over seeds; then
mean over transitions and the count of transitions > 0 (and > 2/sqrt(n_masked), about 2 SE).
Also mean_g(s01_median > 0) per bin.
"""
import json
import os

import numpy as np

import common as C

CTRL = ["lag1", "lag2", "lag3", "rev_box2", "rev_box4", "rev_box8", "rev_kal1", "rev_kal3", "rev_eb", "mom_eb",
        "logse", "frac", "lag1_x_se", "rev4_x_se"]


def resid(D_, v):
    return v - D_ @ np.linalg.lstsq(D_, v, rcond=None)[0]


def pcorr(y, x, D_):
    ry, rx = resid(D_, y), resid(D_, x)
    den = np.sqrt((ry ** 2).sum() * (rx ** 2).sum())
    return float((ry * rx).sum() / den) if den > 0 else np.nan


res = {}
for name in C.NAMES:
    e = C.Ext(name)
    T, G = e.T, e.G
    Y = e.Y
    _, z = C.nass_full_l1(e)
    F = {k: z[f"F_{k}"] for k in CTRL}
    cls = z["cls"].astype(int)
    fcl = np.minimum(e.f, 1 - 1 / (2 * e.n[None, :]))
    lam = -np.log1p(-fcl)
    Llam = np.log1p(1e4 * lam / e.lib[None, :])
    Lmed = np.log1p(1e4 * np.maximum(e.med, 0) / e.lib[None, :])
    disp = np.where(e.m > 0, (e.s2 - e.m) / np.where(e.m > 0, e.m, 1) ** 2 - e.cv2lib[None, :], 0.0)
    cand = {}
    for t in range(1, T):
        cand.setdefault("f_mom", {})[t] = Llam[:, t] - Llam[:, t - 1]
        cand.setdefault("f_rev", {})[t] = Llam[:, t] - e.L[:, t]
        cand.setdefault("med_mom", {})[t] = Lmed[:, t] - Lmed[:, t - 1]
        cand.setdefault("med_rev", {})[t] = Lmed[:, t] - e.L[:, t]
        cand.setdefault("disp", {})[t] = disp[:, t]
        rk = np.argsort(np.argsort(disp[:, t], kind="stable"), kind="stable").astype(float)
        cand.setdefault("disp_rank", {})[t] = rk
    mk = C.masks3(G, T)
    out = {"median_gt0_share_per_bin": np.round((e.med > 0).mean(0), 4).tolist(),
           "frac_genes_m>0_per_bin": np.round((e.m > 0).mean(0), 4).tolist(), "features": {}}
    for design in ("lin", "int"):
        pcs = {k: np.full(T, np.nan) for k in cand}
        pcs_pos = {k: np.full(T, np.nan) for k in cand}             # disp only on genes with m_t > 0
        for t in range(1, T):
            Xc = np.column_stack([F[k][t] for k in CTRL])
            dm = np.column_stack([(cls[t] == c).astype(float) for c in range(5)])
            if design == "lin":
                Dfull = np.hstack([Xc, dm])
            else:
                Dfull = np.hstack([Xc * dm[:, [c]] for c in range(5)] + [dm])
            for k in cand:
                vals, vals_pos = [], []
                for s in C.SEEDS:
                    g = mk[s][t]
                    vals.append(pcorr(Y[t][g], cand[k][t][g], Dfull[g]))
                    if k.startswith("disp"):
                        gp = g[e.m[g, t] > 0]
                        vals_pos.append(pcorr(Y[t][gp], cand[k][t][gp], Dfull[gp]))
                pcs[k][t] = np.nanmean(vals)
                if vals_pos:
                    pcs_pos[k][t] = np.nanmean(vals_pos)
        se2 = 2 / np.sqrt(len(mk[0][0]))
        for k in cand:
            for tag, arr in (("", pcs[k]), ("|m>0", pcs_pos[k])):
                if np.all(np.isnan(arr)):
                    continue
                rec = {}
                for lo in (1, 2):
                    a = arr[lo:]
                    rec[f"i>={lo}"] = {"mean": round(float(np.nanmean(a)), 4), "n_pos": int((a > 0).sum()),
                                       "n_pos_2se": int((a > se2).sum()), "n_neg_2se": int((a < -se2).sum()),
                                       "n": int(np.isfinite(a).sum())}
                rec["per_transition"] = np.round(arr, 4).tolist()
                out["features"][f"{design}:{k}{tag}"] = rec
                print(name, f"{design}:{k}{tag}", rec["i>=1"], rec["i>=2"], flush=True)
    res[name] = out
    print(name, "median>0 share", out["median_gt0_share_per_bin"], flush=True)
json.dump(res, open(os.path.join(C.OUT, "c8_partial_corr.json"), "w"), indent=1)
