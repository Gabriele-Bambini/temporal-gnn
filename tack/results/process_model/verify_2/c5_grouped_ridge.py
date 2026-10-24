"""C5: per-group ridge on own history (lag1..3, box-2/4/8 reversion), three groupings, strict rolling origin, l1.

Features: NASS's own F['lag1..3'], F['rev_box2/4/8'] (nass.build_features; row t uses bins <= t).
Learner: NASS Stacker + run_learned UNCHANGED (features and target z-scored per transition, group-interacted
coefficients + group intercepts, ridge on past transitions j = 1..t-1, recency half-life h, scale-free ridge lam,
small-bin weighting; (h, lam, nw) chosen at every origin by rolling-origin J on past transitions, window 8; i = 2
uses the default; i = 0, 1 the shared fallback). The ONLY thing that changes between rows is the group array.
Groupings (5 groups each, group of row (g, j) is a function of bins <= j):
  (a) level: NASS level_class(m_j) (0, <0.03, <0.3, <3, >= 3 counts/cell).
  (b) R: group 0 = m_j == 0 (R = 0 there), groups 1..4 = quartiles of R_{g,j} over the nonzero genes of bin j,
      R = kappa^2 s^2 / (n - 1) / (1 + CP)^2 (log-scale delta method).
  (c) SNR: group 0 = m_j == 0, groups 1..4 = quartiles of v_g^(j) / R_{g,j} over nonzero genes; v_g^(j) =
      empirical Bayes signal variance of the change from past changes y_k, k + 1 <= j: raw = mean_k (y_k^2 - R_k -
      R_{k+1}), shrunk (nu = 4) to the median of its level class (class of the cell-weighted running mean, bins <= j),
      floored at 5 % of the prior.
  (o) one group (reference), (a5) quintiles of R with zeros tied into the lowest group are not used.
Scores: CONTEXT masks (3 seeds), PCC / tie-aware P@200 up / down, i >= 2 (and i >= 1); also NASS's own 20 masks.
"""
import json
import os

import numpy as np

import common as C

NU = 4.0
FEATS = ["lag1", "lag2", "lag3", "rev_box2", "rev_box4", "rev_box8"]


def quart_groups(val, nz):
    g = np.zeros(len(val), int)
    q = np.quantile(val[nz], [0.25, 0.5, 0.75])
    g[nz] = 1 + np.digitize(val[nz], q)
    return g


def eb_v(e, Y, j):
    """v_g from changes k = 0..j-1 (bins <= j)."""
    R = e.R
    nd = j
    raw = np.mean(Y[:j] ** 2 - R[:, :j].T - R[:, 1:j + 1].T, axis=0)
    w = np.cumsum(e.n[:j + 1])[-1]
    mbar = (e.m[:, :j + 1] * e.n[:j + 1][None, :]).sum(1) / w
    cls = C.N.level_class(mbar)
    prior = np.zeros(e.G)
    for c in range(C.N.N_CLS):
        s = cls == c
        if s.any():
            prior[s] = max(float(np.median(raw[s])), 1e-6)
    return np.maximum((nd * raw + NU * prior) / (nd + NU), 0.05 * prior)


res = {}
for name in C.NAMES:
    e = C.Ext(name)
    d = e.d
    Y = d.Y("l1")
    _, z = C.nass_full_l1(e)
    F = {k: z[f"F_{k}"] if f"F_{k}" in z.files else np.zeros((e.T, e.G)) for k in FEATS}
    fb = C.N.fallback(d, "l1", Y, F)
    T = e.T
    grp = {"a_level": z["cls"].astype(int)}
    gb, gc = np.zeros((T, e.G), int), np.zeros((T, e.G), int)
    sizes_b, sizes_c = [], []
    for j in range(T):
        nz = e.m[:, j] > 0
        gb[j] = quart_groups(e.R[:, j], nz)
        if j >= 1:
            v = eb_v(e, Y, j)
            gc[j] = quart_groups(v / np.where(nz, e.R[:, j], 1.0), nz)
        else:
            gc[j] = gb[j]                       # row 0 is never used for training (j >= 1) nor scored by a learner
        sizes_b.append(np.bincount(gb[j], minlength=5).tolist())
    grp["b_R"] = gb
    grp["c_SNR"] = gc
    grp["o_one_group"] = np.zeros((T, e.G), int)
    # overlap between groupings (share of genes with identical group), averaged over transitions >= 1
    ov = {f"{a}~{b}": round(float(np.mean([(grp[a][j] == grp[b][j]).mean() for j in range(1, T)])), 3)
          for a, b in (("a_level", "b_R"), ("a_level", "c_SNR"), ("b_R", "c_SNR"))}
    mk = C.masks3(e.G, T)
    mk_n = C.D.masks(e.G, T)
    r = {"group_overlap": ov, "group_sizes_b_first_last": [sizes_b[1], sizes_b[-1]], "rows": {}}
    per_t = {}
    for gname, g in grp.items():
        st = C.N.Stacker(F, g, Y, FEATS)
        P, chosen = C.N.run_learned(st, [FEATS], Y, d, fb)
        tab = C.pcc_table(Y, P, mk)
        Sn = C.D.score(Y, P, mk_n)
        r["rows"][gname] = {"ctx_i>=2": np.round(C.mean_from(tab, 2), 4).tolist(),
                            "ctx_i>=1": np.round(C.mean_from(tab, 1), 4).tolist(),
                            "nassmasks_i>=2": C.D.summarize(Sn, list(range(2, T)))}
        per_t[gname] = np.nanmean(tab[:, :, 0], 0)
        print(name, gname, r["rows"][gname]["ctx_i>=2"], "| NASS masks pcc", r["rows"][gname]["nassmasks_i>=2"]["pcc"],
              flush=True)
    Pn, _ = C.nass_full_l1(e)
    tabn = C.pcc_table(Y, list(Pn), mk)
    per_t["NASS_full"] = np.nanmean(tabn[:, :, 0], 0)
    r["rows"]["NASS_full(reference)"] = {"ctx_i>=2": np.round(C.mean_from(tabn, 2), 4).tolist()}
    # paired per-transition differences vs (a), i >= 2
    pr = {}
    for gname in ("b_R", "c_SNR", "o_one_group", "NASS_full"):
        dlt = per_t[gname][2:] - per_t["a_level"][2:]
        pr[f"{gname} - a_level"] = {"mean": round(float(dlt.mean()), 4), "better": int((dlt > 0).sum()),
                                    "worse": int((dlt < 0).sum())}
    r["paired_pcc_i>=2"] = pr
    r["per_transition_pcc_ctx"] = {k: np.round(v, 4).tolist() for k, v in per_t.items()}
    print(name, ov, pr, flush=True)
    res[name] = r
json.dump(res, open(os.path.join(C.OUT, "c5_grouped_ridge.json"), "w"), indent=1)
