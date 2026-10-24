"""C14: NASS stacker trained on one-sided bridged labels vs raw labels (l1, strict rolling origin).

At origin t (predicting transition t, bins <= t): a known-R local-level Kalman smoother (forward filter + RTS) is run
over bins 0..t; bridged label of past transition j: y~_j = E[L_{j+1} | bins <= t] - L_j (raw L_j) for j + 2 <= t;
y_{t-1} stays raw. Features, Gram matrices, z-scoring, class interaction, CONFIGS grid and the rolling-origin J
selection (scored against the RAW truth, window 8, default at i = 2) are NASS's, unchanged; only the label vectors
X_j' zs(y_j) of the training transitions change, and they are rebuilt at every origin.
Smoothers:
  'L_eb'  : on L = log1p(CP10k); R = kappa^2 max(s^2, 1/n) / (n - 1) / (1 + CP)^2 (delta method, s^2 floored at one
            count in one cell so zero bins are not treated as noiseless); Q_g = empirical-Bayes variance of the true
            change from y_k, k + 1 <= t (raw mean of y_k^2 - R_k - R_{k+1}, shrunk nu = 4 to its level-class median,
            floor 5 % of the prior).
  'CP_nass': on CP10k with NASS's own Base('l1') R and q_eb (rev_eb's filter) plus an RTS pass; mapped with tfun.
Baseline: the same loop with raw labels (must equal nass.run_learned exactly; checked).
Scores: CONTEXT masks (3 seeds) PCC / tie-aware P@200, i >= 2; per-transition deltas; also NASS's 20 masks.
"""
import json
import os

import numpy as np
from scipy import stats

import common as C

N, D = C.N, C.D
NU = 4.0


def rts(Z, R, Qfun, upto):
    """Local level filter + RTS smoother over bins 0..upto. Z, R: (G, B). Qfun(b) -> (G,) for b >= 1."""
    G = Z.shape[0]
    xs = np.zeros((G, upto + 1))
    Ps = np.zeros((G, upto + 1))
    Pms = np.zeros((G, upto + 1))
    x, P = Z[:, 0].copy(), R[:, 0].copy()
    xs[:, 0], Ps[:, 0] = x, P
    for b in range(1, upto + 1):
        Pm = P + Qfun(b)
        S = Pm + R[:, b]
        K = np.where(S > 0, Pm / np.where(S > 0, S, 1), 1.0)
        x = x + K * (Z[:, b] - x)
        P = (1 - K) * Pm
        xs[:, b], Ps[:, b], Pms[:, b] = x, P, Pm
    sm = xs.copy()
    for b in range(upto - 1, -1, -1):
        Cg = np.where(Pms[:, b + 1] > 0, Ps[:, b] / np.where(Pms[:, b + 1] > 0, Pms[:, b + 1], 1), 0.0)
        sm[:, b] = xs[:, b] + Cg * (sm[:, b + 1] - xs[:, b])
    return sm


def eb_q(e, Rf, t):
    """Q_g from changes k = 0..t-1 (bins <= t), log scale."""
    Yd = e.Yd
    raw = np.mean(Yd[:t] ** 2 - Rf[:, :t].T - Rf[:, 1:t + 1].T, axis=0)
    w = np.cumsum(e.n[:t + 1])[-1]
    mbar = (e.m[:, :t + 1] * e.n[:t + 1][None, :]).sum(1) / w
    cls = N.level_class(mbar)
    prior = np.zeros(e.G)
    for c in range(N.N_CLS):
        s = cls == c
        if s.any():
            prior[s] = max(float(np.median(raw[s])), 1e-6)
    return np.maximum((t * raw + NU * prior) / (t + NU), 0.05 * prior)


def bridged_labels(e, kind, t, Y, base=None, qeb=None):
    """Labels for j = 0..t-1 at origin t (rows j + 2 <= t bridged, row t-1 raw)."""
    lab = [Y[j].copy() for j in range(t)]
    if t < 2:
        return lab
    if kind == "L_eb":
        Rf = e.r_floor()
        q = eb_q(e, Rf, t)
        sm = rts(e.L, Rf, lambda b: q, t)
        for j in range(t - 1):                       # j + 2 <= t  <=>  j <= t - 2
            lab[j] = sm[:, j + 1] - e.L[:, j]
    else:
        sm = rts(base.Z, base.R, qeb, t)
        for j in range(t - 1):
            lab[j] = D.tfun("l1", base.Z[:, j], np.maximum(sm[:, j + 1], 0))
    return lab


def run_custom(st, sub, Y, d, fb, label_fn):
    """nass.run_learned for one subset, with origin-specific training labels."""
    T = d.T
    idx = st.cols(sub)
    Gs, bs_raw = st.grams(sub)
    cand = list(range(len(N.CONFIGS)))
    P = {c: [None] * T for c in cand}
    J = {c: np.full(T, np.nan) for c in cand}
    for i in range(2, T):
        if label_fn is None:
            bs = bs_raw
        else:
            lab = label_fn(i)
            bs = bs_raw.copy()
            for j in range(1, i):
                bs[j] = st.X[j][:, idx].astype(np.float64).T @ N.zs(lab[j])
        Pi = st.predict_all(i, sub, d.n, Gs, bs)
        for c in cand:
            P[c][i] = Pi[:, c]
            J[c][i] = D.criterion(Y[i], Pi[:, c], d.n_test)
    pred, chosen = [fb[0], fb[1]], [None, None]
    for i in range(2, T):
        if i == 2:
            k = N.DEFAULT
        else:
            lo = max(2, i - N.WINDOW)
            k = cand[int(np.argmax([np.mean(J[c][lo:i]) for c in cand]))]
        pred.append(P[k][i])
        chosen.append(k)
    return pred, chosen


res = {}
for name in C.NAMES:
    e = C.Ext(name)
    d = e.d
    Y = d.Y("l1")
    F, cls = N.build_features(d, "l1", Y, None, None)
    fb = N.fallback(d, "l1", Y, F)
    G_ = N.GROUPS
    full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"]
    st = N.Stacker(F, cls, Y, full)
    base = N.Base(d, "l1")
    qeb = base.q_eb()
    # label diagnostics: how much the bridged labels differ from raw, and how they correlate with the raw ones
    diag = {}
    preds = {}
    p_ref, _ = N.run_learned(st, [full], Y, d, fb)
    preds["raw"], ch_raw = run_custom(st, full, Y, d, fb, None)
    same = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(p_ref[2:], preds["raw"][2:]))
    cache = {}

    def lf(kind):
        def f(i):
            key = (kind, i)
            if key not in cache:
                cache[key] = bridged_labels(e, kind, i, Y, base, qeb)
            return cache[key]
        return f

    for kind in ("L_eb", "CP_nass"):
        preds[kind], _ = run_custom(st, full, Y, d, fb, lf(kind))
        t = e.T - 1
        lab = lf(kind)(t)
        r = [float(np.corrcoef(lab[j], Y[j])[0, 1]) for j in range(1, t - 1)]
        vr = [float(lab[j].var() / Y[j].var()) for j in range(1, t - 1)]
        # does the bridged label correlate with the NEXT raw change better than the raw label? (persistence)
        diag[kind] = {"corr(bridged, raw) mean at last origin": round(float(np.mean(r)), 3),
                      "var(bridged)/var(raw) mean": round(float(np.mean(vr)), 3)}
    mk = C.masks3(e.G, e.T)
    mk_n = D.masks(e.G, e.T)
    tabs = {k: C.pcc_table(Y, v, mk) for k, v in preds.items()}
    out = {"baseline_equals_nass.run_learned_maxabs": same, "label_diag": diag, "rows": {}, "paired_pcc_i>=2": {}}
    for k, tb in tabs.items():
        Sn = D.score(Y, preds[k], mk_n)
        out["rows"][k] = {"ctx_i>=2": np.round(C.mean_from(tb, 2), 4).tolist(),
                          "nassmasks_i>=2": D.summarize(Sn, list(range(2, e.T)))}
        print(name, k, out["rows"][k]["ctx_i>=2"], "| NASS masks", out["rows"][k]["nassmasks_i>=2"]["pcc"], flush=True)
    base_t = np.nanmean(tabs["raw"][:, :, 0], 0)
    for k in ("L_eb", "CP_nass"):
        dl = np.nanmean(tabs[k][:, :, 0], 0)[2:] - base_t[2:]
        nz = dl[np.abs(dl) > 1e-12]
        out["paired_pcc_i>=2"][f"{k} - raw"] = {
            "mean_delta": round(float(dl.mean()), 4), "better": int((dl > 0).sum()), "worse": int((dl < 0).sum()),
            "wilcoxon_p": round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None,
            "per_transition": np.round(dl, 4).tolist()}
        print(name, k, "- raw", {a: b for a, b in out["paired_pcc_i>=2"][f"{k} - raw"].items() if a != "per_transition"},
              flush=True)
    print(name, "baseline == nass.run_learned (max abs diff)", same, diag, flush=True)
    res[name] = out
json.dump(res, open(os.path.join(C.OUT, "c14_bridged_labels.json"), "w"), indent=1)
