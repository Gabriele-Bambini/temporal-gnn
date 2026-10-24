"""Strict rolling-origin evaluation of the latent-SB / state-space forecaster (lsb_model.py) on both '-human' builds.

Loop over origins t = 0..T-1 in increasing order. At origin t every configuration predicts transition t from
snapshots <= t only; the configuration used for the test prediction at t is selected by rolling origin from the
criterion J (lsb_common.J_all) of the SAME configuration at past origins j in [max(1, t-8), t-1], each of which
also used only snapshots <= j. J of origin t is recorded only after its prediction is stored.

Methods (same masks, paired):
  trend            plain 'repeat the last observed change' (zero at t = 0)              [required comparator]
  regime           re-implementation of variance/regime_fit.py (per-gene lags + level deviations x 5 level
                   classes, recency-weighted LS on past transitions, half-life by rolling origin, as there)
  lsb_*            the latent model, RO-selected over a config subset:
                     pergene       k = 0 (per-gene Kalman, known noise, EB q / rho)
                     pergene_nomom k = 0, rho = 0 (local level denoiser)
                     lsb_nodrift   k in {0,2,4,8} with latent dynamics 'level' only (no SB drift)
                     lsb_full      all configs (adds 'ar' and 'var' latent drift)
                     lsb_forced    k > 0 only
  pergene_fixed / lsb_fixed_k4ar   the two a-priori configs F_PG (k=0, EB nu=4, momentum, all bins) and
                   F_LAT (same + k=4 'cur' basis, 'ar' latent drift), no selection
  lsb_stack        recency-weighted LS by 5 level classes on the fixed-config outputs (per-gene denoise term,
                   per-gene forecast change, cross-gene denoise increment F_LAT - F_PG, latent factor change,
                   last observed change [+ library term for linear]); half-life by rolling origin (regime rule)
  regime+lsb       regime features + the four lsb features, same fit (tests whether the latent model adds
                   information to per-gene history)
  orig only:       levelobs (box-pooled law + finite-cell head, alpha = 0: the DistBridge-without-OT equivalent),
                   lsb_obs (law mean moved toward the lsb_full latent level, alpha / pool / source by RO)
  HINDSIGHT        best fixed config in hindsight (NOT strict, labelled)
Output: results_<dataset>.json and a printed table.
"""
import json
import sys
import time

import numpy as np

import lsb_common as C
from lsb_model import LatentSB

KS = (0, 2, 4, 8)
BASES = ("cur", "lag")
DYNS = ("level", "ar", "var")
WS = (6, None)
NUS = (None, 4.0)
MOMS = ("mom", "none")
BETAS = (0.0, 0.5, 1.0)
ORIG_CFGS = [(kp, 0.0, "-") for kp in (2, 4, 8)] + [(kp, a, s) for kp in (2, 4, 8) for a in (0.5, 1.0)
                                                    for s in ("next", "now")]
HS = (0.5, 1.0, 2.0, 4.0, np.inf)
CONTEXT_DISTBRIDGE = {  # CONTEXT, strict, convention 'all', their own masks (not paired with ours)
    "gastrulation-human": {"orig": [0.585, 0.406, 0.541]},
    "pancreas-human": {"orig": [0.488, 0.301, 0.442], "l1": [0.10, 0.17, 0.21], "linear": [0.30, 0.25, 0.30],
                       "l1_noOT": [-0.05, 0.10, 0.15], "linear_noOT": [-0.14, 0.06, 0.06],
                       "l1_trend_context": [0.15, 0.21, 0.21], "linear_trend_context": [0.40, 0.27, 0.35]}}


def latent_configs():
    cf = []
    for k in KS:
        for basis in (BASES if k else ("-",)):
            for dyn in (DYNS if k else ("-",)):
                for W in WS:
                    for nu in NUS:
                        for mom in MOMS:
                            cf.append((k, basis, dyn, W, nu, mom))
    return cf


CFGS = latent_configs()
DEFAULT = CFGS.index((0, "-", "-", None, None, "none"))
# fixed a-priori configs used as stacking features and reported unselected (no hindsight choice)
F_PG = CFGS.index((0, "-", "-", None, 4.0, "mom"))
F_LAT = CFGS.index((4, "cur", "ar", None, 4.0, "mom"))
# latent-drift ablation at fixed settings: Brownian reference (no drift) vs integrated OU vs learned VAR drift
F_ABL = {"k4_level_fixed": CFGS.index((4, "cur", "level", None, 4.0, "mom")),
         "lsb_fixed_k4ar": F_LAT,
         "k4_var_fixed": CFGS.index((4, "cur", "var", None, 4.0, "mom")),
         "k4_lagbasis_ar_fixed": CFGS.index((4, "lag", "ar", None, 4.0, "mom"))}
SUBSETS = {
    "pergene": lambda c: c[0] == 0,
    "pergene_nomom": lambda c: c[0] == 0 and c[5] == "none",
    "lsb_nodrift": lambda c: c[0] == 0 or c[2] == "level",
    "lsb_full": lambda c: True,
    "lsb_forced": lambda c: c[0] > 0,
}


def level_class(m):
    return np.digitize(np.where(m > 0, np.log10(np.where(m > 0, m, 1)), -9),
                       [-8, np.log10(0.03), np.log10(0.3), np.log10(3)])


def interact(X, cls, n_cls=5):
    return np.column_stack([X * (cls == c)[:, None] for c in range(n_cls)] +
                           [(cls == c).astype(float) for c in range(1, n_cls)])


def wls(X, y, w):
    X1 = np.column_stack([np.ones(len(X)), X])
    A = (X1 * w[:, None]).T @ X1
    b = (X1 * w[:, None]).T @ y
    return np.linalg.lstsq(A + 1e-8 * np.eye(len(A)) * np.trace(A) / len(A), b, rcond=None)[0]


def corr(y, p):
    if np.std(p) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def stacked(feats, Y, fallback):
    """Regime-style fit: predictions for t >= 2 with every half-life, h chosen per t >= 3 by mean all-gene
    Pearson over past transitions 2..t-1 (variance/regime_fit.py rule). fallback: T x N used for t < 3."""
    T = len(Y)
    P = {h: {} for h in HS}
    for i in range(2, T):
        Xp = np.vstack([feats[j] for j in range(1, i)])
        yp = np.concatenate([Y[j] for j in range(1, i)])
        G = len(Y[0])
        for h in HS:
            w = np.concatenate([np.full(G, 1.0 if not np.isfinite(h) else 0.5 ** ((i - 1 - j) / h))
                                for j in range(1, i)])
            c = wls(Xp, yp, w)
            P[h][i] = c[0] + feats[i] @ c[1:]
    Cm = {h: {i: corr(Y[i], P[h][i]) for i in P[h]} for h in HS}
    out = np.array(fallback, copy=True)
    sel = {}
    for i in range(3, T):
        h = max(HS, key=lambda hh: np.mean([Cm[hh][j] for j in range(2, i)]))
        sel[i] = str(h)
        out[i] = P[h][i]
    return out, sel


def regime_feats(d, Y, tg):
    T = d.T
    feats = []
    import dileo_protocol
    f = dileo_protocol.TARGET_NP[tg]
    for i in range(T):
        cols = [Y[i - k] if i - k >= 0 else np.zeros(d.N) for k in (1, 2, 3)]
        for K in (4, 8):
            ks = list(range(max(0, i - K + 1), i + 1))
            w = d.nc[ks]
            lev = (d.M[:, ks] * w).sum(1) / w.sum()
            cols.append(np.asarray(f(d.M[:, i].astype(np.float32), lev.astype(np.float32)), np.float64))
        feats.append(np.column_stack(cols))
    return feats


def main(name):
    t_start = time.time()
    d = C.Data(name)
    mod = LatentSB(d)
    mk = C.masks(d)
    ks = C.k_sel(d)
    T, G = d.T, d.N
    Y = {tg: C.all_targets(d, tg) for tg in C.TARGETS}
    ds_id = 1 if name.startswith("gastr") else 2
    nC = len(CFGS)
    J = {"l1": np.full((nC, T), np.nan), "linear": np.full((nC * len(BETAS), T), np.nan),
         "orig": np.full((len(ORIG_CFGS), T), np.nan)}
    idx = {s: [c for c in range(nC) if f(CFGS[c])] for s, f in SUBSETS.items()}
    lin_idx = {s: [c * len(BETAS) + bi for c in v for bi in range(len(BETAS))] for s, v in idx.items()}
    lin_idx["lsb_full_beta0"] = [c * len(BETAS) for c in range(nC)]
    orig_idx = {"levelobs": [i for i, c in enumerate(ORIG_CFGS) if c[1] == 0.0],
                "lsb_obs": list(range(len(ORIG_CFGS)))}
    P = {"l1": {s: np.zeros((T, G)) for s in list(idx) + ["pergene_fixed"] + list(F_ABL)},
         "linear": {s: np.zeros((T, G)) for s in list(lin_idx) + ["pergene_fixed"] + list(F_ABL)},
         "orig": {s: np.zeros((T, G)) for s in orig_idx}}
    SEL = {"l1": {s: [] for s in idx}, "linear": {s: [] for s in lin_idx}, "orig": {s: [] for s in orig_idx}}
    stack_feats = {"l1": [], "linear": []}
    diag = []
    for t in range(T):
        t0 = time.time()
        xs = [mod.levels(t, c) for c in CFGS]
        p_l1 = [mod.head_l1(t, x[1]) for x in xs]
        # ---- l1 selection and storage
        for s, ids in idx.items():
            sub = J["l1"][ids]
            c = ids[C.select(sub, ids.index(DEFAULT) if DEFAULT in ids else 0)[t]] if True else None
            P["l1"][s][t] = p_l1[c]
            SEL["l1"][s].append(c)
        # ---- linear
        for s, ids in lin_idx.items():
            sub = J["linear"][ids]
            dflt = ids.index(DEFAULT * len(BETAS)) if DEFAULT * len(BETAS) in ids else 0
            cb = ids[C.select(sub, dflt)[t]]
            c, bi = divmod(cb, len(BETAS))
            P["linear"][s][t] = mod.head_linear(t, xs[c][1], BETAS[bi])
            SEL["linear"][s].append(cb)
        # ---- orig: law head on the l1-selected lsb_full latent level
        cl = SEL["l1"]["lsb_full"][-1]
        x_now, x_next, parts = xs[cl]
        p_orig = []
        for oi, (kp, a, src) in enumerate(ORIG_CFGS):
            lev = x_next if src == "next" else x_now
            p_orig.append(mod.head_orig(t, lev, kp, a, [ds_id, t, oi]))
        for s, ids in orig_idx.items():
            sub = J["orig"][ids]
            c = ids[C.select(sub, ids.index(1) if 1 in ids else 0)[t]]
            P["orig"][s][t] = p_orig[c]
            SEL["orig"][s].append(c)
        # ---- fixed-config outputs and stacking features
        (pn, px, _), (ln_, lx, lparts) = xs[F_PG], xs[F_LAT]
        P["l1"]["pergene_fixed"][t] = p_l1[F_PG]
        P["linear"]["pergene_fixed"][t] = mod.head_linear(t, px, 0.0)
        for s_, ci in F_ABL.items():
            P["l1"][s_][t] = p_l1[ci]
            P["linear"][s_][t] = mod.head_linear(t, xs[ci][1], 0.0)
        obs_now = d.X[:, t] - mod.R[:, t] / 2
        stack_feats["l1"].append(np.column_stack([pn - obs_now, px - pn, ln_ - pn, lparts["fac_chg"]]))
        to_m = lambda x: np.maximum(d.lib[t] / 1e4 * np.expm1(x), 0)
        stack_feats["linear"].append(np.column_stack([to_m(pn) - d.M[:, t], to_m(px) - to_m(pn),
                                                      to_m(ln_) - to_m(pn), to_m(ln_ + lparts["fac_chg"]) - to_m(ln_),
                                                      d.M[:, t] * ((d.lib[t] / d.lib[t - 1] - 1) if t else 0.0)]))
        # ---- record J of origin t (after storing predictions)
        for c in range(nC):
            J["l1"][c, t] = C.J_all(Y["l1"][t], p_l1[c], ks)
            for bi, b in enumerate(BETAS):
                J["linear"][c * len(BETAS) + bi, t] = C.J_all(Y["linear"][t], mod.head_linear(t, xs[c][1], b), ks)
        for oi in range(len(ORIG_CFGS)):
            J["orig"][oi, t] = C.J_all(Y["orig"][t], p_orig[oi], ks)
        # diagnostics of the selected latent config
        k, basis, dyn = CFGS[cl][:3]
        zf = mod.zforecast(t, k, basis, dyn) if k else None
        rr = mod.resid(t, *CFGS[cl][:1], *CFGS[cl][1:2], *CFGS[cl][3:])
        diag.append({"t": t, "bins": f"{d.bins[t]}->{d.bins[t + 1]}", "l1_cfg": str(CFGS[cl]),
                     "rho_mean_expr": round(float(rr[2][d.M[:, t] > 0].mean()), 3),
                     "latent_score_noise_over_signal": None if zf is None else round(zf[2]["score_noise_over_signal"], 5),
                     "latent_phi_max": None if zf is None else round(zf[2]["phi_norm"], 3),
                     "sec": round(time.time() - t0, 1)})
        mod._fac.clear(); mod._zf.clear(); mod._res.clear()
        print(f"[{name}] t={t} cfg={CFGS[cl]} orig={ORIG_CFGS[SEL['orig']['lsb_obs'][-1]]} "
              f"{time.time() - t0:.1f}s", flush=True)
    # ---- comparators
    trend = {tg: np.vstack([np.zeros((1, G)), Y[tg][:-1]]) for tg in C.TARGETS}
    for tg in ("l1", "linear"):
        rf = [interact(f, level_class(d.M[:, i])) for i, f in enumerate(regime_feats(d, Y[tg], tg))]
        P[tg]["regime"], SEL[tg]["regime"] = stacked(rf, Y[tg], trend[tg])
        lf = [interact(np.column_stack([stack_feats[tg][i], trend[tg][i]]), level_class(d.M[:, i])) for i in range(T)]
        P[tg]["lsb_stack"], SEL[tg]["lsb_stack"] = stacked(lf, Y[tg], P[tg]["lsb_full"])
        rl = [interact(np.column_stack([f, stack_feats[tg][i][:, :4]]), level_class(d.M[:, i]))
              for i, f in enumerate(regime_feats(d, Y[tg], tg))]
        P[tg]["regime+lsb"], SEL[tg]["regime+lsb"] = stacked(rl, Y[tg], P[tg]["lsb_full"])
    # ---- hindsight (NOT strict): best fixed config by mean J over transitions >= 1
    hb = int(np.nanargmax(np.nanmean(J["l1"][:, 1:], 1)))
    P["l1"]["HINDSIGHT_fixed_cfg"] = np.stack([mod.head_l1(t, mod.levels(t, CFGS[hb])[1]) for t in range(T)])
    hbl = int(np.nanargmax(np.nanmean(J["linear"][:, 1:], 1)))
    cH, bH = divmod(hbl, len(BETAS))
    P["linear"]["HINDSIGHT_fixed_cfg"] = np.stack([mod.head_linear(t, mod.levels(t, CFGS[cH])[1], BETAS[bH])
                                                   for t in range(T)])
    out = {"dataset": name, "protocol": __doc__, "model": sys.modules["lsb_model"].__doc__,
           "n_cells": d.nc.astype(int).tolist(), "n_configs_latent": nC,
           "hindsight_cfg": {"l1": str(CFGS[hb]), "linear": str((CFGS[cH], BETAS[bH]))},
           "context_distbridge": CONTEXT_DISTBRIDGE[name], "methods": {}, "paired": {}, "diag": diag}
    scores = {}
    for tg in C.TARGETS:
        P[tg]["trend"] = trend[tg]
        out["methods"][tg] = {}
        scores[tg] = {}
        for s, Pm in P[tg].items():
            sc = C.test_scores(Y[tg], Pm, mk)
            scores[tg][s] = sc
            rep = C.summarize(sc, d)
            if s in SEL[tg]:
                sel = SEL[tg][s]
                if tg == "l1":
                    rep["selected"] = [str(CFGS[c]) for c in sel] if isinstance(sel, list) else sel
                elif tg == "linear":
                    rep["selected"] = ([str((CFGS[c // len(BETAS)], BETAS[c % len(BETAS)])) for c in sel]
                                       if isinstance(sel, list) else sel)
                else:
                    rep["selected"] = [str(ORIG_CFGS[c]) for c in sel]
            out["methods"][tg][s] = rep
        ref = ["trend", "regime", "pergene", "pergene_fixed", "k4_level_fixed"] if tg != "orig" else ["trend", "levelobs"]
        out["paired"][tg] = {}
        for s in P[tg]:
            for r in ref:
                if s != r and r in scores[tg]:
                    for conv in ("all", "i>=1", "i>=3"):
                        out["paired"][tg][f"{s}_vs_{r}_{conv}"] = C.paired(scores[tg][s], scores[tg][r], d, conv)
    with open(f"{C.OUT}/results_{name}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n==== {name}  ({time.time() - t_start:.0f}s)   PCC / P@200 up / down  [mask sd of PCC]")
    for tg in C.TARGETS:
        print(f"-- {tg}")
        for s, rep in out["methods"][tg].items():
            line = "  ".join(f"{cn}: {rep[cn]['pcc']:.3f}/{rep[cn]['up']:.3f}/{rep[cn]['down']:.3f}"
                             for cn in ("all", "i>=1", "i>=3"))
            print(f"  {s:22s} {line}  [sd {rep['i>=1']['pcc_mask_sd']:.3f}]")
        for key, v in out["paired"][tg].items():
            if (key.endswith("_i>=1") and ("_vs_trend" in key or "_vs_levelobs" in key)) or                     (key.endswith("_i>=3") and "_vs_regime" in key):
                print(f"    {key:40s} dPCC {v['pcc']['mean_delta']:+.3f} ({v['pcc']['better']}/{v['pcc']['worse']}, "
                      f"p={v['pcc']['wilcoxon_p']})  dUp {v['up']['mean_delta']:+.3f}  dDn {v['down']['mean_delta']:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
