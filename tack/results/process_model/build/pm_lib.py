"""pm_lib: shared pieces of the process-model v2 build (numpy / scipy / pandas only; never imports torch).

Every runner of this folder imports this module FIRST: it points GRAPH_CACHE at build/edge_cache and imports the
build copies of dsdata.py / nass.py (unchanged copies of design_best-signal), so dsdata.OUT = build (its caches land
here) and nothing outside build is written.

Contents
  data_from_X     dsdata.Data built from a given (B, G, 5) array (same attribute lines as dsdata.Data.__init__),
                  used by the truncation test; checked against dsdata.Data on clean data (exact equality).
  ctx_masks       CONTEXT masks: np.sort(default_rng(1000 s + i).permutation(G)[:round(G/4)]), s = 0, 1, 2.
  score_ctx/summ  dileo_rescore.metrics (pearson, p_ta_up, p_ta_down, spearman) per (seed, transition); mean over
                  transitions (i >= lo), then over the 3 seeds.
  paired_ctx      per-transition mask-averaged deltas a - b (pcc, up, down), Wilcoxon over transitions.
  NEW features    (section B of the revised model; strict: row i uses snapshots <= i only)
                  grnbar_lag1 = Wbar_i y_{i-1}, Wbar_i = rownorm(sum_{j in {i,i-1,i-2}, j>=0} W_{bins[j]})
                  has_regbar  = row of Wbar_i non-empty
                  grnbarnull_lag1 = same with the level-matched null graphs Wn (NASS load_graph)
                  grnbar_mom  = Wbar_i mom_eb_i (ablation only: denoised-slope input)
                  fgap = X_i - L_i, fmom = X_i - X_{i-1} (0 at i = 0), X_b = log1p(1e4 lam_b / lib_b),
                  lam_b = -log(1 - clip(f_b, 0, 1 - 1/(2 n_b))), L_b = log1p(1e4 m_b / lib_b) (= dsdata L)
                  every new row 0 is set to 0 (transition 0 uses the shared fallback).
  l1_models       NASS stacker variants V0..Vauto (one Stacker over all names, NS.run_learned for each subset).
  linear_models   NASS full and full+grn on linear.
  level_obs       DistBridge no-SB (finite-cell observation head + rolling pool), dist_bridge.REGISTRY
                  distbridge_strict_nosb_l1 / _linear, one transition at a time; DB._CACHE cleared before and after.
  garbage_X / garbage_graph   finite garbage for bins > k (gene-permuted, rescaled copies of random bins; graphs
                  with permuted gene labels), for the truncation test.
"""
import os
import sys

import numpy as np
from scipy import stats

B = os.path.dirname(os.path.abspath(__file__))
os.environ["GRAPH_CACHE"] = os.path.join(B, "edge_cache")
if B not in sys.path:
    sys.path.insert(0, B)

import dsdata as D  # noqa: E402
import nass as NS  # noqa: E402
from dsdata import DB  # noqa: E402
import dileo_rescore as RS  # noqa: E402

assert os.path.dirname(os.path.abspath(D.__file__)) == B, D.__file__
assert os.path.dirname(os.path.abspath(NS.__file__)) == B, NS.__file__
assert D.OUT == B, D.OUT
assert D.EDGE_CACHE == os.path.join(B, "edge_cache"), D.EDGE_CACHE

NAMES = ("gastrulation-human", "pancreas-human")
SHORT = {"gastrulation-human": "gastr", "pancreas-human": "pancr"}
SEEDS = (0, 1, 2)
KEYS = ("pearson", "p_ta_up", "p_ta_down", "spearman")
NEW = ["grnbar_lag1", "has_regbar", "grnbarnull_lag1", "grnbar_mom", "fgap", "fmom"]


# ------------------------------------------------------------------ data
def data_from_X(name, X):
    """dsdata.Data with the given X (B, G, 5); the attribute lines are those of dsdata.Data.__init__."""
    d = object.__new__(D.Data)
    Xs = np.asarray(X, np.float64)
    d.name = name
    d.bins = D.SNAPS[name]
    d.X = Xs
    d.B, d.G = Xs.shape[0], Xs.shape[1]
    d.T = d.B - 1
    d.m = np.maximum(Xs[:, :, 0].T, 0)
    d.s = Xs[:, :, 2].T
    d.f = Xs[:, :, 3].T
    d.n = np.array([DB.n_cells(Xs[b]) for b in range(d.B)])
    d.lib = d.m.sum(0)
    d.CP = 1e4 * d.m / d.lib[None, :]
    d.L = np.log1p(d.CP)
    d.n_test = int(round(0.25 * d.G))
    return d


def check_factory(d):
    e = data_from_X(d.name, d.X)
    for a in ("m", "s", "f", "n", "lib", "CP", "L"):
        assert np.array_equal(getattr(e, a), getattr(d, a)), a
    assert (e.B, e.G, e.T, e.n_test, e.bins) == (d.B, d.G, d.T, d.n_test, d.bins)
    return True


def garbage_X(X, k, seed):
    """Bins > k replaced by gene-permuted copies of random bins with mean / median / sd / sum scaled by a common
    random factor (finite, valid-looking, wrong). Bins <= k untouched."""
    rng = np.random.default_rng(seed)
    Xg = X.copy()
    Bn, G = X.shape[0], X.shape[1]
    for b in range(k + 1, Bn):
        src = int(rng.integers(0, Bn))
        x = X[src][rng.permutation(G)].copy()
        c = float(rng.uniform(0.3, 3.0))
        x[:, [0, 1, 2, 4]] *= c
        Xg[b] = x
    return Xg


def garbage_graph(graph, bins, k, seed, G):
    """Graphs of snapshot bins with index > k replaced by a random bin's graph with permuted gene labels."""
    rng = np.random.default_rng(seed)
    g2 = dict(graph)
    for j in range(k + 1, len(bins)):
        src = bins[int(rng.integers(0, len(bins)))]
        p = rng.permutation(G)
        W, has, Wn = graph[src]
        g2[bins[j]] = (W[p][:, p].tocsr(), has[p], Wn[p][:, p].tocsr())
    return g2


# ------------------------------------------------------------------ masks and scoring
def ctx_masks(G, T):
    nt = int(round(0.25 * G))
    return np.array([[np.sort(np.random.default_rng(1000 * s + i).permutation(G)[:nt]) for i in range(T)]
                     for s in SEEDS])


def score_ctx(Y, P, mk):
    """(3, T, 4) pearson / p_ta_up / p_ta_down / spearman on the CONTEXT masks; NaN where P[i] is None."""
    T = len(Y)
    out = np.full((len(SEEDS), T, 4), np.nan)
    for s in range(len(SEEDS)):
        for i in range(T):
            if P[i] is None:
                continue
            t = mk[s, i]
            m = RS.metrics(np.asarray(Y[i])[t], np.asarray(P[i])[t])
            out[s, i] = [m[k] for k in KEYS]
    return out


def summ(rows, lo):
    v = np.nanmean(rows[:, lo:, :], axis=1).mean(0)
    return {k: round(float(x), 4) for k, x in zip(("pcc", "up", "down", "spearman"), v)}


def fmt(s):
    return f"{s['pcc']:+.4f}/{s['up']:.4f}/{s['down']:.4f}"


def pertrans(rows):
    """(T, 3) mask-averaged pcc / up / down per transition."""
    return np.nanmean(rows[:, :, :3], axis=0)


def wil(x):
    nz = x[np.abs(x) > 1e-12]
    return round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None


def paired_ctx(ra, rb, lo):
    """a - b per transition (mask-averaged), transitions >= lo; mean, better / worse, Wilcoxon p."""
    a, b = pertrans(ra)[lo:], pertrans(rb)[lo:]
    out = {}
    for j, k in enumerate(("pcc", "up", "down")):
        dl = np.nan_to_num(a[:, j]) - np.nan_to_num(b[:, j])
        out[k] = {"mean_delta": round(float(dl.mean()), 4), "better": int((dl > 1e-12).sum()),
                  "worse": int((dl < -1e-12).sum()), "n": int(len(dl)), "wilcoxon_p": wil(dl)}
    return out


def fmt_pair(p):
    q = p["pcc"]
    return (f"dPCC {q['mean_delta']:+.4f} ({q['better']}/{q['worse']} of {q['n']}, p={q['wilcoxon_p']}) "
            f"dUp {p['up']['mean_delta']:+.4f} (p={p['up']['wilcoxon_p']}) "
            f"dDown {p['down']['mean_delta']:+.4f} (p={p['down']['wilcoxon_p']})")


# ------------------------------------------------------------------ new features
def det_level(f_b, n_b, lib_b):
    lam = -np.log(1 - np.clip(f_b, 0, 1 - 1 / (2 * n_b)))
    return np.log1p(1e4 * lam / lib_b)


def new_row(i, f, n, lib, L, lag1_i, mom_i, graph, bins):
    """Row i of the new features from column i (and i-1, i-2) of the inputs only."""
    J = [j for j in (i, i - 1, i - 2) if j >= 0]
    Wsum = sum(graph[bins[j]][0] for j in J)
    Wbar, hasb = NS._rownorm(Wsum)
    Wnsum = sum(graph[bins[j]][2] for j in J)
    Wbn, _ = NS._rownorm(Wnsum)
    Xi = det_level(f[:, i], n[i], lib[i])
    Xp = det_level(f[:, i - 1], n[i - 1], lib[i - 1]) if i >= 1 else Xi
    return {"grnbar_lag1": Wbar @ lag1_i, "has_regbar": hasb, "grnbarnull_lag1": Wbn @ lag1_i,
            "grnbar_mom": Wbar @ mom_i, "fgap": Xi - L[:, i], "fmom": Xi - Xp}


def add_new(d, F, graph):
    T, G = d.T, d.G
    for k in NEW:
        F[k] = np.zeros((T, G))
    for i in range(1, T):                    # row 0 stays 0
        r = new_row(i, d.f, d.n, d.lib, d.L, F["lag1"][i], F["mom_eb"][i], graph, d.bins)
        for k in NEW:
            F[k][i] = r[k]
    return F


def strict_nan_check(d, F, graph, tg="l1"):
    """For every row i >= 1: recompute the new features with every bin > i set to NaN (and the graphs of bins > i
    removed, so any access raises), lag1 rebuilt from the NaN-padded means; must equal the stored row exactly."""
    T = d.T
    worst = 0.0
    for i in range(1, T):
        f, n, lib, L, m = d.f.copy(), d.n.copy(), d.lib.copy(), d.L.copy(), d.m.copy()
        for a in (f, L, m):
            a[:, i + 1:] = np.nan
        n[i + 1:] = np.nan
        lib[i + 1:] = np.nan
        g = {d.bins[j]: graph[d.bins[j]] for j in range(i + 1)}
        lag1 = np.asarray(D.dileo_protocol.TARGET_NP[tg](m[:, i - 1], m[:, i]), np.float64)
        assert np.array_equal(lag1, F["lag1"][i]), f"lag1 row {i} != Y[i-1]"
        r = new_row(i, f, n, lib, L, lag1, F["mom_eb"][i], g, d.bins)
        for k in NEW:
            assert np.all(np.isfinite(r[k])), (k, i)
            dd = float(np.max(np.abs(r[k] - F[k][i])))
            worst = max(worst, dd)
            assert dd == 0.0, (k, i, dd)
    return worst


# ------------------------------------------------------------------ models
def subsets():
    G_ = NS.GROUPS
    full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"]
    V1 = full + G_["grn"]
    S = {
        "V0_full": [full],
        "V1_full+grn": [V1],
        "V2_V1+grnbar": [V1 + ["grnbar_lag1", "has_regbar"]],
        "V2b_full+grnbar": [full + ["grnbar_lag1", "has_regbar"]],
        "V2n_V1+grnbarnull": [V1 + ["grnbarnull_lag1", "has_regbar"]],
        "Vd_full+fgap+fmom": [full + ["fgap", "fmom"]],
        "P_primary": [V1 + ["grnbar_lag1", "has_regbar", "fgap", "fmom"]],
        "Pn_null": [full + G_["grn_null"] + ["has_reg"] + ["grnbarnull_lag1", "has_regbar", "fgap", "fmom"]],
        "Vm_V1+grnbar_mom": [V1 + ["grnbar_mom", "has_regbar"]],
    }
    S["Vauto_V1|V2|P"] = [S["V1_full+grn"][0], S["V2_V1+grnbar"][0], S["P_primary"][0]]
    return full, S


def l1_models(d, graph):
    tg = "l1"
    Y = d.Y(tg)
    F, cls = NS.build_features(d, tg, Y, graph, None)
    add_new(d, F, graph)
    fb = NS.fallback(d, tg, Y, F)
    full, S = subsets()
    allnames = full + NS.GROUPS["grn"] + NS.GROUPS["grn_null"] + NEW
    st = NS.Stacker(F, cls, Y, allnames)
    preds, chosen = {}, {}
    for k, subs in S.items():
        preds[k], chosen[k] = NS.run_learned(st, subs, Y, d, fb)
    preds["trend"] = [fb[0]] + [Y[i - 1] for i in range(1, d.T)]
    del st
    return Y, F, cls, preds, chosen


def linear_models(d, graph):
    tg = "linear"
    Y = d.Y(tg)
    F, cls = NS.build_features(d, tg, Y, graph, None)
    fb = NS.fallback(d, tg, Y, F)
    full, _ = subsets()
    st = NS.Stacker(F, cls, Y, full + NS.GROUPS["grn"])
    preds, chosen = {}, {}
    for k, subs in (("V0_full", [full]), ("V1_full+grn", [full + NS.GROUPS["grn"]])):
        preds[k], chosen[k] = NS.run_learned(st, subs, Y, d, fb)
    preds["trend"] = [fb[0]] + [Y[i - 1] for i in range(1, d.T)]
    del st
    return Y, preds, chosen


def level_obs(d, tg):
    """DistBridge no-SB (rolling pool + finite-cell observation head) for tg in l1 / linear, strict."""
    key = {"l1": "distbridge_strict_nosb_l1", "linear": "distbridge_strict_nosb_linear",
           "orig": "distbridge_strict_nosb"}[tg]
    DB._CACHE.clear()
    model = DB.REGISTRY[key]()
    hist = [d.X[b] for b in range(d.B)]
    P = [np.asarray(model.predict(hist[:i + 1], None, None), np.float64) for i in range(d.T)]
    sel = [str(v) for _, v in sorted(model.selected.items(), key=lambda kv: kv[0][1])]
    DB._CACHE.clear()
    return P, sel


# ------------------------------------------------------------------ noise budget (diagnostic, NOT strict: uses bin t+1)
def signal_share(d):
    """Unfloored delta-method R of L = log1p(CP10k): kap^2 s^2 / (n-1) / (1+CP)^2. Per transition t:
    a = mean R_t / var y_t, b = mean R_{t+1} / var y_t, share = 1 - a - b."""
    Y = d.Y("l1")
    kap = 1e4 / d.lib
    R = kap[None, :] ** 2 * d.s ** 2 / np.maximum(d.n - 1, 1)[None, :] / (1 + d.CP) ** 2
    v = np.array([Y[t].var() for t in range(d.T)])
    a = np.array([R[:, t].mean() for t in range(d.T)]) / v
    b = np.array([R[:, t + 1].mean() for t in range(d.T)]) / v
    return a, b, 1 - a - b
