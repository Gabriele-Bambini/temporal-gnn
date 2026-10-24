"""(1) Do THEIR GRNBoost2 edges carry signal for the next per-gene change beyond the gene's own history?

STRICT everywhere: at transition i (bin a -> bin b) every feature uses bins <= a only; the GRN is the edge set of
bin a (the same snapshot graph their models get, train_models_gene.build_snapshot_split_future_node_delta_ranking
uses snapshots[i]). Targets orig / l1 / linear from dileo_protocol.TARGET_NP.

Features of gene g at transition i (y_prev = the target formula applied to (bin a-1, bin a) = y of transition i-1):
  own : o_last = y_prev; o_rev = T(m_a, pool4_a) (move to the cell-weighted mean of the last 4 bins, i.e. mean
        reversion of a noisy bin); o_lvl = log1p(CP10k_a); o_f = frac nonzero at a; o_zero = 1[m_a = 0];
        o_obs (orig only) = expected clipped target under a finite-cell observation model (task2_forecasters.ObsLevel(4)).
  grn : g_reg = importance-weighted mean of the regulators' y_prev (bin-a edges, weights normalised per target);
        g_top10 = plain mean of the 10 strongest regulators' y_prev; g_regrev = weighted mean of regulators' o_rev;
        g_out = weighted mean of the targets' y_prev (for genes that are regulators); log in/out degree.
  controls (same cost, same form):
        shuf  = g_* on a degree-preserving null (regulator ids permuted among the bin-a edges: every target keeps
                its in-degree and weights, every regulator its out-degree);
        prof  = k = 20 regulators (pool = genes that were regulators in some bin <= a) whose log1p(CP10k)
                trajectory over bins <= a correlates most with the gene's; weight = correlation;
        knn   = the same with all genes as pool (cross-gene pooling without any GRN).
Analyses:
  A. per transition, on genes with >= 1 regulator in bin a: raw corr(y, g_reg) and partial corr(y, g_reg | own)
     (OLS residualisation inside the transition; a diagnostic, not a forecaster), for GRN / shuf / prof / knn,
     lag-1 (strict) and lag-0 (regulators' change over the SAME transition a -> b; NOT strict: an upper bound).
  B. strict rolling-origin ridge: coefficients fitted on transitions 1..i-1 (features and targets all from bins
     <= a), applied to transition i; scored with their metrics (Pearson, P@200 up / down) on 25 % random test
     genes per transition (5 draws). Feature sets: own, own+grn, own+shuf, own+prof, own+knn, own+grn+knn.
  C. turn-on: genes with m_a = 0; does the union of bin <= a edges (regulators' y_prev) predict m_b > 0 beyond own
     history (ever expressed, fraction of past bins expressed, bins since last expressed)? AUC, strict rolling
     logistic regression.
Output: grn_signal.json in this folder.
"""
import json
import os
import time

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

import common
import task2_forecasters as tf

RNG_SEED = 7
KPROF = 20
DRAWS = 5


def hist_arrays(d, bins):
    return [np.stack([d["m"][:, b], d["med"][:, b], d["s"][:, b], d["f"][:, b], d["S"][:, b]], 1) for b in bins]


def cp10k(m):
    return 1e4 * m / max(m.sum(), 1e-12)


def row_norm(A):
    s = np.asarray(A.sum(1)).ravel()
    inv = np.where(s > 0, 1.0 / np.where(s > 0, s, 1), 0.0)
    return sparse.diags(inv) @ A


def graph_mats(u, i, w, N):
    """A[target, regulator] = importance (duplicates summed)."""
    return sparse.csr_matrix((w.astype(np.float64), (i, u)), shape=(N, N))


def top_k_mat(u, i, w, N, k):
    order = np.lexsort((-w, i))
    ii = i[order]
    start = np.r_[0, np.flatnonzero(np.diff(ii)) + 1]
    rank = np.arange(len(ii)) - np.repeat(start, np.diff(np.r_[start, len(ii)]))
    keep = order[rank < k]
    return sparse.csr_matrix((np.ones(len(keep)), (i[keep], u[keep])), shape=(N, N))


def corr_neighbours(X, pool, k, block=2000):
    """For every gene: k genes of `pool` (excluding itself) with the largest Pearson correlation of rows of X;
    returns a row-normalised sparse matrix with weights max(corr, 0)."""
    Z = X - X.mean(1, keepdims=True)
    sd = np.sqrt((Z ** 2).sum(1))
    Z = np.where(sd[:, None] > 0, Z / np.where(sd > 0, sd, 1)[:, None], 0.0).astype(np.float32)
    P = Z[pool]
    rows, cols, vals = [], [], []
    N = X.shape[0]
    for lo in range(0, N, block):
        C = Z[lo:lo + block] @ P.T
        gi = np.arange(lo, min(lo + block, N))
        self_hit = np.searchsorted(pool, gi)
        ok = (self_hit < len(pool)) & (pool[np.minimum(self_hit, len(pool) - 1)] == gi)
        C[np.flatnonzero(ok), self_hit[ok]] = -2.0
        idx = np.argpartition(-C, k, axis=1)[:, :k]
        v = np.take_along_axis(C, idx, 1)
        rows.append(np.repeat(gi, k)); cols.append(pool[idx].ravel()); vals.append(np.maximum(v, 0).ravel())
    A = sparse.csr_matrix((np.concatenate(vals).astype(np.float64), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N))
    return row_norm(A)


def zscore(v):
    v = np.asarray(v, float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v * 0.0


def partial_corr(y, x, Z):
    """corr of y and x after OLS removal of Z (with intercept)."""
    D = np.column_stack([np.ones(len(y)), Z])
    by, *_ = np.linalg.lstsq(D, y, rcond=None)
    bx, *_ = np.linalg.lstsq(D, x, rcond=None)
    return common.pearson(y - D @ by, x - D @ bx)


def build(name):
    t0 = time.time()
    d = common.load_node(name)
    N, _ = d["m"].shape
    snaps = common.SNAPS[name]
    trans = common.transitions(name)
    u_all, i_all, ts_all, w_all = common.load_edges(name)
    keep = (u_all < N) & (i_all < N)
    u_all, i_all, ts_all, w_all = u_all[keep], i_all[keep], ts_all[keep], w_all[keep]
    rng = np.random.default_rng(RNG_SEED)
    n = d["n"]
    Y = {T: [common.target(T, d["m"][:, a], d["m"][:, b]) for a, b in trans] for T in common.TARGETS}
    feats = {T: [None] * len(trans) for T in common.TARGETS}   # dict of feature vectors per transition
    lag0 = {T: [None] * len(trans) for T in common.TARGETS}
    cover = []
    turn = []
    regs_seen = np.zeros(N, bool)
    union_A = sparse.csr_matrix((N, N))
    for ti, (a, b) in enumerate(trans):
        hist_bins = [s for s in snaps if s <= a]
        sel = ts_all == a
        u, i, w = u_all[sel], i_all[sel], w_all[sel].astype(np.float64)
        regs_seen[np.unique(u)] = True
        A = graph_mats(u, i, w, N)
        union_A = union_A + row_norm(A)
        W = row_norm(A)
        Wout = row_norm(A.T.tocsr())
        Wtop = row_norm(top_k_mat(u, i, w, N, 10))
        perm = rng.permutation(len(u))
        Wsh = row_norm(graph_mats(u[perm], i, w, N))
        indeg = np.asarray((A > 0).sum(1)).ravel()
        outdeg = np.asarray((A > 0).sum(0)).ravel()
        has_reg = indeg > 0
        cover.append({"transition": f"{a}->{b}", "genes_with_regulator": int(has_reg.sum()),
                      "frac_with_regulator": float(has_reg.mean()), "expressed_at_a": int((d["m"][:, a] > 0).sum()),
                      "regulators": int((outdeg > 0).sum())})
        if ti == 0:
            continue
        p = snaps[snaps.index(a) - 1]
        # trajectory neighbours (strict: bins <= a)
        X = np.log1p(np.stack([cp10k(d["m"][:, s]) for s in hist_bins], 1))
        if X.shape[1] >= 3:
            Wprof = corr_neighbours(X, np.flatnonzero(regs_seen), KPROF)
            Wknn = corr_neighbours(X, np.arange(N), KPROF)
        else:
            Wprof = Wknn = None
        lo = max(0, len(hist_bins) - 4)
        wts = n[hist_bins[lo:]]
        pool4 = (d["m"][:, hist_bins[lo:]] * wts).sum(1) / wts.sum()
        obs = None
        for T in common.TARGETS:
            yprev = Y[T][ti - 1]
            orev = common.target(T, d["m"][:, a], pool4)
            if T == "orig":
                obs = tf.ObsLevel(4).predict(hist_arrays(d, hist_bins), hist_bins, None).astype(np.float64)
            F = {"o_last": yprev, "o_rev": orev, "o_lvl": np.log1p(cp10k(d["m"][:, a])), "o_f": d["f"][:, a],
                 "o_zero": (d["m"][:, a] == 0).astype(float)}
            if T == "orig":
                F["o_obs"] = obs
            F.update({"g_reg": W @ yprev, "g_top10": Wtop @ yprev, "g_regrev": W @ orev, "g_out": Wout @ yprev,
                      "g_lindeg": np.log1p(indeg), "g_loutdeg": np.log1p(outdeg),
                      "shuf_reg": Wsh @ yprev, "shuf_regrev": Wsh @ orev})
            if Wprof is not None:
                F.update({"prof_reg": Wprof @ yprev, "prof_regrev": Wprof @ orev,
                          "knn_reg": Wknn @ yprev, "knn_regrev": Wknn @ orev})
            F["has_reg"] = has_reg.astype(float)
            feats[T][ti] = F
            ynow = Y[T][ti]
            L0 = {"g": W @ ynow, "shuf": Wsh @ ynow}
            if Wprof is not None:
                L0["prof"] = Wprof @ ynow; L0["knn"] = Wknn @ ynow
            lag0[T][ti] = L0
        # turn-on (genes zero at a): own history vs union-of-past-graphs regulators' change (orig units: log)
        z0 = d["m"][:, a] == 0
        past = d["m"][:, hist_bins[:-1]] > 0
        ever = past.any(1)
        frac_past = past.mean(1)
        L = past.shape[1]
        since = np.where(ever, np.argmax(past[:, ::-1], 1) + 1.0, L + 2.0) if L else np.full(N, 2.0)
        cperm = np.arange(N)
        rs = np.flatnonzero(regs_seen)
        cperm[rs] = rng.permutation(rs)
        Wu = row_norm(union_A.tocsr())
        yprev_l1 = Y["l1"][ti - 1]
        hasu = np.asarray((union_A > 0).sum(1)).ravel() > 0
        turn.append({"a": a, "b": b, "z0": z0, "on": d["m"][:, b] > 0, "ever": ever.astype(float),
                     "frac_past": frac_past, "since": since, "f_prevbin": d["f"][:, p],
                     "u_reg": Wu @ yprev_l1, "u_has": hasu.astype(float),
                     "u_shuf": row_norm(union_A.tocsr()[:, cperm]) @ yprev_l1})
    print(name, "features built in", round(time.time() - t0, 1), "s", flush=True)
    return d, trans, Y, feats, lag0, cover, turn


def analysis_A(trans, Y, feats, lag0):
    out = {}
    for T in common.TARGETS:
        own = ["o_last", "o_rev", "o_lvl", "o_f", "o_zero"] + (["o_obs"] if T == "orig" else [])
        rows = []
        for ti in range(1, len(trans)):
            F = feats[T][ti]
            if F is None or "prof_reg" not in F:
                continue
            m = F["has_reg"] > 0
            y = Y[T][ti][m]
            Z = np.column_stack([F[k][m] for k in own])
            r = {"transition": f"{trans[ti][0]}->{trans[ti][1]}", "n_genes": int(m.sum())}
            for tag, key in (("grn", "g_reg"), ("grn_top10", "g_top10"), ("grn_regrev", "g_regrev"),
                             ("shuf", "shuf_reg"), ("prof", "prof_reg"), ("knn", "knn_reg")):
                x = F[key][m]
                r[f"raw_{tag}"] = common.pearson(y, x)
                r[f"partial_{tag}"] = partial_corr(y, x, Z)
            r["raw_own_last"] = common.pearson(y, F["o_last"][m])
            for tag in ("g", "shuf", "prof", "knn"):
                x = lag0[T][ti][tag][m]
                r[f"lag0_raw_{tag}"] = common.pearson(y, x)
                r[f"lag0_partial_{tag}"] = partial_corr(y, x, Z)
            rows.append(r)
        keys = [k for k in rows[0] if k not in ("transition", "n_genes")]
        out[T] = {"mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
                  "per_transition": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]}
    return out


FEATURE_SETS = {
    "own": [],
    "own+grn": ["g_reg", "g_top10", "g_regrev", "g_out", "g_lindeg", "g_loutdeg", "has_reg"],
    "own+shuf": ["shuf_reg", "shuf_regrev", "g_lindeg", "g_loutdeg", "has_reg"],
    "own+deg_only": ["g_lindeg", "g_loutdeg", "has_reg"],
    "own+prof": ["prof_reg", "prof_regrev"],
    "own+knn": ["knn_reg", "knn_regrev"],
    "own+grn+knn": ["g_reg", "g_top10", "g_regrev", "g_out", "g_lindeg", "g_loutdeg", "has_reg", "knn_reg", "knn_regrev"],
    "grn_only": ["g_reg", "g_top10", "g_regrev", "g_out", "g_lindeg", "g_loutdeg", "has_reg", "_noown"],
}


def analysis_B(trans, Y, feats, N):
    """Strict rolling-origin ridge on per-transition z-scored features / target."""
    valid = [ti for ti in range(len(trans)) if feats["orig"][ti] is not None and "prof_reg" in feats["orig"][ti]]
    evals = valid[1:]  # at least one past training transition (~N genes, ample for <= 15 coefficients)
    mk = common.masks(N, len(evals), DRAWS, seed=11)
    out = {"eval_transitions": [f"{trans[t][0]}->{trans[t][1]}" for t in evals]}
    for T in common.TARGETS:
        own = ["o_last", "o_rev", "o_lvl", "o_f", "o_zero"] + (["o_obs"] if T == "orig" else [])
        res = {}
        for fs, extra in FEATURE_SETS.items():
            cols = ([] if "_noown" in extra else own) + [c for c in extra if c != "_noown"]
            Xs = {t: np.column_stack([zscore(feats[T][t][c]) for c in cols]) for t in valid}
            ys = {t: zscore(Y[T][t]) for t in valid}
            P, Yl = [], []
            coefs = []
            for t in evals:
                tr = [s for s in valid if s < t]
                Xtr = np.vstack([Xs[s] for s in tr]); ytr = np.concatenate([ys[s] for s in tr])
                mdl = Ridge(alpha=10.0).fit(Xtr, ytr)
                P.append(mdl.predict(Xs[t])); Yl.append(Y[T][t])
                coefs.append(mdl.coef_)
            sc = common.score_masked(Yl, P, mk)
            sc["mean_coef"] = dict(zip(cols, np.mean(coefs, 0).round(4).tolist()))
            res[fs] = sc
        # single-feature strict references (no fit): own last change, reversion, obs
        for ref in ("o_last", "o_rev") + (("o_obs",) if T == "orig" else ()):
            res[f"raw_{ref}"] = common.score_masked([Y[T][t] for t in evals], [feats[T][t][ref] for t in evals], mk)
        out[T] = res
    return out


def analysis_C(turn):
    """Turn-on of genes zero at a; strict rolling logistic regression own vs own+graph."""
    rows = []
    for k, r in enumerate(turn):
        z = r["z0"]
        y = r["on"][z].astype(int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        row = {"transition": f"{r['a']}->{r['b']}", "n_zero": int(z.sum()), "frac_turn_on": float(y.mean())}
        for f in ("ever", "frac_past", "since", "f_prevbin", "u_reg", "u_has", "u_shuf"):
            x = r[f][z]
            row[f"auc_{f}"] = float(roc_auc_score(y, x)) if np.std(x) > 0 else 0.5
        rows.append(row)
    own = ["ever", "frac_past", "since", "f_prevbin"]
    res = {}
    for name, cols in (("own", own), ("own+graph", own + ["u_reg", "u_has"]), ("own+shuf", own + ["u_shuf", "u_has"])):
        aucs = []
        for k in range(2, len(turn)):
            z = turn[k]["z0"]
            ytest = turn[k]["on"][z].astype(int)
            if ytest.sum() < 5:
                continue
            Xtr = np.vstack([np.column_stack([turn[s][c][turn[s]["z0"]] for c in cols]) for s in range(k)])
            ytr = np.concatenate([turn[s]["on"][turn[s]["z0"]].astype(int) for s in range(k)])
            if ytr.sum() < 5:
                continue
            mdl = LogisticRegression(max_iter=500).fit(Xtr, ytr)
            Xte = np.column_stack([turn[k][c][z] for c in cols])
            aucs.append(roc_auc_score(ytest, mdl.predict_proba(Xte)[:, 1]))
        res[name] = {"mean_auc": float(np.mean(aucs)), "n_transitions": len(aucs)}
    keys = [k for k in rows[0] if k.startswith("auc_") or k == "frac_turn_on"]
    return {"strict_logistic": res, "mean_single_feature": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "per_transition": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]}


def main():
    out = {}
    for name in common.SNAPS:
        d, trans, Y, feats, lag0, cover, turn = build(name)
        N = d["m"].shape[0]
        out[name] = {"coverage": cover,
                     "A_partial_corr": analysis_A(trans, Y, feats, lag0)}
        print(name, "A done", flush=True)
        out[name]["B_strict_ridge"] = analysis_B(trans, Y, feats, N)
        print(name, "B done", flush=True)
        out[name]["C_turn_on"] = analysis_C(turn)
        for T in common.TARGETS:
            A = out[name]["A_partial_corr"][T]["mean"]
            print(f"  {T} A:", {k: round(v, 3) for k, v in A.items()})
            for fs, sc in out[name]["B_strict_ridge"][T].items():
                print(f"  {T} B {fs:14s} pcc={sc['pcc']:.3f} up={sc['p200_up']:.3f} down={sc['p200_down']:.3f}")
        print("  C:", out[name]["C_turn_on"]["strict_logistic"], out[name]["C_turn_on"]["mean_single_feature"], flush=True)
    with open(os.path.join(common.OUT, "grn_signal.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
