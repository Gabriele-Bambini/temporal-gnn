"""Judge (step 6): own truncation / look-ahead test of the process-model v2 pipeline, with the judge's OWN garbage
(pm_lib.garbage_X / garbage_graph are NOT used), plus faithfulness of NASS V0 against the published NASS predictions
and an independent re-scoring of the headline comparisons.

Models under test (copied build code, judge/pm_lib.py, judge/orig/orig_lib.py, never edited):
  l1   : every NASS stacker variant of pm_lib.l1_models (V0, V1, V2, V2b, V2n, Vd, P, Pn, Vm, Vauto) + trend
  orig : orig_lib.orig_all (binomial detection filter, heads base/A/B/C/D with rolling pools, NASS orig full, O2, trend)
(linear and the DistBridge level_obs comparators are skipped: they do not bear on the verdict.)

Garbage kinds (bins > k only; bins <= k untouched; clean and dirty data both built by pm_lib.data_from_X):
  rev   in-distribution: bins k+1..B-1 replaced by the same future bins in REVERSED order, each with its own gene
        permutation; graphs of those bins replaced by the graph of the source bin with the same gene permutation.
  ood   out-of-distribution: every bin > k = bin 0 with mean / median / sd x 50 and sum scaled so n_b = 3 cells;
        graphs of bins > k replaced by EMPTY graphs.
  cut   shorter data: only bins 0..k+1 exist (B = k+2, T = k+1), bin k+1 = gene-permuted copy of the real bin k+1
        (only the label of transition k sees it); d.bins truncated to match. Catches dependence on the series length.
Pass: predictions of transitions 0..k bitwise identical to the clean run (max |diff| = 0) and chosen lists equal up
to k; sanity: predictions at k+1 differ (rev / ood, when k+1 < T).
Usage: python judge_trunc.py <name> <kind:k> ...   e.g.  python judge_trunc.py pancreas-human rev:13 rev:2 ood:5 cut:4
Output: judge_trunc_<name>.json (+ .log), judge_preds_<name>.npz (clean predictions of key models, float64)."""
import json
import os
import sys
import time

import numpy as np
from scipy import sparse, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "orig"))
import pm_lib as L  # noqa: E402
import orig_lib as O  # noqa: E402

D, NS, DB = L.D, L.NS, L.DB
ROOT = D.ROOT
V3 = os.path.join(ROOT, "tack", "results", "process_model", "verify_3")
KEY = {"l1": ["l1/V0_full", "l1/V1_full+grn", "l1/P_primary", "l1/Pn_null", "l1/trend"],
       "orig": ["orig/NASS_full_orig", "orig/O2_nass+C_head", "orig/C_post_all[rolling]", "orig/base[rolling]",
                "orig/trend"]}


def pipeline(d, graph):
    P, C = {}, {}
    DB._CACHE.clear()
    Y, F, cls, p, ch = L.l1_models(d, graph)
    del F
    P.update({f"l1/{k}": v for k, v in p.items()})
    C.update({f"l1/{k}": v for k, v in ch.items()})
    DB._CACHE.clear()
    Yo, p, ch, setting = O.orig_all(d)
    DB._CACHE.clear()
    P.update({f"orig/{k}": v for k, v in p.items()})
    C.update({f"orig/{k}": v for k, v in ch.items()})
    C["orig/detection_setting"] = setting
    return P, C, Y, Yo


# ------------------------------------------------------------------ judge's own garbage
def dirty_rev(name, X, gc, k, seed):
    rng = np.random.default_rng(seed)
    Bn, G = X.shape[0], X.shape[1]
    fut = list(range(k + 1, Bn))
    src = fut[::-1]
    Xg = X.copy()
    perms = {}
    for b, s in zip(fut, src):
        p = rng.permutation(G)
        Xg[b] = X[s][p]
        perms[b] = (s, p)
    d = L.data_from_X(name, Xg)
    g = NS.load_graph(d)
    g0 = gc                                        # real (clean) graphs of the source bins
    for b, (s, p) in perms.items():
        W, has, Wn = g0[d.bins[s]]
        g[d.bins[b]] = (W[p][:, p].tocsr(), has[p], Wn[p][:, p].tocsr())
    return d, g


def dirty_ood(name, X, gc, k, seed):
    Bn, G = X.shape[0], X.shape[1]
    x0 = X[0].copy()
    n0 = float(DB.n_cells(x0))
    Xg = X.copy()
    for b in range(k + 1, Bn):
        x = x0.copy()
        x[:, [0, 1, 2]] *= 50.0
        x[:, 4] *= 50.0 * 3.0 / n0
        Xg[b] = x
    d = L.data_from_X(name, Xg)
    assert all(abs(d.n[b] - 3.0) < 1e-6 for b in range(k + 1, Bn)), d.n
    g = NS.load_graph(d)
    E = sparse.csr_matrix((G, G))
    for j in range(k + 1, Bn):
        g[d.bins[j]] = (E.copy(), np.zeros(G), E.copy())
    return d, g


def dirty_cut(name, X, gc, k, seed):
    rng = np.random.default_rng(seed)
    G = X.shape[1]
    Xt = X[:k + 2].copy()
    Xt[k + 1] = X[k + 1][rng.permutation(G)]
    d = L.data_from_X(name, Xt)
    d.bins = d.bins[:d.B]
    g = NS.load_graph(d)
    return d, g


KINDS = {"rev": dirty_rev, "ood": dirty_ood, "cut": dirty_cut}


# ------------------------------------------------------------------ independent scorer (float64, own code)
def incl(v, k):
    order = np.sort(v)[::-1]
    thr = order[k - 1]
    above = v > thr
    eq = v == thr
    out = above.astype(np.float64)
    out[eq] = (k - above.sum()) / eq.sum()
    return out


def my_metrics(y, p, k=200):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    yc, pc = y - y.mean(), p - p.mean()
    den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
    r = float((yc * pc).sum() / den) if den > 0 else np.nan
    up = float((incl(y, k) * incl(p, k)).sum() / k)
    dn = float((incl(-y, k) * incl(-p, k)).sum() / k)
    return r, up, dn


def my_rows(Y, P, G):
    T = len(Y)
    nt = int(round(0.25 * G))
    out = np.full((3, T, 3), np.nan)
    for s in range(3):
        for i in range(T):
            if P[i] is None:
                continue
            t = np.sort(np.random.default_rng(1000 * s + i).permutation(G)[:nt])
            out[s, i] = my_metrics(Y[i][t], P[i][t])
    return out


def tab(rows, lo):
    v = np.nanmean(rows[:, lo:, :], axis=1).mean(0)
    return [round(float(x), 4) for x in v]


def paired(ra, rb, lo, nperm=20000, seed=7):
    a, b = np.nanmean(ra, 0)[lo:], np.nanmean(rb, 0)[lo:]
    out = {}
    rng = np.random.default_rng(seed)
    for j, kk in enumerate(("pcc", "up", "down")):
        dl = a[:, j] - b[:, j]
        nz = dl[np.abs(dl) > 1e-12]
        sg = rng.choice([-1.0, 1.0], size=(nperm, len(dl)))
        perm_p = float((np.abs((sg * dl).mean(1)) >= abs(dl.mean()) - 1e-15).mean())
        bs = rng.integers(0, len(dl), size=(nperm, len(dl)))
        bm = dl[bs].mean(1)
        out[kk] = {"mean": round(float(dl.mean()), 4), "better": int((dl > 1e-12).sum()),
                   "worse": int((dl < -1e-12).sum()), "n": len(dl),
                   "wilcoxon_p": round(float(stats.wilcoxon(nz).pvalue), 4) if len(nz) >= 5 else None,
                   "signflip_p": round(perm_p, 4),
                   "boot95": [round(float(np.quantile(bm, 0.025)), 4), round(float(np.quantile(bm, 0.975)), 4)],
                   "per_transition": np.round(dl, 4).tolist()}
    return out


def main(name, specs):
    logf = open(os.path.join(HERE, f"judge_trunc_{name}.log"), "a")

    def say(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    t0 = time.time()
    say(f"==== judge_trunc {name} {specs} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    X = D.Data(name).X.copy()
    dc = L.data_from_X(name, X)
    L.check_factory(D.Data(name))
    gc = NS.load_graph(dc)
    Pc, Cc, Y, Yo = pipeline(dc, gc)
    say(f"  clean pipeline {time.time() - t0:.0f}s, {len(Pc)} models")
    res = {"models": sorted(Pc), "cuts": {}}
    # ---- faithfulness: V0 vs the published NASS l1 predictions (verify_3 rerun of design_best-signal's own code)
    ref = np.load(os.path.join(V3, f"nass_l1_preds_{name}.npz"))["pred"]
    V0 = np.stack([np.asarray(p, np.float64) for p in Pc["l1/V0_full"]])
    res["V0_vs_published_NASS_preds"] = {"shape_equal": bool(ref.shape == V0.shape),
                                         "max_abs_diff": float(np.max(np.abs(ref - V0))),
                                         "max_abs_pred": float(np.max(np.abs(ref)))}
    say(f"  V0 vs published NASS l1 predictions: {res['V0_vs_published_NASS_preds']}")
    # ---- independent re-scoring (own float64 scorer, CONTEXT masks)
    sc = {}
    rows = {}
    for tg, Yt in (("l1", Y), ("orig", Yo)):
        for m in KEY[tg]:
            rows[m] = my_rows(Yt, Pc[m], dc.G)
            sc[m] = {"i>=1": tab(rows[m], 1), "i>=2": tab(rows[m], 2)}
            say(f"  own scorer {m:30s} i>=1 {sc[m]['i>=1']} | i>=2 {sc[m]['i>=2']}")
    res["own_scores_ctx"] = sc
    pr = {}
    for a, b in (("l1/P_primary", "l1/V0_full"), ("l1/P_primary", "l1/V1_full+grn"), ("l1/V1_full+grn", "l1/V0_full"),
                 ("l1/P_primary", "l1/Pn_null"), ("l1/V0_full", "l1/trend"),
                 ("orig/O2_nass+C_head", "orig/NASS_full_orig"), ("orig/C_post_all[rolling]", "orig/NASS_full_orig"),
                 ("orig/C_post_all[rolling]", "orig/base[rolling]")):
        for lo in (1, 2):
            pr[f"{a} - {b} | i>={lo}"] = paired(rows[a], rows[b], lo)
            q = pr[f"{a} - {b} | i>={lo}"]["pcc"]
            say(f"  paired {a} - {b} i>={lo}: dPCC {q['mean']:+.4f} ({q['better']}/{q['worse']}) wil p={q['wilcoxon_p']} "
                f"signflip p={q['signflip_p']} boot95 {q['boot95']}")
    res["own_paired_ctx"] = pr
    np.savez_compressed(os.path.join(HERE, f"judge_preds_{name}.npz"),
                        **{m.replace("/", "__").replace("[", "_").replace("]", "").replace("+", "p"):
                           np.stack([np.asarray(p, np.float64) for p in Pc[m]]) for tg in KEY for m in KEY[tg]},
                        Y_l1=Y, Y_orig=Yo)
    json.dump(res, open(os.path.join(HERE, f"judge_trunc_{name}.json"), "w"), indent=1)
    # ---- own truncation tests
    for spec in specs:
        kind, k = spec.split(":")
        k = int(k)
        t1 = time.time()
        dd, gg = KINDS[kind](name, X, gc, k, seed=31 + 7 * k)
        assert np.array_equal(dd.X[:k + 1], X[:k + 1])
        if kind != "cut":
            assert not np.array_equal(dd.X[k + 1:], X[k + 1:])
        Pg, Cg, _, _ = pipeline(dd, gg)
        rec = {"max_abs_diff_upto_k": {}, "chosen_equal_upto_k": {}, "changed_at_k+1": {}}
        for m in Pc:
            dmax = 0.0
            for i in range(k + 1):
                a, b = Pc[m][i], Pg[m][i]
                if a is None or b is None:
                    assert a is None and b is None, (m, i)
                    continue
                a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
                dmax = max(dmax, float(np.max(np.abs(a - b))) if np.all(np.isfinite(b)) else np.inf)
            rec["max_abs_diff_upto_k"][m] = dmax
            if k + 1 < len(Pg[m]):
                rec["changed_at_k+1"][m] = bool(not np.array_equal(np.asarray(Pc[m][k + 1]), np.asarray(Pg[m][k + 1])))
        for m in Cc:
            rec["chosen_equal_upto_k"][m] = bool(list(Cc[m][:k + 1]) == list(Cg[m][:k + 1]))
        n_bad = sum(v != 0.0 for v in rec["max_abs_diff_upto_k"].values())
        n_cbad = sum(not v for v in rec["chosen_equal_upto_k"].values())
        rec["pass"] = bool(n_bad == 0 and n_cbad == 0)
        rec["summary"] = {"kind": kind, "k": k, "T": dc.T, "models": len(Pc), "models_with_any_diff_upto_k": n_bad,
                          "chosen_lists_differing_upto_k": n_cbad,
                          "models_changed_at_k+1": int(sum(rec["changed_at_k+1"].values())),
                          "models_checked_at_k+1": len(rec["changed_at_k+1"]),
                          "worst_diff": max(rec["max_abs_diff_upto_k"].values()), "seconds": round(time.time() - t1)}
        res["cuts"][spec] = rec
        say(f"  {name} {spec}: {'PASS' if rec['pass'] else 'FAIL'} {rec['summary']}")
        if not rec["pass"]:
            for m, v in rec["max_abs_diff_upto_k"].items():
                if v != 0.0:
                    say(f"    diff {m}: {v}")
            for m, v in rec["chosen_equal_upto_k"].items():
                if not v:
                    say(f"    chosen differs {m}")
        json.dump(res, open(os.path.join(HERE, f"judge_trunc_{name}.json"), "w"), indent=1)
        del Pg, Cg, dd, gg
    say(f"==== done {time.time() - t0:.0f}s")
    logf.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
