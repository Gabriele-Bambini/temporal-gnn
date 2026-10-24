"""Truncation test of the whole strict pipeline (every model of pm_run.py and orig/run_orig.py, every comparator).

For each dataset and cut k: bins > k of the node data are replaced by finite garbage (gene-permuted copies of random
bins, mean / median / sd / sum rescaled by a random factor; pm_lib.garbage_X) and the graphs of snapshot bins > k by
a random bin's graph with permuted gene labels (pm_lib.garbage_graph, applied after NS.load_graph on the garbage data).
The whole pipeline is rerun from the data object: l1 (NASS V0..Vauto incl. new features, trend, DistBridge no-SB l1),
linear (NASS full / full+grn, trend, DistBridge no-SB linear), orig (detection filter + heads base/A/B/C/D with
rolling pools, NASS orig full, O2, trend). Pass: every prediction of transitions 0..k is bitwise identical to the clean
run (max |diff| = 0), the chosen configurations / pools / filter settings of transitions 0..k are equal; sanity: the
predictions at k+1 change (the garbage is seen at all).
Clean and garbage runs both use pm_lib.data_from_X (same construction). Usage: python truncation.py <name> k1 k2 ...
Output: truncation_<name>.json, truncation.log (appended)."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "orig"))
import pm_lib as L  # noqa: E402
import orig_lib as O  # noqa: E402

D, NS = L.D, L.NS


def pipeline(d, graph):
    P, C = {}, {}
    Y, F, cls, p, ch = L.l1_models(d, graph)
    del F
    P.update({f"l1/{k}": v for k, v in p.items()})
    C.update({f"l1/{k}": v for k, v in ch.items()})
    P["l1/level_obs"], C["l1/level_obs"] = L.level_obs(d, "l1")
    Yl, p, ch = L.linear_models(d, graph)
    P.update({f"linear/{k}": v for k, v in p.items()})
    C.update({f"linear/{k}": v for k, v in ch.items()})
    P["linear/level_obs"], C["linear/level_obs"] = L.level_obs(d, "linear")
    Yo, p, ch, setting = O.orig_all(d)
    P.update({f"orig/{k}": v for k, v in p.items()})
    C.update({f"orig/{k}": v for k, v in ch.items()})
    C["orig/detection_setting"] = setting
    return P, C


def main(name, ks):
    log = open(os.path.join(HERE, "truncation.log"), "a")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    t0 = time.time()
    X = D.Data(name).X.copy()
    dc = L.data_from_X(name, X)
    gc = NS.load_graph(dc)
    Pc, Cc = pipeline(dc, gc)
    say(f"== {name} clean pipeline {time.time() - t0:.0f}s, {len(Pc)} models")
    res = {}
    for k in ks:
        t1 = time.time()
        Xg = L.garbage_X(X, k, seed=1000 + k)
        assert np.array_equal(Xg[:k + 1], X[:k + 1]) and not np.array_equal(Xg[k + 1:], X[k + 1:])
        dg = L.data_from_X(name, Xg)
        gg = L.garbage_graph(NS.load_graph(dg), dg.bins, k, seed=2000 + k, G=dg.G)
        Pg, Cg = pipeline(dg, gg)
        rec = {"max_abs_diff_upto_k": {}, "chosen_equal_upto_k": {}, "changed_at_k+1": {}}
        for m in Pc:
            dmax = 0.0
            for i in range(k + 1):
                a, b = Pc[m][i], Pg[m][i]
                if a is None or b is None:
                    assert a is None and b is None
                    continue
                dmax = max(dmax, float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))))
            rec["max_abs_diff_upto_k"][m] = dmax
            if k + 1 < len(Pc[m]):
                rec["changed_at_k+1"][m] = bool(not np.array_equal(np.asarray(Pc[m][k + 1]), np.asarray(Pg[m][k + 1])))
        for m in Cc:
            rec["chosen_equal_upto_k"][m] = bool(list(Cc[m][:k + 1]) == list(Cg[m][:k + 1]))
        n_bad = sum(v != 0.0 for v in rec["max_abs_diff_upto_k"].values())
        n_cbad = sum(not v for v in rec["chosen_equal_upto_k"].values())
        n_changed = sum(rec["changed_at_k+1"].values())
        rec["pass"] = bool(n_bad == 0 and n_cbad == 0)
        rec["summary"] = {"models": len(Pc), "models_with_any_diff_upto_k": n_bad,
                          "chosen_lists_differing_upto_k": n_cbad,
                          "models_changed_at_k+1": n_changed,
                          "worst_diff": max(rec["max_abs_diff_upto_k"].values())}
        res[str(k)] = rec
        say(f"  {name} k={k}: {'PASS' if rec['pass'] else 'FAIL'} {rec['summary']} ({time.time() - t1:.0f}s)")
        if not rec["pass"]:
            for m, v in rec["max_abs_diff_upto_k"].items():
                if v != 0.0:
                    say(f"    diff {m}: {v}")
            for m, v in rec["chosen_equal_upto_k"].items():
                if not v:
                    say(f"    chosen differs {m}")
        json.dump(res, open(os.path.join(HERE, f"truncation_{name}.json"), "w"), indent=1)
        del Pg, Cg, dg, gg
    log.close()


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
