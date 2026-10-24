"""Step 1 anchors (stop if any fails). Rebuilds dsdata caches in build/, checks them against design_best-signal's,
checks the edge cache, then NASS full l1 (exactly as verify_2/run_nass.py: Stacker(F, cls, Y, full)) and NASS
full+grn (graph = NS.load_graph(d)) on NASS's own 20 masks (seed 20261022, i >= 2) and on the CONTEXT masks
(1000 s + i, 3 seeds), and the trend on the CONTEXT masks. Output: anchors.json, anchors.log."""
import json
import os
import sys
import time

import numpy as np

import pm_lib as L

D, NS = L.D, L.NS
SA = os.path.join(D.ROOT, "tack", "results", "signal_analysis", "design_best-signal")
EXPECT = {
    "gastrulation-human": {"nass_full_own_i>=2": (0.3351, 0.2755, 0.2905), "nass_full_ctx_i>=2": (0.337, 0.277, 0.295),
                           "trend_ctx_i>=1": (-0.158,)},
    "pancreas-human": {"nass_full_own_i>=2": (0.308, 0.221, 0.283), "nass_full_ctx_i>=2": (0.322, 0.221, 0.280),
                       "nass_full+grn_own_i>=2": (0.327, 0.230, 0.294),
                       "trend_ctx_i>=1": (0.177,), "trend_ctx_i>=2": (0.167,)},
}
EDGES = {"gastrulation-human": 24.8e6, "pancreas-human": 14.1e6}
PER_BIN = {"gastrulation-human": 0.84e6, "pancreas-human": 0.94e6}


def close(got, exp):
    tol = [0.00051 if len(f"{e}".split(".")[-1]) >= 4 else 0.0011 for e in exp]
    return all(abs(g - e) <= t for g, e, t in zip(got, exp, tol))


def main():
    log = open(os.path.join(L.B, "anchors.log"), "w")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    ok_all = True
    out = {}
    for name in L.NAMES:
        t0 = time.time()
        o = {}
        d = D.Data(name)
        ref = np.load(os.path.join(SA, f"_node_{name}.npz"))["X"]
        o["node_cache_equal_design_best_signal"] = bool(np.array_equal(d.X, ref))
        o["factory_equal"] = L.check_factory(d)
        o["n_cells"] = [int(round(x)) for x in d.n]
        z = np.load(os.path.join(D.EDGE_CACHE, f"{name}_edges.npz"))
        ts = z["ts"]
        u, c = np.unique(ts, return_counts=True)
        o["edges_total"] = int(len(ts))
        o["edges_per_bin_mean_snapshot"] = float(np.mean([c[u == b][0] for b in d.bins if (u == b).any()]))
        o["edge_bins"] = u.tolist()
        o["max_index_lt_G"] = bool(max(int(z["u"].max()), int(z["i"].max())) < d.G)
        del z
        e_ok = (abs(o["edges_total"] - EDGES[name]) / EDGES[name] < 0.01 and
                abs(o["edges_per_bin_mean_snapshot"] - PER_BIN[name]) / PER_BIN[name] < 0.03 and o["max_index_lt_G"])
        o["edges_ok"] = bool(e_ok)
        Y = d.Y("l1")
        F, cls = NS.build_features(d, "l1", Y, None, None)
        fb = NS.fallback(d, "l1", Y, F)
        G_ = NS.GROUPS
        full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"]
        st = NS.Stacker(F, cls, Y, full)
        P0, ch0 = NS.run_learned(st, [full], Y, d, fb)
        del st
        graph = NS.load_graph(d)
        Fg, clsg = NS.build_features(d, "l1", Y, graph, None)
        stg = NS.Stacker(Fg, clsg, Y, full + G_["grn"])
        P1, ch1 = NS.run_learned(stg, [full + G_["grn"]], Y, d, fb)
        del stg, graph
        nmk = D.masks(d.G, d.T)
        cmk = L.ctx_masks(d.G, d.T)
        trend = [fb[0]] + [Y[i - 1] for i in range(1, d.T)]
        for lab, P in (("nass_full", P0), ("nass_full+grn", P1), ("trend", trend)):
            S = D.score(Y, P, nmk)
            t2 = D.summarize(S, list(range(2, d.T)))
            o[f"{lab}_own_i>=2"] = (t2["pcc"], t2["pta_up"], t2["pta_down"])
            rows = L.score_ctx(Y, P, cmk)
            for lo in (1, 2):
                s = L.summ(rows, lo)
                o[f"{lab}_ctx_i>={lo}"] = (s["pcc"], s["up"], s["down"])
        o["chosen_nass_full"] = ch0
        checks = {}
        for k, exp in EXPECT[name].items():
            got = o[k][:len(exp)]
            checks[k] = {"expected": exp, "got": got, "pass": close(got, exp)}
        checks["node_cache"] = {"pass": o["node_cache_equal_design_best_signal"]}
        checks["edges"] = {"pass": o["edges_ok"], "got": [o["edges_total"], round(o["edges_per_bin_mean_snapshot"])]}
        o["checks"] = checks
        out[name] = o
        say(f"== {name} ({time.time() - t0:.0f}s)")
        for k in ("nass_full", "nass_full+grn", "trend"):
            say(f"  {k:14s} own i>=2 {o[f'{k}_own_i>=2']} | ctx i>=1 {o[f'{k}_ctx_i>=1']} | ctx i>=2 {o[f'{k}_ctx_i>=2']}")
        for k, v in checks.items():
            say(f"  check {k:24s} {'PASS' if v['pass'] else 'FAIL'} {v.get('got', '')} vs {v.get('expected', '')}")
            ok_all &= bool(v["pass"])
    json.dump(out, open(os.path.join(L.B, "anchors.json"), "w"), indent=1)
    say(f"ANCHORS {'ALL PASS' if ok_all else 'FAILED'}")
    log.close()
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
