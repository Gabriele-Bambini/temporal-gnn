"""Judge 2: frozen NASS (judge2 copy of nass.py, no change to any feature or grid) on builds it was NOT designed on,
and on the '-human' builds with a different mask seed.
Held-out builds: Dileo's mouse builds 'gastrulation' (same 9815 cells, DIFFERENT binning: 35 bins, snapshots 2..34 as in
its y_outdeg header; 6345 genes) and 'pancreas' (same 3696 cells and bins, 7197 mouse genes, snapshots 1..12).
No GRN / OT add-ons (no edge cache for these builds). Strictness is inherited from nass.py (truncation-tested).
Usage: python nass_heldout.py <dataset> [mask_seed]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dsdata as D
import nass as N

D.SNAPS.update({"gastrulation": list(range(2, 35)), "pancreas": list(range(1, 13)),
                "gastr-merged2-off0": list(range(15)), "gastr-merged2-off1": list(range(14))})


def main(name, seed):
    t0 = time.time()
    d = D.Data(name)
    mk = D.masks(d.G, d.T, seed=seed)
    ranges = {"all": list(range(d.T)), "i>=2": list(range(2, d.T)), "i>=3": list(range(3, d.T)),
              "i>=2_bins>=100": [i for i in range(2, d.T) if min(d.n[i], d.n[i + 1]) >= 100]}
    res = {"dataset": name, "mask_seed": seed, "n_cells": d.n.round().tolist(), "G": d.G, "T": d.T, "targets": {}}
    for tg in D.TARGETS:
        Y = d.Y(tg)
        F, cls = N.build_features(d, tg, Y, None, None)
        fb = N.fallback(d, tg, Y, F)
        G_ = N.GROUPS
        full = G_["lag"] + G_["rev"] + G_["kal"] + G_["snr"] + G_["cg"] + (G_["obs"] if tg == "orig" else [])
        regime = G_["lag"] + ["rev_box4", "rev_box8"]
        st = N.Stacker(F, cls, Y, full + [k for k in regime if k not in full])
        minus = lambda grp: [k for k in full if k not in G_[grp]]
        preds = {"trend": [fb[0]] + [Y[i - 1] for i in range(1, d.T)]}
        if tg == "orig":
            preds["level(box4+obs)"] = [F["obs_box4"][i] for i in range(d.T)]
        else:
            preds["level(box4)"] = [fb[0]] + [F["rev_box4"][i] for i in range(1, d.T)]
        variants = {"regime": [regime], "full": [full], "full-kal": [minus("kal")], "full-cg": [minus("cg")]}
        if tg == "orig":
            variants["full-obs"] = [minus("obs")]
        for v, subs in variants.items():
            preds[v], _ = N.run_learned(st, subs, Y, d, fb)
        fx = [N.CONFIGS.index((0.5, 1e-3, 0))]
        preds["regime[h=0.5 fixed]"], _ = N.run_learned(st, [regime], Y, d, fb, cfg_idx=fx)
        preds["full[h=0.5 fixed]"], _ = N.run_learned(st, [full], Y, d, fb, cfg_idx=fx)
        preds["auto(full|trend)"], _ = N.run_learned(st, [full], Y, d, fb, extra=[preds["trend"]])
        S = {k: D.score(Y, v, mk) for k, v in preds.items()}
        tab = {k: {r: D.summarize(S[k], idx) for r, idx in ranges.items()} for k in S}
        pr = {}
        lvl = "level(box4+obs)" if tg == "orig" else "level(box4)"
        for a, b in (("full", "trend"), ("full", "regime"), ("full", lvl), ("full", "regime[h=0.5 fixed]"),
                     ("full", "full-kal"), ("full", "full-cg"), ("full", "full-obs"), ("auto(full|trend)", "trend")):
            if a in S and b in S:
                pr[f"{a} vs {b}"] = N.paired(S[a], S[b], ranges["i>=2"])
        res["targets"][tg] = {"table": tab, "paired_i>=2": pr,
                              "per_transition": {k: np.round(np.nanmean(S[k], 0)[:, :3], 4).tolist() for k in S}}
        print(f"== {name} seed {seed} {tg} ({time.time() - t0:.0f}s)", flush=True)
        for k in S:
            r2, ra = tab[k]["i>=2"], tab[k]["all"]
            print(f"  {k:22s} i>=2 {r2['pcc']:.3f}/{r2['p200_up']:.3f}/{r2['p200_down']:.3f} (sd {r2['pcc_sd']:.3f})"
                  f" | all {ra['pcc']:.3f}/{ra['p200_up']:.3f}/{ra['p200_down']:.3f}", flush=True)
        for k, v in pr.items():
            print(f"  paired {k:32s} dPCC {v['pcc']['mean_delta']:+.3f} ({v['pcc']['better']}/{v['pcc']['worse']},"
                  f" p={v['pcc']['wilcoxon_p']}) dJ {v['J']['mean_delta']:+.3f} ({v['J']['better']}/{v['J']['worse']},"
                  f" p={v['J']['wilcoxon_p']})", flush=True)
    json.dump(res, open(os.path.join(HERE, f"heldout_{name}_seed{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20261022)
