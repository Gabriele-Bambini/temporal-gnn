"""How precisely must detection (alpha) be known? Decoder-oracle world of the precision study (base configuration,
SELF truth, c = 17.86 h/unit): the labels are generated with the true alpha and scored with alpha x s for
s in 0.6 ... 1.6; for each s we record the label clock c_label(s). The slope d log c / d log s says how a
detection error propagates into the hour scale.

Usage: python alpha_scan.py [--quick]
"""
import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

import precision_study as ps  # noqa: E402
import scifate2_clock_probe as probe  # noqa: E402
import sim_recovery as sr  # noqa: E402

SCALES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from Chronocell.models import two_species_ss as model

    real = json.load(open(os.path.join(sr.OUT_REAL, "clock_probe.json")))
    dec = np.load(os.path.join(sr.OUT_REAL, "decoder.npz"), allow_pickle=True)
    theta, tau = dec["theta"], dec["tau"]
    c_true = real["clocks"]["c_A"]
    obs, tot, new, U, S, genes, amb = probe.load(False)
    groups = (obs["timepoint"].astype(str) + "_" + obs["rep"].astype(str)).to_numpy()
    Tcell = obs["timepoint"].astype(str).map(probe.GROUP_T).to_numpy(float)
    keep, alpha, _ = probe.qc(obs, tot, genes)
    fit, score = probe.split(groups, keep, 1000)
    r, gidx, _ = probe.depth_and_genes(U, S, fit, genes, amb, 150)
    del tot, new, U, S
    ambg = np.nan_to_num(amb[gidx], nan=0.0)
    Tc, gc, ac, rc = Tcell[score], groups[score], alpha[score], r[score]
    if a.quick:
        sub = np.random.default_rng(3).random(len(Tc)) < 0.2
        Tc, gc, ac, rc = Tc[sub], gc[sub], ac[sub], rc[sub]
    Uc, Sc, totc, labels = sr.simulate(model, theta, np.ones(len(theta)), tau, real["clocks"]["age"]["theta"],
                                       [c_true], Tc, rc, ac, ambg, np.random.default_rng(100),
                                       np.random.default_rng(201))
    t_grid = np.linspace(tau[0], tau[-1], probe.M)
    lf = model.get_logL(np.stack([Uc, Sc], axis=-1), theta, t_grid, tau, probe.TOPO, {"r": rc})
    q = np.exp(lf - logsumexp(lf, axis=(1, 2), keepdims=True))
    del lf
    res = {}
    for s in SCALES:
        ll = ps.per_cell_ll(model, theta, tau, q, labels[0], totc, np.clip(s * ac, 1e-6, None), ambg)
        ug = np.unique(gc)
        Sprof = np.stack([ll[gc == g].mean(0) for g in ug]).mean(0)
        am = int(np.argmax(Sprof))
        res[str(s)] = dict(c_label=float(probe.C_GRID[am]), at_edge=bool(am in (0, len(Sprof) - 1)))
        print(s, res[str(s)], flush=True)
    ok = [(np.log(float(k)), np.log(v["c_label"])) for k, v in res.items() if not v["at_edge"]]
    x, y = np.array(ok).T
    slope = float(np.polyfit(x, y, 1)[0])
    out = dict(c_true=c_true, scan=res, slope_dlogc_dlogalpha=slope,
               note="decoder-oracle world; labels generated with the true alpha, scored with alpha x s")
    od = os.path.expanduser(f"~/tack/out/precision_study{'_quick' if a.quick else ''}")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "alpha_scan.json"), "w"), indent=1)
    print("SLOPE", slope, flush=True)


if __name__ == "__main__":
    main()
