"""Judge: strictness tests for the population-sink prototype (uot_pop_sink/uot_sink.py, judge copy in p1/).

(A) Future scramble, per transition: for transition i, every bin > i (m, f, n and everything derived) is replaced by
    random values; the predictions of ALL 300 configs at transition i (targets l1, linear, orig) must be bit-identical
    to the unperturbed ones. The oracle composition (reads bin i+1) must change (positive control).
(B) End-to-end truncation: U.main on the data cut to bins 0..k (k = CUT), written to a temp folder; the per-transition
    test scores of every selected group at transitions 1..k-1 must equal those of the full judge rerun (selection,
    windows, fallbacks included).
Usage: python strict_p1.py <dataset> [CUT]   ->  strict_p1_<dataset>.json
"""
import copy
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "p1")
sys.path.insert(0, P1)
import uot_sink as U  # noqa: E402


def derive(d):
    d.S = d.m * d.n[None, :]
    d.lib = d.m.sum(0)
    d.CP = 1e4 * d.m / d.lib[None, :]
    d.Lg = np.log1p(d.CP)


def scrambled(d, i, seed):
    d2 = copy.copy(d)
    rng = np.random.default_rng(seed)
    d2.m, d2.f, d2.n = d.m.copy(), d.f.copy(), d.n.copy()
    nb = d.B - (i + 1)
    d2.m[:, i + 1:] = rng.gamma(0.2, 2.0, size=(d.G, nb))
    d2.f[:, i + 1:] = np.minimum(rng.random((d.G, nb)), d2.m[:, i + 1:])
    d2.n[i + 1:] = rng.uniform(5, 1500, size=nb)
    derive(d2)
    return d2


def preds_at(d, i, cfgs):
    fits = {K: U.Fit(d, i, min(K, i)) for K in U.KS}
    out = np.zeros((len(cfgs), len(U.TARGETS), d.G))
    comp = {}
    for r, cfg in enumerate(cfgs):
        fam, K, L, hp = cfg
        ck = U.comp_key(cfg)
        if ck not in comp:
            comp[ck] = U.pooled_forecast(fits[K], fam, hp)
        mh, fh = U.map_back(fits[K], comp[ck][0], L)
        for a, t in enumerate(U.TARGETS):
            out[r, a] = U.target_pred(d, i, mh, fh, t)
    orc = np.concatenate([fits[K].oracle() for K in U.KS])
    return out, orc


class Trunc(U.Data):
    CUT = None

    def __init__(self, name):
        super().__init__(name)
        k = Trunc.CUT + 1
        self.m, self.f, self.n = self.m[:, :k], self.f[:, :k], self.n[:k]
        self.B, self.T = k, k - 1
        derive(self)


def main():
    name = sys.argv[1]
    cut = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = U.Data(name)
    cfgs = U.build_configs()
    res = {"dataset": name, "A_future_scramble": {}, "B_truncation": {}}
    test_is = [2, 5, 9, 12] if d.T > 13 else [2, 5, 9]
    if d.T > 20:
        test_is = [2, 7, 15, 24]
    for i in test_is:
        p0, o0 = preds_at(d, i, cfgs)
        p1, o1 = preds_at(scrambled(d, i, 100 + i), i, cfgs)
        res["A_future_scramble"][str(i)] = {"max_abs_diff_all_configs_targets": float(np.abs(p0 - p1).max()),
                                            "oracle_max_abs_diff_positive_control": float(np.abs(o0 - o1).max())}
        print(name, "scramble i", i, res["A_future_scramble"][str(i)], flush=True)
    if "noB" in sys.argv:
        with open(os.path.join(HERE, f"strict_p1_{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        return
    # (B) truncation end-to-end
    tmp =os.path.join(HERE, f"_trunc_{name}_{cut}")
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(os.path.join(P1, f"_node_{name}.npz"), tmp)
    full = np.load(os.path.join(P1, f"uot_sink_{name}_scores.npz"))
    Trunc.CUT = cut
    U.Data = Trunc
    U.HERE = tmp
    U.main(name)
    tr = np.load(os.path.join(tmp, f"uot_sink_{name}_scores.npz"))
    worst = 0.0
    nkeys = 0
    for k in tr.files:
        if not k.startswith("S_"):
            continue
        a, b = tr[k][1:cut], full[k][1:cut]
        dd = float(np.nanmax(np.abs(a - b)))
        worst = max(worst, dd)
        nkeys += 1
    jw = max(float(np.nanmax(np.abs(tr[f"J_{t}"][:, 1:cut] - full[f"J_{t}"][:, 1:cut]))) for t in U.TARGETS)
    res["B_truncation"] = {"cut_last_bin": cut, "transitions_compared": f"1..{cut - 1}", "score_arrays_compared": nkeys,
                           "max_abs_diff_scores": worst, "max_abs_diff_J_all_configs": jw}
    print(name, "truncation", res["B_truncation"], flush=True)
    os.remove(os.path.join(tmp, f"_node_{name}.npz"))
    with open(os.path.join(HERE, f"strict_p1_{name}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
