"""Anchor checks: cells per bin vs CONTEXT, trend on l1 (i >= 1) on the 1000 s + i masks vs CONTEXT, and the
calibration of the declared l1 noise R against the CONTEXT noise budget (gastr 0.37 / 0.42 / 0.21, pancr
0.20 / 0.24 / 0.57). Variants of R: floor 1/n^2 (declared), no floor, Poisson floor (max(s^2, running mean) + 1/n)."""
import json
import os

import numpy as np

import common as C

CTX_CELLS = {"gastrulation-human": [9, 182, 792, 929, 982, 616, 340, 217, 225, 248, 290, 304, 431, 440, 333, 247, 246,
                                    230, 275, 316, 355, 350, 264, 233, 228, 306, 231, 138, 46, 7],
             "pancreas-human": [111, 612, 361, 116, 50, 53, 68, 79, 191, 218, 132, 286, 698, 616, 105]}


def budget(Y, R):
    T = len(Y)
    v = np.array([Y[i].var() for i in range(T)])
    a = np.array([R[:, i].mean() for i in range(T)]) / v
    b = np.array([R[:, i + 1].mean() for i in range(T)]) / v
    return a, b, 1 - a - b, v


def main():
    out = {}
    for name in C.NAMES:
        d = C.load(name)
        o = {"cells_match": [int(round(x)) for x in d.n] == CTX_CELLS[name]}
        Y = d.Y("l1")
        mk = C.masks(d.G, d.T)
        P = [None] + [Y[i - 1] for i in range(1, d.T)]
        rows = C.score(Y, P, mk)
        o["trend_l1_i>=1"] = C.summ(rows, range(1, d.T))
        o["trend_l1_i>=2"] = C.summ(rows, range(2, d.T))
        kap = 1e4 / d.lib
        nb = d.n[None, :]
        w = np.cumsum(d.n)
        mbar = np.cumsum(d.m * d.n[None, :], 1) / w[None, :]
        variants = {
            "floor_1/n^2": C.R_l1(d, True),
            "no_floor": C.R_l1(d, False),
            "poisson_floor": ((np.maximum(d.s ** 2, mbar) + 1 / nb) / np.maximum(nb - 1, 1)) * (kap[None, :] / (1 + d.CP)) ** 2,
        }
        for k, R in variants.items():
            a, b, sh, v = budget(Y, R)
            o[f"budget[{k}]"] = {"mean_over_transitions": [round(float(x.mean()), 3) for x in (a, b, sh)],
                                 "ratio_of_sums": [round(float((a * v).sum() / v.sum()), 3),
                                                   round(float((b * v).sum() / v.sum()), 3),
                                                   round(float((sh * v).sum() / v.sum()), 3)]}
        out[name] = o
        print(name, json.dumps(o, indent=1))
    json.dump(out, open(os.path.join(C.HERE, "s0_anchor.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
