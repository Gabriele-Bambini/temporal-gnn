"""JUDGE, EXPLORATORY (descriptive, uses atlas cell-type labels and the held stage): is the held stage's cell-type
composition a mixture of its neighbours'? For every held stage (atlas k = 1..7, erythroid k = 2..5):
TV(held, a prev + (1 - a) next) for a = 0.5 and the best a (grid 0.01), TV(held, prev), TV(held, next), and the held mass
in types with < 10 cells at k-1 or at k+1 (appearing / disappearing types).
Usage: python j_comp.py -> j_comp.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
SETS = {"atlas": (os.path.join(SCR, "atlas"), range(1, 8)), "ery": (os.path.join(SCR, "ery"), range(2, 6))}


def main():
    res = {}
    for name, (data, ks) in SETS.items():
        res[name] = {}
        for k in ks:
            d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
            st, ct, ch = d["stage_fit"], d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
            types = sorted(set(ct) | set(ch))
            cnt = lambda lab: np.array([(lab == t).sum() for t in types], float)  # noqa: E731
            cp, cn, chh = cnt(ct[st == k - 1]), cnt(ct[st == k + 1]), cnt(ch)
            p, n, h = cp / cp.sum(), cn / cn.sum(), chh / chh.sum()
            tv = lambda a, b: float(0.5 * np.abs(a - b).sum())  # noqa: E731
            grid = np.round(np.arange(0, 1.0001, 0.01), 2)
            tvs = [tv(h, a * p + (1 - a) * n) for a in grid]
            ba = float(grid[int(np.argmin(tvs))])
            one_sided = (cp < 10) | (cn < 10)
            res[name][str(k)] = {"TV_held_mix05": tv(h, 0.5 * p + 0.5 * n), "TV_held_bestmix": float(min(tvs)),
                                 "best_alpha_on_prev": ba, "TV_held_prev": tv(h, p), "TV_held_next": tv(h, n),
                                 "TV_prev_next": tv(p, n), "held_mass_in_one_sided_types": float(h[one_sided].sum()),
                                 "mix05_closer_than_nearer_neighbour": tv(h, 0.5 * p + 0.5 * n) < min(tv(h, p), tv(h, n))}
            print(name, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in res[name][str(k)].items()})
        res[name + "_n_mix05_closer"] = sum(v["mix05_closer_than_nearer_neighbour"] for v in res[name].values())
    print({k: v for k, v in res.items() if k.endswith("closer")})
    json.dump(res, open(os.path.join(HERE, "j_comp.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
