"""EXPLORATORY + ORACLE-LABELLED (looks at the held stage; can never change a verdict): why does the 50/50 union beat
the OT pair interpolant on the atlas? Uses the frozen rep subsamples (IB.sub seeds 1000 + r / 2000 + r), the frozen
target (held subsample, seed 1) and IB.plan / IB.draw_pairs / IB.ed_uv unchanged; only t, the union mixing weight
alpha and eps_rel vary. Also: spread (RMS radius about the weighted mean) of each prediction relative to the held
stage, and the per-pool comparison ot_pairs vs union from confirm.json (pool_mean). Output: audit/explore.json.
"""
import numpy as np

from _common import ATLAS, ERY, IB, confirm, dump, load

TS = (0.25, 0.5, 0.75)
ALPHAS = (0.25, 0.5, 0.75)
EPS = (0.01, 0.05, 0.2)


def spread(Z, w=None):
    w = np.full(len(Z), 1.0 / len(Z)) if w is None else w / w.sum()
    mu = w @ Z
    return float(np.sqrt(w @ ((Z - mu) ** 2).sum(1)))


def mix(ZA, ZB, alpha, n, rng):
    mB = int(round(alpha * n))
    iA = rng.choice(len(ZA), n - mB, replace=n - mB > len(ZA))
    iB = rng.choice(len(ZB), mB, replace=mB > len(ZB))
    return np.vstack([ZA[iA], ZB[iB]])


def case(folder, k, reps=3):
    d = load(folder, f"interp{k}.npz")
    st = d["stage_fit"]
    Zf, Zh = d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], None, IB.med_scale(Zf), with_sd=False)
    r_held = spread(Zh[it])
    acc = {}

    def add(key, v):
        acc.setdefault(key, []).append(v)

    for r in range(reps):
        rng = np.random.default_rng(555 + 1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, ZB = Zf[A[iA]], Zf[B[iB]]
        add("spread/prev", spread(ZA) / r_held)
        add("spread/next", spread(ZB) / r_held)
        for e in EPS:
            P, _ = IB.plan(ZA, ZB, e)
            i, j = IB.draw_pairs(P, n_pred, rng)
            for t in (TS if e == 0.05 else (0.5,)):
                Zp = (1 - t) * ZA[i] + t * ZB[j]
                add(f"ot_pairs/eps{e}/t{t}", IB.ed_uv(Zp, IB.unif(n_pred), tgt)[0])
                if t == 0.5:
                    add(f"spread/ot_pairs_eps{e}", spread(Zp) / r_held)
        ii, jj = rng.integers(0, len(ZA), n_pred), rng.integers(0, len(ZB), n_pred)
        add("spread/indep", spread(0.5 * ZA[ii] + 0.5 * ZB[jj]) / r_held)
        for a in ALPHAS:
            Zu = mix(ZA, ZB, a, n_pred, rng)
            add(f"union/alpha{a}", IB.ed_uv(Zu, IB.unif(n_pred), tgt)[0])
            if a == 0.5:
                add("spread/union", spread(Zu) / r_held)
    return {kk: float(np.mean(v)) for kk, v in acc.items()}


def main():
    cj = confirm()
    out = {"atlas": {}, "ery": {}, "pools_ot_pairs_lt_union": {}}
    for k in range(1, 8):
        out["atlas"][k] = case(ATLAS, k)
        print("atlas", k, {kk: round(v, 3) for kk, v in out["atlas"][k].items()}, flush=True)
        pm = cj["interp"][str(k)]["pool_mean"]
        out["pools_ot_pairs_lt_union"][k] = {p: pm["ot_pairs"][p] < pm["union"][p] for p in pm["union"]}
    for k in range(2, 6):
        out["ery"][k] = case(ERY, k)
        print("ery", k, {kk: round(v, 3) for kk, v in out["ery"][k].items()}, flush=True)
    won = [v for d in out["pools_ot_pairs_lt_union"].values() for v in d.values()]
    out["pools_ot_pairs_lt_union_total"] = f"{sum(won)}/{len(won)}"
    print("pools where ot_pairs < union:", out["pools_ot_pairs_lt_union_total"])
    dump("explore.json", out)


if __name__ == "__main__":
    main()
