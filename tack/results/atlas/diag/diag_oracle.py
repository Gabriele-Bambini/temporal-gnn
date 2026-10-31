"""EXPLORATORY diagnosis (not pre-registered): is the OT failure a question, data or architecture problem?

Part A, ceiling decomposition of interpolation, cross-fitted by BIOLOGICAL pool (meta 'pool'; samples of one pool are
technical replicates). For held stage k and pool p: target T = held cells of p, fit set F = held cells of the other pools
(random halves when the stage has one pool). All numbers are ED_U (PCA space of the product) against T:
  floor   F itself (another real replicate of the same stage: the best anything can do from outside the pool)
  union   50/50 mix of prev/next          ot      McCann OT pairs (frozen plan, eps_rel 0.05)
  mix*    best alpha-mixture of prev/next fitted on F
  W*      best simplex reweighting of prev+next cells fitted on F  (ORACLE bound for any 'reweight observed cells' model)
  M*      same over prev+next+McCann points                        (ORACLE bound for 'reweight + transport' models)
Gap union - floor = (union - W*) weighting/composition + (W* - M*) transport adds states + (M* - floor) not reachable
from the neighbour stages (new states, batch, pool effects).
Part B, biological clock of each pool: leave-one-pool-out multinomial logistic regression of stage on the 30 PCs of
origin7 (common space for all 9 stages), mean expected stage per pool; spread within nominal stage; batch.
Part C, per-pool oracle mixture weight alpha_p (fitted on the pool itself): asynchrony inside the interpolation task.
Writes diag_oracle.json next to this file.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = os.environ.get("DIAG_DATA", r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad")
# sample -> biological pool, from ~/rafm_data/pijuan_sala2019/meta.tab.gz (columns sample, pool)
POOL = {1: 1, 5: 5, 18: 15, 7: 7, 10: 9, 14: 12, 15: 12, 30: 23, 31: 23, 32: 23, 23: 18, 26: 20, 27: 20, 2: 2, 3: 3,
        4: 4, 6: 6, 19: 16, 20: 16, 8: 8, 9: 8, 12: 11, 13: 11, 16: 13, 33: 24, 34: 24, 35: 24, 24: 19, 25: 19, 28: 21,
        17: 14, 29: 22, 36: 25, 37: 26, 21: 17, 22: 17}
BATCH = {1: 1, 5: 1, 15: 2, 7: 1, 9: 1, 12: 2, 23: 3, 18: 2, 20: 2, 2: 1, 3: 1, 4: 1, 6: 1, 16: 2, 8: 1, 11: 2, 13: 2,
         24: 3, 19: 2, 21: 2, 14: 2, 22: 3, 25: 3, 26: 3}
MIN_POOL, NC, ITERS = 100, 1000, 800
ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)


def bio(samples):
    return np.array([POOL[int(s)] for s in samples])


def ed(P, w, tgt):
    return IB.ed_uv(P, w, tgt)


def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def oracle_weights(C, Y):
    """argmin over the simplex of ED_V(sum_i w_i delta_{C_i}, uniform Y) = 2 w.a - w'Dw + const (convex, D is CND)."""
    a = np.sqrt(IB.sqdist(C, Y)).mean(1)
    D = np.sqrt(IB.sqdist(C, C))
    x = np.ones(len(C)) / len(C)
    for _ in range(50):
        x = D @ x
        x /= np.linalg.norm(x)
    L = 2.0 * float(x @ D @ x)
    w = np.full(len(C), 1.0 / len(C))
    y, t = w.copy(), 1.0
    for _ in range(ITERS):
        w_new = proj_simplex(y - (2 * a - 2 * D @ y) / L)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = w_new + (t - 1) / t_new * (w_new - w)
        w, t = w_new, t_new
    return w


def wscore(C, w, tgt):
    keep = w > 1e-12
    eu, ev = ed(C[keep], w[keep] / w[keep].sum(), tgt)
    return {"ED_U": eu, "ED_V": ev, "ess": float(1.0 / (w[keep] ** 2).sum())}


def mixture(ZA, ZB, a):
    w = np.r_[np.full(len(ZA), a / len(ZA)), np.full(len(ZB), (1 - a) / len(ZB))]
    return np.vstack([ZA, ZB]), w


def best_alpha(ZA, ZB, tgt):
    vals = {}
    for a in ALPHAS:
        P, w = mixture(ZA, ZB, a)
        keep = w > 0
        vals[float(a)] = ed(P[keep], w[keep] / w[keep].sum(), tgt)[0]
    a = min(vals, key=vals.get)
    return a, vals[a]


def part_a(name, k):
    d = np.load(os.path.join(SCR, name, f"interp{k}.npz"), allow_pickle=True)
    st, Zf, Zh = d["stage_fit"], d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
    hp = bio(d["sample_held"])
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    S = IB.med_scale(Zf)
    ZA, ZB = Zf[A[IB.sub(len(A), NC, seed=11)]], Zf[B[IB.sub(len(B), NC, seed=12)]]
    P, _ = IB.plan(Zf[A[IB.sub(len(A), seed=1000)]], Zf[B[IB.sub(len(B), seed=2000)]], 0.05)
    a_sub, b_sub = Zf[A[IB.sub(len(A), seed=1000)]], Zf[B[IB.sub(len(B), seed=2000)]]
    i, j = IB.draw_pairs(P, 1500, np.random.default_rng(7))
    ZM = 0.5 * a_sub[i] + 0.5 * b_sub[j]
    groups = [p for p in sorted(set(hp)) if (hp == p).sum() >= MIN_POOL]
    if len(groups) >= 2:
        splits = {str(p): (np.where(hp == p)[0], np.where(hp != p)[0]) for p in groups}
    else:
        perm = np.random.default_rng(13).permutation(len(Zh))
        h1, h2 = np.sort(perm[: len(Zh) // 2]), np.sort(perm[len(Zh) // 2:])
        splits = {"half1": (h1, h2), "half2": (h2, h1)}
    union_P, union_w = mixture(ZA, ZB, 0.5)
    CW, CM = np.vstack([ZA, ZB]), np.vstack([ZA, ZB, ZM])
    res = {"k": k, "groups": {}}
    for g, (it_, if_) in splits.items():
        T = IB.Target(Zh[it_[IB.sub(len(it_), seed=3)]], None, S, with_sd=False)
        Fc = Zh[if_[IB.sub(len(if_), seed=4)]]
        Fu = IB.unif(len(Fc))
        aF, _ = best_alpha(ZA, ZB, IB.Target(Fc, None, S, with_sd=False))
        mP, mw = mixture(ZA, ZB, aF)
        keep = mw > 0
        a_own, ed_own = best_alpha(ZA, ZB, T)
        wW, wM = oracle_weights(CW, Fc), oracle_weights(CM, Fc)
        r = {"n": int(len(it_)), "batch": BATCH.get(int(g), None) if g.isdigit() else None,
             "floor": ed(Fc, Fu, T)[0], "union": ed(union_P, union_w, T)[0],
             "ot": ed(ZM, IB.unif(len(ZM)), T)[0],
             "mix_fit": {"alpha_prev": aF, "ED_U": ed(mP[keep], mw[keep] / mw[keep].sum(), T)[0]},
             "W": wscore(CW, wW, T), "M": wscore(CM, wM, T),
             "W_insample": wscore(CW, wW, IB.Target(Fc, None, S, with_sd=False))["ED_U"],
             "M_mass_on_transport": float(wM[2 * NC:].sum()),
             "W_mass_on_prev": float(wW[:NC].sum()),
             "alpha_own": {"alpha_prev": a_own, "ED_U": ed_own}}
        res["groups"][g] = r
    keys = ("floor", "union", "ot")
    res["mean"] = {kk: float(np.mean([v[kk] for v in res["groups"].values()])) for kk in keys}
    res["mean"].update({"mix_fit": float(np.mean([v["mix_fit"]["ED_U"] for v in res["groups"].values()])),
                        "W": float(np.mean([v["W"]["ED_U"] for v in res["groups"].values()])),
                        "M": float(np.mean([v["M"]["ED_U"] for v in res["groups"].values()]))})
    m = res["mean"]
    gap = m["union"] - m["floor"]
    res["decomp"] = {"gap_union_floor": gap, "weighting": m["union"] - m["W"], "transport": m["W"] - m["M"],
                     "unreachable": m["M"] - m["floor"]}
    return res


def part_b():
    d = np.load(os.path.join(SCR, "atlas", "origin7.npz"), allow_pickle=True)
    Z = np.vstack([d["Z_past"], d["Z_fut"]]).astype(np.float64)
    st = np.r_[d["stage_past"], d["stage_fut"]]
    pl = bio(np.r_[d["sample_past"], d["sample_fut"]])
    Zs = StandardScaler().fit_transform(Z)
    out = {}
    for p in sorted(set(pl)):
        tr, te = pl != p, pl == p
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(Zs[tr], st[tr])
        age = clf.predict_proba(Zs[te]) @ clf.classes_
        out[int(p)] = {"stage": int(np.bincount(st[te]).argmax()), "batch": BATCH.get(int(p)), "n": int(te.sum()),
                       "age_mean": float(age.mean()), "age_sd": float(age.std())}
        print("pool", p, out[int(p)], flush=True)
    nom = np.array([v["stage"] for v in out.values()])
    age = np.array([v["age_mean"] for v in out.values()])
    summ = {"spearman": float(spearmanr(nom, age)[0]),
            "mean_abs_err_stage_units": float(np.abs(age - nom).mean()),
            "within_stage_range": {int(s): float(np.ptp(age[nom == s])) for s in sorted(set(nom)) if (nom == s).sum() > 1},
            "closer_to_other_stage": int(sum(int(np.argmin(np.abs(np.arange(9) - a))) != s for s, a in zip(nom, age)))}
    return {"pools": out, "summary": summ}


def main():
    out = {"A": {"atlas": {}, "ery": {}}}
    for name, ks in (("atlas", range(1, 8)), ("ery", range(2, 6))):
        for k in ks:
            r = part_a(name, k)
            out["A"][name][k] = r
            print(name, k, {kk: round(v, 3) for kk, v in r["mean"].items()}, {kk: round(v, 3) for kk, v in r["decomp"].items()}, flush=True)
    out["B"] = part_b()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_oracle.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
