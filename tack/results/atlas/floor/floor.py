"""EXPLORATORY (not pre-registered): noise floor and signal size behind the atlas confirmation verdict.

The pre-registered verdict (tack/results/atlas/confirm.json: H1 / H1b / H2 / H2b FAIL, H1c PASS) is final; nothing
computed here changes it. Question: on the atlas, is the 50/50 union already at (or near) the floor set by embryo-pool
variation, so that no interpolant can beat it, or is there room that OT fails to use? Same computation on the
erythroid products, to explain why the erythroid study looked positive.

Conventions (frozen, imported read-only from tack/hpc): IB.ed_uv (ED_U = energy distance, U-statistic, weighted),
IB.Target (with_sd=False: ED_U never uses S, the Sinkhorn scale; S = IB.med_scale(fit or past cells) is passed only
for fidelity), IB.sub (seeded subsample <= 1500), pools = embryo samples with >= MIN_POOL (20) cells, PCA space of
each product. The stage target is exactly the confirm.json one: Z_held[IB.sub(n, seed=1)] (interp) /
Z_fut[IB.sub(n, seed=1)] (forecast); per-pool targets are the full pools (as confirm.json pool_ED_U).

Per held stage k (interp<k>.npz) and per forecast target s+1 (origin<s>.npz):
  halves       ED_U between two disjoint random halves of the target stage (frozen seed-4 scheme of
               interp_bench_frozen.py + 20 more seeds). SANITY CHECK ONLY: it is ~0 by construction (same embryos on
               both sides, unbiased U-statistic) and is NOT an achievable floor for a prediction made from other
               embryos. Its SD is the sampling noise of ED_U at these sizes.
  complement   ED_U(held cells NOT in the confirm target, confirm target): same, against the exact target.
  lopo         leave-one-pool-out: ED_U(other cells of the target stage (subsample seed 5), full pool p) = score of a
               real same-stage replicate from other embryos against pool p (frozen own_loso convention). Directly
               comparable with confirm.json pool_ED_U[p] (same target object).
  pair / tau2  ED_U between every two pools of one stage; tau2 = mean / 2 (diag_batch.py convention). Because ED is a
               squared Hilbert distance (2 x MMD^2, distance kernel), with exchangeable pool offsets of variance tau2:
                 E ED(population, mixture with pool shares w) = tau2 * sum w^2              ('oracle')
               = the smallest expected ED_U of ANY prediction that cannot know the held embryos' idiosyncrasies.
  split        ED_U between two disjoint groups of whole pools (every bipartition; 'balanced' = pool counts differ
               by <= 1).
  noise_union  oracle_k + 1/4 tau2_{k-1} sum w_{k-1}^2 + 1/4 tau2_{k+1} sum w_{k+1}^2 = expected ED_U of the 50/50
               union that comes ONLY from pool idiosyncrasies of the three stages (model-based; it assumes pool
               offsets independent across stages - shared sequencing batches would violate it). union - noise_union
               = the systematic (stage-level) error of the union: the most an interpolant built from the same
               source embryos could remove.
  mixture      Hilbert identity ED(1/2 P + 1/2 N, H) = 1/2 ED(P, H) + 1/2 ED(N, H) - 1/4 ED(P, N), checked against
               the union's measured ED_U; best mixture weight alpha* = (c + b - a) / (2c) on [0, 1] (uses the held
               stage: oracle-tuned, descriptive) and its ED_U = how close the best prev/next mixture gets.
  stage size   ED_U(prev, next) (rep subsamples seeds 1000 + r / 2000 + r), and pool-level
               ratio_cross_within = mean cross-stage pool-pair ED_U / mean within-stage pool-pair ED_U (diag_batch.py
               line 289: the erythroid '1.2-2.1x' number).
Forecast: persist recomputed with atlas_confirm.Problem (must equal confirm.json), ED_U(stage s-1, stage s),
noise_persist = oracle_{s+1} + tau2_s sum w_s^2, persist - noise_persist = the real stage change above pool noise.
Missing tau2 (a stage with < 2 usable pools: atlas E6.75) is BORROWED as the mean of the other stages' tau2 in the
same product, and flagged.
Validation: erythroid own_halves / own_loso of tack/results/ery/attempt_interp/results.json reproduced; confirm.json
(atlas) and parity_erythroid.json (erythroid) interpolation numbers reproduced by re-running atlas_confirm.run_interp
locally (--repro), persist reproduced for every origin.
Usage: python floor.py [--atlas DIR] [--ery DIR] [--no-repro]   (writes floor.json next to this file)
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402  (frozen, read-only)
import atlas_confirm as AC  # noqa: E402  (frozen, read-only)

SCR = (r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN"
       r"\264f6820-d374-441d-a750-fb2a7c436807\scratchpad")
ATLAS_STAGES = ["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
ERY_STAGES = IB.STAGES
MIN_POOL = AC.MIN_POOL
HALF_SEEDS = list(range(100, 120))
METHODS = AC.INTERP_METHODS


def f64(a):
    return np.asarray(a, np.float64)


def tgt(Z, S):
    return IB.Target(f64(Z), None, S, with_sd=False)


def edu(P, t):
    P = f64(P)
    return float(IB.ed_uv(P, IB.unif(len(P)), t)[0])


def cap(idx, seed):
    return idx[IB.sub(len(idx), seed=seed)]


def mean_or_none(v):
    v = list(v)
    return float(np.mean(v)) if v else None


class Block:
    """Pools of one stage inside one product space (indices into a combined array Z)."""

    def __init__(self, Z, smp, idx, S, w_idx=None):
        lab = smp[idx]
        self.n = int(len(idx))
        self.pools = [p for p in sorted(set(lab)) if (lab == p).sum() >= MIN_POOL]
        self.P = {p: idx[lab == p] for p in self.pools}
        self.tg = {p: tgt(Z[self.P[p]], S) for p in self.pools}           # full pool, as confirm.json ptg
        wl = smp[idx if w_idx is None else w_idx]
        u, c = np.unique(wl, return_counts=True)
        self.shares = {str(a): float(b / c.sum()) for a, b in zip(u, c)}
        self.sum_w2 = float(((c / c.sum()) ** 2).sum())
        self.pairs = {f"{p}|{q}": edu(Z[cap(self.P[p], 5)], self.tg[q]) for p, q in itertools.combinations(self.pools, 2)}
        self.pair_mean = mean_or_none(self.pairs.values())
        self.tau2 = None if self.pair_mean is None else self.pair_mean / 2
        self.tau2_borrowed = False

    def summary(self):
        return {"n": self.n, "pools_usable": self.pools, "pool_sizes": {p: int(len(v)) for p, v in self.P.items()},
                "shares": self.shares, "sum_w2": self.sum_w2, "pair_ED_U": self.pairs, "pair_mean_ED_U": self.pair_mean,
                "pair_min_ED_U": min(self.pairs.values()) if self.pairs else None,
                "pair_max_ED_U": max(self.pairs.values()) if self.pairs else None,
                "tau2": self.tau2, "tau2_borrowed": self.tau2_borrowed}


def borrow(blocks):
    have = [b.tau2 for b in blocks.values() if b.tau2 is not None and not b.tau2_borrowed]
    for b in blocks.values():
        if b.tau2 is None:
            b.tau2, b.tau2_borrowed = float(np.mean(have)), True


def cross(Z, ba, bb):
    v = [edu(Z[cap(ba.P[p], 5)], bb.tg[q]) for p in ba.pools for q in bb.pools]
    out = {"cross_mean_ED_U": mean_or_none(v), "n_cross": len(v),
           "cross_min_ED_U": min(v) if v else None, "cross_max_ED_U": max(v) if v else None}
    wa, wb = list(ba.pairs.values()), list(bb.pairs.values())
    within = [x for x in (mean_or_none(wa), mean_or_none(wb)) if x is not None]
    out["within_mean_ED_U"] = float(np.mean(within)) if within else None
    out["within_from"] = "both" if len(within) == 2 else ("one stage only" if within else "none")
    out["ratio_cross_within"] = (out["cross_mean_ED_U"] / out["within_mean_ED_U"]
                                 if v and within else None)
    out["frac_cross_gt_within"] = (float(np.mean([c > w for c in v for w in wa + wb])) if v and (wa + wb) else None)
    return out


def target_floors(Z, smp, idxT, S, blk):
    """Floors of the target stage (cells Z[idxT] in file order); blk = its Block."""
    n = len(idxT)
    Zt = Z[idxT]
    it = IB.sub(n, seed=1)
    T0 = tgt(Zt[it], S)
    out = {"n": int(n), "n_target": int(len(it))}
    perm = np.random.default_rng(4).permutation(n)                       # frozen own_halves scheme
    hA, hB = perm[: n // 2][: IB.NSUB], perm[n // 2:][: IB.NSUB]
    out["halves_frozen_seed4_ED_U"] = edu(Zt[hA], tgt(Zt[hB], S))
    hv = []
    for sd in HALF_SEEDS:
        perm = np.random.default_rng(sd).permutation(n)
        hA, hB = perm[: n // 2][: IB.NSUB], perm[n // 2:][: IB.NSUB]
        hv.append(edu(Zt[hA], tgt(Zt[hB], S)))
    out["halves_ED_U"] = {"mean": float(np.mean(hv)), "sd": float(np.std(hv, ddof=1)), "min": float(min(hv)),
                          "max": float(max(hv)), "n_each": int(min(n // 2, IB.NSUB)), "seeds": len(hv)}
    comp = np.setdiff1d(np.arange(n), it)
    out["complement_vs_target_ED_U"] = edu(Zt[cap(comp, 5)], T0) if len(comp) >= MIN_POOL else None
    out["complement_n"] = int(min(len(comp), IB.NSUB))
    lab = smp[idxT]
    lo = {}
    for p in blk.pools:
        rest = np.where(lab != p)[0]
        if len(rest) < MIN_POOL:                                          # single-pool stage: no replicate
            continue
        rest = rest[IB.sub(len(rest), seed=5)]
        lo[p] = edu(Zt[rest], blk.tg[p])
    out["lopo_ED_U"] = lo
    out["lopo_ED_U_mean"] = mean_or_none(lo.values())
    out["lopo_ED_U_min"] = min(lo.values()) if lo else None
    out["lopo_ED_U_max"] = max(lo.values()) if lo else None
    sp, spb = [], []
    us = blk.pools
    for r_ in range(1, len(us)):
        for A in itertools.combinations(us, r_):
            if us[0] not in A:
                continue
            Bs = [p for p in us if p not in A]
            ia = np.concatenate([blk.P[p] for p in A])
            ib = np.concatenate([blk.P[p] for p in Bs])
            e = edu(Z[cap(ia, 5)], tgt(Z[cap(ib, 1)], S))
            sp.append(e)
            if abs(len(A) - len(Bs)) <= 1:
                spb.append(e)
    out["split_ED_U"] = ({"mean": float(np.mean(sp)), "min": float(min(sp)), "max": float(max(sp)),
                          "balanced_mean": mean_or_none(spb), "n_splits": len(sp)} if sp else None)
    # pool shares of the exact confirm target subsample
    u, c = np.unique(lab[it], return_counts=True)
    w = c / c.sum()
    out["target_sum_w2"] = float((w ** 2).sum())
    out["tau2"], out["tau2_borrowed"] = blk.tau2, blk.tau2_borrowed
    out["oracle_ED_U"] = blk.tau2 * out["target_sum_w2"]
    exp_lo = {}
    for p in lo:                                                         # E ED(rest, p) = tau2 (1 + sum_rest v^2)
        cr = np.unique(lab[lab != p], return_counts=True)[1]
        exp_lo[p] = blk.tau2 * (1 + float(((cr / cr.sum()) ** 2).sum()))
    out["lopo_expected_from_tau2"] = exp_lo
    return out, T0, it


# ---------------------------------------------------------------- interpolation units

def interp_unit(data, k, stages, conf_k, repro):
    t0 = time.time()
    d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
    st = d["stage_fit"]
    Zf, Zh = f64(d["Z_fit"]), f64(d["Z_held"])
    smp_fit, smp_held = d["sample_fit"].astype(str), d["sample_held"].astype(str)
    S = IB.med_scale(Zf)
    nf = len(Zf)
    Z = np.vstack([Zf, Zh])
    stage = np.r_[st, np.full(len(Zh), k)]
    smp = np.r_[smp_fit, smp_held]
    A, B = np.where(stage == k - 1)[0], np.where(stage == k + 1)[0]
    idxT = np.arange(nf, nf + len(Zh))
    blocks = {k - 1: Block(Z, smp, A, S, w_idx=A[IB.sub(len(A), seed=1000)]),
              k: Block(Z, smp, idxT, S, w_idx=idxT[IB.sub(len(idxT), seed=1)]),
              k + 1: Block(Z, smp, B, S, w_idx=B[IB.sub(len(B), seed=2000)])}
    borrow(blocks)
    fl, T0, it = target_floors(Z, smp, idxT, S, blocks[k])
    out = {"held": stages[k], "prev": stages[k - 1], "next": stages[k + 1], "S_fit_unused_by_ED_U": S,
           "stages": {stages[s]: b.summary() for s, b in blocks.items()}, "floors": fl}
    # ---- stage size ----
    pn = []
    for r in range(3):
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        pn.append(edu(Z[A[iA]], tgt(Z[B[iB]], S)))
    out["ED_U_prev_next"] = {"mean": float(np.mean(pn)), "reps": pn}
    out["prev_vs_held_rep0_recomputed"] = edu(Z[A[IB.sub(len(A), seed=1000)]], T0)
    out["pool_level"] = {f"{stages[k - 1]}->{stages[k]}": cross(Z, blocks[k - 1], blocks[k]),
                         f"{stages[k]}->{stages[k + 1]}": cross(Z, blocks[k], blocks[k + 1]),
                         f"{stages[k - 1]}->{stages[k + 1]}": cross(Z, blocks[k - 1], blocks[k + 1])}
    # ---- confirm.json numbers and ratios ----
    m = {mm: conf_k["mean"][mm]["ED_U"] for mm in METHODS}
    rep_sd = {mm: float(np.std([rp[mm]["ED_U"] for rp in conf_k["reps"]], ddof=1)) for mm in METHODS}
    out["confirm_mean_ED_U"], out["confirm_rep_sd_ED_U"] = m, rep_sd
    a, b, c = m["prev"], m["next"], out["ED_U_prev_next"]["mean"]
    pred_union = 0.5 * a + 0.5 * b - 0.25 * c
    al = min(max((c + b - a) / (2 * c), 0.0), 1.0)
    best_mix = al * a + (1 - al) * b - al * (1 - al) * c
    oracle = fl["oracle_ED_U"]
    bp, bn = blocks[k - 1], blocks[k + 1]
    noise_union = oracle + 0.25 * bp.tau2 * bp.sum_w2 + 0.25 * bn.tau2 * bn.sum_w2
    noise_mix = oracle + al ** 2 * bp.tau2 * bp.sum_w2 + (1 - al) ** 2 * bn.tau2 * bn.sum_w2
    best = min(METHODS, key=lambda x: m[x])
    out["mixture"] = {"union_predicted_by_identity": pred_union, "union_measured": m["union"],
                      "identity_error": m["union"] - pred_union, "alpha_star_prev_weight": al,
                      "best_mixture_ED_U": best_mix, "best_mixture_noise_floor": noise_mix,
                      "best_mixture_minus_noise": best_mix - noise_mix}
    out["ratios"] = {
        "oracle_ED_U": oracle, "noise_union_ED_U": noise_union,
        "union_over_oracle": m["union"] / oracle, "ot_pairs_over_oracle": m["ot_pairs"] / oracle,
        "uot_over_oracle": m["uot"] / oracle, "best_method": best, "best_over_oracle": m[best] / oracle,
        "union_over_noise_union": m["union"] / noise_union, "ot_pairs_over_noise_union": m["ot_pairs"] / noise_union,
        "union_minus_oracle_room_any": m["union"] - oracle,
        "union_minus_noise_union_systematic": m["union"] - noise_union,
        "ot_pairs_minus_noise_union": m["ot_pairs"] - noise_union,
        "ot_pairs_minus_union": m["ot_pairs"] - m["union"], "uot_minus_union": m["uot"] - m["union"],
        "union_over_lopo_mean": m["union"] / fl["lopo_ED_U_mean"] if fl["lopo_ED_U_mean"] else None,
        "ot_pairs_over_lopo_mean": m["ot_pairs"] / fl["lopo_ED_U_mean"] if fl["lopo_ED_U_mean"] else None,
        "prev_next_over_pair_mean_held": (c / blocks[k].pair_mean) if blocks[k].pair_mean else None,
        "prev_next_over_oracle": c / oracle,
        "halves_sd_over_union": fl["halves_ED_U"]["sd"] / m["union"]}
    pools = conf_k["pools"]
    out["per_pool"] = {p: {"lopo_ED_U": fl["lopo_ED_U"].get(p),
                           **{f"{mm}_ED_U": conf_k["pool_mean"][mm][p] for mm in ("prev", "next", "union", "ot_pairs", "uot")},
                           **({f"{mm}_over_lopo": conf_k["pool_mean"][mm][p] / fl["lopo_ED_U"][p]
                               for mm in ("union", "ot_pairs", "uot")} if p in fl["lopo_ED_U"] else {})}
                       for p in pools}
    if len(fl["lopo_ED_U"]):
        out["per_pool_summary"] = {
            "union_beats_lopo_in": int(sum(conf_k["pool_mean"]["union"][p] < fl["lopo_ED_U"][p] for p in fl["lopo_ED_U"])),
            "ot_pairs_beats_lopo_in": int(sum(conf_k["pool_mean"]["ot_pairs"][p] < fl["lopo_ED_U"][p] for p in fl["lopo_ED_U"])),
            "ot_pairs_beats_union_in": int(sum(conf_k["pool_mean"]["ot_pairs"][p] < conf_k["pool_mean"]["union"][p] for p in pools)),
            "n_pools": len(fl["lopo_ED_U"])}
    # ---- validation ----
    val = {"prev_rep0_recomputed": out["prev_vs_held_rep0_recomputed"], "prev_rep0_confirm": conf_k["reps"][0]["prev"]["ED_U"]}
    if repro:
        rr = AC.run_interp(data, k)
        diffs = [abs(rr["reps"][r][mm]["ED_U"] - conf_k["reps"][r][mm]["ED_U"]) for r in range(3) for mm in METHODS]
        pdiff = [abs(rr["reps"][r][mm]["pool_ED_U"][p] - conf_k["reps"][r][mm]["pool_ED_U"][p])
                 for r in range(3) for mm in METHODS for p in conf_k["pools"]]
        val["run_interp_max_abs_diff_ED_U"] = float(max(diffs))
        val["run_interp_max_abs_diff_pool_ED_U"] = float(max(pdiff)) if pdiff else None
    out["validation"] = val
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- forecasting units

def forecast_unit(data, s, stages, conf_s, smp_past=None, smp_fut=None):
    t0 = time.time()
    d = np.load(os.path.join(data, f"origin{s}.npz"), allow_pickle=True)
    stp = d["stage_past"]
    if smp_past is None:
        smp_past, smp_fut = d["sample_past"].astype(str), d["sample_fut"].astype(str)
    Zp, Zt = f64(d["Z_past"]), f64(d["Z_fut"])
    S = IB.med_scale(Zp)
    Z = np.vstack([Zp, Zt])
    stage = np.r_[stp, np.full(len(Zt), s + 1)]
    smp = np.r_[smp_past, smp_fut]
    iN, iP = np.where(stage == s)[0], np.where(stage == s - 1)[0]
    idxT = np.arange(len(Zp), len(Zp) + len(Zt))
    blocks = {s - 1: Block(Z, smp, iP, S), s: Block(Z, smp, iN, S, w_idx=iN[IB.sub(len(iN))]),
              s + 1: Block(Z, smp, idxT, S, w_idx=idxT[IB.sub(len(idxT), seed=1)])}
    borrow(blocks)
    fl, T0, it = target_floors(Z, smp, idxT, S, blocks[s + 1])
    prob = AC.Problem(d, s, d["Z_fut"], d["X_fut"].astype(np.float32), smp_fut)
    ps = prob.score(*prob.forecast("persist", {}), pools=True)
    out = {"now": stages[s], "target": stages[s + 1], "S_past_unused_by_ED_U": S,
           "stages": {stages[x]: b.summary() for x, b in blocks.items()}, "floors": fl}
    out["persist_recomputed_ED_U"] = ps["ED_U"]
    out["ED_U_prev_stage_to_now"] = edu(Z[cap(iP, 0)], tgt(Z[cap(iN, 1)], S))
    out["pool_level"] = {f"{stages[s - 1]}->{stages[s]}": cross(Z, blocks[s - 1], blocks[s]),
                         f"{stages[s]}->{stages[s + 1]}": cross(Z, blocks[s], blocks[s + 1])}
    fam = {f: v["ED_U"] for f, v in conf_s.items()}
    best = min(fam, key=fam.get)
    oracle = fl["oracle_ED_U"]
    bn = blocks[s]
    noise_persist = oracle + bn.tau2 * bn.sum_w2
    pe = fam["persist"]
    out["confirm_ED_U"] = fam
    out["ratios"] = {"oracle_ED_U": oracle, "noise_persist_ED_U": noise_persist,
                     "persist_over_oracle": pe / oracle, "persist_over_noise_persist": pe / noise_persist,
                     "persist_minus_noise_persist_stage_change": pe - noise_persist,
                     "persist_over_lopo_mean": pe / fl["lopo_ED_U_mean"] if fl["lopo_ED_U_mean"] else None,
                     "best_family": best, "best_ED_U": fam[best], "best_over_oracle": fam[best] / oracle,
                     "uot_rw_minus_persist": fam["uot_rw"] - pe,
                     "change_over_pair_mean_target": (pe / blocks[s + 1].pair_mean) if blocks[s + 1].pair_mean else None,
                     "prev_change_ED_U": out["ED_U_prev_stage_to_now"]}
    # per pool: persist from the recomputation (pools mapped here; the erythroid parity run had no sample arrays in
    # its origin files, so its pool_ED_U is a single 'all' entry), the other families from the json when present
    out["per_pool"] = {p: {"lopo_ED_U": fl["lopo_ED_U"].get(p), "persist_ED_U": ps["pool_ED_U"][p],
                           **{f"{f}_ED_U": conf_s[f]["pool_ED_U"][p] for f in conf_s if p in conf_s[f]["pool_ED_U"]},
                           **({"persist_over_lopo": ps["pool_ED_U"][p] / fl["lopo_ED_U"][p]}
                              if p in fl["lopo_ED_U"] else {})}
                       for p in ps["pool_ED_U"]}
    if fl["lopo_ED_U"]:
        out["per_pool_summary"] = {"persist_beats_lopo_in": int(sum(ps["pool_ED_U"][p] < fl["lopo_ED_U"][p]
                                                                    for p in fl["lopo_ED_U"])),
                                   "n_pools": len(fl["lopo_ED_U"])}
    common = [p for p in ps["pool_ED_U"] if p in conf_s["persist"]["pool_ED_U"]]
    out["validation"] = {"persist_recomputed": ps["ED_U"], "persist_confirm": fam["persist"],
                         "abs_diff": abs(ps["ED_U"] - fam["persist"]),
                         "pool_abs_diff_max": (max(abs(ps["pool_ED_U"][p] - conf_s["persist"]["pool_ED_U"][p])
                                                   for p in common) if common else None)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- erythroid frozen-floor check

def ery_frozen_check(ery_res, units):
    out = {}
    for k, u in units.items():
        fr = ery_res["k"][str(k)]["floors"]
        own = {p: v["ED_U"] for p, v in fr["own_loso"].items()}
        mine = u["floors"]["lopo_ED_U"]
        out[k] = {"own_halves_frozen": fr["own_halves"]["ED_U"], "own_halves_mine": u["floors"]["halves_frozen_seed4_ED_U"],
                  "own_loso_max_abs_diff": max(abs(own[p] - mine[p]) for p in own),
                  "pools_frozen": sorted(own), "pools_mine": sorted(mine)}
    return out


def interp_row(u):
    r, fl = u["ratios"], u["floors"]
    pl = list(u["pool_level"].values())
    return {"held": u["held"], "pools": len(u["stages"][u["held"]]["pools_usable"]), "tau2": fl["tau2"],
            "tau2_borrowed": fl["tau2_borrowed"], "sum_w2": fl["target_sum_w2"], "oracle": r["oracle_ED_U"],
            "noise_union": r["noise_union_ED_U"], "union": u["confirm_mean_ED_U"]["union"],
            "ot_pairs": u["confirm_mean_ED_U"]["ot_pairs"], "uot": u["confirm_mean_ED_U"]["uot"],
            "best_method": r["best_method"], "union/oracle": r["union_over_oracle"],
            "ot_pairs/oracle": r["ot_pairs_over_oracle"], "union/noise_union": r["union_over_noise_union"],
            "union-noise_union": r["union_minus_noise_union_systematic"], "ot_pairs-union": r["ot_pairs_minus_union"],
            "lopo_mean": fl["lopo_ED_U_mean"], "halves_mean": fl["halves_ED_U"]["mean"], "halves_sd": fl["halves_ED_U"]["sd"],
            "prev_next": u["ED_U_prev_next"]["mean"], "prev_next/oracle": r["prev_next_over_oracle"],
            "cross/within_adjacent": [x["ratio_cross_within"] for x in pl[:2]],
            "cross/within_prev_next": pl[2]["ratio_cross_within"],
            "union_identity_error": u["mixture"]["identity_error"], "alpha_star": u["mixture"]["alpha_star_prev_weight"],
            "best_mixture": u["mixture"]["best_mixture_ED_U"],
            "per_pool": u.get("per_pool_summary")}


def forecast_row(u):
    r, fl = u["ratios"], u["floors"]
    pl = list(u["pool_level"].values())
    return {"target": u["target"], "pools": len(u["stages"][u["target"]]["pools_usable"]), "tau2": fl["tau2"],
            "oracle": r["oracle_ED_U"], "noise_persist": r["noise_persist_ED_U"], "persist": u["confirm_ED_U"]["persist"],
            "uot_rw": u["confirm_ED_U"]["uot_rw"], "best": r["best_family"], "best_ED_U": r["best_ED_U"],
            "persist/oracle": r["persist_over_oracle"], "persist/noise_persist": r["persist_over_noise_persist"],
            "persist-noise_persist": r["persist_minus_noise_persist_stage_change"], "lopo_mean": fl["lopo_ED_U_mean"],
            "halves_sd": fl["halves_ED_U"]["sd"], "prev_change": r["prev_change_ED_U"],
            "cross/within_now_target": pl[1]["ratio_cross_within"], "cross/within_prev_now": pl[0]["ratio_cross_within"],
            "per_pool": u.get("per_pool_summary")}


# ---------------------------------------------------------------- post-processing (no new distances)

def batch_view(stage_summary, bmap):
    """Pool -> sequencing batch (from the erythroid obs.csv: the two products share the embryo pools; atlas E6.5,
    E6.75 and three E7.0 pools have no erythroid cells, so their batch is unknown '?')."""
    pools = stage_summary["pools_usable"]
    bt = {p: bmap.get(p, "?") for p in pools}
    same, diff = [], []
    for key, v in stage_summary["pair_ED_U"].items():
        p, q = key.split("|")
        if "?" in (bt[p], bt[q]):
            continue
        (same if bt[p] == bt[q] else diff).append(v)
    share = {}
    for p, w in stage_summary["shares"].items():
        share[bmap.get(p, "?")] = share.get(bmap.get(p, "?"), 0.0) + w
    return {"pool_batch": bt, "batch_shares": share, "same_batch_pair_mean_ED_U": mean_or_none(same),
            "n_same": len(same), "diff_batch_pair_mean_ED_U": mean_or_none(diff), "n_diff": len(diff)}


def summarize(res, bmap):
    from scipy.stats import spearmanr
    s = {}
    units = []
    for ds in ("atlas", "ery"):
        for k, u in res[ds]["interp"].items():
            for st_name, st in u["stages"].items():
                u["stages"][st_name]["batch"] = batch_view(st, bmap)
            m, r, mx = u["confirm_mean_ED_U"], u["ratios"], u["mixture"]
            row = {"unit": f"{ds}_k{k}", "held": u["held"], "tau2_borrowed_any": any(x["tau2_borrowed"] for x in u["stages"].values()),
                   "union": m["union"], "ot_pairs": m["ot_pairs"], "uot": m["uot"], "noise_union": r["noise_union_ED_U"],
                   "oracle": r["oracle_ED_U"], "union_over_noise_union": r["union_over_noise_union"],
                   "union_systematic": r["union_minus_noise_union_systematic"],
                   "union_systematic_over_prev_next": r["union_minus_noise_union_systematic"] / u["ED_U_prev_next"]["mean"],
                   "off_chord": mx["best_mixture_minus_noise"], "alpha_star": mx["alpha_star_prev_weight"],
                   "best_mixture": mx["best_mixture_ED_U"],
                   "gain_ot_pairs_over_union": m["union"] - m["ot_pairs"],
                   "gain_best_ot_over_union": m["union"] - min(m["ot_pairs"], m["uot"]),
                   "rep_sd_union": u["confirm_rep_sd_ED_U"]["union"], "rep_sd_ot_pairs": u["confirm_rep_sd_ED_U"]["ot_pairs"],
                   "halves_sd": u["floors"]["halves_ED_U"]["sd"],
                   "target_batch": u["stages"][u["held"]]["batch"]["batch_shares"],
                   "prev_batch": u["stages"][u["prev"]]["batch"]["batch_shares"],
                   "next_batch": u["stages"][u["next"]]["batch"]["batch_shares"],
                   "held_same_vs_diff_batch_pair": (u["stages"][u["held"]]["batch"]["same_batch_pair_mean_ED_U"],
                                                    u["stages"][u["held"]]["batch"]["diff_batch_pair_mean_ED_U"])}
            pp = u.get("per_pool")
            if pp:
                lo = [p for p in pp if p in u["floors"]["lopo_ED_U"]]
                row["per_pool_union_over_lopo"] = {p: pp[p]["union_over_lopo"] for p in lo}
                row["per_pool_ot_pairs_minus_union"] = {p: pp[p]["ot_pairs_ED_U"] - pp[p]["union_ED_U"] for p in pp}
            units.append(row)
    s["interp_units"] = units
    use = [x for x in units if not x["tau2_borrowed_any"]]
    for ds in ("atlas", "ery"):
        sel = [x for x in use if x["unit"].startswith(ds)]
        allp = [v for x in units if x["unit"].startswith(ds) for v in x.get("per_pool_union_over_lopo", {}).values()]
        otp = [v for x in units if x["unit"].startswith(ds) for v in x.get("per_pool_ot_pairs_minus_union", {}).values()]
        s[f"{ds}_interp_summary"] = {
            "units_without_borrowed_tau2": [x["unit"] for x in sel],
            "median_union_over_noise_union": float(np.median([x["union_over_noise_union"] for x in sel])),
            "median_union_systematic": float(np.median([x["union_systematic"] for x in sel])),
            "median_union_systematic_over_prev_next": float(np.median([x["union_systematic_over_prev_next"] for x in sel])),
            "median_off_chord": float(np.median([x["off_chord"] for x in sel])),
            "ot_pairs_beats_union_units": int(sum(x["gain_ot_pairs_over_union"] > 0 for x in units if x["unit"].startswith(ds))),
            "n_units": int(sum(x["unit"].startswith(ds) for x in units)),
            "pools_union_below_lopo": int(sum(v < 1 for v in allp)), "pools_with_lopo": len(allp),
            "median_pool_union_over_lopo": float(np.median(allp)) if allp else None,
            "pools_ot_pairs_below_union": int(sum(v < 0 for v in otp)), "pools": len(otp)}
    for key in ("gain_ot_pairs_over_union", "gain_best_ot_over_union"):
        for tag, sel in (("", use), ("_within_atlas", [x for x in use if x["unit"].startswith("atlas")]),
                         ("_within_ery", [x for x in use if x["unit"].startswith("ery")])):
            rho, p = spearmanr([x["off_chord"] for x in sel], [x[key] for x in sel])
            s[f"spearman_off_chord_vs_{key}{tag}"] = {"rho": float(rho), "p": float(p), "n": len(sel),
                                                      "units": [x["unit"] for x in sel]}
    fc = []
    for ds in ("atlas", "ery"):
        for t, u in res[ds]["forecast"].items():
            for st_name, st in u["stages"].items():
                u["stages"][st_name]["batch"] = batch_view(st, bmap)
            r, f = u["ratios"], u["confirm_ED_U"]
            room = r["persist_minus_noise_persist_stage_change"]
            fc.append({"unit": f"{ds}_s{t}", "target": u["target"], "persist": f["persist"], "uot_rw": f["uot_rw"],
                       "tilt_rw": f["tilt_rw"], "noise_persist": r["noise_persist_ED_U"], "oracle": r["oracle_ED_U"],
                       "persist_over_noise_persist": r["persist_over_noise_persist"], "room_above_noise": room,
                       "gain_uot_rw": f["persist"] - f["uot_rw"], "gain_best_rw": f["persist"] - min(f["uot_rw"], f["tilt_rw"]),
                       "gain_uot_rw_over_room": (f["persist"] - f["uot_rw"]) / room if room > 0 else None,
                       "cross_within_now_target": list(u["pool_level"].values())[1]["ratio_cross_within"],
                       "target_batch": u["stages"][u["target"]]["batch"]["batch_shares"],
                       "now_batch": u["stages"][u["now"]]["batch"]["batch_shares"]})
    s["forecast_units"] = fc
    s["answer"] = answer(res, s)
    return s


def answer(res, s):
    U = {x["unit"]: x for x in s["interp_units"]}
    F = {x["unit"]: x for x in s["forecast_units"]}
    j = lambda xs, key, fmt="{:.3f}": "/".join(fmt.format(U[x][key]) for x in xs)  # noqa: E731
    jf = lambda xs, key, fmt="{:.2f}": "/".join("NA" if F[x][key] is None else fmt.format(F[x][key]) for x in xs)  # noqa: E731
    at = [f"atlas_k{k}" for k in range(2, 8)]
    er = [f"ery_k{k}" for k in range(2, 6)]
    hv = [u["floors"]["halves_ED_U"] for d in ("atlas", "ery") for u in res[d]["interp"].values()]
    a, e = s["atlas_interp_summary"], s["ery_interp_summary"]
    sp = s["spearman_off_chord_vs_gain_ot_pairs_over_union"]
    sa = s["spearman_off_chord_vs_gain_ot_pairs_over_union_within_atlas"]
    se = s["spearman_off_chord_vs_gain_ot_pairs_over_union_within_ery"]
    fa = [f"atlas_s{t}" for t in range(2, 8)]
    fe = [f"ery_s{t}" for t in range(2, 6)]
    br = [b["diff_batch_pair_mean_ED_U"] / b["same_batch_pair_mean_ED_U"]
          for d in ("atlas", "ery") for kind in ("interp", "forecast") for u in res[d][kind].values()
          for b in [u["stages"][u["held"] if kind == "interp" else u["target"]]["batch"]]
          if b["diff_batch_pair_mean_ED_U"] is not None and b["same_batch_pair_mean_ED_U"] is not None]
    s["batch_ratio_diff_over_same_range"] = [float(min(br)), float(max(br)), len(br)]
    return (
        "EXPLORATORY (pre-registered verdict unchanged). "
        f"Random halves are ~0 (|mean| <= {max(abs(h['mean']) for h in hv):.3f}, SD {min(h['sd'] for h in hv):.3f}-"
        f"{max(h['sd'] for h in hv):.3f}): a sanity check, not an achievable floor. "
        f"ATLAS interpolation k2..k7: union / pool-noise floor of a 50/50 mix (noise_union) = {j(at, 'union_over_noise_union', '{:.2f}')}; "
        f"systematic error of the union above that floor = {j(at, 'union_systematic')} ED_U "
        f"(median {a['median_union_systematic_over_prev_next']:.1%} of the prev-next distance over k3..k7; k2 uses a borrowed tau2). "
        f"The largest room (k3, E7.25) is a timing asymmetry: the pre-registered prev baseline alone scores "
        f"{res['atlas']['interp']['3']['confirm_mean_ED_U']['prev'] if '3' in res['atlas']['interp'] else res['atlas']['interp'][3]['confirm_mean_ED_U']['prev']:.3f} "
        f"vs union {U['atlas_k3']['union']:.3f}, and the best prev/next mixture (oracle-tuned on the held stage, descriptive) "
        f"puts {U['atlas_k3']['alpha_star']:.0%} on prev and scores {U['atlas_k3']['best_mixture']:.3f}; a symmetric t = 0.5 "
        "interpolant cannot use it. "
        f"The part no prev/next mixture explains (off-chord residual above noise, the intermediate-state signal a displacement "
        f"interpolant could exploit) is {j(at, 'off_chord')} on the atlas vs {j(er, 'off_chord')} on the erythroid "
        f"(medians {a['median_off_chord']:.3f} vs {e['median_off_chord']:.3f}; atlas values down to "
        f"{min(U[x]['off_chord'] for x in at):.3f} show the noise model's own error is about +-0.1, so the atlas residuals "
        "are indistinguishable from 0 except perhaps k3 / k6). "
        f"Erythroid ot_pairs beat union exactly where off-chord was large (gain {j(er, 'gain_ot_pairs_over_union')}); "
        f"across {sp['n']} units Spearman(off-chord, union - ot_pairs) = {sp['rho']:.2f} (p {sp['p']:.3f}, post hoc, units not "
        f"independent, largely dataset membership; within the atlas {sa['rho']:.2f} (n {sa['n']}), within the erythroid "
        f"{se['rho']:.2f} (n {se['n']}): same ordering, too few units for inference). "
        f"On the atlas ot_pairs is worse than union in {a['n_units'] - a['ot_pairs_beats_union_units']}/{a['n_units']} held stages "
        "(ot_pairs - union = " + "/".join(f"{-U[x]['gain_ot_pairs_over_union']:.3f}" for x in at)
        + "; k2 within rep SD), also where union is at the floor: the McCann pairing adds "
        f"error when the held stage is a mixture of its neighbours. Per pool (model-free): union beats the real same-stage "
        f"leave-one-pool-out replicate in {a['pools_union_below_lopo']}/{a['pools_with_lopo']} atlas pools vs "
        f"{e['pools_union_below_lopo']}/{e['pools_with_lopo']} erythroid pools (median union / replicate "
        f"{a['median_pool_union_over_lopo']:.2f} vs {e['median_pool_union_over_lopo']:.2f}): on this check the two datasets look "
        "the SAME, because the replicate of a pool is mostly a same-batch pool, which no prediction built from neighbour "
        "stages with another batch mix can match (e.g. the atlas E8.0 target is "
        f"{U['atlas_k6']['target_batch'].get('3', 0):.0%} batch 3, its neighbours {U['atlas_k6']['prev_batch'].get('3', 0):.0%} "
        f"and {U['atlas_k6']['next_batch'].get('3', 0):.0%}). ot_pairs beats union in "
        f"{a['pools_ot_pairs_below_union']}/{a['pools']} atlas pools vs {e['pools_ot_pairs_below_union']}/{e['pools']} erythroid. "
        "So 'at the floor' below is the MODEL-based claim (noise_union); what discriminates the datasets is the kind of gap "
        "that remains: on the atlas a prev/next mixture closes it, on the erythroid it does not. "
        f"Verdict: on the atlas union is at or near the model pool-noise floor in most held stages; the room left is small and "
        f"mostly asymmetric timing, not intermediate states, so OT has little to use; on the erythroid OT won in the 3 "
        f"held stages whose true cells sat off the prev/next chord by "
        f"{min(U[x]['off_chord'] for x in ('ery_k2', 'ery_k4', 'ery_k5')):.2f}-{max(U[x]['off_chord'] for x in ('ery_k2', 'ery_k4', 'ery_k5')):.2f} ED_U. "
        f"FORECAST atlas s2..s7: persist / its noise floor = {jf(fa, 'persist_over_noise_persist')}; room above noise "
        f"= {jf(fa, 'room_above_noise')}; uot_rw gain = {jf(fa, 'gain_uot_rw')} (rolling-origin lambda 0 at s2, s3, s5, so "
        f"the largest room, s3 = {F['atlas_s3']['room_above_noise']:.2f}, and s5 = {F['atlas_s5']['room_above_noise']:.2f} "
        f"were not used; at s6 (room {F['atlas_s6']['room_above_noise']:.2f}) uot_rw took "
        f"{F['atlas_s6']['gain_uot_rw_over_room']:.0%}; lambda 2 at s4 overshot where persist was already at the floor). "
        "Erythroid s2..s5 room "
        f"{jf(fe, 'room_above_noise')} grew with origin; uot_rw took {jf(fe, 'gain_uot_rw_over_room', '{:.0%}')} of it. "
        "Caveats: tau2 assumes exchangeable embryo pools, but within-stage pool differences are dominated by sequencing "
        f"batch (stages[*].batch: different-batch pool pairs are {min(br):.1f}-{max(br):.0f}x the same-batch ones, median "
        f"{float(np.median(br)):.1f}x over {len(br)} target stages), so noise floors are model estimates; atlas k1 (single held "
        "pool, borrowed tau2) is "
        "uninformative.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=os.path.join(SCR, "atlas"))
    ap.add_argument("--ery", default=os.path.join(SCR, "ery"))
    ap.add_argument("--no-repro", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=list(range(1, 8)))
    ap.add_argument("--origins", type=int, nargs="+", default=list(range(2, 8)))
    ap.add_argument("--out", default=os.path.join(HERE, "floor.json"))
    ap.add_argument("--summarize-only", action="store_true", help="re-run only the post-processing on --out")
    args = ap.parse_args()
    obs = pd.read_csv(os.path.join(args.ery, "obs.csv"))
    bmap = obs.groupby(obs["sample"].astype(str))["sequencing.batch"].first().astype(int).astype(str).to_dict()
    if args.summarize_only:
        res = json.load(open(args.out))
        res["summary"] = summarize(res, bmap)
        json.dump(res, open(args.out, "w"), indent=1, default=float)
        print(json.dumps({k: v for k, v in res["summary"].items() if k not in ("interp_units", "forecast_units")}, indent=1))
        return
    t0 = time.time()
    conf = json.load(open(os.path.join(ROOT, "tack", "results", "atlas", "confirm.json")))
    par = json.load(open(os.path.join(ROOT, "tack", "results", "atlas", "parity_erythroid.json")))
    ery_res = json.load(open(os.path.join(ROOT, "tack", "results", "ery", "attempt_interp", "results.json")))
    res = {"exploratory": True,
           "label": "EXPLORATORY, not pre-registered. The pre-registered verdict in confirm.json is final and unchanged.",
           "script": "tack/results/atlas/floor/floor.py", "conventions": __doc__.split("Usage:")[0],
           "atlas": {"interp": {}, "forecast": {}}, "ery": {"interp": {}, "forecast": {}}}

    def dump():
        json.dump(res, open(args.out, "w"), indent=1, default=float)

    for k in args.ks:
        res["atlas"]["interp"][k] = interp_unit(args.atlas, k, ATLAS_STAGES, conf["interp"][str(k)], not args.no_repro)
        print("atlas interp", k, json.dumps(interp_row(res["atlas"]["interp"][k]), default=float), flush=True)
        dump()
    for s in args.origins:
        res["atlas"]["forecast"][s] = forecast_unit(args.atlas, s, ATLAS_STAGES, conf["forecast"][str(s)])
        print("atlas forecast", s, json.dumps(forecast_row(res["atlas"]["forecast"][s]), default=float), flush=True)
        dump()
    for k in (2, 3, 4, 5):
        res["ery"]["interp"][k] = interp_unit(args.ery, k, ERY_STAGES, par["interp"][str(k)], not args.no_repro)
        print("ery interp", k, json.dumps(interp_row(res["ery"]["interp"][k]), default=float), flush=True)
        dump()
    # erythroid origins: samples are not stored in the origin files; map them from obs.csv (origin files are in obs
    # order: checked through the cell types)
    ost =np.array([ERY_STAGES.index(x) for x in obs["stage"].astype(str)])
    osm, oct_ = obs["sample"].astype(str).to_numpy(), obs["celltype"].astype(str).to_numpy()
    for s in (2, 3, 4, 5):
        d = np.load(os.path.join(args.ery, f"origin{s}.npz"), allow_pickle=True)
        assert (d["celltype_past"].astype(str) == oct_[ost <= s]).all() and (d["stage_past"] == ost[ost <= s]).all()
        assert (d["celltype_fut"].astype(str) == oct_[ost == s + 1]).all()
        res["ery"]["forecast"][s] = forecast_unit(args.ery, s, ERY_STAGES, par["forecast"][str(s)],
                                                  smp_past=osm[ost <= s], smp_fut=osm[ost == s + 1])
        print("ery forecast", s, json.dumps(forecast_row(res["ery"]["forecast"][s]), default=float), flush=True)
        dump()
    res["validation_ery_frozen_floors"] = ery_frozen_check(ery_res, res["ery"]["interp"])
    res["tables"] = {"atlas_interp": {k: interp_row(u) for k, u in res["atlas"]["interp"].items()},
                     "atlas_forecast": {s: forecast_row(u) for s, u in res["atlas"]["forecast"].items()},
                     "ery_interp": {k: interp_row(u) for k, u in res["ery"]["interp"].items()},
                     "ery_forecast": {s: forecast_row(u) for s, u in res["ery"]["forecast"].items()}}
    res["summary"] = summarize(res, bmap)
    res["seconds"] = round(time.time() - t0, 1)
    dump()
    print("validation ery frozen floors", json.dumps(res["validation_ery_frozen_floors"], default=float))
    print("done", res["seconds"], "s")


if __name__ == "__main__":
    main()
