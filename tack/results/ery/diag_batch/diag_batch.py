"""BATCH vs STAGE diagnosis for the erythroid forecasting benchmark. DESCRIPTIVE ONLY, not a forecast.

Question: in each strict origin space (origin<s>.npz, s = 1..5; origin 1 is degenerate, reported only) and each
interpolation space (interp<k>.npz, k = 2..5), is the change from one real embryonic stage to the next larger than
the change between SAMPLES (embryo pools) of the same stage? What is the smallest error that any forecast of a new
stage can expect, given that the new stage is made of embryo pools that the forecast has never seen?

Everything here is descriptive: the distances use the true cells of every stage present in a space, INCLUDING the
stage that origin s forecasts (s+1) and the held-out stage k of interp k. Nothing is fitted; the spaces are exactly
the ones the forecasts use (PCA 30 fitted on past / neighbour stages only, cells of the other stages only
transformed). The 2000-HVG log space of the same file is a robustness check ("genes").

Distance: energy distance ED(A, B) = 2 E|a - b| - E|a - a'| - E|b - b'| (Euclidean).
  ED_U  within terms exclude the zero self-pairs (U-statistic; ~0 for two groups drawn from one distribution).
  ED_V  within terms include them, as ery_forecast.energy does (V-statistic): ED_V = ED_U + m_A / n_A + m_B / n_B,
        m = mean within-group distance. The forecast.json random-halves floor is almost entirely this bias.
  ED_V_fs  ED_V at the forecast's own sizes (each side capped at 1500 cells; prediction side = n of stage s).
Group-level ED come from exact block sums of all pairwise distances over (sample, cell type) blocks, so no cells are
dropped; composition-matched ED ("comp") re-weights each cell type of both groups to a common composition (the mean
of the two compositions over the types with >= MIN_TYPE cells in both; "cover" = share of cells in those types).

Sample-level floor for a forecast of stage T (the true stage s+1 of origin s; the held stage k of interp k):
  oracle     if the pools of T are exchangeable draws around a stage-level population and a forecast cannot know
             their idiosyncrasies, then E ED(forecast, observed T) >= E ED(population, observed T)
             = tau2 * sum_j w_j^2, w_j = cell share of pool j, tau2 = E ED(pool, population) = (mean ED_U between two
             pools of T) / 2, because ED is a squared Hilbert-space distance (2 x MMD^2, distance kernel), so the
             cross term vanishes. Reported as ED_U and as ED_V_fs (+ m_T / n_pred + m_T / n_true).
  replicate  leave-one-sample-out: ED(one pool of T, the other pools of T) = the score of a perfect replicate embryo
             pool of the right stage. Larger than the oracle (one pool carries its full idiosyncrasy).
  split      ED between two disjoint groups of whole pools of T (every bipartition).
Permutation test (cell labels shuffled within a pair, 199 permutations, <= 1000 cells per sample): are two samples of
one stage distinguishable at all? Run in the origin-5 and interp PCA spaces.

Usage: python -I diag_batch.py --data <folder with origin*.npz, interp*.npz, obs.csv, forecast.json> [--no-genes]
Outputs next to this script: *.csv tables, summary.json, run.log.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
TYPES = ["Blood progenitors 1", "Blood progenitors 2", "Erythroid1", "Erythroid2", "Erythroid3"]
MIN_CELLS = 20   # samples below this stay in their stage pool but are never used as a pool on their own
MIN_TYPE = 20    # cells of a type needed in both groups for within-type and composition-matched ED
NSUB = 1500
NPERM = 199
PERM_CAP = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "run.log")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


# ---------------- distances ----------------

def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def energy_v(X, Y):
    """Exactly ery_forecast.energy with uniform weights (V-statistic)."""
    d = lambda A, B: np.sqrt(sqdist(A, B))
    return float(2 * d(X, Y).mean() - d(X, X).mean() - d(Y, Y).mean())


def energy_u(X, Y):
    nx, ny = len(X), len(Y)
    return float(2 * np.sqrt(sqdist(X, Y)).mean() - np.sqrt(sqdist(X, X)).sum() / (nx * (nx - 1))
                 - np.sqrt(sqdist(Y, Y)).sum() / (ny * (ny - 1)))


def sub(n, k=NSUB, seed=0):
    r = np.random.default_rng(seed)
    return np.sort(r.choice(n, min(n, k), replace=False))


def block_sums(E, grp, G, chunk=1024):
    """S[g, h] = sum of Euclidean distances between the cells of block g and block h (self-pairs = 0)."""
    order = np.argsort(grp, kind="stable")
    E, g = E[order], grp[order]
    starts = np.searchsorted(g, np.arange(G))
    nrm = (E * E).sum(1)
    S = np.zeros((G, G))
    for a in range(0, len(E), chunk):
        b = min(a + chunk, len(E))
        D = nrm[a:b, None] + nrm[None, :] - 2 * E[a:b] @ E.T
        np.maximum(D, 0, out=D)
        np.sqrt(D, out=D)
        D[np.arange(b - a), np.arange(a, b)] = 0.0
        np.add.at(S, g[a:b], np.add.reduceat(D, starts, axis=1))
    return S


class Blocks:
    """Pairwise-distance block sums over (sample, cell type) blocks of one space; groups = per-cell weights."""

    def __init__(self, E, smp, typ, stg, bat):
        key = np.array([f"{a}|{b}" for a, b in zip(smp, typ)])
        uniq, inv = np.unique(key, return_inverse=True)
        self.G = len(uniq)
        self.n = np.bincount(inv, minlength=self.G).astype(float)
        first = np.array([np.where(inv == g)[0][0] for g in range(self.G)])
        self.smp, self.typ, self.stg, self.bat = smp[first], typ[first], stg[first], bat[first]
        self.S = block_sums(E, inv, self.G)

    def mask(self, samples=None, stages=None, types=None):
        m = np.ones(self.G, bool)
        if samples is not None:
            m &= np.isin(self.smp, list(samples))
        if stages is not None:
            m &= np.isin(self.stg, list(stages))
        if types is not None:
            m &= np.isin(self.typ, list(types))
        return m

    def w_uniform(self, m):
        w = m.astype(float)
        return w / (w * self.n).sum()

    def comp(self, m):
        return {t: float(self.n[m & (self.typ == t)].sum() / self.n[m].sum()) for t in TYPES}

    def w_comp(self, m, frac):
        w = np.zeros(self.G)
        for t, f in frac.items():
            mt = m & (self.typ == t)
            w[mt] = f / self.n[mt].sum()
        return w

    def within(self, w):
        s2 = float((self.n * w * w).sum())
        mv = float(w @ self.S @ w)
        return mv / (1 - s2), 1 / s2

    def ed(self, wa, wb):
        mA, neA = self.within(wa)
        mB, neB = self.within(wb)
        u = 2 * float(wa @ self.S @ wb) - mA - mB
        return {"ED_U": u, "ED_V": u + mA / neA + mB / neB, "m_A": mA, "m_B": mB, "n_A": neA, "n_B": neB}

    def ed_masks(self, ma, mb):
        return self.ed(self.w_uniform(ma), self.w_uniform(mb))

    def ed_comp(self, ma, mb):
        """Composition-matched ED over the types with >= MIN_TYPE cells in both groups."""
        shared = [t for t in TYPES if self.n[ma & (self.typ == t)].sum() >= MIN_TYPE
                  and self.n[mb & (self.typ == t)].sum() >= MIN_TYPE]
        if not shared:
            return np.nan, 0.0, ""
        ca, cb = self.comp(ma), self.comp(mb)
        tot = sum(ca[t] + cb[t] for t in shared)
        frac = {t: (ca[t] + cb[t]) / tot for t in shared}
        cover = min(sum(ca[t] for t in shared), sum(cb[t] for t in shared))
        r = self.ed(self.w_comp(ma, frac), self.w_comp(mb, frac))
        return r["ED_U"], cover, ";".join(t.replace("Blood progenitors ", "BP").replace("Erythroid", "Ery")
                                         for t in shared)


def tv(ca, cb):
    return 0.5 * sum(abs(ca[t] - cb[t]) for t in TYPES)


# ---------------- spaces ----------------

def load_space(data, kind, k, rep, obs, codes):
    d = np.load(os.path.join(data, f"{kind}{k}.npz"), allow_pickle=True)
    if kind == "origin":
        a, b = codes <= k, codes == k + 1
        assert (d["stage_past"] == codes[a]).all() and (d["stage_fut"] == codes[b]).all()
        assert (d["celltype_past"].astype(str) == obs["celltype"].to_numpy()[a]).all()
        assert (d["celltype_fut"].astype(str) == obs["celltype"].to_numpy()[b]).all()
        A, B = ("Z_past", "Z_fut") if rep == "pca" else ("X_past", "X_fut")
        target = k + 1
    else:
        a, b = (codes == k - 1) | (codes == k + 1), codes == k
        assert (d["stage_fit"] == codes[a]).all()
        assert (d["sample_fit"].astype(str) == obs["sample"].to_numpy()[a]).all()
        assert (d["sample_held"].astype(str) == obs["sample"].to_numpy()[b]).all()
        assert (d["celltype_held"].astype(str) == obs["celltype"].to_numpy()[b]).all()
        A, B = ("Z_fit", "Z_held") if rep == "pca" else ("X_fit", "X_held")
        target = k
    E = np.vstack([d[A].astype(np.float64), d[B].astype(np.float64)])
    idx = np.concatenate([np.where(a)[0], np.where(b)[0]])
    return E, idx, target


def analyse(name, kind, k, rep, E, idx, target, obs, codes, fc, out):
    t0 = time.time()
    smp = obs["sample"].to_numpy()[idx]
    typ = obs["celltype"].to_numpy()[idx]
    bat = obs["sequencing.batch"].to_numpy()[idx]
    stg = codes[idx]
    bl = Blocks(E, smp, typ, stg, bat)
    stages = sorted(set(stg.tolist()))
    nsmp = pd.Series(smp).value_counts()
    smp_stage = {s: int(stg[smp == s][0]) for s in nsmp.index}
    smp_batch = {s: int(bat[smp == s][0]) for s in nsmp.index}
    usable = {t: sorted([s for s in nsmp.index if smp_stage[s] == t and nsmp[s] >= MIN_CELLS], key=int) for t in stages}
    tag = {"space": name, "kind": kind, "k": k, "rep": rep, "dim": E.shape[1]}
    stg_n = {t: int((stg == t).sum()) for t in stages}

    # 1. random halves of every stage (cell level, forecast.json procedure, seed 4)
    for t in stages:
        Zt = E[stg == t]
        h = np.random.default_rng(4).permutation(len(Zt))
        A, B = Zt[h[: len(Zt) // 2]][:NSUB], Zt[h[len(Zt) // 2:]][:NSUB]
        out["halves"].append({**tag, "stage": STAGES[t], "n": len(Zt), "n_half": len(A),
                              "ED_V": energy_v(A, B), "ED_U": energy_u(A, B), "is_target": t == target})

    # 2. stage pairs (pooled, all cells), gap 1 and 2
    for t1, t2 in itertools.combinations(stages, 2):
        if t2 - t1 > 2:
            continue
        ma, mb = bl.mask(stages=[t1]), bl.mask(stages=[t2])
        r = bl.ed_masks(ma, mb)
        nfa, nfb = min(stg_n[t1], NSUB), min(stg_n[t2], NSUB)
        cu, cov, shared = bl.ed_comp(ma, mb)
        out["stage_pairs"].append({**tag, "stage_a": STAGES[t1], "stage_b": STAGES[t2], "gap": t2 - t1,
                                   "n_a": stg_n[t1], "n_b": stg_n[t2], "ED_U": r["ED_U"], "ED_V": r["ED_V"],
                                   "ED_V_fs": r["ED_U"] + r["m_A"] / nfa + r["m_B"] / nfb,
                                   "ED_U_comp": cu, "comp_cover": cov, "comp_types": shared,
                                   "TV_comp": tv(bl.comp(ma), bl.comp(mb)),
                                   "uses_target": target in (t1, t2)})

    # 3. sample pairs: within stage (gap 0) and across stages (gap 1, 2); within-type for gap 0 and 1
    allu = [(t, s) for t in stages for s in usable[t]]
    pair_ed = {}
    for (t1, s1), (t2, s2) in itertools.combinations(allu, 2):
        if t2 - t1 > 2:
            continue
        ma, mb = bl.mask(samples=[s1]), bl.mask(samples=[s2])
        r = bl.ed_masks(ma, mb)
        cu, cov, shared = bl.ed_comp(ma, mb)
        pair_ed[(s1, s2)] = r["ED_U"]
        out["sample_pairs"].append({**tag, "gap": t2 - t1, "stage_a": STAGES[t1], "stage_b": STAGES[t2],
                                    "sample_a": s1, "sample_b": s2, "n_a": int(nsmp[s1]), "n_b": int(nsmp[s2]),
                                    "batch_a": smp_batch[s1], "batch_b": smp_batch[s2],
                                    "same_batch": smp_batch[s1] == smp_batch[s2], "ED_U": r["ED_U"],
                                    "ED_V": r["ED_V"], "ED_U_comp": cu, "comp_cover": cov, "comp_types": shared,
                                    "TV_comp": tv(bl.comp(ma), bl.comp(mb)), "uses_target": target in (t1, t2)})
        if t2 - t1 <= 1:
            for ty in TYPES:
                mta, mtb = ma & (bl.typ == ty), mb & (bl.typ == ty)
                if bl.n[mta].sum() >= MIN_TYPE and bl.n[mtb].sum() >= MIN_TYPE:
                    rt = bl.ed_masks(mta, mtb)
                    out["type_pairs"].append({**tag, "gap": t2 - t1, "celltype": ty, "stage_a": STAGES[t1],
                                              "stage_b": STAGES[t2], "sample_a": s1, "sample_b": s2,
                                              "n_a": int(bl.n[mta].sum()), "n_b": int(bl.n[mtb].sum()),
                                              "same_batch": smp_batch[s1] == smp_batch[s2], "ED_U": rt["ED_U"]})

    def pe(a, b):
        return pair_ed.get((a, b), pair_ed.get((b, a)))

    def within_list(t):
        return [pe(a, b) for a, b in itertools.combinations(usable[t], 2)]

    # 4. decomposition per stage pair: within-stage sample pairs vs cross-stage sample pairs
    for gap in (1, 2):
        for t1 in stages:
            t2 = t1 + gap
            if t2 not in stages or len(usable[t1]) < 2 or len(usable[t2]) < 2:
                continue
            w1, w2 = within_list(t1), within_list(t2)
            cross = [pe(a, b) for a in usable[t1] for b in usable[t2]]
            cross_sb = [pe(a, b) for a in usable[t1] for b in usable[t2] if smp_batch[a] == smp_batch[b]]
            cross_db = [pe(a, b) for a in usable[t1] for b in usable[t2] if smp_batch[a] != smp_batch[b]]
            wall = w1 + w2
            within_m = 0.5 * (np.mean(w1) + np.mean(w2))
            pooled = [r for r in out["stage_pairs"] if r["space"] == name and r["rep"] == rep
                      and r["stage_a"] == STAGES[t1] and r["stage_b"] == STAGES[t2]][0]
            out["decomp"].append({
                **tag, "gap": gap, "stage_a": STAGES[t1], "stage_b": STAGES[t2],
                "n_samples_a": len(usable[t1]), "n_samples_b": len(usable[t2]),
                "within_a_mean": float(np.mean(w1)), "within_b_mean": float(np.mean(w2)),
                "within_max": float(np.max(wall)), "cross_mean": float(np.mean(cross)),
                "cross_min": float(np.min(cross)), "cross_same_batch_mean": float(np.mean(cross_sb)) if cross_sb else np.nan,
                "cross_diff_batch_mean": float(np.mean(cross_db)) if cross_db else np.nan,
                "stage_shift": float(np.mean(cross) - within_m),
                "ratio_cross_within": float(np.mean(cross) / within_m),
                "frac_cross_gt_within": float(np.mean([c > w for c in cross for w in wall])),
                "pooled_ED_U": pooled["ED_U"], "pooled_over_median_within": pooled["ED_U"] / float(np.median(wall)),
                "uses_target": target in (t1, t2)})
    for t in stages:
        if len(usable[t]) < 2:
            continue
        sb = [pe(a, b) for a, b in itertools.combinations(usable[t], 2) if smp_batch[a] == smp_batch[b]]
        db = [pe(a, b) for a, b in itertools.combinations(usable[t], 2) if smp_batch[a] != smp_batch[b]]
        out["within_batch"].append({**tag, "stage": STAGES[t], "n_samples": len(usable[t]),
                                    "batches": ";".join(f"{s}:b{smp_batch[s]}" for s in usable[t]),
                                    "n_same_batch_pairs": len(sb), "same_batch_mean": float(np.mean(sb)) if sb else np.nan,
                                    "n_diff_batch_pairs": len(db), "diff_batch_mean": float(np.mean(db)) if db else np.nan,
                                    "uses_target": t == target})

    # 5. sample positions: each pool vs the previous / next stage pool and vs the rest of its own stage
    for t in stages:
        for s in usable[t]:
            ms = bl.mask(samples=[s])
            row = {**tag, "stage": STAGES[t], "sample": s, "n": int(nsmp[s]), "batch": smp_batch[s],
                   "uses_target": t == target or target in (t - 1, t + 1)}
            for lab, tt in (("prev", t - 1), ("next", t + 1)):
                row[f"ED_U_{lab}_stage"] = bl.ed_masks(ms, bl.mask(stages=[tt]))["ED_U"] if tt in stages else np.nan
            rest = bl.mask(stages=[t]) & ~ms
            row["ED_U_rest_own_stage"] = bl.ed_masks(ms, rest)["ED_U"] if rest.any() else np.nan
            out["positions"].append(row)

    # 6. sample-level floors for the target stage
    T = target
    mT = bl.mask(stages=[T])
    m_T, _ = bl.within(bl.w_uniform(mT))
    n_true = min(stg_n[T], NSUB)
    n_pred = min(stg_n[T - 1], NSUB) if kind == "origin" else n_true
    w = np.array([nsmp[s] for s in nsmp.index if smp_stage[s] == T], float)
    w /= w.sum()
    fl = {**tag, "target": STAGES[T], "n_target": stg_n[T], "n_samples": int(len(w)), "usable": ";".join(usable[T]),
          "sum_w2": float((w * w).sum()), "m_T": m_T, "n_pred": n_pred, "n_true": n_true,
          "vbias_fs": m_T / n_pred + m_T / n_true}
    h = [r for r in out["halves"] if r["space"] == name and r["rep"] == rep and r["stage"] == STAGES[T]][0]
    fl["halves_ED_V"], fl["halves_ED_U"] = h["ED_V"], h["ED_U"]
    wl = within_list(T)
    if wl:
        tau2 = float(np.mean(wl)) / 2
        fl.update(pair_mean_ED_U=float(np.mean(wl)), tau2=tau2, oracle_ED_U=tau2 * fl["sum_w2"],
                  oracle_ED_V_fs=tau2 * fl["sum_w2"] + fl["vbias_fs"])
        loso = []
        for s in usable[T]:
            ms = bl.mask(samples=[s])
            r = bl.ed_masks(ms, mT & ~ms)
            loso.append((r["ED_U"], r["ED_U"] + r["m_A"] / min(nsmp[s], NSUB) + r["m_B"] / min(stg_n[T] - nsmp[s], NSUB)))
            out["loso"].append({**tag, "target": STAGES[T], "sample": s, "n": int(nsmp[s]), "batch": smp_batch[s],
                                "ED_U": r["ED_U"], "ED_V_fs": loso[-1][1]})
        lu, lv = np.array(loso).T
        fl.update(loso_ED_U_mean=float(lu.mean()), loso_ED_U_min=float(lu.min()), loso_ED_U_max=float(lu.max()),
                  loso_ED_V_fs_mean=float(lv.mean()), loso_ED_V_fs_min=float(lv.min()), loso_ED_V_fs_max=float(lv.max()))
        us = usable[T]
        sp_u, sp_bal = [], []
        for r_ in range(1, len(us)):
            for A in itertools.combinations(us, r_):
                if us[0] not in A:
                    continue
                Bs = [s for s in us if s not in A]
                ma, mb = bl.mask(samples=A), bl.mask(samples=Bs)
                rr = bl.ed_masks(ma, mb)
                sp_u.append(rr["ED_U"])
                if abs(len(A) - len(Bs)) <= 1:
                    sp_bal.append(rr["ED_U"])
        fl.update(split_ED_U_mean=float(np.mean(sp_u)), split_ED_U_min=float(np.min(sp_u)),
                  split_balanced_ED_U_mean=float(np.mean(sp_bal)), n_splits=len(sp_u))
    if kind == "origin" and rep == "pca":
        # exact reproduction of the forecast.json persist score and floor in this space
        Zn, Zt = E[stg == T - 1], E[stg == T]
        i, j = sub(len(Zn)), sub(len(Zt), seed=1)
        fl["persist_reproduced"] = energy_v(Zn[i], Zt[j])
        fo = fc["origins"][str(k)]
        fams = [f for f in fc["families"] if f in fo]
        best = min(fams, key=lambda f: fo[f]["energy"])
        fl.update(persist_forecast_json=fo["persist"]["energy"], floor_forecast_json=fo["floor"]["energy"],
                  best_family=best, best_energy=fo[best]["energy"],
                  ot_energy=fo["ot"]["energy"], meanshift_energy=fo["meanshift"]["energy"])
        if wl:
            fl["best_over_oracle"] = fo[best]["energy"] / fl["oracle_ED_V_fs"]
            fl["best_over_loso_mean"] = fo[best]["energy"] / fl["loso_ED_V_fs_mean"]
    out["floors"].append(fl)
    log(f"{name} {rep}: {len(E)} cells, {bl.G} blocks, {time.time() - t0:.1f}s")


def perm_tests(name, E, idx, obs, codes, out, rng):
    smp = obs["sample"].to_numpy()[idx]
    stg = codes[idx]
    bat = obs["sequencing.batch"].to_numpy()[idx]
    nsmp = pd.Series(smp).value_counts()
    for t in sorted(set(stg.tolist())):
        us = sorted([s for s in nsmp.index if nsmp[s] >= MIN_CELLS and stg[smp == s][0] == t], key=int)
        for a, b in itertools.combinations(us, 2):
            ia, ib = np.where(smp == a)[0], np.where(smp == b)[0]
            ia = rng.choice(ia, min(len(ia), PERM_CAP), replace=False)
            ib = rng.choice(ib, min(len(ib), PERM_CAP), replace=False)
            Z = E[np.concatenate([ia, ib])]
            D = np.sqrt(sqdist(Z, Z))
            np.fill_diagonal(D, 0.0)
            na, n = len(ia), len(ia) + len(ib)
            nb = n - na
            L = np.zeros((n, NPERM + 1))
            L[:na, 0] = 1
            for p in range(1, NPERM + 1):
                L[rng.permutation(n)[:na], p] = 1
            r, tot = D.sum(1), D.sum()
            sA = (L * (D @ L)).sum(0)
            sAB = r @ L - sA
            sB = tot - 2 * sAB - sA
            u = 2 * sAB / (na * nb) - sA / (na * (na - 1)) - sB / (nb * (nb - 1))
            null = u[1:]
            out["perm"].append({"space": name, "rep": "pca", "stage": STAGES[t], "sample_a": a, "sample_b": b,
                                "n_a": na, "n_b": nb, "same_batch": bat[smp == a][0] == bat[smp == b][0],
                                "ED_U": float(u[0]), "null_mean": float(null.mean()), "null_sd": float(null.std()),
                                "z": float((u[0] - null.mean()) / null.std()),
                                "p": float((1 + (null >= u[0]).sum()) / (1 + NPERM))})


def composition_tables(obs):
    ct = pd.crosstab(obs["sample"], obs["celltype"]).reindex(columns=TYPES, fill_value=0)
    info = obs.groupby("sample").agg(stage=("stage", "first"), batch=("sequencing.batch", "first"), n=("stage", "size"))
    by_s = info.join(ct).join((ct.div(ct.sum(axis=1), axis=0)).add_prefix("frac_"))
    by_s["stage_i"] = [STAGES.index(s) for s in by_s["stage"]]
    by_s = by_s.sort_values(["stage_i", "batch"]).drop(columns="stage_i")
    by_s.to_csv(os.path.join(HERE, "composition_by_sample.csv"))
    ct2 = pd.crosstab([obs["stage"], obs["sequencing.batch"]], obs["celltype"]).reindex(columns=TYPES, fill_value=0)
    ct1 = pd.crosstab(obs["stage"], obs["celltype"]).reindex(index=STAGES, columns=TYPES, fill_value=0)
    ct1.index = pd.MultiIndex.from_arrays([ct1.index, ["all"] * len(ct1)], names=["stage", "sequencing.batch"])
    by_t = pd.concat([ct1, ct2])
    by_t.insert(0, "n", by_t.sum(axis=1))
    by_t = by_t.join(by_t[TYPES].div(by_t["n"], axis=0).add_prefix("frac_"))
    by_t = by_t.reset_index()
    by_t["stage_i"] = [STAGES.index(s) for s in by_t["stage"]]
    by_t["b"] = by_t["sequencing.batch"].astype(str)
    by_t = by_t.sort_values(["stage_i", "b"]).drop(columns=["stage_i", "b"])
    by_t.to_csv(os.path.join(HERE, "composition_by_stage.csv"), index=False)
    return by_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--no-genes", action="store_true")
    args = ap.parse_args()
    open(LOG, "w").close()
    t_all = time.time()
    obs = pd.read_csv(os.path.join(args.data, "obs.csv"))
    obs["sample"] = obs["sample"].astype(str)
    obs["celltype"] = obs["celltype"].astype(str)
    codes = np.array([STAGES.index(s) for s in obs["stage"].astype(str)])
    with open(os.path.join(args.data, "forecast.json")) as fh:
        fc = json.load(fh)
    composition_tables(obs)
    out = {k: [] for k in ("halves", "stage_pairs", "sample_pairs", "type_pairs", "decomp", "within_batch",
                           "positions", "loso", "floors", "perm")}
    reps = ["pca"] if args.no_genes else ["pca", "genes"]
    rng = np.random.default_rng(0)
    for kind, ks in (("origin", range(1, 6)), ("interp", range(2, 6))):
        for k in ks:
            for rep in reps:
                E, idx, target = load_space(args.data, kind, k, rep, obs, codes)
                analyse(f"{kind}{k}", kind, k, rep, E, idx, target, obs, codes, fc, out)
                if rep == "pca" and (kind == "interp" or k == 5):
                    t0 = time.time()
                    perm_tests(f"{kind}{k}", E, idx, obs, codes, out, rng)
                    log(f"{kind}{k} permutation tests {time.time() - t0:.1f}s")
            for name, rows in out.items():  # checkpoint
                if rows:
                    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"{name}.csv"), index=False)
    summ = {"note": "DESCRIPTIVE diagnostics: distances use the true target stage (s+1 / held k). Not a forecast.",
            "min_cells_per_sample": MIN_CELLS, "min_cells_per_type": MIN_TYPE,
            "decomp_pca": [r for r in out["decomp"] if r["rep"] == "pca"],
            "floors": out["floors"], "within_batch_pca": [r for r in out["within_batch"] if r["rep"] == "pca"],
            "seconds": round(time.time() - t_all, 1)}
    with open(os.path.join(HERE, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log(f"done in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
