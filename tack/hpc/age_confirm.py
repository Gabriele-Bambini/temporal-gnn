"""Pre-registered test: does the estimated EMBRYO AGE beat the nominal STAGE label? (tack/prereg/2026-10-26-embryo-age-prereg.md)

Motivation (exploratory diagnosis, tack/results/atlas/diag/diagnosis.md): on the Pijuan-Sala atlas the stage label is a
noisy clock (pools of one nominal stage differ by up to 1.5 stages in estimated age), and the 50/50 mix of the
neighbour stages beats every OT interpolation. Here every biological embryo pool (meta 'pool') gets an age estimated
from its OWN cells by a model trained on the OTHER pools only (leave-one-pool-out multinomial logistic regression of the
nominal stage on the standardised PCs, expected stage). Ages are calibrated to the stage scale by a linear fit of age on
nominal stage over the reference pools (alpha, beta, residual sd sigma). Predictions are mixtures of real reference
cells; each pool's mass is kernel weight x its number of cells. No graph, no transport, no tuning.

H1, interpolation of held stage k = 1..7, product fitted on all stages != k (atlas_prep.transform, fit-only):
  union   50/50 of stages k-1 and k+1, cells uniform within stage (the champion of the confirmation study)
  age     kernel exp(-(a_p - t_k)^2 / 2 sigma^2) on the estimated ages, t_k = alpha + beta k
  stagek  same kernel on the calibrated NOMINAL stage (alpha + beta nom_p): more pools, no asynchrony information
  shufw   age with ages permuted among pools of the same nominal stage (20 permutations, mean ED_U)
H2, forecasting stage s+1 from stages <= s (origin<s>.npz), s = 2..7:
  persist stage s cells uniform     agef  kernel on ages around t_{s+1} = alpha + beta (s+1)
  stagef  same kernel on the calibrated nominal stage           shuff agef with within-stage permuted ages
Metric ED_U (interp_bench_frozen.ed_uv) in the PCA space of the product, target = held / future stage subsample
(<= 1500, seed 1); secondary: per-pool ED_U. Usage: python age_confirm.py [--out PATH] [--ks ..] [--origins ..]
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_bench_frozen as IB  # noqa: E402

SUB_PER_POOL, TRAIN_PER_POOL, REPS, NPERM, C_REG, MIN_POOL = 300, 600, 3, 20, 0.1, 20
ORIGIN_DIR = os.path.expanduser("~/tack/out/atlas")


# ---------------------------------------------------------------- age model

def loo_ages(Z, pools, labels, seed=0):
    """Leave-one-pool-out expected nominal stage of each pool (mean over its cells)."""
    Zs = StandardScaler().fit_transform(Z)
    rng = np.random.default_rng(seed)
    uniq = sorted(set(pools))
    idx = {p: np.where(pools == p)[0] for p in uniq}
    tr_idx = {p: rng.choice(ix, min(len(ix), TRAIN_PER_POOL), replace=False) for p, ix in idx.items()}
    ages = {}
    for p in uniq:
        tr = np.concatenate([tr_idx[q] for q in uniq if q != p])
        clf = LogisticRegression(C=C_REG, max_iter=3000).fit(Zs[tr], labels[tr])
        ages[p] = float((clf.predict_proba(Zs[idx[p]]) @ clf.classes_).mean())
    return ages


def calibrate(ages, nominal):
    """Linear map nominal stage -> estimated age over the reference pools; sigma = residual sd (floor 0.05)."""
    ps = sorted(ages)
    a = np.array([ages[p] for p in ps])
    n = np.array([nominal[p] for p in ps], float)
    beta, alpha = np.polyfit(n, a, 1)
    res = a - (alpha + beta * n)
    sigma = float(max(np.sqrt((res ** 2).sum() / max(len(ps) - 2, 1)), 0.05))
    return float(alpha), float(beta), sigma


def kernel(vals, t, sigma):
    return {p: float(np.exp(-(v - t) ** 2 / (2 * sigma ** 2))) for p, v in vals.items()}


def shuffle_within(ages, nominal, rng):
    out = dict(ages)
    for s in sorted(set(nominal.values())):
        ps = [p for p in ages if nominal[p] == s]
        vals = rng.permutation([ages[p] for p in ps])
        out.update(dict(zip(ps, vals)))
    return out


# ---------------------------------------------------------------- predictions

def candidates(pools, rep):
    """Per pool <= SUB_PER_POOL cell indices (seeded by rep and pool order)."""
    out = {}
    for i, p in enumerate(sorted(set(pools))):
        ix = np.where(pools == p)[0]
        r = np.random.default_rng(10007 * rep + i)
        out[p] = np.sort(r.choice(ix, min(len(ix), SUB_PER_POOL), replace=False))
    return out


def mixture(Z, cand, mass):
    """Weighted point set: pool p carries total mass[p], spread uniformly over its candidate cells."""
    tot = sum(mass.values())
    keep = [p for p in cand if mass.get(p, 0.0) > 1e-8 * tot]
    P = np.vstack([Z[cand[p]] for p in keep])
    w = np.concatenate([np.full(len(cand[p]), mass[p] / len(cand[p])) for p in keep])
    return P, w / w.sum()


def cell_mass(pools, wpool):
    """Pool mass = kernel weight x number of cells of the pool."""
    return {p: wpool[p] * float((pools == p).sum()) for p in wpool}


def stage_mass(pools, nominal, stages_w):
    """Cells uniform within each listed stage, stage s carrying total weight stages_w[s]."""
    out = {}
    for s, ws in stages_w.items():
        ps = [p for p in nominal if nominal[p] == s]
        n = sum((pools == p).sum() for p in ps)
        out.update({p: ws * float((pools == p).sum()) / n for p in ps})
    return out


def make_target(Zt, pools_t, S):
    it = IB.sub(len(Zt), seed=1)
    tgt = IB.Target(Zt[it], None, S, with_sd=False)
    ptg = {p: IB.Target(Zt[pools_t == p], None, S, with_sd=False)
           for p in sorted(set(pools_t)) if (pools_t == p).sum() >= MIN_POOL}
    return tgt, ptg


def score(P, w, tgt, ptg):
    return IB.ed_uv(P, w, tgt)[0], {p: IB.ed_uv(P, w, t)[0] for p, t in ptg.items()}


def run_unit(Zr, pools_r, nominal, t_stage, base_mass, Zt, pools_t, ages=None, seed=0):
    """All methods for one unit. Zr/pools_r: reference cells; nominal: pool -> nominal stage; t_stage: target stage;
    base_mass: the champion baseline (union or persist) as pool masses."""
    ages = loo_ages(Zr, pools_r, np.array([nominal[p] for p in pools_r]), seed=seed) if ages is None else ages
    alpha, beta, sigma = calibrate(ages, nominal)
    t = alpha + beta * t_stage
    w_age = kernel(ages, t, sigma)
    w_stage = kernel({p: alpha + beta * nominal[p] for p in ages}, t, sigma)
    S = IB.med_scale(Zr)
    tgt, ptg = make_target(Zt, pools_t, S)
    rng = np.random.default_rng(seed + 99)
    perms = [shuffle_within(ages, nominal, rng) for _ in range(NPERM)]
    names = ("base", "age", "stagek", "shufw")
    rec = {m: {"ED_U": [], "pool": []} for m in names}
    for rep in range(REPS):
        cand = candidates(pools_r, rep)
        masses = {"base": base_mass, "age": cell_mass(pools_r, w_age), "stagek": cell_mass(pools_r, w_stage)}
        for m, ms in masses.items():
            e, pe = score(*mixture(Zr, cand, ms), tgt, ptg)
            rec[m]["ED_U"].append(e)
            rec[m]["pool"].append(pe)
        es = [score(*mixture(Zr, cand, cell_mass(pools_r, kernel(pa, t, sigma))), tgt, ptg) for pa in perms]
        rec["shufw"]["ED_U"].append(float(np.mean([e for e, _ in es])))
        rec["shufw"]["pool"].append({p: float(np.mean([pe[p] for _, pe in es])) for p in ptg})
    out = {m: {"ED_U": float(np.mean(v["ED_U"])), "ED_U_reps": v["ED_U"],
               "pool_ED_U": {p: float(np.mean([r[p] for r in v["pool"]])) for p in ptg}} for m, v in rec.items()}
    tot = sum(w_age.values())
    out["_info"] = {"ages": ages, "nominal": nominal, "alpha": alpha, "beta": beta, "sigma": sigma, "t": t,
                    "w_age": {p: v / tot for p, v in w_age.items()},
                    "eff_pools_age": float(tot ** 2 / sum(v * v for v in w_age.values()))}
    return out


# ---------------------------------------------------------------- data

def pool_maps():
    import pandas as pd
    import atlas_prep as AP
    meta = pd.read_csv(AP.META, sep="\t", usecols=["cell", "sample", "pool"])
    s2p = dict(zip(meta["sample"].astype(str), meta["pool"].astype(str)))
    return meta, s2p


def interp_units(ks):
    import anndata as ad
    import pandas as pd
    import atlas_prep as AP
    X, obs, genes = AP.load()
    meta, _ = pool_maps()
    obs = obs.merge(meta[["cell", "pool"]], on="cell", how="left")
    stage = np.array([AP.STAGES.index(s) for s in obs["stage"]])
    pool = obs["pool"].astype(str).to_numpy()
    probe = ad.AnnData(X=np.zeros((2, len(genes)), np.float32))
    probe.var_names = pd.Index(genes).astype(str)
    probe.var_names_make_unique()
    s_g, g2m = importlib.import_module("scvelo.tools.score_genes_cell_cycle").get_phase_marker_genes(probe)
    cc_genes = set(str(x).capitalize() for x in list(s_g) + list(g2m))
    res = {}
    for k in ks:
        t0 = time.time()
        fit, held = np.where(stage != k)[0], np.where(stage == k)[0]
        r = AP.transform(X, fit, held, genes, cc_genes)
        Zr, Zt = r["Z_fit"].astype(np.float64), r["Z_other"].astype(np.float64)
        pr, pt, sr = pool[fit], pool[held], stage[fit]
        nominal = {p: int(np.bincount(sr[pr == p]).argmax()) for p in sorted(set(pr))}
        base = stage_mass(pr, nominal, {k - 1: 0.5, k + 1: 0.5})
        res[k] = run_unit(Zr, pr, nominal, k, base, Zt, pt, seed=k)
        res[k]["_info"]["sec"] = round(time.time() - t0, 1)
        print("interp", k, {m: round(v["ED_U"], 3) for m, v in res[k].items() if m != "_info"},
              "sigma", round(res[k]["_info"]["sigma"], 3), f"[{res[k]['_info']['sec']}s]", flush=True)
    return res


def forecast_units(origins):
    _, s2p = pool_maps()
    res = {}
    for s in origins:
        t0 = time.time()
        d = np.load(os.path.join(ORIGIN_DIR, f"origin{s}.npz"), allow_pickle=True)
        Zr, Zt = d["Z_past"].astype(np.float64), d["Z_fut"].astype(np.float64)
        pr = np.array([s2p[str(x)] for x in d["sample_past"]])
        pt = np.array([s2p[str(x)] for x in d["sample_fut"]])
        sr = d["stage_past"]
        nominal = {p: int(np.bincount(sr[pr == p]).argmax()) for p in sorted(set(pr))}
        base = stage_mass(pr, nominal, {s: 1.0})
        res[s] = run_unit(Zr, pr, nominal, s + 1, base, Zt, pt, seed=100 + s)
        res[s]["_info"]["sec"] = round(time.time() - t0, 1)
        print("forecast", s, {m: round(v["ED_U"], 3) for m, v in res[s].items() if m != "_info"},
              "sigma", round(res[s]["_info"]["sigma"], 3), f"[{res[s]['_info']['sec']}s]", flush=True)
    return res


# ---------------------------------------------------------------- verdicts

def beats(res, a, bs):
    units = sorted(res)
    n = len(units)
    need = int(np.ceil(2 * n / 3))
    out = {}
    for b in bs:
        wins = sum(res[u][a]["ED_U"] < res[u][b]["ED_U"] for u in units)
        mean_ok = np.mean([res[u][a]["ED_U"] for u in units]) < np.mean([res[u][b]["ED_U"] for u in units])
        out[b] = {"wins": int(wins), "of": n, "need": need, "mean_ok": bool(mean_ok), "pass": bool(wins >= need and mean_ok)}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def verdicts(ri, rf):
    v = {}
    if ri:
        v["H1_age_vs_union"] = beats(ri, "age", ["base"])
        v["H1b_age_vs_stagek"] = beats(ri, "age", ["stagek"])
        v["H1c_age_vs_shufw"] = beats(ri, "age", ["shufw"])
    if rf:
        v["H2_age_vs_persist"] = beats(rf, "age", ["base"])
        v["H2b_age_vs_shufw"] = beats(rf, "age", ["shufw"])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ORIGIN_DIR, "age_confirm.json"))
    ap.add_argument("--ks", type=int, nargs="*", default=list(range(1, 8)))
    ap.add_argument("--origins", type=int, nargs="*", default=list(range(2, 8)))
    a = ap.parse_args()
    t0 = time.time()
    ri = interp_units(a.ks) if a.ks else {}
    rf = forecast_units(a.origins) if a.origins else {}
    out = {"interp": ri, "forecast": rf, "verdicts": verdicts(ri, rf), "seconds": round(time.time() - t0, 1)}
    print("VERDICTS", json.dumps(out["verdicts"]), flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
