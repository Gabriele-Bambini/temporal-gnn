"""Pre-registered CONFIRMATION of the two erythroid OT signals (tack/prereg/2026-10-25-atlas-confirmation-prereg.md).

H1  interpolation of a held-out stage k from stages k-1 and k+1: ot_pairs (McCann pairs from the entropic plan) and
    uot (KL-unbalanced) against prev, next and the 50/50 union. Methods, subsamples, seeds and scoring are those of the
    erythroid study: interp_bench_frozen.py (byte copy of tack/results/ery/attempt_interp/interp_bench.py) is imported
    and its methods(), Target, ed_uv, subsample_pred, med_scale, sub are used unchanged (3 reps, DEFAULT
    eps_rel 0.05, tau 1, beta 2, proliferation from its S/G2M lists among the HVGs, centred on fit cells).
H2  forecasting stage s+1 from stages <= s: uot_rw (stage-s cells unmoved, reweighted by UOT growth^lambda, the
    erythroid definition) against persist and meanshift; control tilt_rw (kNN density ratio^lambda, no OT).
    Same grids as tack/results/ery/attempt_forecast; hyperparameters by ROLLING ORIGIN: every grid point forecasts
    stage s from origin s-1's file and space, and the argmin of ED_U is used at origin s.
Primary metric ED_U (energy distance, U-statistic, weighted) in the PCA space of the product; secondary ED_U in gene
space, per-pool ED_U, gene-level PCC of the mean change. Verdict rules are in the pre-registration and in main().
Usage: python atlas_confirm.py --data <dir with origin*.npz, interp*.npz> --out <out.json> [--ks ..] [--origins ..]
       [--parity]  (parity: the erythroid products, k = 2..5 and origins 2..5, to reproduce the erythroid numbers)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_bench_frozen as IB  # noqa: E402
from ery_forecast import kernel_regress  # noqa: E402

DT = 0.25
K_NN = 30
MIN_POOL = 20
UOT_GRID = [{"lam": 0.0, "tau": 1.0, "beta": 0.0}] + [
    {"lam": x, "tau": t, "beta": b} for t in (0.1, 1.0) for b in (0.0, 2.0) for x in (0.5, 1.0, 2.0)]
TILT_GRID = [{"lam": x} for x in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)]
INTERP_METHODS = ("prev", "next", "union", "indep", "ot_pairs", "sb_half", "uot")


# ---------------------------------------------------------------- H1: interpolation (frozen erythroid pipeline)

def run_interp(data, k, reps=3):
    d = np.load(os.path.join(data, f"interp{k}.npz"), allow_pickle=True)
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
    comps = d["pca_comps"].astype(np.float64)
    smp_held = d["sample_held"].astype(str)
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in set(IB.S_GENES + IB.G2M_GENES) for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    cc_fit = (Xcc - Xcc.mean(0)).mean(1)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], Xh[it], S, with_sd=False)
    pools = [p for p in sorted(set(smp_held)) if (smp_held == p).sum() >= MIN_POOL]
    ptg = {p: IB.Target(Zh[smp_held == p], None, S, with_sd=False) for p in pools}
    out = {"n_prev": int(len(A)), "n_next": int(len(B)), "n_held": int(len(Zh)), "cc_genes": int(ccm.sum()),
           "pools": pools, "reps": []}
    for r in range(reps):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        M.pop("_plan", None)
        M.pop("_uot", None)
        rec = {}
        for name, (Z, X, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
            if name not in INTERP_METHODS:
                continue
            rec[name] = {"ED_U": IB.ed_uv(Zs, ws, tgt)[0], "ED_U_gene": IB.ed_uv(Xs, ws, tgt, gene=True)[0],
                         "pool_ED_U": {p: IB.ed_uv(Zs, ws, ptg[p])[0] for p in pools}}
        out["reps"].append(rec)
    out["mean"] = {m: {key: float(np.mean([rp[m][key] for rp in out["reps"]])) for key in ("ED_U", "ED_U_gene")}
                   for m in INTERP_METHODS}
    out["pool_mean"] = {m: {p: float(np.mean([rp[m]["pool_ED_U"][p] for rp in out["reps"]])) for p in pools}
                        for m in INTERP_METHODS}
    return out


# ---------------------------------------------------------------- H2: forecasting by reweighting

class Problem:
    """Forecast stage i_now + 1 from stages <= i_now with the arrays of one origin file (all strict)."""

    def __init__(self, d, i_now, Zt, Xt, smp_t):
        st = d["stage_past"]
        now, prev = st == i_now, st == i_now - 1
        self.Zn, self.Xn = d["Z_past"][now].astype(np.float64), d["X_past"][now].astype(np.float32)
        self.Zp = d["Z_past"][prev].astype(np.float64)
        self.Xp = d["X_past"][prev].astype(np.float32)
        self.cc_p = d["cc_past"][prev]
        self.S = IB.med_scale(d["Z_past"].astype(np.float64))
        it = IB.sub(len(Zt), seed=1)
        self.tgt = IB.Target(Zt[it].astype(np.float64), Xt[it].astype(np.float64), self.S, with_sd=False)
        self.Xt = Xt.astype(np.float64)
        pools = [p for p in sorted(set(smp_t)) if (smp_t == p).sum() >= MIN_POOL]
        self.ptg = {p: IB.Target(Zt[smp_t == p].astype(np.float64), None, self.S, with_sd=False) for p in pools}
        self.cache = {}

    def growth(self, tau, beta):
        key = (tau, beta)
        if key not in self.cache:
            a = np.exp(beta * self.cc_p * DT)
            P, _ = IB.plan(self.Zp, self.Zn, 0.05, tau=tau, a=a)
            gr = P.sum(1) / (a / a.sum())
            gr = gr / gr.mean()
            self.cache[key] = np.maximum(kernel_regress(self.Zp, gr[:, None], self.Zn)[:, 0], 1e-6)
        return self.cache[key]

    def ratio(self):
        if "ratio" not in self.cache:
            Zall = np.vstack([self.Zp, self.Zn])
            lab = np.r_[np.zeros(len(self.Zp)), np.ones(len(self.Zn))]
            k = min(K_NN, len(Zall) - 1)
            _, ind = NearestNeighbors(n_neighbors=k + 1).fit(Zall).kneighbors(self.Zn)
            q = (lab[ind[:, 1:]].sum(1) + 1) / (k + 2)
            self.cache["ratio"] = np.clip(q / (1 - q) * len(self.Zp) / len(self.Zn), 0.05, 20.0)
        return self.cache["ratio"]

    def forecast(self, fam, hp):
        if fam == "persist":
            return self.Zn, self.Xn, None
        if fam == "meanshift":
            return self.Zn + (self.Zn.mean(0) - self.Zp.mean(0)), self.Xn + (self.Xn.mean(0) - self.Xp.mean(0)), None
        if hp["lam"] == 0:
            return self.Zn, self.Xn, None
        w = self.growth(hp["tau"], hp["beta"]) if fam == "uot_rw" else self.ratio()
        return self.Zn, self.Xn, w ** hp["lam"]

    def score(self, Zf, Xf, w, pools=False):
        i = IB.sub(len(Zf))
        ww = np.full(len(Zf), 1.0 / len(Zf)) if w is None else w / w.sum()
        if w is not None:
            i = i[ww[i] > 0]
        wi = ww[i] / ww[i].sum()
        eu, ev = IB.ed_uv(Zf[i], wi, self.tgt)
        out = {"ED_U": eu, "ED_V": ev, "ED_U_gene": IB.ed_uv(Xf[i].astype(np.float64), wi, self.tgt, gene=True)[0],
               "ess": float(1.0 / (wi ** 2).sum())}
        pred = ww @ Xf.astype(np.float64) - self.Xn.astype(np.float64).mean(0)
        true = self.Xt.mean(0) - self.Xn.astype(np.float64).mean(0)
        out["pcc"] = float("nan") if np.linalg.norm(pred) <= 1e-4 * np.linalg.norm(true) else float(np.corrcoef(pred, true)[0, 1])
        if pools:
            out["pool_ED_U"] = {p: IB.ed_uv(Zf[i], wi, t)[0] for p, t in self.ptg.items()}
        return out


def origin_problem(data, s):
    d = np.load(os.path.join(data, f"origin{s}.npz"), allow_pickle=True)
    smp_t = d["sample_fut"].astype(str) if "sample_fut" in d.files else np.full(len(d["Z_fut"]), "all")
    return Problem(d, s, d["Z_fut"], d["X_fut"].astype(np.float32), smp_t)


def run_forecast(data, s):
    final, back = origin_problem(data, s), origin_problem(data, s - 1)
    out = {}
    for fam, grid in (("persist", [{}]), ("meanshift", [{}]), ("uot_rw", UOT_GRID), ("tilt_rw", TILT_GRID)):
        bt = [back.score(*back.forecast(fam, hp))["ED_U"] for hp in grid]
        hp = grid[int(np.argmin(bt))]
        out[fam] = {"hp": hp, "backtest_ED_U": float(min(bt)), **final.score(*final.forecast(fam, hp), pools=True)}
    return out


# ---------------------------------------------------------------- verdicts (pre-registered)

def beats(res, a, b_list, need):
    """a beats every b: strictly lower in >= need units and lower on the mean."""
    units = sorted(res)
    wins = [all(res[u][a] < res[u][b] for b in b_list) for u in units]
    mean_ok = all(np.mean([res[u][a] for u in units]) < np.mean([res[u][b] for u in units]) for b in b_list)
    return {"wins": int(sum(wins)), "of": len(units), "need": need, "mean_ok": bool(mean_ok),
            "pass": bool(sum(wins) >= need and mean_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=list(range(1, 8)))
    ap.add_argument("--origins", type=int, nargs="+", default=list(range(2, 8)))
    ap.add_argument("--parity", action="store_true", help="erythroid products: k = 2..5, origins 2..5")
    args = ap.parse_args()
    if args.parity:
        args.ks, args.origins = [2, 3, 4, 5], [2, 3, 4, 5]
    t0 = time.time()
    res = {"interp": {}, "forecast": {}, "config": vars(args)}
    for k in args.ks:
        res["interp"][k] = run_interp(args.data, k)
        print("interp", k, {m: round(v["ED_U"], 3) for m, v in res["interp"][k]["mean"].items()}, f"[{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(args.out, "w"), indent=1, default=float)
    for s in args.origins:
        res["forecast"][s] = run_forecast(args.data, s)
        print("forecast", s, {f: (round(v["ED_U"], 3), v["hp"]) for f, v in res["forecast"][s].items()}, f"[{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(args.out, "w"), indent=1, default=float)
    ki = {k: {m: v["ED_U"] for m, v in r["mean"].items()} for k, r in res["interp"].items()}
    fo = {s: {f: v["ED_U"] for f, v in r.items()} for s, r in res["forecast"].items()}
    need_i = int(np.ceil(2 * len(ki) / 3)) if ki else 0
    need_f = int(np.ceil(2 * len(fo) / 3)) if fo else 0
    res["verdicts"] = {
        "H1_ot_pairs": beats(ki, "ot_pairs", ["prev", "next", "union"], need_i) if ki else None,
        "H1b_uot": beats(ki, "uot", ["prev", "next", "union"], need_i) if ki else None,
        "H1c_pairing": beats(ki, "ot_pairs", ["indep"], need_i) if ki else None,
        "H2_uot_rw": beats(fo, "uot_rw", ["persist", "meanshift"], need_f) if fo else None,
        "H2b_ot_needed": beats(fo, "uot_rw", ["tilt_rw"], need_f) if fo else None,
    }
    res["seconds"] = round(time.time() - t0, 1)
    print("VERDICTS", json.dumps(res["verdicts"]), flush=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
