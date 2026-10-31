"""Judge: independent re-implementation of the forecasting benchmark for ONE origin (strict: stages <= s only).

Re-coded from the method descriptions (ery_forecast.py docstring, attempt_forecast.py docstring), using only
judge_lib (own Sinkhorn / energy / sd_fixed). Nothing is imported from the audited scripts.
  First-run families: persist, meanshift, ot (+noise), uot_growth, vel_ss, vel_dyn, ot_vel_ss, ot_vel_dyn, with the
  first-run backtest ('same' design: predict stage s from s-1 with the drift fitted on s-2 -> s-1, origin-s space,
  scored against stage s) and hp = argmin of sd_fixed / energy on that backtest (Mode B), plus the hp reported by
  diag_metric (rescore.json) and the recorded first-run hp (Mode A), all scored against the true stage s+1.
  Post-hoc families of attempt_forecast (only the ones that carry its conclusions): uot_rw, tilt_rw, mk_km,
  contract, ms_smooth, full grid on the final problem of this origin (the 'roll' selection for origin s+1 uses
  this origin's final grid, done in judge_report.py).
Scoring 'proto' = the audited protocol (S from sub(n_past, seed 7), prediction subsample seed 0, target seed 1);
'alt' = an independent protocol (S from 3000 past cells seed 123, prediction seed 10, target seed 11).
Output: judge/indep/forecast_o<s>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_lib as J  # noqa: E402

DT = 0.25
GRIDS = {
    "persist": [{}], "meanshift": [{}],
    "ot": [{"noise": n} for n in (0.0, 0.5, 1.0)],
    "uot_growth": [{"tau": t, "beta": b, "noise": 0.0} for t in (0.1, 1.0) for b in (0.0, 2.0, 5.0)],
    "vel_ss": [{"c": c} for c in (0.25, 0.5, 1, 2, 4, 8)],
    "vel_dyn": [{"c": c} for c in (0.25, 0.5, 1, 2, 4, 8)],
    "ot_vel_ss": [{"c": c, "w": w} for c in (0.5, 2, 8) for w in (0.25, 0.5, 0.75)],
    "ot_vel_dyn": [{"c": c, "w": w} for c in (0.5, 2, 8) for w in (0.25, 0.5, 0.75)],
}
NEW = {
    "uot_rw": [{"lam": 0.0, "tau": 1.0, "beta": 0.0}] +
              [{"lam": x, "tau": t, "beta": b} for t in (0.1, 1.0) for b in (0.0, 2.0) for x in (0.5, 1.0, 2.0)],
    "tilt_rw": [{"lam": x} for x in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)],
    "mk_km": [{"lam": x} for x in (0.0, 0.5, 1.0, 1.5, 2.0)],
    "contract": [{"a": x} for x in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)],
    "ms_smooth": [{"lam": x} for x in (0.0, 0.25, 0.5, 0.75, 1.0)],
}


def hk(hp):
    return json.dumps(hp, sort_keys=True)


class Origin:
    def __init__(self, d):
        self.d = d
        self.st = d["stage_past"]
        self.Z = d["Z_past"].astype(np.float64)
        self.X = d["X_past"].astype(np.float32)
        self.cache = {}

    def coupling(self, i_now, i_prev, tau=None, beta=None, genes=False):
        key = (i_now, i_prev, tau, beta)
        st = self.st
        if key in self.cache:
            out = self.cache[key]
            if genes and "dx" not in out:
                P, row, Zp = out["P"], out["P"].sum(1), self.Z[st == i_prev]
                Xp, Xn = self.X[st == i_prev].astype(np.float64), self.X[st == i_now].astype(np.float64)
                out["dx"] = J.nw_regress(Zp, (P @ Xn) / row[:, None] - Xp, self.Z[st == i_now])
            return out
        Zp, Zn = self.Z[st == i_prev], self.Z[st == i_now]
        C = J.sqdist(Zp, Zn)
        med = float(np.median(C))
        if tau is None:
            a = J.unif(len(Zp))
        else:
            a = np.exp(beta * self.d["cc_past"][st == i_prev].astype(np.float64) * DT)
            a = a / a.sum()
        P = J.plan(a, J.unif(len(Zn)), C / med, 0.05, tau)
        row = P.sum(1)
        DZ = (P @ Zn) / row[:, None] - Zp
        gr = row / a
        gr = gr / gr.mean()
        out = {"P": P, "dz": J.nw_regress(Zp, DZ, Zn),
               "gw": np.maximum(J.nw_regress(Zp, gr[:, None], Zn)[:, 0], 1e-6)}
        if genes:
            Xp, Xn = self.X[st == i_prev].astype(np.float64), self.X[st == i_now].astype(np.float64)
            out["dx"] = J.nw_regress(Zp, (P @ Xn) / row[:, None] - Xp, Zn)
        self.cache[key] = out
        return out

    def vel(self, i_now, kind):
        m = self.st == i_now
        V = self.d[kind][m].astype(np.float64) / (1.0 + self.d["Ms"][m].astype(np.float64))
        return V @ self.d["pca_comps"].astype(np.float64).T, V


def forecast_old(O, fam, hp, i_now, i_prev, rng, genes=False):
    """returns (Z, w, gene-space mean change or None)."""
    st = O.st
    Zn, Zp = O.Z[st == i_now], O.Z[st == i_prev]
    w, dxm = None, None
    if fam == "persist":
        return Zn, None, (np.zeros(O.X.shape[1]) if genes else None)
    if fam == "meanshift":
        dxm = (O.X[st == i_now].astype(np.float64).mean(0) - O.X[st == i_prev].astype(np.float64).mean(0)) if genes else None
        return Zn + (Zn.mean(0) - Zp.mean(0)), None, dxm
    dz = np.zeros_like(Zn)
    dx = 0.0
    if fam in ("ot", "uot_growth") or fam.startswith("ot_vel"):
        tau = hp["tau"] if fam == "uot_growth" else None
        beta = hp["beta"] if fam == "uot_growth" else None
        cp = O.coupling(i_now, i_prev, tau, beta, genes)
        dz = cp["dz"]
        dx = cp["dx"] if genes else 0.0
        if fam == "uot_growth":
            w = cp["gw"]
    if fam.startswith("vel") or fam.startswith("ot_vel"):
        vz, vx = O.vel(i_now, "v_ss" if fam.endswith("ss") else "v_dyn")
        if fam.startswith("vel"):
            dz, dx = hp["c"] * vz, hp["c"] * vx
        else:
            dz = hp["w"] * dz + (1 - hp["w"]) * hp["c"] * vz
            dx = hp["w"] * dx + (1 - hp["w"]) * hp["c"] * vx
    Zf = Zn + dz
    if hp.get("noise", 0) > 0:
        Zf = Zf + rng.normal(0.0, hp["noise"], Zf.shape)
    if genes:
        ww = J.unif(len(Zn)) if w is None else w / w.sum()
        Xn = O.X[st == i_now].astype(np.float64)
        dxm = ww @ (Xn + dx) - Xn.mean(0)
    return Zf, w, dxm


def comp_markov(P, lab_p, lab_n, lam):
    types = np.unique(np.r_[lab_p, lab_n])
    ip, inn = np.searchsorted(types, lab_p), np.searchsorted(types, lab_n)
    K = len(types)
    T = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            T[i, j] = P[np.ix_(ip == i, inn == j)].sum()
    M = np.eye(K)
    rs = T.sum(1)
    M[rs > 0] = T[rs > 0] / rs[rs > 0, None]
    pn = np.bincount(inn, minlength=K) / len(inn)
    phat = np.maximum(pn + lam * (pn @ M - pn), 0.0)
    phat = np.where(pn > 0, phat, 0.0)
    phat = phat / phat.sum()
    fac = np.where(pn > 0, phat / np.maximum(pn, 1e-300), 0.0)
    w = fac[inn] / len(inn)
    return w / w.sum()


def forecast_new(O, fam, hp, i_now, km):
    st = O.st
    Zn = O.Z[st == i_now]
    Zp = O.Z[st == i_now - 1]
    if fam == "contract":
        c = Zn.mean(0)
        return c + (1 - hp["a"]) * (Zn - c), None
    if fam == "ms_smooth":
        if i_now >= 2:
            drift = (Zn.mean(0) - O.Z[st == i_now - 2].mean(0)) / 2
        else:
            drift = Zn.mean(0) - Zp.mean(0)
        return Zn + hp["lam"] * drift, None
    if hp.get("lam", 0) == 0:
        return Zn, None
    if fam == "uot_rw":
        return Zn, O.coupling(i_now, i_now - 1, hp["tau"], hp["beta"])["gw"] ** hp["lam"]
    if fam == "tilt_rw":
        key = ("ratio", i_now)
        if key not in O.cache:
            Zall = np.vstack([Zp, Zn])
            lab = np.r_[np.zeros(len(Zp)), np.ones(len(Zn))]
            k = min(30, len(Zall) - 1)
            _, ind = NearestNeighbors(n_neighbors=k + 1).fit(Zall).kneighbors(Zn)
            q = (lab[ind[:, 1:]].sum(1) + 1) / (k + 2)
            O.cache[key] = np.clip(q / (1 - q) * len(Zp) / len(Zn), 0.05, 20.0)
        return Zn, O.cache[key] ** hp["lam"]
    if fam == "mk_km":
        P = O.coupling(i_now, i_now - 1)["P"]
        return Zn, comp_markov(P, km[st == i_now - 1], km[st == i_now], hp["lam"])
    raise KeyError(fam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "indep"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = a.s
    t0 = time.time()
    d = dict(np.load(os.path.join(a.data, f"origin{s}.npz"), allow_pickle=True))
    # strictness by construction: drop every array about the future stage before any forecast is built
    Zfut = d.pop("Z_fut").astype(np.float64)
    Xfut = d.pop("X_fut").astype(np.float64)
    for k_ in [k_ for k_ in d if k_.endswith("_fut")]:
        d.pop(k_)
    O = Origin(d)
    st = O.st
    S_proto = float(np.median(J.sqdist(O.Z[J.sub(len(O.Z), 7)], O.Z[J.sub(len(O.Z), 7)])[
        np.triu_indices(min(len(O.Z), 1500), 1)]))
    S_alt = J.past_scale(O.Z)
    out = {"origin": s, "S_proto": S_proto, "S_alt": S_alt, "old": {}, "new": {}, "backtest": {}}
    tg_fin = {"proto": J.Tgt(Zfut[J.sub(len(Zfut), 1)], S_proto), "alt": J.Tgt(Zfut[J.sub(len(Zfut), 11)], S_alt)}
    Xn_mean = O.X[st == s].astype(np.float64).mean(0)
    true_dx = Xfut.mean(0) - Xn_mean

    def sc(Z, w, tg, seed):
        return J.score(Z, w, tg, seed=seed)

    rescore = json.load(open(os.path.join(os.path.dirname(HERE), "diag_metric", "rescore.json")))
    rr = rescore["origins"].get(str(s))
    # ---------------- first-run families: backtest + finals
    if s >= 2:
        Zb = O.Z[st == s]
        tg_bt = J.Tgt(Zb[J.sub(len(Zb), 1)], S_proto)
        for fam, grid in GRIDS.items():
            rng = np.random.default_rng([s, 1234, len(fam)])
            bt = []
            if len(grid) > 1:
                for g in grid:
                    Zf, w, _ = forecast_old(O, fam, g, s - 1, s - 2, rng)
                    r = sc(Zf, w, tg_bt, 0)
                    bt.append({"hp": g, **r})
                sel = {m: grid[int(np.argmin([b[m] for b in bt]))] for m in ("sd_fixed", "energy", "energy_U")}
            else:
                sel = {m: grid[0] for m in ("sd_fixed", "energy", "energy_U")}
            out["backtest"][fam] = bt
            hps = {"judge_" + m: h for m, h in sel.items()}
            if rr is not None:
                fr = rr["families"][fam]["sel"]
                hps.update({"reported_" + m: fr[m] for m in ("sd_fixed", "energy")})
                hps["recorded"] = fr["recorded"]
            fin = {}
            for lab, hp in hps.items():
                key = hk(hp)
                if key in fin:
                    fin[key]["labels"].append(lab)
                    continue
                rng = np.random.default_rng([s, 99, len(fam), len(key)])
                Zf, w, dxm = forecast_old(O, fam, hp, s, s - 1, rng, genes=True)
                rec = {"labels": [lab], "hp": hp, "proto": sc(Zf, w, tg_fin["proto"], 0),
                       "alt": sc(Zf, w, tg_fin["alt"], 10)}
                rec["pcc"] = (float(np.corrcoef(dxm, true_dx)[0, 1]) if np.linalg.norm(dxm) > 1e-4 * np.linalg.norm(true_dx)
                              else float("nan"))
                fin[key] = rec
            out["old"][fam] = {"sel_judge": {m: hk(h) for m, h in sel.items()}, "final": fin}
            print(s, fam, {m: hk(h) for m, h in sel.items()}, f"{time.time() - t0:.0f}s", flush=True)
    # ---------------- post-hoc families: full grid on the final problem (proto scoring)
    km = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(O.Z)
    for fam, grid in NEW.items():
        res = {}
        for g in grid:
            Zf, w = forecast_new(O, fam, g, s, km)
            res[hk(g)] = sc(Zf, w, tg_fin["proto"], 0)
        out["new"][fam] = res
        print(s, fam, "grid done", f"{time.time() - t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["solver"] = dict(J.STATS)
    with open(os.path.join(a.out, f"forecast_o{s}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("done", s, out["seconds"], J.STATS, flush=True)


if __name__ == "__main__":
    main()
