"""Judge: independent re-implementation of the held-out-stage INTERPOLATION benchmark for one held stage k.

Re-coded from the method list in attempt_interp/interp_bench.py's docstring with judge_lib only (own Sinkhorn,
energies). Own subsample / pair-draw seeds (5000+r, 6000+r, 7000+r), so agreement with the audited numbers is a
reproduction, not a replay. Predictions use ONLY the fit stages k-1, k+1 (the held arrays are loaded into a separate
variable that the prediction code never receives); n_pred = min(n_held, 1500) as in the audited protocol.
Also: the descriptive metric probe (half A of the held stage shrunk toward its mean by lambda vs half B).
Output: judge/indep/interp_k<k>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_lib as J  # noqa: E402

T = 0.5
C_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
# Tirosh et al. 2016 S / G2M lists (public gene lists, as shipped with scVelo)
CC = set(("Mcm5 Pcna Tyms Fen1 Mcm2 Mcm4 Rrm1 Ung Gins2 Mcm6 Cdca7 Dtl Prim1 Uhrf1 Mlf1ip Hells Rfc2 Rpa2 Nasp "
          "Rad51ap1 Gmnn Wdr76 Slbp Ccne2 Ubr7 Pold3 Msh2 Atad2 Rad51 Rrm2 Cdc45 Cdc6 Exo1 Tipin Dscc1 Blm Casp8ap2 "
          "Usp1 Clspn Pola1 Chaf1b Brip1 E2f8 Hmgb2 Cdk1 Nusap1 Ube2c Birc5 Tpx2 Top2a Ndc80 Cks2 Nuf2 Cks1b Mki67 "
          "Tmpo Cenpf Tacc3 Fam64a Smc4 Ccnb2 Ckap2l Ckap2 Aurkb Bub1 Kif11 Anp32e Tubb4b Gtse1 Kif20b Hjurp Cdca3 "
          "Hn1 Cdc20 Ttk Cdc25c Kif2c Rangap1 Ncapd2 Dlgap5 Cdca2 Cdca8 Ect2 Kif23 Hmmr Aurka Psrc1 Anln Lbr Ckap5 "
          "Cenpe Ctcf Nek2 G2e3 Gas2l3 Cbx5 Cenpa").split())


class GTgt:
    """energy-only target (gene space)."""

    def __init__(self, Y):
        self.Y = np.asarray(Y, np.float64)
        self.w = J.unif(len(Y))
        self.yyD = float(self.w @ np.sqrt(J.sqdist(self.Y, self.Y)) @ self.w)
        self.sw2 = float((self.w ** 2).sum())


def predict(ZA, XA, ZB, XB, ccA, comps, n, rng, VA, c_fit):
    """All interpolants from the two fit stages only. Returns {name: (Z, X, w)}."""
    out = {}
    nA, nB = len(ZA), len(ZB)
    out["prev"] = (ZA, XA, J.unif(nA))
    out["next"] = (ZB, XB, J.unif(nB))
    mA = min(nA, n // 2)
    mB = min(nB, n - mA)
    iA, iB = np.sort(rng.choice(nA, mA, replace=False)), np.sort(rng.choice(nB, mB, replace=False))
    out["union"] = (np.vstack([ZA[iA], ZB[iB]]), np.vstack([XA[iA], XB[iB]]),
                    np.r_[np.full(mA, 0.5 / mA), np.full(mB, 0.5 / mB)])
    i, j = rng.integers(0, nA, n), rng.integers(0, nB, n)
    out["indep"] = ((1 - T) * ZA[i] + T * ZB[j], (1 - T) * XA[i] + T * XB[j], J.unif(n))
    C = J.sqdist(ZA, ZB)
    med = float(np.median(C))
    P = J.plan(J.unif(nA), J.unif(nB), C / med, 0.05)
    p = P.ravel() / P.sum()
    i, j = np.unravel_index(rng.choice(p.size, n, p=p), P.shape)
    Zp, Xp = (1 - T) * ZA[i] + T * ZB[j], (1 - T) * XA[i] + T * XB[j]
    out["ot_pairs"] = (Zp, Xp, J.unif(n))
    eps_abs = 0.05 * med
    for name, var in (("sb", T * (1 - T) * eps_abs), ("sb_half", T * (1 - T) * eps_abs / 2)):
        nz = rng.normal(0, np.sqrt(var), Zp.shape)
        out[name] = (Zp + nz, Xp + nz @ comps, J.unif(n))
    r, c = P.sum(1), P.sum(0)
    fZ = ZA + T * ((P @ ZB) / r[:, None] - ZA)
    fX = XA + T * ((P @ XB) / r[:, None] - XA)
    out["ot_bary"] = (fZ, fX, r / r.sum())
    bZ = ZB + (1 - T) * ((P.T @ ZA) / c[:, None] - ZB)
    bX = XB + (1 - T) * ((P.T @ XA) / c[:, None] - XB)
    mA = min(nA, n // 2)
    mB = min(nB, n - mA)
    iA, iB = np.sort(rng.choice(nA, mA, replace=False)), np.sort(rng.choice(nB, mB, replace=False))
    out["ot_bary_sym"] = (np.vstack([fZ[iA], bZ[iB]]), np.vstack([fX[iA], bX[iB]]),
                          np.r_[r[iA] / r[iA].sum() * 0.5, c[iB] / c[iB].sum() * 0.5])
    for name, beta in (("uot", 2.0), ("uot_beta0", 0.0)):
        a = np.exp(beta * ccA * 0.5)
        a = a / a.sum()
        Pu = J.plan(a, J.unif(nB), C / med, 0.05, tau=1.0)
        ru = Pu.sum(1)
        u = np.full(nA, Pu.sum() / nA)
        W = Pu * ((u / ru) ** (1 - T))[:, None]
        q = W.ravel() / W.sum()
        i, j = np.unravel_index(rng.choice(q.size, n, p=q), W.shape)
        out[name] = ((1 - T) * ZA[i] + T * ZB[j], (1 - T) * XA[i] + T * XB[j], J.unif(n))
    vz, vx = VA
    for m, cc in c_fit.items():
        out[f"vel_ss_{m}"] = (ZA + cc / 2 * vz, XA + cc / 2 * vx, J.unif(nA))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "indep"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    k = a.k
    t0 = time.time()
    d = dict(np.load(os.path.join(a.data, f"interp{k}.npz"), allow_pickle=True))
    held_Z = d.pop("Z_held").astype(np.float64)       # scoring only
    held_X = d.pop("X_held").astype(np.float64)
    n_held = len(held_Z)
    for k_ in [x for x in d if x.endswith("_held")]:
        d.pop(k_)
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float64)
    comps = d["pca_comps"].astype(np.float64)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in CC for g in genes])
    Xcc = Xf[:, ccm]
    cc = (Xcc - Xcc.mean(0)).mean(1)
    S_proto = float(np.median(J.sqdist(Zf[J.sub(len(Zf), 7)], Zf[J.sub(len(Zf), 7)])[
        np.triu_indices(min(len(Zf), 1500), 1)]))
    S_alt = J.past_scale(Zf)
    S = S_alt
    n = min(n_held, 1500)
    it = J.sub(n_held, 11)
    tgt = J.Tgt(held_Z[it], S)
    gtgt = GTgt(held_X[it])
    V = d["v_ss"].astype(np.float64) / (1.0 + d["Ms"].astype(np.float64))
    # strict velocity scale: k-1 -> k+1 only
    iA0, iB0 = J.sub(len(A), 5000), J.sub(len(B), 6000)
    tB = J.Tgt(Zf[B[iB0]], S)
    vzA0 = V[A[iA0]] @ comps.T
    curve = {"sd_fixed": [], "energy_U": []}
    for c_ in C_GRID:
        r = J.score(Zf[A[iA0]] + c_ * vzA0, None, tB)
        curve["sd_fixed"].append(r["sd_fixed"])
        curve["energy_U"].append(r["energy_U"])
    c_fit = {m: C_GRID[int(np.argmin(v))] for m, v in curve.items()}
    res = {"k": k, "n_held": n_held, "n_pred": n, "S_proto": S_proto, "S_alt": S_alt, "cc_genes": int(ccm.sum()),
           "c_fit": c_fit, "reps": []}
    for rep in range(a.reps):
        rng = np.random.default_rng(7000 + rep)
        iA, iB = A[J.sub(len(A), 5000 + rep)], B[J.sub(len(B), 6000 + rep)]
        M = predict(Zf[iA], Xf[iA], Zf[iB], Xf[iB], cc[iA], comps, n, rng, (V[iA] @ comps.T, V[iA]), c_fit)
        rr = {}
        for name, (Z, X, w) in M.items():
            if len(Z) > n:
                ii = np.sort(rng.choice(len(Z), n, replace=False))
                Z, X, w = Z[ii], X[ii], w[ii]
            w = w / w.sum()
            ev, eu = J.energies(Z, w, tgt)
            cross = float(w @ np.sqrt(J.sqdist(X, gtgt.Y)) @ gtgt.w)
            xx = float(w @ np.sqrt(J.sqdist(X, X)) @ w)
            eug = 2 * cross - xx / (1 - (w ** 2).sum()) - gtgt.yyD / (1 - gtgt.sw2)
            mu = w @ Z
            rr[name] = {"ED_U": eu, "ED_V": ev, "sd_fixed": J.sd_fixed(Z, w, tgt), "ED_U_gene": eug,
                        "spread_ratio": float(np.sqrt(w @ ((Z - mu) ** 2).sum(1)) /
                                              np.sqrt(((tgt.Y - tgt.Y.mean(0)) ** 2).sum(1).mean()))}
        res["reps"].append(rr)
        print(k, "rep", rep, {m: round(v["ED_U"], 3) for m, v in rr.items()}, f"{time.time() - t0:.0f}s", flush=True)
    # descriptive metric probe: shrink half A of the held stage by lambda, score against half B
    perm = np.random.default_rng(44).permutation(n_held)
    hA, hB = perm[: n_held // 2][:1500], perm[n_held // 2:][:1500]
    tH = J.Tgt(held_Z[hB], S)
    probe = []
    for lam in (1.0, 0.9, 0.85, 0.8, 0.7):
        Z = held_Z[hA]
        Z = Z.mean(0) + lam * (Z - Z.mean(0))
        ev, eu = J.energies(Z, J.unif(len(Z)), tH)
        probe.append({"lambda": lam, "ED_U": eu, "ED_V": ev, "sd_fixed": J.sd_fixed(Z, J.unif(len(Z)), tH)})
    res["probe"] = probe
    res["seconds"] = round(time.time() - t0, 1)
    res["solver"] = dict(J.STATS)
    with open(os.path.join(a.out, f"interp_k{k}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("done", k, res["seconds"], J.STATS, flush=True)


if __name__ == "__main__":
    main()
