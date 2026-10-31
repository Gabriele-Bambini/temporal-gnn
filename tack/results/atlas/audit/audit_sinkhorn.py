"""AUDIT 2 (OT-only code path): Sinkhorn convergence, marginals and plan sharpness of the frozen IB.plan on the atlas,
next to the erythroid products. Every plan the frozen run builds is rebuilt here:
  interp k (rep-0 subsamples, exactly as atlas_confirm.run_interp): balanced plan (ot_pairs / sb_half / indep-free) and
  KL-unbalanced plan (uot, tau 1, beta 2, a = exp(beta * cc * 0.5));
  origin s (full stage s-1 -> stage s, exactly as atlas_confirm.Problem.growth): the 4 (tau, beta) growth plans.
A faithful instrumented copy of IB.sinkhorn (asserted bit-identical to the frozen function) reports the iteration
count; a reference run of the same updates (iters 20000, tol 1e-12) measures how far the frozen plan is from the
fixed point (TV distance of the normalised plans, max relative error of the growth weights after kernel regression).
Sharpness (descriptive): effective partners per source cell exp(H(P_i. / r_i)) / n_target; eps_abs / median 15-NN
squared distance of the target cells; transported cost / independent cost; plan mass on same-cell-type pairs vs the
independent coupling. Output: audit/sinkhorn.json.
"""
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from _common import AC, ATLAS, ERY, IB, dump, kernel_regress, load


def sinkhorn_instr(a, b, C, eps, tau=None, iters=2000, tol=1e-7):
    """identical updates and stopping rule as IB.sinkhorn; returns (P, iterations, converged, fallback)."""
    Ce = np.asarray(C, np.float64) / eps
    if Ce.min(1).max() > IB.UNDERFLOW or Ce.min(0).max() > IB.UNDERFLOW:
        return IB.sinkhorn_log(a, b, C, eps, tau, iters, tol), -1, None, "underflow"
    K = np.exp(-Ce)
    k = 1.0 if tau is None else tau / (tau + eps)
    f, g = np.zeros(len(a)), np.zeros(len(b))
    v = np.ones(len(b))
    done = False
    with np.errstate(all="ignore"):
        for it in range(iters):
            u = a / (K @ v)
            if tau is not None:
                u = u ** k
            v = b / (K.T @ u)
            if tau is not None:
                v = v ** k
            if not (np.isfinite(u).all() and np.isfinite(v).all() and u.min() > 0 and v.min() > 0):
                return IB.sinkhorn_log(a, b, C, eps, tau, iters, tol), it + 1, None, "nonfinite"
            f_new, g_new = eps * np.log(u), eps * np.log(v)
            done = max(np.abs(f_new - f).max(), np.abs(g_new - g).max()) < tol
            f, g = f_new, g_new
            if done:
                break
    return u[:, None] * K * v[None, :], it + 1, bool(done), None


def plan_instr(ZA, ZB, eps_rel, tau=None, a=None, iters=2000, tol=1e-7):
    """same as IB.plan, instrumented."""
    C = IB.sqdist(ZA, ZB)
    med = float(np.median(C))
    a = IB.unif(len(ZA)) if a is None else a / a.sum()
    b = IB.unif(len(ZB))
    P, n_it, conv, fb = sinkhorn_instr(a, b, C / med, eps_rel, tau, iters, tol)
    return P, med, C, a, b, n_it, conv, fb


def sharp(P, C, med, eps_rel, ZB, same=None):
    r = P.sum(1)
    Q = P / r[:, None]
    H = -(Q * np.log(np.maximum(Q, 1e-300))).sum(1)
    nn = NearestNeighbors(n_neighbors=16).fit(ZB)
    dist, _ = nn.kneighbors(ZB)
    out = {"eff_partners_frac": float(np.mean(np.exp(H)) / P.shape[1]),
           "eps_abs_over_med15nn_sq": float(eps_rel * med / np.median(dist[:, 15] ** 2)),
           "cost_plan_over_indep": float((P * C).sum() / P.sum() / C.mean())}
    if same is not None:
        rr, cc = P.sum(1) / P.sum(), P.sum(0) / P.sum()
        out["same_type_mass_plan"] = float((P * same).sum() / P.sum())
        out["same_type_mass_indep"] = float(rr @ same @ cc)
    return out


def compare(P, Pref):
    p, q = P / P.sum(), Pref / Pref.sum()
    return {"tv_vs_ref": float(0.5 * np.abs(p - q).sum()), "mass_rel_err": float(abs(P.sum() / Pref.sum() - 1))}


def check(tag, ZA, ZB, eps_rel, tau, a, ctA=None, ctB=None, growth_tgt=None, first=False):
    t0 = time.time()
    P, med, C, an, b, n_it, conv, fb = plan_instr(ZA, ZB, eps_rel, tau, a)
    if first:  # faithfulness of the copy: bit-identical to the frozen plan
        Pf, medf = IB.plan(ZA, ZB, eps_rel, tau=tau, a=a)
        assert medf == med and np.array_equal(Pf, P), tag
    Pref, _, _, _, _, n_ref, conv_ref, _ = plan_instr(ZA, ZB, eps_rel, tau, a, iters=20000, tol=1e-12)
    out = {"iters": n_it, "converged": conv, "fallback": fb, "ref_iters": n_ref, "ref_converged": conv_ref,
           **compare(P, Pref)}
    if tau is None:
        out["max_marg_rel_err"] = float(max(np.abs(P.sum(1) / an - 1).max(), np.abs(P.sum(0) / b - 1).max()))
    else:
        out["row_mass_cv"] = float(np.std(P.sum(1) / an) / np.mean(P.sum(1) / an))
    same = None if ctA is None else (ctA[:, None] == ctB[None, :]).astype(float)
    out.update(sharp(P, C, med, eps_rel, ZB, same))
    if growth_tgt is not None:  # forecast: growth weights exactly as Problem.growth, frozen vs reference plan
        gws = []
        for PP in (P, Pref):
            gr = PP.sum(1) / an
            gr = gr / gr.mean()
            gws.append(np.maximum(kernel_regress(ZA, gr[:, None], growth_tgt)[:, 0], 1e-6))
        out["growth_w_max_rel_err_vs_ref"] = float(np.max(np.abs(gws[0] - gws[1]) / gws[1]))
    out["sec"] = round(time.time() - t0, 1)
    print(tag, out, flush=True)
    return out


def interp_case(folder, k, res, first):
    d = load(folder, f"interp{k}.npz")
    st = d["stage_fit"]
    Zf = d["Z_fit"].astype(np.float64)
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in set(IB.S_GENES + IB.G2M_GENES) for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    cc_fit = (Xcc - Xcc.mean(0)).mean(1)
    A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
    iA, iB = IB.sub(len(A), seed=1000), IB.sub(len(B), seed=2000)
    ct = d["celltype_fit"].astype(str)
    ZA, ZB = Zf[A[iA]], Zf[B[iB]]
    a = np.exp(IB.DEFAULT["beta"] * cc_fit[A[iA]] * IB.DT_INT)
    res[f"k{k}_balanced"] = check(f"{folder[-5:]} k{k} balanced", ZA, ZB, 0.05, None, None, ct[A[iA]], ct[B[iB]],
                                  first=first)
    res[f"k{k}_uot"] = check(f"{folder[-5:]} k{k} uot", ZA, ZB, 0.05, 1.0, a, ct[A[iA]], ct[B[iB]], first=first)


def origin_case(folder, s, res, first):
    d = load(folder, f"origin{s}.npz")
    st = d["stage_past"]
    Zn = d["Z_past"][st == s].astype(np.float64)
    Zp = d["Z_past"][st == s - 1].astype(np.float64)
    cc_p = d["cc_past"][st == s - 1]
    ct = d["celltype_past"].astype(str)
    for tau in (0.1, 1.0):
        for beta in (0.0, 2.0):
            a = np.exp(beta * cc_p * AC.DT)
            res[f"o{s}_tau{tau}_beta{beta}"] = check(f"{folder[-5:]} o{s} tau{tau} beta{beta}", Zp, Zn, 0.05, tau, a,
                                                     ct[st == s - 1], ct[st == s], growth_tgt=Zn,
                                                     first=first and tau == 0.1 and beta == 0.0)


def main():
    out = {"atlas": {}, "ery": {}}
    for k in range(1, 8):
        interp_case(ATLAS, k, out["atlas"], first=(k == 4))
    for k in range(2, 6):
        interp_case(ERY, k, out["ery"], first=False)
    for s in range(1, 8):
        origin_case(ATLAS, s, out["atlas"], first=(s == 4))
    for s in range(2, 6):
        origin_case(ERY, s, out["ery"], first=False)
    for grp in ("atlas", "ery"):
        v = out[grp].values()
        out[f"{grp}_summary"] = {"max_iters": max(x["iters"] for x in v),
                                 "n_unconverged": int(sum(x["converged"] is False for x in v)),
                                 "n_fallback": int(sum(x["fallback"] is not None for x in v)),
                                 "max_tv_vs_ref": max(x["tv_vs_ref"] for x in v),
                                 "max_marg_rel_err_balanced": max(x.get("max_marg_rel_err", 0.0) for x in v),
                                 "max_growth_w_rel_err": max(x.get("growth_w_max_rel_err_vs_ref", 0.0) for x in v)}
        print(grp, out[f"{grp}_summary"], flush=True)
    dump("sinkhorn.json", out)


if __name__ == "__main__":
    main()
