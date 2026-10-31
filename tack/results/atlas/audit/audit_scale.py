"""AUDIT 7 (med_scale / S): the primary metric ED_U never reads the scale S = IB.med_scale(Z_fit) (S only enters the
unused sd_fixed). Test: for interp k = 4 (rep-0 frozen methods) and the origin-4 forecast target, ED_U and ED_U_gene of
every method are computed against Targets built with the frozen S, S = 1 and S = 1e6; they must be bit-identical.
The plan's own eps (eps_rel * median cross cost) is audited in audit_sinkhorn.py. Output: audit/scale.json.
"""
import numpy as np

from _common import AC, ATLAS, IB, dump, load


def main():
    d = load(ATLAS, "interp4.npz")
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
    A, B = np.where(st == 3)[0], np.where(st == 5)[0]
    iA, iB = IB.sub(len(A), seed=1000), IB.sub(len(B), seed=2000)
    rng = np.random.default_rng(4000)
    cc = np.zeros(len(st))
    M = IB.methods(Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64), cc[A[iA]],
                   d["pca_comps"].astype(np.float64), 1500, rng)
    M.pop("_plan"), M.pop("_uot")
    it = IB.sub(len(Zh), seed=1)
    S0 = IB.med_scale(Zf)
    tg = {S: IB.Target(Zh[it], Xh[it], S, with_sd=False) for S in (S0, 1.0, 1e6)}
    vals = {}
    for name, (Z, X, w, _) in M.items():
        Zs, Xs, ws = IB.subsample_pred(Z, X, w, 1500, np.random.default_rng(1))
        vals[name] = [(IB.ed_uv(Zs, ws, t)[0], IB.ed_uv(Xs, ws, t, gene=True)[0]) for t in tg.values()]
    interp_ok = all(len(set(v)) == 1 for v in vals.values())
    # forecast: Problem.S is only passed to Target; same test on the origin-4 final target
    P = AC.origin_problem(ATLAS, 4)
    tF = [P.tgt] + [IB.Target(P.tgt.Z, P.tgt.X, S, with_sd=False) for S in (1.0, 1e6)]
    Zn = P.Zn[IB.sub(len(P.Zn))]
    w = IB.unif(len(Zn))
    fvals = [IB.ed_uv(Zn, w, t)[0] for t in tF]
    out = {"S_frozen_interp4": S0, "interp_bit_identical": interp_ok, "forecast_bit_identical": len(set(fvals)) == 1,
           "example_interp4_union": vals["union"][0], "example_origin4_persist": fvals[0]}
    print(out)
    dump("scale.json", out)


if __name__ == "__main__":
    main()
