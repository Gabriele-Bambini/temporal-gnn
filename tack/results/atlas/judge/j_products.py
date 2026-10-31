"""JUDGE (not exploratory): spot checks of the audit's product / solver claims, on the local atlas products.
  - prereg file sha256 vs the value logged in decisions.md (7cb27ba3...), and file mtimes (weak provenance only);
  - interp<k>: stage_fit takes exactly {k-1, k+1}; held cells equal origin<k-1>.fut cells (cell type and sample, in order);
    next block equals origin<k>.fut; pools per stage;
  - S/G2M genes among the 2000 HVGs per interp file (the uot growth prior), vs confirm.json cc_genes;
  - PCA of each interp file vs an SVD of its own fit cells (largest principal angle), and vs fit + held;
  - balanced Sinkhorn marginal error and log-domain fallbacks for the 7 rep-0 interpolation plans (IB.STATS);
  - kernel_regress underflow margin (largest row-minimum exponent d^2 / 2h^2) for the origin growth regressions.
Usage: python j_products.py -> j_products.json
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy.linalg import subspace_angles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tack", "hpc"))
import interp_bench_frozen as IB  # noqa: E402

SCR = r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad"
DATA = os.path.join(SCR, "atlas")
C = json.load(open(os.path.join(HERE, "..", "confirm.json")))


def main():
    out = {}
    pre = os.path.join(ROOT, "tack", "prereg", "2026-10-25-atlas-confirmation-prereg.md")
    h = hashlib.sha256(open(pre, "rb").read()).hexdigest()
    h_lf = hashlib.sha256(open(pre, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    logged = "7cb27ba3d237775ed13c5a6707960e4f23509addc97ca26cb18e9dd854b5ff3b"
    out["prereg_sha256"] = {"raw": h, "lf": h_lf, "logged_in_decisions": logged, "match": logged in (h, h_lf)}
    out["mtimes_utc"] = {os.path.relpath(p, ROOT): float(os.path.getmtime(p)) for p in
                         (pre, os.path.join(HERE, "..", "confirm.json"), os.path.join(HERE, "..", "parity_erythroid.json"))}
    cc_set = set(IB.S_GENES + IB.G2M_GENES)
    prod = {}
    for k in range(1, 8):
        d = np.load(os.path.join(DATA, f"interp{k}.npz"), allow_pickle=True)
        st = d["stage_fit"]
        r = {"stage_fit_values": sorted(set(st.tolist())), "prev_block_first": bool((np.where(st == k - 1)[0] < np.where(st == k + 1)[0].min()).all())}
        o_prev = np.load(os.path.join(DATA, f"origin{k - 1}.npz"), allow_pickle=True) if k >= 2 else None
        o_cur = np.load(os.path.join(DATA, f"origin{k}.npz"), allow_pickle=True)
        if o_prev is not None:
            r["held_eq_origin_prev_fut"] = bool((o_prev["celltype_fut"] == d["celltype_held"]).all() and (o_prev["sample_fut"] == d["sample_held"]).all())
        else:  # k = 1: held stage 1 = origin1's stage-1 block
            m = o_cur["stage_past"] == 1
            r["held_eq_origin1_stage1"] = bool((o_cur["celltype_past"][m] == d["celltype_held"]).all() and (o_cur["sample_past"][m] == d["sample_held"]).all())
        nb = st == k + 1
        r["next_eq_origin_k_fut"] = bool((o_cur["celltype_fut"] == d["celltype_fit"][nb]).all() and (o_cur["sample_fut"] == d["sample_fit"][nb]).all())
        r["n_pools_held"] = int(len(set(d["sample_held"].astype(str))))
        genes = [g.capitalize() for g in d["genes"].astype(str)]
        r["cc_genes_among_hvg"] = int(sum(g in cc_set for g in genes))
        r["cc_genes_confirm_json"] = int(C["interp"][str(k)]["cc_genes"])
        Xf = d["X_fit"].astype(np.float64)
        Xh = d["X_held"].astype(np.float64)
        comps = d["pca_comps"].astype(np.float64)
        _, _, Vt = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)
        Xa = np.vstack([Xf, Xh])
        _, _, Va = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)
        r["pca_angle_deg_vs_fit_svd"] = float(np.degrees(subspace_angles(comps.T, Vt[:30].T).max()))
        r["pca_angle_deg_vs_fit_plus_held_svd"] = float(np.degrees(subspace_angles(comps.T, Va[:30].T).max()))
        # rep-0 balanced plan marginals
        Zf = d["Z_fit"].astype(np.float64)
        A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
        iA, iB = IB.sub(len(A), seed=1000), IB.sub(len(B), seed=2000)
        IB.STATS.update({"kernel_calls": 0, "log_fallbacks": 0, "max_marg_err": 0.0})
        P, med = IB.plan(Zf[A[iA]], Zf[B[iB]], 0.05)
        r["plan_rep0"] = {"max_marg_rel_err": IB.STATS["max_marg_err"], "log_fallbacks": IB.STATS["log_fallbacks"],
                          "mass": float(P.sum())}
        prod[str(k)] = r
        print("interp", k, r, flush=True)
    out["interp_products"] = prod
    pools = []
    for s in range(1, 8):
        o = np.load(os.path.join(DATA, f"origin{s}.npz"), allow_pickle=True)
        if s == 1:
            for q in (0, 1):
                pools.append(int(len(set(o["sample_past"][o["stage_past"] == q].astype(str)))))
        pools.append(int(len(set(o["sample_fut"].astype(str)))))
        # kernel_regress underflow margin for the stage s-1 -> s growth regression (src = stage s-1, tgt = stage s)
        st = o["stage_past"]
        Zp = o["Z_past"][st == s - 1].astype(np.float64)
        Zn = o["Z_past"][st == s].astype(np.float64)
        d_ss = np.sqrt(IB.sqdist(Zp[:min(len(Zp), 800)], Zp))
        hbw = np.median(np.sort(d_ss, 1)[:, min(15, d_ss.shape[1] - 1)]) + 1e-9
        expo = IB.sqdist(Zn, Zp).min(1) / (2 * hbw * hbw)
        out.setdefault("kernel_underflow", {})[str(s)] = {"max_row_min_exponent": float(expo.max()), "threshold": 745.0}
    out["pools_per_stage"] = pools
    out["pools_match_prereg"] = pools == [3, 1, 6, 3, 6, 4, 4, 3, 4]
    print(json.dumps({k: v for k, v in out.items() if k != "interp_products"}, indent=0))
    json.dump(out, open(os.path.join(HERE, "j_products.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
