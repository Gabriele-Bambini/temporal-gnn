"""Controls for diag_direction.py (DESCRIPTIVE: uses the true stage s+1; not a forecast).

Question: is the per-cell agreement between RNA velocity and the OT displacement toward stage s+1 about REAL TIME,
or about geometry that any transport map shows (entropic blur and finite-sample regression to the mean both pull a
cell's barycentric image toward the centre / the dense part of the cloud, and a steady-state velocity can also point
inward)? For each origin s (and, strictly, for the previous transition s-1 -> s):
  stage-s cells split at random into halves A and B (seed 11);
  D_time = displacement of the A cells toward stage s+1   (balanced Sinkhorn eps_rel 0.05 / exact assignment)
  D_free = displacement of the A cells toward half B of the SAME stage (time-free control, same method)
  D_time - D_free = the time-specific part of the displacement (same source cells, same method).
  inward direction r_i = -(z_i - mu_stage).
Reported per velocity variant (v_ss / v_dyn, with / without MURK) and cell type: median cosine and fraction > 0 of
V with D_time, D_free, D_time - D_free, r; cosine of the residual displacement with r; partial cosine of V and
D_time after removing from both the component along r_i; within-stage / within-type permutation nulls (100).
Output: controls.json, controls.csv (next to this file).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "hpc")))
import diag_direction as dd  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ERY_DATA", "")
NPERM = 100
ROWS = []


def stats(c, groups):
    return dd.group_stats(c, groups)


def null_p(fn, V, groups, obs, rng):
    """fn(Vperm) -> per-cell cosines; returns per group (null median mean, p) for within-stage and within-type."""
    out = {}
    for kind, wt in (("stage", False), ("type", True)):
        med = {g: [] for g in obs}
        for _ in range(NPERM):
            c = fn(V[dd.permuted(len(V), groups, rng, wt)])
            for g in obs:
                m = np.ones(len(c), bool) if g == "all" else groups == g
                med[g].append(np.median(c[m]))
        for g in obs:
            mm = np.array(med[g])
            out.setdefault(g, {})[f"null_{kind}_median_mean"] = float(mm.mean())
            out[g][f"p_{kind}"] = float((1 + (mm >= obs[g]["median"]).sum()) / (1 + NPERM))
    return out


def perp(A, r):
    ru = dd.unit(r)
    return A - (A * ru).sum(1, keepdims=True) * ru


def displacement(Z0, Z1, method, seed):
    if method == "sinkhorn":
        j = dd.sub(len(Z1), seed=seed)
        D, _, marg = dd.barycentric(Z0, Z1[j], dd.EPS_MAIN)
        return D, np.arange(len(Z0))
    n = min(len(Z0), len(Z1), dd.NSUB)
    i, j = dd.sub(len(Z0), n, seed=seed), dd.sub(len(Z1), n, seed=seed + 1)
    Dex, _, r = dd.exact_assignment(Z0[i], Z1[j])
    return Dex, i[r]


def run(origin, label, Zsrc_full, ct_full, vel, Ztgt, rng):
    """Zsrc_full: all cells of the source stage; vel[(vk, mk)] = (Vz, Vg) for those cells; Ztgt: next-stage cells."""
    mu = Zsrc_full.mean(0)
    perm = np.random.default_rng(11).permutation(len(Zsrc_full))
    A, B = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
    A = A[dd.sub(len(A), seed=3)]
    res = {"n_A": int(len(A)), "n_B": int(len(B)), "n_tgt": int(len(Ztgt))}
    for method in ("sinkhorn", "exact"):
        Dt, kt = displacement(Zsrc_full[A], Ztgt, method, 0)
        Df, kf = displacement(Zsrc_full[A], Zsrc_full[B], method, 5)
        # keep the A cells present in both (exact assignment may subsample)
        common = np.intersect1d(kt, kf)
        Dt = Dt[np.searchsorted(kt, common)] if method == "exact" else Dt[common]
        Df = Df[np.searchsorted(kf, common)] if method == "exact" else Df[common]
        idx = A[common]
        Z = Zsrc_full[idx]
        g = ct_full[idx]
        r = -(Z - mu)
        Dres = Dt - Dt.mean(0)
        geo = {"n": int(len(idx)),
               "cos_Dtime_res_inward": stats(dd.cos_rows(Dres, r), g)["all"],
               "cos_Dfree_res_inward": stats(dd.cos_rows(Df - Df.mean(0), r), g)["all"],
               "cos_Dtime_Dfree": stats(dd.cos_rows(Dt, Df), g)["all"],
               "norm_Dfree_over_Dtime_median": float(np.median(np.linalg.norm(Df, axis=1) / np.linalg.norm(Dt, axis=1)))}
        res[f"{method}_geometry"] = geo
        for (vk, mk), (Vz_full, _) in vel.items():
            V = Vz_full[idx]
            comps = {
                "D_time": lambda W: dd.cos_rows(Dt, W),
                "D_free": lambda W: dd.cos_rows(Df, W),
                "D_time_minus_D_free": lambda W: dd.cos_rows(Dt - Df, W),
                "inward": lambda W: dd.cos_rows(r, W),
                "D_time_perp_inward": lambda W: dd.cos_rows(perp(Dt, r), perp(W, r)),
                "D_time_res_perp_inward_vs_V_res": lambda W: dd.cos_rows(perp(Dres, r), perp(W - W.mean(0), r)),
            }
            for name, fn in comps.items():
                c = fn(V)
                st = stats(c, g)
                nul = null_p(fn, V, g, st, rng) if name != "inward" else {}
                for grp in st:
                    st[grp].update(nul.get(grp, {}))
                    ROWS.append({"origin": origin, "transition": label, "method": method, "velocity": vk,
                                 "murk": mk, "comparison": name, "celltype": grp, **st[grp]})
                res.setdefault(f"{method}|{vk}|{mk}", {})[name] = st["all"]
    return res


def main():
    if not DATA or not os.path.isdir(DATA):
        sys.exit("give the folder with origin<s>.npz as first argument")
    T0 = time.time()
    rng = np.random.default_rng(2024)
    out = {"note": "DESCRIPTIVE controls (use the true stage s+1)", "origins": {}}
    for s in range(1, 6):
        t0 = time.time()
        f = np.load(os.path.join(DATA, f"origin{s}.npz"), allow_pickle=True)
        st_p = f["stage_past"]
        ct_p = f["celltype_past"].astype(str)
        Zp = f["Z_past"].astype(np.float64)
        Zf = f["Z_fut"].astype(np.float64)
        comps = f["pca_comps"].astype(np.float64)
        murk = f["murk"].astype(bool)
        now, prev = np.where(st_p == s)[0], np.where(st_p == s - 1)[0]
        o = {"future": run(s, "s->s+1", Zp[now], ct_p[now], dd.velocity_pca(f, now, comps, murk), Zf, rng),
             "past_strict": run(s, "s-1->s", Zp[prev], ct_p[prev], dd.velocity_pca(f, prev, comps, murk), Zp[now], rng)}
        o["seconds"] = round(time.time() - t0, 1)
        out["origins"][s] = o
        print("origin", s, "done", o["seconds"], "s", flush=True)
        with open(os.path.join(HERE, "controls.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        pd.DataFrame(ROWS).to_csv(os.path.join(HERE, "controls.csv"), index=False)
    out["seconds_total"] = round(time.time() - T0, 1)
    with open(os.path.join(HERE, "controls.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("total", out["seconds_total"])


if __name__ == "__main__":
    main()
