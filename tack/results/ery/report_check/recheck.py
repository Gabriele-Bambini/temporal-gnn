"""Report-time re-check of the cheapest load-bearing diagnosis numbers (diag_direction, diag_batch).

DESCRIPTIVE: uses the true stage s+1. Own code, nothing imported from the audited scripts.
Usage: python -I recheck.py --data <folder with origin<s>.npz and obs.csv>
Writes recheck.json next to this file.
"""
import argparse, itertools, json, os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

STAGES = ["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]
HERE = os.path.dirname(os.path.abspath(__file__))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def dsum(A, B, step=1500):
    s = 0.0
    for i in range(0, len(A), step):
        s += cdist(A[i:i + step], B).sum()
    return s


def ed_u(A, B):
    na, nb = len(A), len(B)
    return float(2 * dsum(A, B) / (na * nb) - dsum(A, A) / (na * (na - 1)) - dsum(B, B) / (nb * (nb - 1)))


def comp_share(Z0, c0, Z1, c1):
    tot = Z1.mean(0) - Z0.mean(0)
    C = np.zeros_like(tot)
    for t in set(c0) | set(c1):
        p0, p1 = (c0 == t).mean(), (c1 == t).mean()
        m0 = Z0[c0 == t].mean(0) if p0 > 0 else Z1[c1 == t].mean(0)
        m1 = Z1[c1 == t].mean(0) if p1 > 0 else m0
        C += (p1 - p0) * 0.5 * (m0 + m1)
    return float(C @ tot / (tot @ tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    obs = pd.read_csv(os.path.join(a.data, "obs.csv"))
    ost = np.array([STAGES.index(x) for x in obs["stage"]])
    out = {"origins": {}}
    out["batch_by_stage_cells"] = {STAGES[k]: {str(b): int(n) for b, n in
                                               obs.loc[ost == k, "sequencing.batch"].value_counts().sort_index().items()}
                                   for k in range(7)}
    for s in (2, 3, 4, 5):
        f = np.load(os.path.join(a.data, f"origin{s}.npz"), allow_pickle=True)
        st = f["stage_past"]
        Zp, Zf = f["Z_past"].astype(np.float64), f["Z_fut"].astype(np.float64)
        cp, cf = f["celltype_past"].astype(str), f["celltype_fut"].astype(str)
        assert (obs.loc[ost <= s, "celltype"].astype(str).to_numpy() == cp).all()
        assert (obs.loc[ost == s + 1, "celltype"].astype(str).to_numpy() == cf).all()
        comps = f["pca_comps"].astype(np.float64)
        murk = f["murk"].astype(bool)
        now, prev = st == s, st == s - 1
        mu_prev, mu_now, mu_fut = Zp[prev].mean(0), Zp[now].mean(0), Zf.mean(0)
        sh_prev, sh_next = mu_now - mu_prev, mu_fut - mu_now
        r_now = float(np.sqrt(((Zp[now] - mu_now) ** 2).sum(1).mean()))
        r_fut = float(np.sqrt(((Zf - mu_fut) ** 2).sum(1).mean()))
        o = {"n_murk_hvg": int(murk.sum()),
             "cos_shift_prev_next": cos(sh_prev, sh_next),
             "norm_next_over_prev": float(np.linalg.norm(sh_next) / np.linalg.norm(sh_prev)),
             "rms_radius_now": r_now, "rms_radius_fut": r_fut,
             "shift_over_radius": float(np.linalg.norm(sh_next) / r_now),
             "composition_share": comp_share(Zp[now], cp[now], Zf, cf),
             "meanshift_err_over_persist_err": float(np.linalg.norm(sh_next - sh_prev) / np.linalg.norm(sh_next)),
             "vel": {}}
        Ms = f["Ms"][now].astype(np.float64)
        for vk in ("v_ss", "v_dyn"):
            V = f[vk][now].astype(np.float64) / (1.0 + Ms)
            Vn = V.copy()
            Vn[:, murk] = 0.0
            P, Pn = V @ comps.T, Vn @ comps.T
            gn = np.linalg.norm(V, axis=1)
            ok = gn > 0
            pc = np.linalg.norm(P, axis=1)
            percos = (P @ sh_next) / (pc * np.linalg.norm(sh_next) + 1e-300)
            d = {"pop_cos_with_murk": cos(P.mean(0), sh_next), "pop_cos_without_murk": cos(Pn.mean(0), sh_next),
                 "pop_cos_with_vs_without": cos(P.mean(0), Pn.mean(0)),
                 "median_norm_share_in_pca": float(np.median(pc[ok] / gn[ok])),
                 "n_genes_nonzero": int((np.abs(f[vk]).sum(0) > 0).sum())}
            if s == 5:
                d["per_type_median_cos_vs_shift"] = {t: [float(np.median(percos[cp[now] == t])),
                                                         float((percos[cp[now] == t] > 0).mean())]
                                                     for t in sorted(set(cp[now]))}
            o["vel"][vk] = d
        out["origins"][str(s)] = o

        if s == 5:  # sample-level structure in origin-5 PCA space
            Zall = np.vstack([Zp, Zf])
            sall = np.concatenate([st, np.full(len(Zf), 6)])
            ob = pd.concat([obs[ost <= 5], obs[ost == 6]])
            samp, bat = ob["sample"].to_numpy(), ob["sequencing.batch"].to_numpy()
            chain = [Zall[sall == k].mean(0) for k in range(2, 7)]
            shifts = [chain[i + 1] - chain[i] for i in range(4)]
            out["origin5_chain_adjacent_cos"] = [cos(shifts[i], shifts[i + 1]) for i in range(3)]
            out["origin5_chain_gap2_cos"] = [cos(shifts[0], shifts[2]), cos(shifts[1], shifts[3])]
            usable = {k: [p for p in np.unique(samp[sall == k]) if ((samp == p) & (sall == k)).sum() >= 20]
                      for k in range(2, 7)}
            pb = {p: int(bat[samp == p][0]) for k in usable for p in usable[k]}
            cache = {}

            def pe(p, q):
                key = tuple(sorted((p, q)))
                if key not in cache:
                    cache[key] = ed_u(Zall[samp == p], Zall[samp == q])
                return cache[key]

            wb = {}
            for k in range(2, 7):
                pr = list(itertools.combinations(usable[k], 2))
                sb = [pe(p, q) for p, q in pr if pb[p] == pb[q]]
                db = [pe(p, q) for p, q in pr if pb[p] != pb[q]]
                wb[STAGES[k]] = {"pools": {str(p): pb[p] for p in usable[k]},
                                 "same_batch_mean": float(np.mean(sb)) if sb else None,
                                 "diff_batch_mean": float(np.mean(db)) if db else None}
            out["origin5_within_stage_batch"] = wb
            dec = {}
            for k in range(2, 6):
                w1 = [pe(p, q) for p, q in itertools.combinations(usable[k], 2)]
                w2 = [pe(p, q) for p, q in itertools.combinations(usable[k + 1], 2)]
                cr = [pe(p, q) for p in usable[k] for q in usable[k + 1]]
                pooled = ed_u(Zall[sall == k], Zall[sall == k + 1])
                dec[f"{STAGES[k]}->{STAGES[k + 1]}"] = {
                    "ratio_cross_within": float(np.mean(cr) / (0.5 * (np.mean(w1) + np.mean(w2)))),
                    "frac_cross_gt_within": float(np.mean([c > w for c in cr for w in w1 + w2])),
                    "pooled_ED_U": pooled, "pooled_over_median_within": float(pooled / np.median(w1 + w2))}
            out["origin5_stage_vs_sample"] = dec
    with open(os.path.join(HERE, "recheck.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
