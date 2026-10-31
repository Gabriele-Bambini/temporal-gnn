"""AUDIT 1 (products, atlas-specific): stage indexing, cross-file cell identity, pools, float16, PCA strictness,
cc centring, blood exclusion, label/expression alignment. Compares with the erythroid products where meaningful.
Output: audit/products.json. Every number here is a check of the frozen data products; nothing changes a verdict.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from _common import ATLAS, ERY, confirm, dump, load

BLOOD = {"Haematoendothelial progenitors", "Blood progenitors 1", "Blood progenitors 2",
         "Erythroid1", "Erythroid2", "Erythroid3"}
N_STAGE = [2999, 2072, 3000, 3000, 3000, 3001, 3000, 3000, 3000]
N_POOL = [3, 1, 6, 3, 6, 4, 4, 3, 4]


def angles(A, B):
    """principal angles (degrees) between row spaces of A and B (rows orthonormalised)."""
    qa, _ = np.linalg.qr(A.T)
    qb, _ = np.linalg.qr(B.T)
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))


def pca_check(Xfit16, Xother16, mu, comps, Zfit, Zother):
    Xf = Xfit16.astype(np.float64)
    Xo = Xother16.astype(np.float64)
    out = {}
    out["comps_orthonormal_maxdev"] = float(np.abs(comps @ comps.T - np.eye(len(comps))).max())
    out["mean_vs_fit_mean_maxabs"] = float(np.abs(Xf.mean(0) - mu).max())
    zr_f = (Xf - mu) @ comps.T
    zr_o = (Xo - mu) @ comps.T
    sc = float(np.sqrt((Zfit.astype(np.float64) ** 2).sum(1).mean()))
    out["Z_fit_rebuild_maxabs"] = float(np.abs(zr_f - Zfit).max())
    out["Z_other_rebuild_maxabs"] = float(np.abs(zr_o - Zother).max())
    out["Z_rms_norm"] = sc
    out["Z_other_rebuild_rel"] = float(np.abs(zr_o - Zother).max() / sc)
    _, _, Vf = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)
    Xa = np.vstack([Xf, Xo])
    _, _, Va = np.linalg.svd(Xa - Xa.mean(0), full_matrices=False)
    n = len(comps)
    out["max_angle_vs_fit_svd_deg"] = float(angles(comps, Vf[:n]).max())
    out["max_angle_vs_fit+other_svd_deg"] = float(angles(comps, Va[:n]).max())
    return out


def stage_probe(Zf, st, a, b, Zh):
    """logistic regression a vs b on fit cells; mean P(b) on held cells (should sit between the two fit stages)."""
    y = (st == b).astype(int)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Zf, y)
    pf = clf.predict_proba(Zf)[:, 1]
    return {"P_next_on_prev": float(pf[y == 0].mean()), "P_next_on_next": float(pf[y == 1].mean()),
            "P_next_on_held": float(clf.predict_proba(Zh)[:, 1].mean())}


def main():
    cj = confirm()
    res = {"interp": {}, "origin": {}, "sample_stage": {}, "ery_interp": {}}
    sample_stage = {}
    ok_all = True
    for k in range(1, 8):
        d = load(ATLAS, f"interp{k}.npz")
        st = d["stage_fit"]
        smp_f, smp_h = d["sample_fit"].astype(str), d["sample_held"].astype(str)
        ct_f, ct_h = d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
        A, B = np.where(st == k - 1)[0], np.where(st == k + 1)[0]
        r = {"stage_values": sorted(set(st.tolist())), "n_prev": len(A), "n_next": len(B), "n_held": len(smp_h)}
        r["stage_values_ok"] = r["stage_values"] == [k - 1, k + 1]
        r["counts_match_prereg"] = (len(A), len(B), len(smp_h)) == (N_STAGE[k - 1], N_STAGE[k + 1], N_STAGE[k])
        cr = cj["interp"][str(k)]
        r["counts_match_confirm"] = (len(A), len(B), len(smp_h)) == (cr["n_prev"], cr["n_next"], cr["n_held"])
        r["prev_block_first"] = bool(A.max() < B.min())
        # pools: disjoint across the three stages, pool counts as in the prereg
        sa, sb, sh = set(smp_f[A]), set(smp_f[B]), set(smp_h)
        r["pools_prev_next_held"] = [len(sa), len(sb), len(sh)]
        r["pools_match_prereg"] = [len(sa), len(sb), len(sh)] == [N_POOL[k - 1], N_POOL[k + 1], N_POOL[k]]
        r["pools_disjoint"] = not (sa & sb or sa & sh or sb & sh)
        r["held_pool_sizes"] = {p: int((smp_h == p).sum()) for p in sorted(sh)}
        r["pools_used_match_confirm"] = sorted(p for p, n in r["held_pool_sizes"].items() if n >= 20) == sorted(cr["pools"])
        for p in sa:
            sample_stage.setdefault(p, set()).add(k - 1)
        for p in sb:
            sample_stage.setdefault(p, set()).add(k + 1)
        for p in sh:
            sample_stage.setdefault(p, set()).add(k)
        # cross-file identity: prev block = origin(k-1) stage k-1 cells; next block = origin(k).fut; held = origin(k-1).fut
        o_prev = load(ATLAS, f"origin{k - 1}.npz") if k >= 2 else None
        o_k = load(ATLAS, f"origin{k}.npz")
        if o_prev is not None:
            sp = o_prev["stage_past"]
            r["prev_cells_eq_origin_km1"] = bool((o_prev["celltype_past"][sp == k - 1].astype(str) == ct_f[A]).all()
                                                and (o_prev["sample_past"][sp == k - 1].astype(str) == smp_f[A]).all())
            r["held_cells_eq_origin_km1_fut"] = bool((o_prev["celltype_fut"].astype(str) == ct_h).all()
                                                    and (o_prev["sample_fut"].astype(str) == smp_h).all())
        else:
            sp = o_k["stage_past"]
            r["prev_cells_eq_origin1_stage0"] = bool((o_k["celltype_past"][sp == 0].astype(str) == ct_f[A]).all())
            r["held_cells_eq_origin1_stage1"] = bool((o_k["celltype_past"][sp == 1].astype(str) == ct_h).all())
        r["next_cells_eq_origin_k_fut"] = bool((o_k["celltype_fut"].astype(str) == ct_f[B]).all()
                                               and (o_k["sample_fut"].astype(str) == smp_f[B]).all())
        r["blood_cells"] = int(np.isin(ct_f, list(BLOOD)).sum() + np.isin(ct_h, list(BLOOD)).sum())
        # float16 / PCA strictness
        r["pca"] = pca_check(d["X_fit"], d["X_held"], d["pca_mean"].astype(np.float64), d["pca_comps"].astype(np.float64),
                             d["Z_fit"], d["Z_held"])
        r["X_fit_max"] = float(d["X_fit"].astype(np.float32).max())
        r["cc_fit_stored_mean"] = float(d["cc_fit"].astype(np.float64).mean())
        # label / expression alignment (descriptive): 15-NN cell type from fit cells, accuracy on held
        Zf, Zh = d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
        knn = KNeighborsClassifier(n_neighbors=15).fit(Zf, ct_f)
        lab = knn.predict(Zh)
        r["knn_celltype_acc_held"] = float((lab == ct_h).mean())
        shared = np.isin(ct_h, list(set(ct_f)))
        r["knn_celltype_acc_held_shared_types"] = float((lab[shared] == ct_h[shared]).mean())
        perm = np.random.default_rng(0).permutation(len(ct_h))
        r["knn_celltype_acc_held_if_labels_shuffled"] = float((lab == ct_h[perm]).mean())
        r["stage_probe"] = stage_probe(Zf, st, k - 1, k + 1, Zh)
        flags = [r["stage_values_ok"], r["counts_match_prereg"], r["counts_match_confirm"], r["pools_match_prereg"],
                 r["pools_disjoint"], r["pools_used_match_confirm"], r["next_cells_eq_origin_k_fut"], r["blood_cells"] == 0]
        flags += [v for kk, v in r.items() if kk.startswith(("prev_cells_eq", "held_cells_eq"))]
        r["all_ok"] = bool(all(flags))
        ok_all &= r["all_ok"]
        res["interp"][k] = r
        print("interp", k, {kk: r[kk] for kk in ("all_ok", "n_prev", "n_next", "n_held", "pools_prev_next_held",
                                                 "knn_celltype_acc_held", "stage_probe")},
              {kk: round(v, 5) for kk, v in r["pca"].items()}, flush=True)
    for s in range(1, 8):
        d = load(ATLAS, f"origin{s}.npz")
        sp, sf = d["stage_past"], d["stage_fut"]
        r = {"stage_past_values": sorted(set(sp.tolist())), "stage_fut_values": sorted(set(sf.tolist()))}
        r["stages_ok"] = r["stage_past_values"] == list(range(s + 1)) and r["stage_fut_values"] == [s + 1]
        r["counts_ok"] = [int((sp == i).sum()) for i in range(s + 1)] == N_STAGE[: s + 1] and len(sf) == N_STAGE[s + 1]
        if s < 7:
            o2 = load(ATLAS, f"origin{s + 1}.npz")
            r["fut_cells_eq_next_origin_now"] = bool(
                (o2["celltype_past"][o2["stage_past"] == s + 1].astype(str) == d["celltype_fut"].astype(str)).all()
                and (o2["sample_past"][o2["stage_past"] == s + 1].astype(str) == d["sample_fut"].astype(str)).all())
        r["cc_past_mean"] = float(d["cc_past"].astype(np.float64).mean())
        r["cc_past_sd"] = float(d["cc_past"].astype(np.float64).std())
        r["pca"] = pca_check(d["X_past"], d["X_fut"], d["pca_mean"].astype(np.float64), d["pca_comps"].astype(np.float64),
                             d["Z_past"], d["Z_fut"])
        r["blood_cells"] = int(np.isin(d["celltype_past"].astype(str), list(BLOOD)).sum()
                               + np.isin(d["celltype_fut"].astype(str), list(BLOOD)).sum())
        for p, i in zip(d["sample_past"].astype(str), sp):
            sample_stage.setdefault(p, set()).add(int(i))
        for p in set(d["sample_fut"].astype(str)):
            sample_stage.setdefault(p, set()).add(s + 1)
        flags = [r["stages_ok"], r["counts_ok"], r["blood_cells"] == 0, abs(r["cc_past_mean"]) < 1e-4]
        if s < 7:
            flags.append(r["fut_cells_eq_next_origin_now"])
        r["all_ok"] = bool(all(flags))
        ok_all &= r["all_ok"]
        res["origin"][s] = r
        print("origin", s, {kk: r[kk] for kk in ("all_ok", "cc_past_mean", "cc_past_sd")},
              {kk: round(v, 5) for kk, v in r["pca"].items()}, flush=True)
    res["sample_stage"] = {p: sorted(v) for p, v in sorted(sample_stage.items(), key=lambda x: int(x[0]))}
    res["every_sample_one_stage"] = all(len(v) == 1 for v in sample_stage.values())
    ok_all &= res["every_sample_one_stage"]
    # erythroid reference for the descriptive numbers
    for k in range(2, 6):
        d = load(ERY, f"interp{k}.npz")
        st = d["stage_fit"]
        Zf, Zh = d["Z_fit"].astype(np.float64), d["Z_held"].astype(np.float64)
        ct_f, ct_h = d["celltype_fit"].astype(str), d["celltype_held"].astype(str)
        lab = KNeighborsClassifier(n_neighbors=15).fit(Zf, ct_f).predict(Zh)
        res["ery_interp"][k] = {"knn_celltype_acc_held": float((lab == ct_h).mean()),
                                "stage_probe": stage_probe(Zf, st, k - 1, k + 1, Zh),
                                "Z_held_rebuild_rel": pca_check(d["X_fit"], d["X_held"], d["pca_mean"].astype(np.float64),
                                                                d["pca_comps"].astype(np.float64), d["Z_fit"],
                                                                d["Z_held"])["Z_other_rebuild_rel"]}
        print("ery interp", k, res["ery_interp"][k], flush=True)
    res["ALL_OK"] = bool(ok_all)
    print("ALL_OK", ok_all, "every_sample_one_stage", res["every_sample_one_stage"])
    dump("products.json", res)


if __name__ == "__main__":
    main()
