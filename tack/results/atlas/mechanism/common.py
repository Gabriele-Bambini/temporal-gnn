"""EXPLORATORY helpers for the atlas mechanism analysis (tack/results/atlas/mechanism/). Not pre-registered.

Imports the frozen modules read-only (tack/hpc/interp_bench_frozen.py, tack/hpc/atlas_confirm.py); never modifies them.
run_interp_subset() is a line-by-line mirror of atlas_confirm.run_interp with two optional boolean masks (fit cells,
held cells); with no masks it reproduces confirm.json exactly (checked in check_repro.py).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TACK = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(TACK, "hpc"))
import interp_bench_frozen as IB  # noqa: E402
import atlas_confirm as AC  # noqa: E402

DATA = os.environ.get("ATLAS_DATA", r"C:\Users\Gabriele\AppData\Local\Temp\claude\C--Users-Gabriele-My-Drive-PhD---Research-Temporal-GNN\264f6820-d374-441d-a750-fb2a7c436807\scratchpad\atlas")
CONFIRM = os.path.join(TACK, "results", "atlas", "confirm.json")
METHODS = AC.INTERP_METHODS
STAGES = ["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"]

# coarse groups defined from the atlas type names (EXPLORATORY; PGC left out). Early epiblast feeds every layer, so
# these groups are not closed lineages at k = 1..3.
GROUPS = {
    "epiblast_ectoderm": ["Epiblast", "Rostral neurectoderm", "Caudal epiblast", "Caudal neurectoderm",
                          "Forebrain/Midbrain/Hindbrain", "Spinal cord", "NMP", "Neural crest", "Surface ectoderm"],
    "streak_mesoderm": ["Primitive Streak", "Anterior Primitive Streak", "Nascent mesoderm", "Mixed mesoderm",
                        "Caudal Mesoderm", "Paraxial mesoderm", "Somitic mesoderm", "Intermediate mesoderm",
                        "Pharyngeal mesoderm", "Mesenchyme", "Cardiomyocytes", "Endothelium", "Allantois",
                        "ExE mesoderm", "Notochord"],
    "endoderm": ["Def. endoderm", "Gut"],
    "extraembryonic_ecto_endo": ["ExE ectoderm", "ExE endoderm", "Visceral endoderm", "Parietal endoderm"],
    "early_gastrulation_axis": ["Epiblast", "Primitive Streak", "Anterior Primitive Streak", "Nascent mesoderm",
                                "Mixed mesoderm"],
}


def load_interp(k):
    return np.load(os.path.join(DATA, f"interp{k}.npz"), allow_pickle=True)


def confirm():
    return json.load(open(CONFIRM))


def cc_score(d):
    """Frozen proliferation score of run_interp (HVG cc genes, centred on ALL fit cells)."""
    genes = [g.capitalize() for g in d["genes"].astype(str)]
    ccm = np.array([g in set(IB.S_GENES + IB.G2M_GENES) for g in genes])
    Xcc = d["X_fit"][:, ccm].astype(np.float64)
    return (Xcc - Xcc.mean(0)).mean(1), int(ccm.sum())


def run_interp_subset(d, k, fit_keep=None, held_keep=None, reps=3, pools=True, keep_rep0=False):
    """Mirror of atlas_confirm.run_interp restricted to fit cells fit_keep and held cells held_keep (same space,
    same seeds: rng 1000k+r, A/B subsample seeds 1000+r / 2000+r, target subsample seed 1, n_pred = min(n_held, 1500))."""
    st = d["stage_fit"]
    Zf, Xf = d["Z_fit"].astype(np.float64), d["X_fit"].astype(np.float32)
    Zh, Xh = d["Z_held"].astype(np.float64), d["X_held"].astype(np.float32)
    comps = d["pca_comps"].astype(np.float64)
    smp_held = d["sample_held"].astype(str)
    cc_fit, ncc = cc_score(d)
    fk = np.ones(len(st), bool) if fit_keep is None else np.asarray(fit_keep, bool)
    hk = np.ones(len(Zh), bool) if held_keep is None else np.asarray(held_keep, bool)
    Zh, Xh, smp_held = Zh[hk], Xh[hk], smp_held[hk]
    A, B = np.where((st == k - 1) & fk)[0], np.where((st == k + 1) & fk)[0]
    S = IB.med_scale(Zf)
    n_pred = min(len(Zh), IB.NSUB)
    it = IB.sub(len(Zh), seed=1)
    tgt = IB.Target(Zh[it], Xh[it], S, with_sd=False)
    pl = [p for p in sorted(set(smp_held)) if (smp_held == p).sum() >= AC.MIN_POOL] if pools else []
    ptg = {p: IB.Target(Zh[smp_held == p], None, S, with_sd=False) for p in pl}
    out = {"n_prev": int(len(A)), "n_next": int(len(B)), "n_held": int(len(Zh)), "pools": pl, "reps": []}
    rep0 = None
    for r in range(reps):
        rng = np.random.default_rng(1000 * k + r)
        iA, iB = IB.sub(len(A), seed=1000 + r), IB.sub(len(B), seed=2000 + r)
        ZA, XA, ZB, XB = Zf[A[iA]], Xf[A[iA]].astype(np.float64), Zf[B[iB]], Xf[B[iB]].astype(np.float64)
        M = IB.methods(ZA, XA, ZB, XB, cc_fit[A[iA]], comps, n_pred, rng)
        M.pop("_plan", None)
        M.pop("_uot", None)
        rec, clouds = {}, {}
        for name, (Z, X, w, gm) in M.items():
            Zs, Xs, ws = IB.subsample_pred(Z, X, w, n_pred, rng)
            if name not in METHODS:
                continue
            rec[name] = {"ED_U": IB.ed_uv(Zs, ws, tgt)[0], "ED_U_gene": IB.ed_uv(Xs, ws, tgt, gene=True)[0],
                         "pool_ED_U": {p: IB.ed_uv(Zs, ws, ptg[p])[0] for p in pl}}
            if r == 0 and keep_rep0:
                clouds[name] = (Zs, ws)
        out["reps"].append(rec)
        if r == 0 and keep_rep0:
            rep0 = {"clouds": clouds, "A_idx": A[iA], "B_idx": B[iB], "tgt": tgt, "held_sub_idx": np.where(hk)[0][it]}
    out["mean"] = {m: {key: float(np.mean([rp[m][key] for rp in out["reps"]])) for key in ("ED_U", "ED_U_gene")}
                   for m in METHODS}
    out["sd"] = {m: float(np.std([rp[m]["ED_U"] for rp in out["reps"]])) for m in METHODS}
    out["pool_mean"] = {m: {p: float(np.mean([rp[m]["pool_ED_U"][p] for rp in out["reps"]])) for p in pl}
                        for m in METHODS}
    if keep_rep0:
        return out, rep0
    return out


def tv(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def dump(path, obj):
    def conv(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=conv)
