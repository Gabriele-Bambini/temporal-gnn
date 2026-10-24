"""Gene-level unbalanced OT with a sink / source for Dileo & Sottoriva Task 2 (numpy only, own Sinkhorn).

Law. Per gene and bin the cell-count law of tack/hpc/dist_bridge.py: P(c = 0) = 1 - f, c | c > 0 = 1 + E, E
negative binomial (or Poisson) matching the positive-part mean and variance (positive_part). Bins t-L and t share a
per-gene COUNT-scale grid built like dist_bridge.grid_masses (integer grid 1..K when the 99.9 % quantile cmax <= 48,
else 48 points evenly spaced in counts from 1 to cmax, last cell takes the tail); the only change is that integer
grids stop at K = the smallest of 2, 3, 4, 6, 8, 12, 16, 24, 32, 48 that is >= cmax (genes bucketed by K), instead of
always 48: the extra levels carry < 0.1 % of the mass and made the batched solve 50x slower.

Unbalanced OT (the model). Transported measure = EXPRESSING mass per cell:
    a_j = f_A pA_j (bin t-L), b_l = f_B pB_l (bin t); sum a = f_A != sum b = f_B in general.
Sink / source problem (own eps-scaled log-domain solver uot_sinkhorn.solve_tv, batched over genes):
    min <T, C> + eps KL(T | R) + lambda_g (sum a - sum T) + lambda_g (sum b - sum T),  T 1 <= a,  T^T 1 <= b,
    C_jl = (c_j - c_l)^2 / 2; s = a - T 1 is sunk at A's levels, beta = b - T^T 1 is sourced at B's levels, each
    priced lambda_g per unit (the eps -> 0 limit is the dummy-node problem: level -> sink lambda, source -> level
    lambda, source -> sink 0; a first attempt with explicit dummy nodes and plain Sinkhorn did not converge for
    lambda >> eps and its entropy pushed 20-40 % of the mass through the dummies at small lambda, so the entropy is
    kept on the transported part only, reference R = min(f_A, f_B) (a / f_A) x (b / f_B)).
    lambda_g = lam0 max(mu_B, (var_A + var_B) / 2) (positive-part moments: a cell is sunk and re-sourced rather than
    moved when its count would change by more than ~2 sqrt(lam0) sd); eps_g = 0.25 L mu_B,g (dist_bridge count
    scale). Plan -> T_jl (expressing cells kept, moved from level j to l), s_j (cells at level j that stop expressing
    or are lost relative to the population: SINK), beta_l (cells that start expressing / appear at level l: SOURCE).
Extrapolation delta bins ahead, per level l of today's grid:
    death probability over L bins d_l = s_l / a_l (A's mass at that level; gene average where A has none), survival
    (1 - d_l)^(delta / L) applied to all mass now at l; the transported part c_l = sum_j T_jl moves by delta v_l,
    v_l = (c_l - src_l) / L with src_l the barycentre of its origin (v = 0 where c_l < 1e-6 b_l); the newborn part
    beta_l stays at c_l and births continue at the same rate (delta beta_l / L added at c_l). Levels pushed below
    1/2 switch off. Forecast nonzero fraction = surviving + born mass (capped at 1); positive-part mean / second
    moment from the forecast masses. f_A = 0 < f_B: all mass is birth (forecast f_B (1 + delta / L), same positive
    law); f_A > 0 = f_B: all death (0); both 0: 0.

Balanced ablation (no sink): dist_bridge.coupling + sb_step on the count scale, re-implemented on the same bucketed
grid and solver: OT between the NORMALISED positive parts, barycentric velocity, levels below 1/2 switch off, and the
nonzero fraction follows its own log-linear trend f_B (f_B / f_A)^(delta / L) (genes with a zero bin: unchanged).
"""
import os
import sys

import numpy as np

ROOT = r"C:\Users\Gabriele\My Drive\PhD & Research\Temporal GNN"
HPC = os.path.join(ROOT, "tack", "hpc")
sys.path.insert(0, HPC)
import dist_bridge as DB  # noqa: E402  (laws and observation heads; numpy + scipy only)
import uot_sinkhorn  # noqa: E402

NODE_CACHE = os.path.join(ROOT, "tack", "results", "signal_analysis", "design_best-signal", "_node_{}.npz")
SNAPS = {"gastrulation-human": list(range(2, 32)), "pancreas-human": list(range(0, 15))}
N_GENES = {"gastrulation-human": 21430, "pancreas-human": 18561}
SIGMA2 = DB.SIGMA2_SB  # 0.25
TOL = 1e-7
MAX_FINAL = 1500
BUCKETS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
FMIN = 1e-6


def load(name):
    X = np.load(NODE_CACHE.format(name))["X"]           # (B, G, 5): mean, median, sd, frac_nonzero, sum
    assert X.shape[1] == N_GENES[name], X.shape
    trip = [DB.triplet(X[b]) for b in range(X.shape[0])]
    n = np.array([DB.n_cells(X[b]) for b in range(X.shape[0])])
    m = np.maximum(X[:, :, 0], 0)                         # (B, G)
    return X, trip, n, m


def lam_scale(muB, vA, vB):
    """Per-gene price unit: the positive-part variance (mean of the two bins), at least the Poisson variance mu_B.
    A cell is then sunk / sourced rather than moved when its count would change by more than ~2 sqrt(lam0) sd."""
    return np.maximum(muB, 0.5 * (vA + vB))


def _laws(A, B):
    fA, muA, vA = DB.positive_part(A)
    fB, muB, vB = DB.positive_part(B)
    fA, fB = np.where(fA >= FMIN, fA, 0.0), np.where(fB >= FMIN, fB, 0.0)
    return fA, muA, vA, fB, muB, vB


def grids(muA, vA, muB, vB):
    """Buckets of genes with a shared count grid: list of (positions into the inputs, z (g, K), pA, pB)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        kA, kB = DB._law_kind(muA, vA), DB._law_kind(muB, vB)
        cmax = np.maximum(np.maximum(DB._law_ppf(kA, muA, vA, 0.999), DB._law_ppf(kB, muB, vB, 0.999)), 1.0)
    integer = cmax <= BUCKETS[-1]
    Kg = np.where(integer, np.array(BUCKETS)[np.searchsorted(BUCKETS, np.minimum(cmax, BUCKETS[-1]))], -1)
    out = []
    for K in list(BUCKETS) + [-1]:
        sel = np.where(Kg == K)[0]
        if len(sel) == 0:
            continue
        g = len(sel)
        if K > 0:
            j = np.arange(K, dtype=float)[None, :]
            z = np.broadcast_to(1 + j, (g, K)).copy()
            c_hi = np.concatenate([np.broadcast_to(1 + j[:, :-1], (g, K - 1)), np.full((g, 1), 1e12)], 1)
        else:
            K = DB.GRID
            j = np.arange(K, dtype=float)[None, :]
            z = 1 + (cmax[sel][:, None] - 1) * j / (K - 1)
            c_hi = np.concatenate([(z[:, 1:] + z[:, :-1]) / 2, np.full((g, 1), 1e12)], 1)
        ms = []
        for kind, mu, var in ((kA[sel], muA[sel], vA[sel]), (kB[sel], muB[sel], vB[sel])):
            with np.errstate(divide="ignore", invalid="ignore"):
                cdf = DB._law_cdf(kind, mu, var, c_hi)
            cdf[:, -1] = 1.0
            pm = np.maximum(np.diff(np.concatenate([np.zeros((g, 1)), cdf], 1), axis=1), 0)
            ms.append(pm / pm.sum(1, keepdims=True))
        out.append((sel, z, ms[0], ms[1]))
    return out


# ------------------------------------------------------------------ unbalanced coupling with sink / source
def uot_coupling(A, B, L, lam0, tol=TOL):
    fA, muA, vA, fB, muB, vB = _laws(A, B)
    idx = np.where((fA > 0) & (fB > 0))[0]
    res = {"on": (fA == 0) & (fB > 0), "fA": fA, "fB": fB, "err": 0.0, "iters": 0, "buckets": []}
    if len(idx) == 0:
        return res
    for sel, z, pA, pB in grids(muA[idx], vA[idx], muB[idx], vB[idx]):
        ii = idx[sel]
        a, b = fA[ii, None] * pA, fB[ii, None] * pB
        lam = lam0 * lam_scale(muB[ii], vA[ii], vB[ii])
        C = (z[:, :, None] - z[:, None, :]) ** 2 / 2
        eps = np.maximum(SIGMA2 * L * muB[ii], 1e-3)
        T, move, over, it = uot_sinkhorn.solve_tv(a, b, C, eps, lam, tol=tol, max_final=MAX_FINAL)
        res["err"], res["iters"] = max(res["err"], move), max(res["iters"], it)
        res["over"] = max(res.get("over", 0.0), over)
        c = np.minimum(T.sum(1), b)                      # transported mass arriving at B's level l
        beta = b - c                                     # SOURCE: created at B's level l
        s = np.maximum(a - T.sum(2), 0.0)                # SINK: destroyed at A's level j
        a_pl = a
        dbar = s.sum(1) / np.maximum(a_pl.sum(1), 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(a_pl > 1e-12, s / np.maximum(a_pl, 1e-300), dbar[:, None])
            src = np.where(c > 1e-14, np.einsum("gjk,gj->gk", T, z) / np.maximum(c, 1e-300), z)
        d = np.clip(d, 0.0, 1.0 - 1e-12)
        v = np.where(c > 1e-6 * np.maximum(c + beta, 1e-300), (z - src) / L, 0.0)
        res["buckets"].append({"ii": ii, "z": z, "c": c, "beta": beta, "d": d, "v": v, "s": s, "a": a_pl})
    return res


def uot_step(B, cp, L, delta):
    """Forecast triplet (m, f, e = E[c^2]) delta bins ahead of today's law B along the unbalanced coupling."""
    H = B.copy()
    if delta == 0:
        return H
    on = cp["on"]
    if on.any():
        fB = cp["fB"][on]
        H[on] *= (np.minimum(1.0, fB * (1 + delta / L)) / fB)[:, None]
    for bk in cp["buckets"]:
        ii, z, c, beta, d, v = bk["ii"], bk["z"], bk["c"], bk["beta"], bk["d"], bk["v"]
        S = (1.0 - d) ** (delta / L)
        pos = z + delta * v
        mc = np.where(pos >= 0.5, c * S, 0.0)
        cnt = np.maximum(1.0, pos)
        mb = beta * S + delta * beta / L
        W = mc.sum(1) + mb.sum(1)
        Wd = np.where(W > 0, W, 1.0)
        mu = np.where(W > 0, ((mc * cnt).sum(1) + (mb * z).sum(1)) / Wd, 1.0)
        e2 = np.where(W > 0, ((mc * cnt ** 2).sum(1) + (mb * z ** 2).sum(1)) / Wd, 1.0)
        f = np.minimum(1.0, W)
        H[ii, 0], H[ii, 1], H[ii, 2] = f * mu, f, f * e2
    return H


def uot_diag(cp):
    """Where the sink / source mass sits (sums over genes; per-third rates use the gene's own A (death) or B (birth)
    law split into lower / middle / upper thirds of its mass)."""
    fA, fB = cp["fA"], cp["fB"]
    if not cp["buckets"]:
        return {}
    ii = np.concatenate([bk["ii"] for bk in cp["buckets"]])
    sink = np.concatenate([bk["s"].sum(1) for bk in cp["buckets"]])
    srcm = np.concatenate([bk["beta"].sum(1) for bk in cp["buckets"]])
    fa, fb = fA[ii], fB[ii]
    out = {"n_genes_ot": int(len(ii)), "n_genes_on": int(cp["on"].sum()),
           "n_genes_off": int(((fA > 0) & (fB == 0)).sum()),
           "sink_over_fA": float(sink.sum() / fa.sum()), "source_over_fB": float(srcm.sum() / fb.sum()),
           "forced_death_over_fA": float(np.maximum(fa - fb, 0).sum() / fa.sum()),
           "forced_birth_over_fB": float(np.maximum(fb - fa, 0).sum() / fb.sum())}
    out["excess_death_over_fA"] = out["sink_over_fA"] - out["forced_death_over_fA"]
    out["excess_birth_over_fB"] = out["source_over_fB"] - out["forced_birth_over_fB"]
    ratios = {"death": [], "birth": []}
    acc = {"death": np.zeros((3, 2)), "birth": np.zeros((3, 2))}
    for bk in cp["buckets"]:
        z = bk["z"]
        bmass = bk["c"] + bk["beta"]
        for nm, mass, ref in (("death", bk["s"], bk["a"]), ("birth", bk["beta"], bmass)):
            tot_m, tot_r = mass.sum(1), ref.sum(1)
            w = tot_m > 1e-6
            lv = (mass * z).sum(1) / np.maximum(tot_m, 1e-300)
            lr = (ref * z).sum(1) / np.maximum(tot_r, 1e-300)
            ratios[nm].append((lv / lr)[w])
            rn = ref / np.maximum(tot_r[:, None], 1e-300)
            mid = np.cumsum(rn, 1) - rn / 2
            for k, (lo, hi) in enumerate(((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))):
                sel = (mid >= lo) & (mid < hi)
                acc[nm][k] += [(mass * sel).sum(), (ref * sel).sum()]
    # integer-grid genes: rate at count 1 versus counts >= 2 (death: share of A's mass at that count destroyed;
    # birth: share of B's mass at that count newly created)
    one = {"death": np.zeros(4), "birth": np.zeros(4)}
    for bk in cp["buckets"]:
        z = bk["z"]
        if z.shape[1] < 2 or not np.allclose(z[:, 1], 2.0):
            continue
        bmass = bk["c"] + bk["beta"]
        for nm, mass, ref in (("death", bk["s"], bk["a"]), ("birth", bk["beta"], bmass)):
            one[nm] += [mass[:, 0].sum(), ref[:, 0].sum(), mass[:, 1:].sum(), ref[:, 1:].sum()]
    for nm in ("death", "birth"):
        o = one[nm]
        out[f"{nm}_rate_count1_vs_ge2"] = [float(o[0] / max(o[1], 1e-300)), float(o[2] / max(o[3], 1e-300))]
    for nm in ("death", "birth"):
        r = np.concatenate(ratios[nm])
        out[f"{nm}_level_ratio_median"] = float(np.median(r)) if len(r) else float("nan")
        out[f"{nm}_rate_by_third"] = [float(x[0] / max(x[1], 1e-300)) for x in acc[nm]]
    return out


# ------------------------------------------------------------------ balanced ablation (dist_bridge count scale)
def bal_coupling(A, B, L, tol=TOL):
    fA, muA, vA, fB, muB, vB = _laws(A, B)
    idx = np.where((fA > 0) & (fB > 0))[0]
    res = {"fA": fA, "fB": fB, "err": 0.0, "iters": 0, "buckets": []}
    if len(idx) == 0:
        return res
    for sel, z, pA, pB in grids(muA[idx], vA[idx], muB[idx], vB[idx]):
        ii = idx[sel]
        C = (z[:, :, None] - z[:, None, :]) ** 2 / 2
        eps = np.maximum(SIGMA2 * L * muB[ii], 1e-3)
        pi, err, it = uot_sinkhorn.solve(pA, pB, C, eps, tol=tol, max_final=MAX_FINAL)
        res["err"], res["iters"] = max(res["err"], err), max(res["iters"], it)
        with np.errstate(divide="ignore", invalid="ignore"):
            src = np.where(pB > 0, np.einsum("gjk,gj->gk", pi, z) / np.where(pB > 0, pB, 1), z)
        res["buckets"].append({"ii": ii, "z": z, "pB": pB, "v": (z - src) / L})
    return res


def bal_step(B, cp, L, delta):
    """dist_bridge.sb_step on the count scale (same formulas), on the bucketed grid."""
    H = B.copy()
    if delta == 0:
        return H
    for bk in cp["buckets"]:
        ii, z, pB, v = bk["ii"], bk["z"], bk["pB"], bk["v"]
        zp = z + delta * v
        pa = np.where(zp >= 0.5, pB, 0.0)
        surv = pa.sum(1)
        gr = np.log(cp["fB"][ii] / cp["fA"][ii]) / L
        f = np.minimum(1.0, cp["fB"][ii] * np.exp(delta * gr)) * surv
        cnt = np.maximum(1.0, zp)
        sd = np.where(surv > 0, surv, 1.0)
        mu = np.where(surv > 0, (pa * cnt).sum(1) / sd, 1.0)
        var = np.where(surv > 0, (pa * (cnt - mu[:, None]) ** 2).sum(1) / sd, 0.0)
        H[ii, 0], H[ii, 1], H[ii, 2] = f * mu, f, f * (var + mu ** 2)
    return H


# ------------------------------------------------------------------ observation heads -> targets
def heads(H, m_now, n_now, rng_seed, targets, draws=DB.DRAWS):
    """Predictions of the requested targets when bin t+1 is n_t cells drawn from the forecast law H."""
    out = {}
    Lt = max(float(m_now.sum()), 1e-12)
    Lh = max(float(H[:, 0].sum()), 1e-12)
    for tg in targets:
        rng = np.random.default_rng(list(rng_seed) + [hash_t(tg)])
        if tg == "linear":
            out[tg] = (H[:, 0] - m_now).astype(np.float32)
        elif tg == "l1_plugin":
            out[tg] = (np.log1p(1e4 * H[:, 0] / Lh) - np.log1p(1e4 * m_now / Lt)).astype(np.float32)
        elif tg == "l1":
            tfn = lambda m0, s: (np.log1p(1e4 * s / Lh) - np.log1p(1e4 * np.maximum(m0, 0) / Lt)).astype(np.float32)
            out[tg] = DB.expected_value(H, m_now, n_now, rng, tfn, draws=draws)
        elif tg == "orig":
            out[tg] = DB.expected_target(H, m_now, n_now, rng, draws=draws)
    return out


def hash_t(s):
    import zlib
    return zlib.crc32(str(s).encode()) & 0x7FFFFFFF
