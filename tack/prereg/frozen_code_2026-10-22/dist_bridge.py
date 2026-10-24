"""DistBridge: a per-gene population model for Dileo & Sottoriva Task 2, built only on their node files.

Each bin of their '-human' builds is summarised per gene by raw-count statistics (mean m, sd s, nonzero
fraction f, n cells). A bin is read as a cell-count law: P(c = 0) = 1 - f and, given c > 0, c = 1 + E with E
negative binomial (or Poisson) matching the positive-part mean and variance. The forecast of bin t+1 is:
  1. pool: a denoised law at t from the recent bins (Box: cell-weighted mixture of the last k bins; or a causal
     local-level Kalman filter on the mean channel, the same gain applied to the whole law);
  2. OT / Schroedinger-bridge step: per gene, an entropic OT plan (own Sinkhorn, Brownian cost on the Anscombe
     scale z = 2 sqrt(c + 3/8)) couples the positive part of the pooled law L bins ago with today's; the barycentric
     velocity of each count level is extrapolated delta bins ahead, levels pushed below c = 1/2 switch off, and the
     nonzero mass follows its log-linear trend. delta = 0 leaves the law unchanged. For one gene the mean drift on
     z is the same for every coupling; the plan only shapes the spread, hence how many cells cross zero;
  3. observation: bin t+1 is n = n_t cells drawn from the forecast law; the prediction is the expected clipped
     target E[clip(log(S / n + 1e-8) - log(m_t + 1e-8), -5, 5)], zero-crossings analytic, the rest Monte Carlo;
  4. selection (strict): the configuration is chosen per transition by rolling origin on past transitions only
     (criterion J = Pearson + tie-aware expected P@200 up + down, all genes, targets rebuilt from bins <= t).
Transition i = 0 uses its own bin (pool = x_0, delta = 0). Everything is deterministic: Monte Carlo draws come
from numpy generators keyed by (dataset, transition, config), never from torch.
"""
import hashlib

import numpy as np
from scipy import stats

import sinkhorn

EPS = 1e-8
CLIP = 5.0
GRID = 48
Z_HALF = 2.0 * np.sqrt(0.5 + 3 / 8)
Z_ONE = 2.0 * np.sqrt(1.0 + 3 / 8)
DRAWS = 256
WINDOW = 8
DEFAULT = ("kalman", 0.2, 0.0)
POOLS = (("box", 2), ("box", 4), ("box", 6), ("box", 8), ("kalman", 0.1), ("kalman", 0.2), ("kalman", 0.35))
DELTAS = (0.0, 0.5, 1.0)
C_R, C_Q = 1.0, 1 / 50
L_SB, SIGMA2_SB = 3, 0.25


def z_of(c):
    return 2.0 * np.sqrt(np.asarray(c, float) + 3 / 8)


def c_of(z):
    return np.maximum(1.0, (np.asarray(z, float) / 2) ** 2 - 3 / 8)


def log_delta(m_now, m_next):
    d = np.log(np.maximum(m_next, 0).astype(np.float32) + np.float32(EPS)) \
        - np.log(np.maximum(m_now, 0).astype(np.float32) + np.float32(EPS))
    return np.clip(d, -CLIP, CLIP)


def n_cells(x):
    ok = x[:, 0] > 0
    return float(np.median(x[ok, 4] / x[ok, 0])) if ok.any() else 0.0


def triplet(x):
    """(m, f, e) with e = E[c^2] = s^2 + m^2 (s is the population sd)."""
    return np.stack([x[:, 0], x[:, 3], x[:, 2] ** 2 + x[:, 0] ** 2], 1)


def positive_part(H):
    """Mean and variance of c given c > 0, from a triplet; 0 where f = 0."""
    m, f, e = H[:, 0], np.clip(H[:, 1], 0, 1), H[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(f > 0, m / np.where(f > 0, f, 1), 0.0)
        var = np.where(f > 0, np.maximum(e / np.where(f > 0, f, 1) - mu ** 2, 0.0), 0.0)
    return f, mu, var


# ---------------- pools (causal) ----------------

def box_pool(trip, ns, k):
    """H_b = cell-weighted mixture of the last k bins <= b, for every b."""
    out = []
    for b in range(len(trip)):
        lo = max(0, b - k + 1)
        w = np.array(ns[lo:b + 1])
        out.append(sum(wi * t for wi, t in zip(w, trip[lo:b + 1])) / w.sum())
    return out


def kalman_pool(trip, ns, theta):
    """Local-level filter on the bin mean (count-scale noise R, drift variance q); the same gain mixes the law."""
    H = trip[0].copy()
    out = [H.copy()]
    msum, nsum = trip[0][:, 0] * ns[0], ns[0]
    s2 = lambda t: np.maximum(t[:, 2] - t[:, 0] ** 2, 0.0)
    P = (np.maximum(s2(trip[0]), trip[0][:, 0]) + C_R / max(ns[0], 1)) / max(ns[0], 1)
    for b in range(1, len(trip)):
        msum, nsum = msum + trip[b][:, 0] * ns[b], nsum + ns[b]
        mbar = msum / nsum
        R = (np.maximum(s2(trip[b]), mbar) + C_R / max(ns[b], 1)) / max(ns[b], 1)
        Pm = P + theta ** 2 * (mbar ** 2 + C_Q * mbar)
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.where(Pm + R > 0, Pm / (Pm + R), 1.0)
        H = (1 - K)[:, None] * H + K[:, None] * trip[b]
        P = (1 - K) * Pm
        out.append(H.copy())
    return out


# ---------------- OT / Schroedinger-bridge step ----------------

def _law_kind(mu, var):
    lam = mu - 1
    return np.where(lam <= 1e-9, 0, np.where(var > lam, 2, 1))


def _law_cdf(kind, mu, var, c):
    """P(count <= c) of the positive-part law (count = 1 + E), vectorised over (genes, points)."""
    e = np.floor(c) - 1
    out = np.zeros_like(c, dtype=float)
    lam = mu - 1
    nb, po, dg = kind == 2, kind == 1, kind == 0
    if nb.any():
        out[nb] = stats.nbinom.cdf(e[nb], (lam ** 2 / (var - lam))[nb][:, None], (lam / var)[nb][:, None])
    if po.any():
        out[po] = stats.poisson.cdf(e[po], lam[po][:, None])
    if dg.any():
        out[dg] = (e[dg] >= 0).astype(float)
    return out


def _law_ppf(kind, mu, var, q):
    lam = mu - 1
    out = np.ones_like(mu)
    nb, po = kind == 2, kind == 1
    if nb.any():
        out[nb] = 1 + stats.nbinom.ppf(q, lam[nb] ** 2 / (var[nb] - lam[nb]), lam[nb] / var[nb])
    if po.any():
        out[po] = 1 + stats.poisson.ppf(q, lam[po])
    return out


def grid_masses(muA, vA, muB, vB):
    """Shared per-gene grid on the Anscombe scale and the positive-part pmfs of A and B on it."""
    kA, kB = _law_kind(muA, vA), _law_kind(muB, vB)
    cmax = np.maximum(np.maximum(_law_ppf(kA, muA, vA, 0.999), _law_ppf(kB, muB, vB, 0.999)), 1.0)
    G = len(muA)
    j = np.arange(GRID, dtype=float)[None, :]
    integer = cmax <= GRID
    z = np.where(integer[:, None], z_of(1 + j), z_of(1) + (z_of(cmax)[:, None] - z_of(1)) * j / (GRID - 1))
    # cell edges halfway between grid points (in z), mapped back to counts; the last cell takes the tail
    zmid = (z[:, 1:] + z[:, :-1]) / 2
    c_hi = np.concatenate([np.where(integer[:, None], 1 + j[:, :-1], (zmid / 2) ** 2 - 3 / 8), np.full((G, 1), 1e12)], 1)
    masses = []
    for kind, mu, var in ((kA, muA, vA), (kB, muB, vB)):
        cdf = _law_cdf(kind, mu, var, c_hi)
        cdf[:, -1] = 1.0
        pm = np.maximum(np.diff(np.concatenate([np.zeros((G, 1)), cdf], 1), axis=1), 0)
        masses.append(pm / pm.sum(1, keepdims=True))
    return z, masses[0], masses[1]


def coupling(A, B, L, chunk=3000):
    """Per-gene entropic OT plans between the positive parts of laws A and B (Brownian cost, eps = sigma^2 L)."""
    fA, muA, vA = positive_part(A)
    fB, muB, vB = positive_part(B)
    idx = np.where((fA >= 1e-6) & (fB >= 1e-6))[0]
    res = {"idx": idx, "fA": fA, "fB": fB, "z": None, "pB": None, "v": None, "err": 0.0}
    if len(idx) == 0:
        return res
    zs, pBs, vs = [], [], []
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        z, pA, pB = grid_masses(muA[ii], vA[ii], muB[ii], vB[ii])
        C = (z[:, :, None] - z[:, None, :]) ** 2 / 2
        pi, err = sinkhorn.solve(pA, pB, C, max(SIGMA2_SB * L, 1e-3))
        res["err"] = max(res["err"], err)
        with np.errstate(divide="ignore", invalid="ignore"):
            src = np.where(pB > 0, np.einsum("gjk,gj->gk", pi, z) / np.where(pB > 0, pB, 1), z)
        zs.append(z)
        pBs.append(pB)
        vs.append((z - src) / L)
    res["z"], res["pB"], res["v"] = np.concatenate(zs), np.concatenate(pBs), np.concatenate(vs)
    return res


def sb_step(B, cp, L, delta):
    """Push today's law delta bins ahead along the coupling's velocity field; delta = 0 returns B unchanged."""
    if delta == 0 or len(cp["idx"]) == 0:
        return B
    H = B.copy()
    ii = cp["idx"]
    zp = cp["z"] + delta * cp["v"]
    pa = np.where(zp >= Z_HALF, cp["pB"], 0.0)
    surv = pa.sum(1)
    g = np.log(cp["fB"][ii] / cp["fA"][ii]) / L
    f = np.minimum(1.0, cp["fB"][ii] * np.exp(delta * g)) * surv
    cnt = c_of(zp)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(surv > 0, (pa * cnt).sum(1) / np.where(surv > 0, surv, 1), 1.0)
        var = np.where(surv > 0, (pa * (cnt - mu[:, None]) ** 2).sum(1) / np.where(surv > 0, surv, 1), 0.0)
    H[ii, 0] = f * mu
    H[ii, 1] = f
    H[ii, 2] = f * (var + mu ** 2)
    return H


# ---------------- observation model ----------------

def expected_target(H, m_now, n, rng, draws=DRAWS):
    """E[clip(log(S/n + eps) - log(m_now + eps))] when bin t+1 is n cells from law H (counts).

    No cell expressed (probability P0): the target is -5 if m_now > 0, 0 otherwise; if m_now = 0 any S >= 1 gives
    +5. Otherwise K (expressing cells | K >= 1) is drawn exactly (first success truncated-geometric, the rest
    binomial) and S = K + NB (or Poisson) extra counts."""
    n = max(int(round(n)), 1)
    f, mu, var = positive_part(H)
    P0 = (1 - f) ** n
    y = np.where(m_now > 0, -CLIP * P0, CLIP * (1 - P0))
    act = np.where((m_now > 0) & (f > 0))[0]
    if len(act):
        fa, mua, va = f[act], mu[act], var[act]
        u = rng.random((len(act), draws))
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.ceil(np.log1p(-u * (1 - P0[act, None])) / np.log1p(-fa[:, None]))
        J = np.clip(np.nan_to_num(J, nan=1.0, posinf=n, neginf=1.0), 1, n).astype(np.int64)
        Kc = 1 + rng.binomial(n - J, fa[:, None])
        lam = (mua - 1)[:, None] * Kc
        vv = va[:, None] * Kc
        kind = np.broadcast_to(_law_kind(mua, va)[:, None], lam.shape)
        extra = np.zeros_like(lam)
        nb, po = kind == 2, kind == 1
        if nb.any():
            ln, vn = lam[nb], vv[nb]
            extra[nb] = rng.negative_binomial(ln ** 2 / (vn - ln), ln / vn)
        if po.any():
            extra[po] = rng.poisson(lam[po])
        cond = log_delta(np.repeat(m_now[act, None], draws, 1), (Kc + extra) / n).mean(1)
        y[act] = y[act] + (1 - P0[act]) * cond
    return y.astype(np.float32)


# ---------------- selection criterion ----------------

def _incl(v, K):
    """Inclusion probability in the top K under uniform random tie-breaking."""
    thr = np.sort(v)[::-1][K - 1]
    above = v > thr
    eq = v == thr
    p = above.astype(float)
    p[eq] = (K - above.sum()) / eq.sum()
    return p


def criterion(y, p, q):
    """J = (Pearson + expected P@K up + expected P@K down) / 3 with K = round(q N)."""
    K = max(1, int(round(q * len(y))))
    yc, pc = y - y.mean(), p - p.mean()
    den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
    r = float((yc * pc).sum() / den) if den > 0 else 0.0
    up = float((_incl(y, K) * _incl(p, K)).sum() / K)
    dn = float((_incl(-y, K) * _incl(-p, K)).sum() / K)
    return (r + up + dn) / 3


# ---------------- forecaster ----------------

_CACHE = {}
ALL_CONFIGS = [(p[0], p[1], d) for p in POOLS for d in DELTAS]


class DistBridge:
    """Strict forecaster: reads only the raw node stats of snapshots <= t."""
    strict = True

    def __init__(self, name="distbridge_strict", deltas=DELTAS, pools=POOLS, fixed=None, obs="count"):
        """obs: 'count' (the model), 'clt' (normal sum of positives, reproduces task2_forecasters.ObsLevel),
        'inf' (infinitely many cells: plain log-ratio of means, reproduces task2_forecasters.Level)."""
        self.name = name
        self.configs = [(p[0], p[1], d) for p in pools for d in deltas]
        self.fixed = fixed
        self.obs = obs
        self.selected = {}

    def _state(self, hist):
        key = hashlib.sha1(np.ascontiguousarray(hist[0]).tobytes()).hexdigest() + self.obs
        st = _CACHE.setdefault(key, {"trip": [], "ns": [], "pools": {}, "cpl": {}, "pred": {}, "id": int(key[:8], 16)})
        while len(st["trip"]) < len(hist):
            x = hist[len(st["trip"])]
            st["trip"].append(triplet(x))
            st["ns"].append(n_cells(x))
        return st

    def _pool(self, st, p, b):
        """Pooled law at bin b; computed from bins <= b only (both pools are causal)."""
        k = (p, b)
        if k not in st["pools"]:
            fn = box_pool if p[0] == "box" else kalman_pool
            st["pools"][k] = fn(st["trip"][:b + 1], st["ns"][:b + 1], p[1])[b]
        return st["pools"][k]

    def _forecast(self, st, hist, i, cfg):
        key = (i, cfg)
        if key in st["pred"]:
            return st["pred"][key]
        rng = np.random.default_rng([st["id"], i, ALL_CONFIGS.index(cfg) if cfg in ALL_CONFIGS else 999])
        if i == 0:
            H = st["trip"][0]
        else:
            p = cfg[:2]
            H = self._pool(st, p, i)
            if cfg[2] > 0:
                L = min(L_SB, i)
                ck = (p, i, L)
                if ck not in st["cpl"]:
                    st["cpl"][ck] = coupling(self._pool(st, p, i - L), H, L)
                H = sb_step(H, st["cpl"][ck], L, cfg[2])
        m_now = hist[i][:, 0]
        if self.obs == "inf":
            pred = log_delta(m_now, H[:, 0])
        elif self.obs == "clt":
            import task2_forecasters as tf
            pred = tf.expected_target(m_now, H[:, 1], H[:, 0], H[:, 2], int(round(st["ns"][i])), DRAWS,
                                      np.random.default_rng(0 + i + 1))
        else:
            pred = expected_target(H, m_now, st["ns"][i], rng)
        st["pred"][key] = pred
        return pred

    def predict(self, hist, ts, ctx):
        st = self._state(hist)
        i = len(hist) - 1
        q = 200.0 / round(0.25 * hist[0].shape[0])
        if self.fixed is not None:
            cfg = self.fixed
        elif i <= 2:
            cfg = DEFAULT if DEFAULT in self.configs else self.configs[0]
        else:
            W = range(max(1, i - WINDOW), i)
            scores = []
            for c in self.configs:
                js = [criterion(log_delta(hist[j][:, 0], hist[j + 1][:, 0]), self._forecast(st, hist, j, c), q) for j in W]
                scores.append(float(np.mean(js)))
            best = max(scores)
            # ties within 1e-3: prefer no transport step, then the earlier configuration
            cand = [c for c, s in zip(self.configs, scores) if s >= best - 1e-3]
            cfg = sorted(cand, key=lambda c: (c[2] > 0, self.configs.index(c)))[0]
        self.selected[(st["id"], i)] = cfg
        return self._forecast(st, hist, i, cfg)


REGISTRY = {
    "distbridge_strict": lambda: DistBridge("distbridge_strict"),
    "distbridge_strict_nosb": lambda: DistBridge("distbridge_strict_nosb", deltas=(0.0,)),
    "obslevel_rollk": lambda: DistBridge("obslevel_rollk", deltas=(0.0,), pools=POOLS[:4]),
}
