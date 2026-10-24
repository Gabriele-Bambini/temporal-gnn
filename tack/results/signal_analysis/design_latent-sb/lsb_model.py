"""Latent-space Schroedinger bridge / state-space forecaster (strict, numpy only).

Level space X_gb = log1p(CP10k_gb). Measurement model (known, not learned):
    Yb_gb = X_gb + R_gb / 2 = x_gb + e_gb,   e_gb ~ N(0, R_gb),
    R_gb = (1e4 / L_b)^2 * V_gb / n_b / (1 + c_gb)^2      (delta method; R/2 removes the Jensen bias of log1p),
with V_gb = s_gb^2 n_b / (n_b - 1) (s02_std is the population sd) floored by the Poisson value of a causal local
mean (cell-weighted bins b-2..b, at least half a count over that window); c_gb is the CP10k at the average of
the observed and the local mean. R uses only bins <= b.
State (true level) x_gb = mu_g + U_g z_b + r_gb:
  * latent part: k-dim factor scores z_b of the bin profiles. mu, U at origin t from a precision-weighted SVD of
    Yb over bins <= t ('cur') or <= t-1 ('lag'); per-bin GLS scores z_b^obs with covariance Lambda_b^{-1},
    Lambda_b = U^T diag(1 / (R_b + tau)) U (tau = idiosyncratic variance by expression class, two passes).
    Dynamics = the reference linear SDE of a Gaussian Schroedinger bridge in latent space, discretised:
      'level': z_{b+1} = z_b + w                                  (Brownian reference, no drift)
      'ar'   : dz_{b+1} = Phi dz_b + w, Phi = diag(phi_j)          (integrated OU; phi_j by noise-corrected MoM)
      'var'  : dz_{b+1} = A dz_b + w, A = ridge regression on consecutive score changes (learned SB drift)
    run as a Kalman filter on z^obs; forecast z_{t+1|t}. For Gaussian marginals the SB between the filtered
    law at t and the propagated law at t+1 IS the reference transition, so its forecasting content is the
    drift (Phi / A) and the diffusion (Q); the mean displacement of any coupling equals the difference of means.
  * per-gene residual r_gb = Yb_gb - mu_g - U_g z_b^obs: ARIMA(1,1,0) state space per gene
      d_{b+1} = rho_g d_b + w,  r_{b+1} = r_b + d_{b+1},  Var(d) = q_g,
    q_g, rho_g by method of moments on bins <= t with the known noise subtracted
      E[dr_b^2] = q + R_b + R_{b-1},   E[dr_b dr_{b+1}] = rho q - R_b,
    pooled by expression class (EB shrinkage of per-gene q toward the class value with strength nu),
    over the last W differences whose bins have >= 20 cells. mom='none' forces rho = 0 (local level).
Forecast x_{t+1} = mu + U z_{t+1|t} + l_t + rho d_t ; current denoised level x_t = mu + U z_{t|t} + l_t.
Heads: l1 = x_{t+1} - R_t / 2 - X_t (expected observed log1p CP10k of a bin of n_t cells minus today's observed);
linear = (Lhat / 1e4) expm1(x_{t+1}) - m_t, Lhat = L_t (L_t / L_{t-1})^beta; orig = finite-cell law head
(dist_bridge.expected_target, n = n_t) on a causal box-pooled law whose mean is moved toward the latent forecast.
"""
import numpy as np

import dist_bridge as DB  # noqa: E402  (tack/hpc on sys.path via lsb_common; numpy + scipy only)

FLOOR_COUNTS = 0.5
MIN_CELLS_MOM = 20
N_CLASSES = 8


class LatentSB:
    def __init__(self, d):
        self.d = d
        G, S = d.M.shape
        n = d.nc
        R = np.zeros((G, S))
        for b in range(S):
            lo = max(0, b - 2)
            w = n[lo:b + 1]
            mloc = np.maximum((d.M[:, lo:b + 1] * w).sum(1) / w.sum(), FLOOR_COUNTS / w.sum())
            V = np.maximum(d.SD[:, b] ** 2 * n[b] / max(n[b] - 1, 1), mloc)
            c = 1e4 * 0.5 * (d.M[:, b] + mloc) / d.lib[b]
            R[:, b] = (1e4 / d.lib[b]) ** 2 * V / n[b] / (1 + c) ** 2
        self.R = R
        self.Yb = d.X + R / 2
        self.nmed = float(np.median(n))
        self._fac, self._zf, self._res, self._cls = {}, {}, {}, {}

    # ---------------------------------------------------------------- expression classes (causal)
    def classes(self, t):
        if t not in self._cls:
            d = self.d
            lo = max(0, t - 3)
            w = d.nc[lo:t + 1]
            lv = (d.CP[:, lo:t + 1] * w).sum(1) / w.sum()
            cls = np.zeros(d.N, int)
            pos = lv > 0
            if pos.sum() > N_CLASSES:
                q = np.quantile(np.log(lv[pos]), np.linspace(0, 1, N_CLASSES + 1)[1:-1])
                cls[pos] = 1 + np.digitize(np.log(lv[pos]), q)
            self._cls[t] = cls
        return self._cls[t]

    @staticmethod
    def _by_class(v, cls, floor=0.0):
        out = np.zeros(cls.max() + 1)
        for c in range(len(out)):
            s = cls == c
            out[c] = max(float(np.mean(v[s])), floor) if s.any() else floor
        return out

    # ---------------------------------------------------------------- latent factor part
    def factor(self, t, k, basis):
        """mu (G), U (G,k), zobs (t+1,k), Rz (t+1,k,k). Basis from bins <= t ('cur') or <= t-1 ('lag')."""
        key = (t, k, basis)
        if key in self._fac:
            return self._fac[key]
        d = self.d
        hi = t if basis == "lag" else t + 1
        k_eff = min(k, hi - 2)
        if k_eff < 1:
            self._fac[key] = None
            return None
        Y, R = self.Yb[:, :hi], self.R[:, :hi]
        cls = self.classes(t)
        cw = np.sqrt(np.minimum(d.nc[:hi], self.nmed) / self.nmed)
        # pass 1: tau from within-gene temporal variance minus noise
        v0 = np.maximum(Y.var(1) - R.mean(1), 0)
        tau = self._by_class(v0, cls, 1e-6)[cls]
        for _ in range(2):
            w = 1.0 / (R + tau[:, None])
            mu = (w * Y).sum(1) / w.sum(1)
            om = 1.0 / (R.mean(1) + tau)
            A = np.sqrt(om)[:, None] * (Y - mu[:, None]) * cw[None, :]
            Uu, sv, _ = np.linalg.svd(A, full_matrices=False)
            U = Uu[:, :k_eff] / np.sqrt(om)[:, None]
            Wfull = 1.0 / (self.R[:, :t + 1] + tau[:, None])
            UU = (U[:, :, None] * U[:, None, :]).reshape(len(U), k_eff * k_eff)
            Lam = (Wfull.T @ UU).reshape(t + 1, k_eff, k_eff)
            eta = (Wfull * (self.Yb[:, :t + 1] - mu[:, None])).T @ U
            z = np.linalg.solve(Lam, eta[:, :, None])[:, :, 0]
            res = self.Yb[:, :hi] - mu[:, None] - U @ z[:hi].T
            tau = self._by_class(np.maximum(res ** 2 - R, 0).mean(1), cls, 1e-6)[cls]
        Rz = np.linalg.inv(Lam)
        out = {"mu": mu, "U": U, "z": z, "Rz": Rz, "tau": tau, "k": k_eff}
        self._fac[key] = out
        return out

    def zforecast(self, t, k, basis, dyn):
        """Kalman filter on GLS scores with the latent reference dynamics; returns (z_t|t, z_{t+1|t}, info)."""
        key = (t, k, basis, dyn)
        if key in self._zf:
            return self._zf[key]
        F = self.factor(t, k, basis)
        if F is None:
            self._zf[key] = None
            return None
        z, Rz, ke = F["z"], F["Rz"], F["k"]
        nb = z.shape[0]
        dz = np.diff(z, axis=0)
        ndz = np.array([np.diag(Rz[b]) + np.diag(Rz[b - 1]) for b in range(1, nb)])
        q = np.maximum((dz ** 2).mean(0) - ndz.mean(0), 1e-3 * (dz ** 2).mean(0) + 1e-12)
        if dyn == "level":
            Phi = np.zeros((ke, ke))
        elif dyn == "ar":
            if nb >= 3:
                num = (dz[1:] * dz[:-1]).mean(0) + np.array([np.diag(Rz[b]) for b in range(1, nb - 1)]).mean(0)
                phi = np.clip(num / q, 0.0, 0.95)
            else:
                phi = np.zeros(ke)
            Phi = np.diag(phi)
        else:  # 'var': ridge regression of dz_{b+1} on dz_b
            if nb >= 3:
                Xr, Yr = dz[:-1], dz[1:]
                lam = 1.0 * np.trace(Xr.T @ Xr) / ke
                Phi = np.linalg.solve(Xr.T @ Xr + lam * np.eye(ke), Xr.T @ Yr).T
                sr = np.max(np.abs(np.linalg.eigvals(Phi)))
                if sr > 0.95:
                    Phi = Phi * 0.95 / sr
            else:
                Phi = np.zeros((ke, ke))
        # state s = (z, dz) ; z_{b+1} = z_b + Phi dz_b + w ; dz_{b+1} = Phi dz_b + w ; Var(dz) ~ q
        I = np.eye(ke)
        Fm = np.block([[I, Phi], [np.zeros((ke, ke)), Phi]])
        qd = np.diag(q)
        Qw = qd - Phi @ qd @ Phi.T if dyn != "level" else qd
        Qw = 0.5 * (Qw + Qw.T)
        ev, evec = np.linalg.eigh(Qw)
        Qw = (evec * np.maximum(ev, 1e-3 * q.min())) @ evec.T
        Q = np.block([[Qw, Qw], [Qw, Qw]])
        H = np.hstack([I, np.zeros((ke, ke))])
        s = np.concatenate([z[0], np.zeros(ke)])
        P = np.block([[Rz[0] + 10 * qd, np.zeros((ke, ke))], [np.zeros((ke, ke)), qd]])
        for b in range(nb):
            if b > 0:
                s = Fm @ s
                P = Fm @ P @ Fm.T + Q
            Sm = H @ P @ H.T + Rz[b]
            K = P @ H.T @ np.linalg.inv(Sm)
            s = s + K @ (z[b] - H @ s)
            P = P - K @ H @ P
        z_now = s[:ke]
        z_next = (Fm @ s)[:ke]
        info = {"score_noise_over_signal": float(np.mean(ndz.mean(0) / np.maximum((dz ** 2).mean(0), 1e-12))),
                "phi_norm": float(np.abs(Phi).max()) if ke else 0.0}
        out = (z_now, z_next, info)
        self._zf[key] = out
        return out

    # ---------------------------------------------------------------- per-gene residual part
    def resid(self, t, k, basis, W, nu, mom):
        """Per-gene ARIMA(1,1,0) Kalman on the residual; returns (l_t, l_t + rho d_t, rho, q)."""
        key = (t, k, basis, W, nu, mom)
        if key in self._res:
            return self._res[key]
        d = self.d
        F = self.factor(t, k, basis) if k > 0 else None
        Y = self.Yb[:, :t + 1]
        if F is not None:
            r = Y - F["mu"][:, None] - F["U"] @ F["z"].T
        else:
            r = Y
        R = self.R[:, :t + 1]
        cls = self.classes(t)
        G = d.N
        rho = np.zeros(G)
        q = np.full(G, 1e-4)
        if t >= 1:
            dr = np.diff(r, axis=1)  # dr[:, b-1] = r_b - r_{b-1}
            nd = R[:, 1:] + R[:, :-1]
            ok = np.array([min(d.nc[b - 1], d.nc[b]) >= MIN_CELLS_MOM for b in range(1, t + 1)])
            if not ok.any():
                ok[:] = True
            idx = np.where(ok)[0]
            if W is not None:
                idx = idx[-W:]
            ex = dr[:, idx] ** 2 - nd[:, idx]
            qg = ex.mean(1)
            qc = self._by_class(qg, cls, 1e-8)
            if nu is None:
                q = qc[cls].copy()
            else:
                ne = len(idx)
                q = np.maximum((ne * qg + nu * qc[cls]) / (ne + nu), 0.05 * qc[cls])
            if mom == "mom":
                pairs = [j for j in idx if j + 1 <= dr.shape[1] - 1 and ok[j + 1]]
                if len(pairs) >= 1:
                    pj = np.array(pairs)
                    num = (dr[:, pj] * dr[:, pj + 1] + R[:, pj + 1]).mean(1)
                    nc_ = self._by_class(num, cls, -np.inf)
                    rho = np.clip(nc_ / qc, 0.0, 0.9)[cls]
        # Kalman over bins 0..t
        ell = r[:, 0].copy()
        dd = np.zeros(G)
        P11, P12, P22 = R[:, 0].copy(), np.zeros(G), q.copy()
        Qs = q * (1 - rho ** 2)
        for b in range(1, t + 1):
            ell = ell + rho * dd
            dd = rho * dd
            p11 = P11 + 2 * rho * P12 + rho ** 2 * P22 + Qs
            p12 = rho * P12 + rho ** 2 * P22 + Qs
            p22 = rho ** 2 * P22 + Qs
            Sv = p11 + R[:, b]
            K1, K2 = p11 / Sv, p12 / Sv
            e = r[:, b] - ell
            ell = ell + K1 * e
            dd = dd + K2 * e
            P11, P12, P22 = (1 - K1) * p11, (1 - K1) * p12, p22 - K2 * p12
        out = (ell, ell + rho * dd, rho, q)
        self._res[key] = out
        return out

    # ---------------------------------------------------------------- levels and heads
    def levels(self, t, cfg):
        """cfg = (k, basis, dyn, W, nu, mom). Returns (x_t denoised, x_{t+1} forecast, parts)."""
        k, basis, dyn, W, nu, mom = cfg
        ell_now, ell_next, rho, q = self.resid(t, k, basis, W, nu, mom)
        x_now, x_next = ell_now.copy(), ell_next.copy()
        fac_now = fac_chg = np.zeros(self.d.N)
        if k > 0:
            F = self.factor(t, k, basis)
            zf = self.zforecast(t, k, basis, dyn)
            if F is not None and zf is not None:
                fac_now = F["mu"] + F["U"] @ zf[0]
                fac_chg = F["U"] @ (zf[1] - zf[0])
                x_now = x_now + fac_now
                x_next = x_next + fac_now + fac_chg
        return x_now, x_next, {"fac_chg": fac_chg, "res_chg": ell_next - ell_now}

    def head_l1(self, t, x_next):
        return x_next - self.R[:, t] / 2 - self.d.X[:, t]

    def head_linear(self, t, x_next, beta):
        d = self.d
        L = d.lib[t] * ((d.lib[t] / d.lib[t - 1]) ** beta if t >= 1 else 1.0)
        return np.maximum(L / 1e4 * np.expm1(x_next), 0) - d.M[:, t]

    def law_orig(self, t, src_level, kpool, alpha):
        """Causal box pool of the last kpool bin laws, mean moved by (latent / pooled)^alpha via the nonzero
        fraction (positive-part mean kept unless the fraction would exceed 1)."""
        d = self.d
        lo = max(0, t - kpool + 1)
        w = d.nc[lo:t + 1]
        H = sum(wi * d.trip[b] for wi, b in zip(w, range(lo, t + 1))) / w.sum()
        if alpha == 0:
            return H
        mp, fp, ep = H[:, 0], H[:, 1], H[:, 2]
        mhat = d.lib[t] / 1e4 * np.expm1(np.maximum(src_level, 0))
        pos = (mp > 0) & (fp > 0)
        ratio = np.ones(d.N)
        ratio[pos] = np.clip(mhat[pos] / mp[pos], 0.05, 20.0) ** alpha
        f_, mu_pp, var_pp = DB.positive_part(H)
        m2 = mp * ratio
        f2 = np.minimum(1.0, fp * ratio)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu2 = np.where(f2 > 0, m2 / np.where(f2 > 0, f2, 1), 0.0)
        low = pos & (mu2 < 1)
        f2[low] = m2[low]
        mu2[low] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(mu_pp > 0, mu2 / np.where(mu_pp > 0, mu_pp, 1), 1.0)
        var2 = var_pp * sc ** 2
        H2 = H.copy()
        H2[pos, 0] = m2[pos]
        H2[pos, 1] = f2[pos]
        H2[pos, 2] = (f2 * (var2 + mu2 ** 2))[pos]
        return H2

    def head_orig(self, t, src_level, kpool, alpha, seed):
        H = self.law_orig(t, src_level, kpool, alpha)
        rng = np.random.default_rng(seed)
        return np.asarray(DB.expected_target(H, self.d.M[:, t], self.d.nc[t], rng), np.float64)
