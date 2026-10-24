"""Factor-space state-space model (the architecture suggested by the structure analysis), strict.

At transition i: the gene-centred, cell-weighted SVD basis of L over bins <= i (forecasts.SVDFactor.basis), scores
z_b = U_k^T (L_b - mu). Each score dimension j follows a local-linear-trend Kalman filter over the bins with
observation noise r_bj = sum_g U_gj^2 kappa_g / n_b (kappa_g = strict running per-cell variance of gene g in L units,
so small bins are down-weighted by their cell count) and process noise lambda, mu x the dimension's mean squared
past score change. Forecast z_hat = level + slope at i, mapped back with the residual zero / avg3 / last.
Selection: rolling origin as everywhere (common.select), fallback = per-gene trend at i <= 1.
Compared on identical masks with the per-gene selection and the SVD / NMF selections of forecasts_<dataset>.json.
Output: extra_svdkf_<dataset>.json.
"""
import json
import sys

import numpy as np

import common
import forecasts as F


class SVDKF:
    name = "svdkf"

    def __init__(self, d):
        self.d = d
        self.svd = F.SVDFactor(d)
        self.configs = [(k, lam, mu, r) for k in (2, 3, 4, 6, 8) for lam in (0.03, 0.1, 0.3, 1.0)
                        for mu in (0.0, 0.03, 0.3) for r in ("zero", "avg3", "last")]
        pc = (1e4 / d.lib[None, :]) ** 2 * d.SD ** 2 / (1 + d.CP) ** 2
        self.kappa = np.cumsum(pc * d.nc[None, :], 1) / np.cumsum(d.nc)[None, :]
        self._z = {}

    def zhat(self, i, k, lam, mu):
        key = (i, k, lam, mu)
        if key in self._z:
            return self._z[key]
        d = self.d
        m, U, Z, w = self.svd.basis(i)
        kk = min(k, U.shape[1])
        Uk, Zk = U[:, :kk], Z[:kk]
        RU = (Uk ** 2).T @ self.kappa[:, i]                     # per-cell noise of each score
        T = Zk.shape[1]
        dz2 = np.diff(Zk, axis=1) ** 2
        V = dz2.mean(1) if T > 1 else np.full(kk, 1e-4)
        q1, q2 = lam * V, mu * V
        l, b = Zk[:, 0].copy(), np.zeros(kk)
        P11, P12, P22 = np.full(kk, 10.0) * (V + 1e-8), np.zeros(kk), np.full(kk, 1.0) * (V + 1e-8)
        for s in range(T):
            if s > 0:
                l, b = l + b, b
                P11, P12, P22 = P11 + 2 * P12 + P22 + q1, P12 + P22, P22 + q2
            R = RU / d.nc[s]
            Sv = P11 + R
            K1, K2 = P11 / Sv, P12 / Sv
            e = Zk[:, s] - l
            l, b = l + K1 * e, b + K2 * e
            P11, P12, P22 = (1 - K1) * P11, (1 - K1) * P12, P22 - K2 * P12
        out = (l + b, kk)
        self._z[key] = out
        return out

    def predict(self, c, i, oracle=False):
        d = self.d
        k, lam, mu, r = self.configs[c]
        if i == 0:
            return d.L[:, 0].copy()
        m, U, Z, w = self.svd.basis(i)
        zh, kk = self.zhat(i, k, lam, mu)
        Uk, Zk = U[:, :kk], Z[:kk]
        base = m + Uk @ zh
        if r == "zero":
            return base
        Rb = (d.L[:, :i + 1] - m[:, None]) - Uk @ Zk
        if r == "last":
            return base + Rb[:, -1]
        lo = max(0, i - 2)
        return base + (Rb[:, lo:] * w[lo:]).sum(1) / w[lo:].sum()


def from_json(rep, T):
    a = np.full((T, 3), np.nan)
    for r in rep["per_transition"]:
        a[r["i"]] = [np.nan if r["pearson"] is None else r["pearson"], r["up"], r["down"]]
    return a


def main(name):
    d = common.Data(name)
    mk = common.masks(d)
    Y = {t: common.all_targets(d, t) for t in ("l1", "linear")}
    base = F.baselines(d)
    prev = json.load(open(f"{common.OUT}/forecasts_{name}.json"))
    fam = SVDKF(d)
    J = F.j_matrix(d, fam, Y)
    out = {"dataset": name, "protocol": __doc__, "methods": {}, "paired": {}}
    for tname in ("l1", "linear"):
        fb = base[(tname, "trend")]
        res = {}
        for sn, pr in (("svdkf_all", lambda c: True), ("svdkf_zero", lambda c: c[3] == "zero"),
                       ("svdkf_avg3", lambda c: c[3] == "avg3"), ("svdkf_last", lambda c: c[3] == "last")):
            rows = F.rows_for(fam, tname, pr)
            P, labels = F.selected_preds(d, fam, J[tname], tname, rows, fb)
            sc = common.test_scores(Y[tname], P, mk)
            res[sn] = sc
            out["methods"].setdefault(tname, {})[sn] = common.report(sc, d)
            out["methods"][tname][sn]["selected"] = labels
            Ph, lab = F.hindsight(d, fam, J[tname], tname, rows)
            sch = common.test_scores(Y[tname], Ph, mk)
            out["methods"][tname][sn + "__HINDSIGHT_not_strict"] = common.report(sch, d)
            out["methods"][tname][sn + "__HINDSIGHT_not_strict"]["config"] = lab
        refs = {k: from_json(prev["methods"][tname][k], d.T) for k in
                ("trend", "pergene_all", "svd_all", "nmf_all", "nmf_K8", "nmf_persist_denoise")}
        out["paired"][tname] = {f"svdkf_all_vs_{k}": common.paired(res["svdkf_all"], v) for k, v in refs.items()}
        print(f"== {name} {tname}")
        for k, r in out["methods"][tname].items():
            a = r["i>=1"]
            print(f"  {k:36s} {a['pearson']:.3f} {a['p_at_200_up']:.3f} {a['p_at_200_down']:.3f}")
        for k, v in out["paired"][tname].items():
            print(f"  {k:36s} dPCC {v['pearson']['mean_delta']:+.3f} ({v['pearson']['n_better']}/{v['pearson']['n_worse']})"
                  f" dJ {v['J']['mean_delta']:+.3f} ({v['J']['n_better']}/{v['J']['n_worse']}, p={v['J']['wilcoxon_two_sided_p']})")
    json.dump(out, open(f"{common.OUT}/extra_svdkf_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
