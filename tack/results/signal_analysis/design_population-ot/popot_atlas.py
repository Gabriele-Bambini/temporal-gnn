"""'Known cell-state dictionary' probes for the population-OT prototype (NOT strict: upper-bound probes).

Question: is population OT weak only because a strict NMF cannot know the cell states that appear later (pancreas
keeps opening new directions), or because composition change does not carry the per-gene change at all? And when
the composition channel IS informative, does the OT coupling add anything over plain extrapolation of proportions?
The dictionary (CP10k NMF -> cell fractions -> archetype count laws A, exactly as popot.PopOT.mix) is fitted on a
bin set that depends on the mode:
  full      all bins (leaks everything, including the sampling noise of the target bin i+1);
  loo       all bins except the target bin i+1 (future states known, target bin unseen; later bins still inform
            the direction each gene will move, so this is interpolation-like);
  terminal  bins 0..i plus ONE terminal bin (the last bin with >= 100 cells: gastrulation bin 29, pancreas bin 14),
            i.e. the strict past plus an 'external atlas' of the end state; when i+1 >= the terminal index the
            terminal bin is not added (target or already past) and the probe is strict.
Given the dictionary, the composition of bin b depends on bin b only, so the composition forecasters read the past
compositions pi_0..pi_i only: persist, lin1, loglin1 (no OT), ot0.05, uot0.05t1, sb3 (OT / SB, as popot.py), and
bridge: pi_i + (pi_term - pi_i) / (t_term - i), the time marginal of ANY jump-process bridge (coupling-independent:
on discrete states the marginal at fraction tau of a bridge between a and b is (1 - tau) a + tau b whatever the plan)
towards the dictionary composition of the terminal bin (lin1 when the terminal bin is not ahead or not in the
dictionary); oracle: the dictionary composition of bin i+1 (its KL projection when bin i+1 is not in the dictionary).
The rest is popot.py unchanged (mixture shift of the gene's own pooled law, observation model for orig, plug-in for
l1 / linear, rolling-origin choice of state x K within each forecaster, paired Wilcoxon over transitions).
Usage: python popot_atlas.py <dataset> [full|loo|terminal]  ->  popot_atlas[_loo|_terminal]_<dataset>.json
"""
import json
import os
import sys
import time

import numpy as np

import popot
from popot import common, dist_bridge, nmf_kl

FCS = ("persist", "lin1", "loglin1", "ot0.05", "uot0.05t1", "sb3", "bridge", "oracle")
STATES_A = (("box1", 0.0), ("box4", 0.0), ("box1", 1.0))
PAIRS = (("ot0.05", "lin1"), ("ot0.05", "persist"), ("uot0.05t1", "loglin1"), ("uot0.05t1", "lin1"),
         ("sb3", "lin1"), ("bridge", "lin1"), ("lin1", "persist"), ("oracle", "lin1"))


def fit_dictionary(d, T3, idx, K):
    """popot.PopOT.mix on an arbitrary bin set idx (sorted): returns pi (len(idx) x K), cost C, laws A, H, c."""
    idx = np.asarray(idx)
    Ke = min(K, len(idx))
    om = d.nc[idx] * d.lib[idx] / 1e4
    W, H = nmf_kl.fit_subset(d.CP[:, idx].T, Ke, om)
    P = nmf_kl.to_props(W)
    LH = np.log1p(H)
    C = ((LH[:, None, :] - LH[None, :, :]) ** 2).mean(2)
    if Ke > 1:
        C = C / C[~np.eye(Ke, dtype=bool)].mean()
    n, lib = d.nc[idx], d.lib[idx]
    X = P * lib[:, None]
    c0 = n.sum() / (n * lib).sum()
    G = X.T @ (n[:, None] * X)
    lam = 0.1 * np.trace(G) / Ke
    c = np.clip(np.linalg.solve(G + lam * np.eye(Ke), X.T @ n + lam * c0), 0.25 * c0, 4 * c0)
    pi = X * c[None, :]
    pi = pi / pi.sum(1, keepdims=True)
    m0 = (1.0 / c)[:, None] * H / 1e4
    Am = popot.fixed_w_kl(pi, d.M[:, idx].T, n, m0)
    Af = np.minimum(popot.fixed_w_kl(pi, d.F[:, idx].T, n, 1 - np.exp(-Am)), 1.0)
    Ae = popot.fixed_w_kl(pi, T3[2][:, idx].T, n, Am + 2 * Am ** 2)
    return {"idx": idx, "pi_rows": pi, "C": C, "A": np.stack([Am, Af, Ae]), "H": H, "c": c}


class AtlasPopOT(popot.PopOT):
    def __init__(self, d, mode):
        super().__init__(d)
        self.mode = mode
        ok = np.where(d.nc >= 100)[0]
        self.t_term = int(ok.max())

    def _comp_of(self, D, b):
        """Composition of bin b in dictionary D: its row if b was fitted, else its KL projection."""
        hit = np.where(D["idx"] == b)[0]
        if len(hit):
            return D["pi_rows"][hit[0]]
        Pn = nmf_kl.to_props(nmf_kl.project(self.d.CP[:, b][None, :], D["H"]))[0]
        x = Pn * self.d.lib[b] * D["c"]
        return x / x.sum()

    def mix(self, i, K):
        key = (self.mode, i, K)
        if key in self._mix:
            return self._mix[key]
        d, S = self.d, self.d.S
        if self.mode == "full":
            fk = ("full", K)
            if fk not in self._mix:
                self._mix[fk] = fit_dictionary(d, self.T3, np.arange(S), K)
            D = self._mix[fk]
        elif self.mode == "loo":
            D = fit_dictionary(d, self.T3, np.array([b for b in range(S) if b != i + 1]), K)
        else:
            extra = [self.t_term] if self.t_term > i + 1 else []
            D = fit_dictionary(d, self.T3, np.array(list(range(i + 1)) + extra), K)
        past = np.array([np.where(D["idx"] == b)[0][0] for b in range(i + 1)])
        r = {"pi": D["pi_rows"][past], "C": D["C"], "A": D["A"], "H": D["H"], "c": D["c"],
             "pi_next": self._comp_of(D, i + 1),
             "pi_term": D["pi_rows"][np.where(D["idx"] == self.t_term)[0][0]]
             if (self.t_term in D["idx"] and self.t_term > i + 1) else None}
        self._mix[key] = r
        return r

    def forecast(self, i, K, fc, oracle=False):
        """No cache: the parent's cache key uses min(K, i + 1), which would mix dictionaries of different K."""
        mx = self.mix(i, K)
        if fc == "oracle" or oracle:
            return mx["pi_next"]
        if fc == "bridge":
            if mx["pi_term"] is not None:
                return mx["pi"][-1] + (mx["pi_term"] - mx["pi"][-1]) / (self.t_term - i)
            fc = "lin1"
        if fc in ("sb3", "sbW"):
            return popot.sb_forecast(mx["pi"], mx["C"], 3 if fc == "sb3" else common.WINDOW, self.d.nc[:i + 1])
        return popot.forecasts.NMFMix.forecast_props(mx["pi"], mx["C"], fc)


def main(name, mode="full"):
    t0 = time.time()
    d = common.Data(name)
    mk = common.masks(d)
    q = 200.0 / mk.shape[2]
    Y = {t: common.all_targets(d, t) for t in popot.TARGETS}
    m = AtlasPopOT(d, mode)
    cfgs = [(s, K, fc, b) for s, b in STATES_A for K in popot.KS for fc in FCS]
    S = {t: np.full((len(cfgs), d.T, 3), np.nan) for t in popot.TARGETS}
    J = {t: np.full((len(cfgs), d.T), np.nan) for t in popot.TARGETS}
    for i in range(1, d.T):
        for r, cfg in enumerate(cfgs):
            fc = cfg[2]
            if fc == "persist" and cfg[0].startswith("box") and cfg[3] == 0 and cfg[1] != popot.KS[0]:
                r0 = cfgs.index((cfg[0], popot.KS[0], fc, cfg[3]))   # independent of K: copy the K = KS[0] row
                for t in popot.TARGETS:
                    S[t][r, i], J[t][r, i] = S[t][r0, i], J[t][r0, i]
                continue
            # 'oracle' goes through the law with a non-persist label so the mixture shift is applied
            Th = m.law(i, (cfg[0], cfg[1], "lin1" if fc == "oracle" else fc, cfg[3]), oracle=(fc == "oracle"))
            for t in popot.TARGETS:
                if t == "orig" and cfg[0] != "box4":
                    continue
                p = m.target_pred(i, Th, t, f"atlas_{mode}|" + popot.label(cfg))
                J[t][r, i] = dist_bridge.criterion(Y[t][i], p, q) * 3
                S[t][r, i] = common.test_scores(Y[t][i:i + 1], p[None, :], mk[:, i:i + 1])[0]
        print(name, mode, "transition", i, round(time.time() - t0, 1), "s", flush=True)
    trend = {t: np.zeros((d.T, d.N)) for t in popot.TARGETS}
    for t in popot.TARGETS:
        trend[t][1:] = Y[t][:-1]
    out = {"dataset": name, "mode": mode, "terminal_snapshot_index": m.t_term, "terminal_bin": d.bins[m.t_term],
           "protocol": __doc__, "targets": {}}
    for t in popot.TARGETS:
        St = common.test_scores(Y[t], trend[t], mk)
        res = {"trend_repeat_last_change": popot.summ(St), "paired": {}, "fixed": {}}
        lvl = cfgs.index(("box4", popot.KS[0], "persist", 0.0))
        gsc = {}
        for fc in FCS:
            rows = [r for r, c in enumerate(cfgs) if c[2] == fc and np.isfinite(J[t][r, 1:]).all()]
            sc = np.zeros((d.T, 3))
            sel = []
            for i in range(d.T):
                r = popot.select(J[t], rows, cfgs, i, None)
                if r is None:
                    sc[i] = S[t][lvl, i] if (t == "orig" and i >= 1) else St[i]
                    sel.append("fallback")
                else:
                    sc[i] = S[t][r, i]
                    sel.append(popot.label(cfgs[r]))
            gsc[fc] = sc
            res[f"fc={fc}"] = {**popot.summ(sc), "selected": sel}
            for r in rows:
                res["fixed"][popot.label(cfgs[r])] = popot.summ(S[t][r])
        for a, b in PAIRS:
            res["paired"][f"{a} - {b}"] = common.paired(gsc[a], gsc[b])
        for a in ("lin1", "ot0.05", "uot0.05t1", "bridge", "oracle"):
            res["paired"][f"{a} - trend"] = common.paired(gsc[a], St)
        out["targets"][t] = res
    out["seconds"] = round(time.time() - t0, 1)
    tag = {"full": "atlas", "loo": "atlas_loo", "terminal": "atlas_terminal"}[mode]
    json.dump(out, open(os.path.join(popot.HERE, f"popot_{tag}_{name}.json"), "w"), indent=1)
    for t, v in out["targets"].items():
        print(name, mode, t, {k: x["i>=1"] for k, x in v.items() if k.startswith(("fc=", "trend"))}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "full")
