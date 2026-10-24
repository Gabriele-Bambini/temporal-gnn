"""Dileo & Sottoriva Task 2 (future per-gene expression change): their definitions, my implementation.

Read from tgl-grn-1CCD/train_models_gene.py and dataset.py (the gene runs pass --gene_features):
  snapshots   = bins that have edges; node feature 0 = s00_mean (bin mean of the expression layer); the number of
                genes is max node id in the edge file + 1 (= rows of *_y_outdeg.csv), extra node-file rows unused;
  transitions = each of snapshots[:train_split] to the next snapshot;
  target      = clip(log(m_next + 1e-8) - log(m_now + 1e-8), -5, 5) per gene, float32;
  test genes  = a fresh random 25 % of genes per transition (round(0.25 N)), the other 75 % are the training genes
                whose targets their models are fitted on; metrics on the test genes in node order: MAE, Pearson (nan
                if either side is constant), P@200 up/down = overlap of the 200 largest / smallest values
                (np.argsort, default kind, as theirs); score = nanmean over transitions.
The paper's Table 2 numbers are reproduced by the '-human' builds (their Linear row equals the July HPC logs of
gastrulation-human / pancreas-human to 3 decimals), so those are the builds to beat; the mouse builds are kept.
A forecaster is f(hist_ts, hist, t_next, train_idx, y_train) -> predicted target for all genes, where hist holds
the bin means of the snapshots <= t_now only. Strict forecasters ignore train_idx / y_train; same-information
forecasters may use them, exactly as Dileo's models do.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

EPS = 1e-8
CLIP = 5.0
K = 200
DATASETS = {
    "gastrulation-human": dict(snapshots=list(range(2, 32)), train_split=29),
    "pancreas-human": dict(snapshots=list(range(0, 15)), train_split=14),
    "gastrulation": dict(snapshots=list(range(2, 35)), train_split=31),
    "pancreas": dict(snapshots=list(range(1, 13)), train_split=11),
}
METRICS = ("mae", "pearson", f"p_at_{K}_up", f"p_at_{K}_down")


def load_means(grn_dir, name):
    node = pd.read_csv(os.path.join(grn_dir, f"{name}_node.csv"))
    n = len(pd.read_csv(os.path.join(grn_dir, f"{name}_y_outdeg.csv"), usecols=["node_id"]))
    cols = sorted(c for c in node.columns if c.endswith("_s00_mean"))
    assert [int(c[1:4]) for c in cols] == list(range(len(cols))), "bins must be t000..t{B-1}"
    assert (node["node_id"].to_numpy() == np.arange(len(node))).all(), "node ids must be 0..N-1 in order"
    return node["gene"].astype(str).to_numpy()[:n], node[cols].to_numpy(np.float32)[:n]


def transitions(name):
    s = DATASETS[name]["snapshots"]
    return [(s[i], s[i + 1]) for i in range(DATASETS[name]["train_split"])]


def log_delta(m_now, m_next):
    d = np.log(np.maximum(m_next, 0).astype(np.float32) + np.float32(EPS)) \
        - np.log(np.maximum(m_now, 0).astype(np.float32) + np.float32(EPS))
    return np.clip(d, -CLIP, CLIP).astype(np.float32)


def metrics(y, p):
    y = np.asarray(y, np.float32).reshape(-1)
    p = np.asarray(p, np.float32).reshape(-1)
    out = {"mae": float(np.mean(np.abs(y - p)))}
    if np.all(y == y[0]) or np.all(p == p[0]):
        out["pearson"] = float("nan")
    else:
        yc, pc = y - y.mean(), p - p.mean()
        den = np.sqrt((yc ** 2).sum() * (pc ** 2).sum())
        out["pearson"] = float((yc * pc).sum() / den) if den > 0 else float("nan")
    k = min(K, len(y))
    out[f"p_at_{K}_up"] = len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k
    out[f"p_at_{K}_down"] = len(set(np.argsort(y)[:k]) & set(np.argsort(p)[:k])) / k
    return out


def score(means, name, forecast, n_draws=20, seed=0, test_frac=0.25):
    """Mean and sd over draws (one draw = one set of random test masks) of the transition-averaged metrics."""
    n = means.shape[0]
    n_test = int(round(test_frac * n))
    snaps = DATASETS[name]["snapshots"]
    trans = transitions(name)
    rng = np.random.default_rng(seed)
    per_draw, per_trans = [], np.zeros((len(trans), len(METRICS)))
    cache = {}
    for _ in range(n_draws):
        rows = []
        for i, (a, b) in enumerate(trans):
            perm = rng.permutation(n)
            test, train = np.sort(perm[n - n_test:]), np.sort(perm[:n - n_test])
            y = log_delta(means[:, a], means[:, b])
            hist_ts = [t for t in snaps if t <= a]
            key = (i, getattr(forecast, "strict", False))
            if key[1] and i in cache:
                p = cache[i]
            else:
                p = np.asarray(forecast(hist_ts, means[:, hist_ts], b, train, y[train]), np.float32)
                if key[1]:
                    cache[i] = p
            m = metrics(y[test], p[test])
            rows.append([m[k] for k in METRICS])
        rows = np.array(rows, float)
        per_trans += rows / n_draws
        per_draw.append(np.nanmean(rows, axis=0))
    per_draw = np.array(per_draw)
    res = {k: float(per_draw[:, j].mean()) for j, k in enumerate(METRICS)}
    res.update({f"{k}_sd": float(per_draw[:, j].std(ddof=1)) for j, k in enumerate(METRICS)})
    res["per_transition"] = {f"{a}->{b}": dict(zip(METRICS, map(float, per_trans[i]))) for i, (a, b) in enumerate(trans)}
    return res


def clip_mass(means, name):
    """Per transition: fraction of genes whose target sits at the +/-5 clip."""
    return {f"{a}->{b}": float((np.abs(log_delta(means[:, a], means[:, b])) >= CLIP - 1e-4).mean())
            for a, b in transitions(name)}


# ---- strict forecasters: only the snapshot bins <= t_now ----

def strict(f):
    f.strict = True
    return f


def trend_log(window=1):
    """Last observed log change between snapshots, averaged over `window` steps; zeros without history."""
    @strict
    def f(hist_ts, hist, t_next, train_idx, y_train):
        if hist.shape[1] < 2:
            return np.zeros(hist.shape[0], np.float32)
        w = min(window, hist.shape[1] - 1)
        ds = [log_delta(hist[:, -2 - j], hist[:, -1 - j]) / (hist_ts[-1 - j] - hist_ts[-2 - j]) for j in range(w)]
        return np.mean(ds, axis=0) * (t_next - hist_ts[-1])
    return f


@strict
def trend_lin(hist_ts, hist, t_next, train_idx, y_train):
    """Linear extrapolation of the bin mean, then the same log-delta as the target."""
    if hist.shape[1] < 2:
        return np.zeros(hist.shape[0], np.float32)
    slope = (hist[:, -1] - hist[:, -2]) / (hist_ts[-1] - hist_ts[-2])
    return log_delta(hist[:, -1], np.maximum(hist[:, -1] + slope * (t_next - hist_ts[-1]), 0))


def level(k):
    """Next mean = average of the last k snapshot means (noisy bins revert to their recent level)."""
    @strict
    def f(hist_ts, hist, t_next, train_idx, y_train):
        if hist.shape[1] < 2:
            return np.zeros(hist.shape[0], np.float32)
        return log_delta(hist[:, -1], hist[:, -k:].mean(1))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grn_dir", default=os.path.expanduser("~/tgl-grn/GRN"))
    ap.add_argument("--out", default="task2_node_baselines.json")
    ap.add_argument("--draws", type=int, default=20)
    args = ap.parse_args()
    out = {}
    for name in DATASETS:
        _, means = load_means(args.grn_dir, name)
        cm = clip_mass(means, name)
        out[name] = {"n_genes": int(means.shape[0]), "clip_mass": cm}
        print(name, "genes", means.shape[0], "clip mass mean", round(float(np.mean(list(cm.values()))), 3), flush=True)
        for label, fc in (("trend_log_1", trend_log(1)), ("trend_log_2", trend_log(2)), ("trend_lin", trend_lin),
                          ("level_2", level(2)), ("level_4", level(4)), ("level_6", level(6))):
            r = score(means, name, fc, n_draws=args.draws)
            out[name][label] = r
            print(name, label, {k: round(r[k], 4) for k in METRICS}, flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
