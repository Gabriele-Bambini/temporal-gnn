"""EXPLORATORY sanity check after the pre-registered LARRY in vivo test (job 728790) found every coupling at chance
(clonal enrichment ~1, vs 2-10 in vitro). Rules out a row-misalignment bug before reading it as biology:
  (a) marker genes per annotated cell type (normed counts, log1p mean): rows of the matrix must match the metadata;
  (b) clonal coherence WITHIN one time point in the test's own embedding: kNN (K = 10, self excluded) inside the day-2
      cohort sample (Z_a) and inside each mouse (Z_b); enrichment >> 1 means expression and clone labels are aligned and
      the embedding carries clonal information, so the day-2 -> mouse failure is not a bug.
Output ~/tack/out/larry_vivo/larry_vivo_sanity.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import larry_robust as LR  # noqa: E402
import larry_vivo_prep as VP  # noqa: E402

MARKERS = {"Neu": ["Elane", "Mpo", "Ltf", "Camp", "S100a8", "Lcn2"], "Ery": ["Hba-a1", "Hbb-bs", "Gypa", "Klf1", "Car1"],
           "B": ["Cd79a", "Cd19", "Ebf1", "Vpreb3"], "Mono": ["Csf1r", "Ly6c2", "F13a1", "Ms4a6c"],
           "Undifferentiated": ["Procr", "Hlf", "Mecom", "Cd34"]}
K = 10


def within(Z, c):
    _, ind = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z)
    ind = ind[:, 1:]
    P = np.zeros((len(Z), len(Z)))
    np.put_along_axis(P, ind, 1.0 / K, axis=1)
    np.fill_diagonal(P, 0.0)
    return LR.enrichment(P, c, c)[0]


def main():
    res = {"markers": {}, "within": {}}
    X, genes, meta, _ = VP.load()
    ct = meta["Cell type annotation"].astype(str).to_numpy()
    gi = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(0)
    for t in MARKERS:
        idx = np.where(ct == t)[0]
        idx = rng.choice(idx, min(3000, len(idx)), replace=False)
        sub = X[idx]
        res["markers"][t] = {}
        for s, gl in MARKERS.items():
            cols = [gi[g] for g in gl if g in gi]
            res["markers"][t][s] = float(np.log1p(sub[:, cols].toarray()).mean()) if cols else float("nan")
        print("cells", t, {s: round(v, 3) for s, v in res["markers"][t].items()}, flush=True)
    meta_u = json.load(open(os.path.join(VP.OUT, "meta.json")))
    for name in sorted(meta_u["units"]):
        c = np.load(os.path.join(VP.OUT, f"{name}.npz"), allow_pickle=True)
        ea = within(c["Z_a"].astype(np.float64), c["clone_a"])
        eb = within(c["Z_b"].astype(np.float64), c["clone_b"])
        res["within"][name] = {"day2_cohort": ea, "mouse": eb}
        print(name, "within day-2 cohort", round(ea, 2), "within mouse", round(eb, 2), flush=True)
    with open(os.path.join(VP.OUT, "larry_vivo_sanity.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
