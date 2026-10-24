"""Pre-freeze integrity checks on sci-FATE2 count layers (no kinetics, no arm, no time-resolved look).

Checks: new + old == total (counting file); total(counting) == total(estimate); total >= unspliced + spliced
(splicing file) entrywise; per-gene ambiguous share a_g = 1 - sum(U+S) / sum(total); per-cell detection metadata
(total_TC_alpha, p_c_TC, p_e): NaNs, quantiles, by (timepoint, rep); effective K_i = ln(1-alpha)/ln(1-p_c).
Writes ~/tack/out/integrity_counts.json.
"""
import json

import anndata
import numpy as np
import scipy.sparse as sp

D = "/home/3393519/tack/data/scifate2/"
out = {}


def csr(a):
    return a.tocsr() if sp.issparse(a) else sp.csr_matrix(a)


C = anndata.read_h5ad(D + "GSE236512_processed_data_counting.h5ad")
new, old, tot = csr(C.layers["new"]), csr(C.layers["old"]), csr(C.layers["total"])
d1 = (new + old - tot)
out["new_plus_old_minus_total_max_abs"] = float(abs(d1).max()) if d1.nnz else 0.0
out["new_le_total"] = bool(((tot - new).min() >= 0))
obs = C.obs.copy()
del C, old, d1

E = anndata.read_h5ad(D + "GSE236512_processed_data_estimate.h5ad")
d2 = csr(E.layers["total"]) - tot
out["total_counting_minus_total_estimate_max_abs"] = float(abs(d2).max()) if d2.nnz else 0.0
del E, d2

S = anndata.read_h5ad(D + "GSE236512_processed_data_splicing.h5ad")
us = csr(S.layers["unspliced"]) + csr(S.layers["spliced"])
del S
d3 = tot - us
d3min = float(d3.min())
neg = d3.copy()
neg.data = np.minimum(neg.data, 0)
neg.eliminate_zeros()
out["total_minus_US_min"] = d3min
out["n_entries_total_lt_US"] = int(neg.nnz)
out["frac_entries_total_lt_US_of_nonzero_total"] = float(neg.nnz / max(tot.nnz, 1))
gene_tot = np.asarray(tot.sum(0)).ravel()
gene_us = np.asarray(us.sum(0)).ravel()
with np.errstate(divide="ignore", invalid="ignore"):
    a_g = 1 - gene_us / gene_tot
ok = gene_tot >= 1000
out["ambiguous_share_genes_with_total_ge_1000"] = dict(
    n=int(ok.sum()), q05=float(np.nanquantile(a_g[ok], 0.05)), q25=float(np.nanquantile(a_g[ok], 0.25)),
    median=float(np.nanmedian(a_g[ok])), q75=float(np.nanquantile(a_g[ok], 0.75)),
    q95=float(np.nanquantile(a_g[ok], 0.95)), frac_gt_0_3=float(np.mean(a_g[ok] > 0.3)))
out["ambiguous_share_overall"] = float(1 - gene_us.sum() / gene_tot.sum())
np.save("/home/3393519/tack/out/ambiguous_share_per_gene.npy", a_g)

alpha = obs["total_TC_alpha"].to_numpy(float)
pc = obs["p_c_TC"].to_numpy(float)
pe = obs["p_e"].to_numpy(float)
out["n_cells"] = int(len(obs))
out["n_nan_alpha"] = int(np.isnan(alpha).sum())
out["n_nan_pc"] = int(np.isnan(pc).sum())
out["n_alpha_gt_1"] = int(np.sum(alpha > 1))
qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
out["alpha_quantiles"] = {str(q): float(np.nanquantile(alpha, q)) for q in qs}
out["pc_quantiles"] = {str(q): float(np.nanquantile(pc, q)) for q in qs}
with np.errstate(divide="ignore", invalid="ignore"):
    K = np.log(1 - np.clip(alpha, None, 0.999)) / np.log(1 - pc)
out["K_eff_quantiles"] = {str(q): float(np.nanquantile(K, q)) for q in qs}
grp = obs["timepoint"].astype(str) + "_" + obs["rep"].astype(str)
by = {}
for g in sorted(grp.unique()):
    m = (grp == g).to_numpy()
    by[g] = dict(n=int(m.sum()), alpha_median=float(np.nanmedian(alpha[m])), pc_median=float(np.nanmedian(pc[m])),
                 K_median=float(np.nanmedian(K[m])))
out["by_group"] = by
json.dump(out, open("/home/3393519/tack/out/integrity_counts.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "by_group"}, indent=1))
for g, v in by.items():
    print(g, v)
