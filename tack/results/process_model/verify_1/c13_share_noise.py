"""C13 part A, noise check: share of y_i in the top-3 past change subspace when the basis excludes y_{i-1}
(basis from y_0 .. y_{i-2}: its noise is independent of y_i's, which involves bins i, i+1 only), raw and
noise-corrected: (||P3 y||^2 - sum_g P3_gg (R_i + R_{i+1})) / (||y||^2 - sum_g (R_i + R_{i+1})), R = plain delta-method
noise of L (the definition that reproduces the CONTEXT noise budget). Mean over i where the basis has >= 4 columns
(i >= 5). Also the same with the basis including y_{i-1} (raw only; that basis shares bin-i noise with y_i)."""
import json
import os

import numpy as np

import common as C


def main():
    res = {}
    for name in C.NAMES:
        d = C.Data(name)
        Y = d.Y("l1")
        R = d.R_L(False)
        rows = []
        for i in range(5, d.T):
            y = Y[i]
            U, s, _ = np.linalg.svd(Y[:i - 1].T, full_matrices=False)
            U3 = U[:, :3]
            noise = R[:, i] + R[:, i + 1]
            num = ((U3.T @ y) ** 2).sum()
            pgg = (U3 ** 2).sum(1)
            raw = num / (y ** 2).sum()
            corr = (num - (pgg * noise).sum()) / ((y ** 2).sum() - noise.sum())
            Ui, _, _ = np.linalg.svd(Y[:i].T, full_matrices=False)
            raw_incl = ((Ui[:, :3].T @ y) ** 2).sum() / (y ** 2).sum()
            rows.append((i, float(raw), float(corr), float(raw_incl), float(noise.sum() / (y ** 2).sum())))
        a = np.array(rows)
        out = {"per_i": [dict(zip(("i", "share3_excl_lag1", "share3_excl_lag1_noisecorr", "share3_incl_lag1",
                                    "noise_frac_of_ss"), r)) for r in rows],
               "mean_share3_excl_lag1": round(float(a[:, 1].mean()), 4),
               "mean_share3_excl_lag1_noisecorr": round(float(a[:, 2].mean()), 4),
               "median_share3_excl_lag1_noisecorr": round(float(np.median(a[:, 2])), 4),
               "mean_share3_incl_lag1_same_i": round(float(a[:, 3].mean()), 4)}
        res[name] = out
        print(f"== {name} (i >= 5): share3 basis without y_(i-1): raw {out['mean_share3_excl_lag1']} "
              f"noise-corrected {out['mean_share3_excl_lag1_noisecorr']} (median {out['median_share3_excl_lag1_noisecorr']}); "
              f"basis incl y_(i-1) raw {out['mean_share3_incl_lag1_same_i']}")
        print("   noise-corrected per i:", np.round(a[:, 2], 3).tolist())
    json.dump(res, open(os.path.join(C.OUT, "c13_share_noise.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
