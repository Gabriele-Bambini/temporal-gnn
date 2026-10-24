"""Judge 1: quick independent check of the l1 noise budget (delta method, population sd -> Var(mean) = s^2/(n-1)).
share_t = sum_g R_gt / sum_g (Delta_g - mean)^2, share_t1 likewise with R_g,t+1; R_gb = (1e4/L_b)^2 s_gb^2/(n_b-1) / (1+CP_gb)^2.
Also the lag-1 correlation of consecutive observed l1 changes (cross-gene, mean over transitions)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "indep"))
import indep as I
out = {}
for name in ("gastrulation-human", "pancreas-human"):
    d = I.Data(name)
    kap = 1e4 / d.lib
    R = (kap[None, :] * d.s) ** 2 / np.maximum(d.n - 1, 1)[None, :] / (1 + d.CP) ** 2
    L = np.log1p(d.CP)
    Dl = np.diff(L, axis=1)
    st, st1, lag = [], [], []
    for i in range(1, d.T):
        v = ((Dl[:, i] - Dl[:, i].mean()) ** 2).sum()
        st.append(R[:, i].sum() / v)
        st1.append(R[:, i + 1].sum() / v)
        lag.append(np.corrcoef(Dl[:, i - 1], Dl[:, i])[0, 1])
    st, st1 = np.array(st), np.array(st1)
    out[name] = {"mean_share_noise_bin_t": round(float(st.mean()), 3), "mean_share_noise_bin_t+1": round(float(st1.mean()), 3),
                 "median_share_t": round(float(np.median(st)), 3), "median_share_t+1": round(float(np.median(st1)), 3),
                 "implied_signal_share_mean": round(float(1 - st.mean() - st1.mean()), 3),
                 "lag1_corr_observed_l1_changes": round(float(np.mean(lag)), 3),
                 "pure_noise_expected_lag1": round(float(np.mean(-st[1:] / np.sqrt(1.0))) if False else float(np.mean([-st[k] for k in range(len(st))])), 3)}
    print(name, out[name])
json.dump(out, open(os.path.join(HERE, "noise_share.json"), "w"), indent=1)
