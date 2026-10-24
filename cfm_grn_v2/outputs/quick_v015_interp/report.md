# Forecast benchmark report

Metrics in the evaluation space standardised on observed cells; mean ± CI95 over seeds (groups averaged within seed). Lower is better.

## synthetic_grn_interp_t3

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| noise_floor | -0.002±0.000 | -0.000±0.000 | 2.002±0.000 | 0.334±0.000 | 0.089±0.000 | 2 |
| mean_interp | 0.224±0.000 | 0.017±0.000 | 2.145±0.000 | 0.861±0.000 | 0.163±0.000 | 2 |
| ot_interp | 0.241±0.000 | 0.018±0.000 | 1.975±0.000 | 0.861±0.000 | 0.156±0.000 | 2 |
| flow_matching | 0.294±0.585 | 0.022±0.045 | 2.069±0.447 | 0.949±0.934 | 0.171±0.081 | 2 |
| linear | 0.437±0.000 | 0.022±0.000 | 1.888±0.000 | 0.905±0.000 | 0.219±0.000 | 2 |
| mlp | 0.477±0.233 | 0.031±0.013 | 2.056±0.093 | 1.110±0.198 | 0.226±0.041 | 2 |
| freefm_replay | 0.639±0.000 | 0.047±0.000 | 3.256±0.000 | 1.268±0.000 | 0.372±0.000 | 2 |
| flow_matching_shuffled_time | 0.831±0.672 | 0.069±0.056 | 2.629±0.266 | 1.701±0.678 | 0.301±0.142 | 2 |
| persistence | 0.875±0.000 | 0.072±0.000 | 2.624±0.000 | 1.740±0.000 | 0.306±0.000 | 2 |

Verdict (benchmark): **FAIL** — best control `mean_interp`, paired ΔED = 0.070 [-0.515, 0.655] (n=2), uses_time = False, noise floor = -0.002, null q95 = 0.013, matches target = False

