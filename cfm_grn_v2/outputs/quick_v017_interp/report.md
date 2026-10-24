# Forecast benchmark report

Metrics in the evaluation space standardised on observed cells; mean ± CI95 over seeds (groups averaged within seed). Lower is better.

## synthetic_grn_interp_t3

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| noise_floor | -0.002±0.000 | -0.000±0.000 | 2.002±0.000 | 0.334±0.000 | 0.089±0.000 | 2 |
| flow_matching | 0.073±0.018 | 0.005±0.001 | 1.969±0.005 | 0.520±0.059 | 0.108±0.004 | 2 |
| mean_interp | 0.224±0.000 | 0.017±0.000 | 2.145±0.000 | 0.861±0.000 | 0.163±0.000 | 2 |
| ot_interp | 0.241±0.000 | 0.018±0.000 | 1.975±0.000 | 0.861±0.000 | 0.156±0.000 | 2 |
| linear | 0.437±0.000 | 0.022±0.000 | 1.888±0.000 | 0.905±0.000 | 0.219±0.000 | 2 |
| mlp | 0.477±0.233 | 0.031±0.013 | 2.056±0.093 | 1.110±0.198 | 0.226±0.041 | 2 |
| freefm_replay | 0.639±0.000 | 0.047±0.000 | 3.256±0.000 | 1.268±0.000 | 0.372±0.000 | 2 |
| persistence | 0.875±0.000 | 0.072±0.000 | 2.624±0.000 | 1.740±0.000 | 0.306±0.000 | 2 |
| flow_matching_shuffled_time | 0.918±0.655 | 0.076±0.057 | 2.682±0.511 | 1.787±0.676 | 0.320±0.119 | 2 |

Verdict (benchmark): **PASS** — best control `mean_interp`, paired ΔED = -0.151 [-0.169, -0.133] (n=2), uses_time = True, noise floor = -0.002, null q95 = 0.014, matches target = False

