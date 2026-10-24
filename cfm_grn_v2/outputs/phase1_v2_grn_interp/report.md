# Forecast benchmark report

Metrics in the evaluation space standardised on observed cells; mean ± CI95 over seeds (groups averaged within seed). Lower is better.

## synthetic_grn_interp_t3

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| noise_floor | 0.000±0.000 | 0.000±0.000 | 1.556±0.000 | 0.162±0.000 | 0.026±0.000 | 5 |
| flow_matching | 0.641±0.002 | 0.023±0.000 | 2.048±0.005 | 1.336±0.003 | 0.136±0.000 | 5 |
| mean_interp | 1.102±0.000 | 0.042±0.000 | 2.413±0.000 | 1.806±0.000 | 0.203±0.000 | 5 |
| ot_interp | 1.171±0.000 | 0.042±0.000 | 2.290±0.000 | 1.806±0.000 | 0.202±0.000 | 5 |
| linear | 2.747±0.000 | 0.098±0.000 | 3.055±0.000 | 2.758±0.000 | 0.287±0.000 | 5 |
| mlp | 2.908±0.183 | 0.110±0.008 | 3.224±0.104 | 2.931±0.113 | 0.316±0.012 | 5 |
| flow_matching_shuffled_time | 4.117±0.084 | 0.191±0.005 | 4.252±0.052 | 3.938±0.055 | 0.515±0.009 | 5 |
| persistence | 4.155±0.000 | 0.193±0.000 | 4.268±0.000 | 3.957±0.000 | 0.518±0.000 | 5 |
| freefm_replay | 4.402±0.000 | 0.228±0.000 | 5.022±0.000 | 4.395±0.000 | 0.510±0.000 | 5 |

Verdict (benchmark): **PASS** — best control `mean_interp`, paired ΔED = -0.461 [-0.463, -0.459] (n=5), uses_time = True, noise floor = 0.000, null q95 = 0.004, matches target = False

