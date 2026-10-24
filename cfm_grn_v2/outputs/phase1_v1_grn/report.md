# Forecast benchmark report

Metrics in the evaluation space standardised on observed cells; mean ± CI95 over seeds (groups averaged within seed). Lower is better.

## synthetic_grn

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| noise_floor | 0.000±0.000 | 0.000±0.000 | 1.615±0.000 | 0.169±0.000 | 0.027±0.000 | 5 |
| linear | 0.559±0.000 | 0.015±0.000 | 1.752±0.000 | 1.029±0.000 | 0.155±0.000 | 5 |
| freefm_replay | 1.121±0.000 | 0.045±0.000 | 2.577±0.000 | 1.843±0.000 | 0.231±0.000 | 5 |
| mlp | 1.148±0.332 | 0.040±0.013 | 2.211±0.221 | 1.709±0.282 | 0.210±0.030 | 5 |
| mean_shift | 1.343±0.000 | 0.053±0.000 | 2.506±0.000 | 1.985±0.000 | 0.257±0.000 | 5 |
| flow_matching_shuffled_time | 2.872±0.104 | 0.128±0.006 | 3.549±0.117 | 3.131±0.075 | 0.324±0.011 | 5 |
| persistence | 2.874±0.000 | 0.127±0.000 | 3.459±0.000 | 3.102±0.000 | 0.315±0.000 | 5 |
| train_mean | 4.605±0.000 | 0.131±0.000 | 3.635±0.000 | 3.102±0.000 | 0.396±0.000 | 5 |
| flow_matching | 5.668±8.797 | 0.345±0.555 | 4.826±4.855 | 4.294±5.305 | 0.537±0.677 | 5 |

Verdict (benchmark): **FAIL** — best control `linear`, paired ΔED = 5.109 [-3.689, 13.906] (n=5), uses_time = False, noise floor = 0.000, null q95 = 0.022, matches target = False

