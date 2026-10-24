# Forecast benchmark report

Metrics in the evaluation space standardised on observed cells; mean ± CI95 over seeds (groups averaged within seed). Lower is better.

## synthetic_grn

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| noise_floor | -0.006±0.000 | -0.000±0.000 | 2.091±0.000 | 0.322±0.000 | 0.093±0.000 | 2 |
| freefm_replay | 0.293±0.000 | 0.024±0.000 | 2.491±0.000 | 0.991±0.000 | 0.202±0.000 | 2 |
| linear | 0.407±0.000 | 0.016±0.000 | 1.901±0.000 | 0.702±0.000 | 0.217±0.000 | 2 |
| mean_shift | 0.441±0.000 | 0.036±0.000 | 2.289±0.000 | 1.201±0.000 | 0.203±0.000 | 2 |
| flow_matching | 0.447±0.117 | 0.037±0.011 | 2.318±0.184 | 1.213±0.197 | 0.215±0.036 | 2 |
| mlp | 0.534±0.601 | 0.040±0.043 | 2.214±0.314 | 1.221±0.645 | 0.217±0.130 | 2 |
| flow_matching_shuffled_time | 0.654±0.631 | 0.056±0.056 | 2.458±0.480 | 1.487±0.737 | 0.248±0.098 | 2 |
| persistence | 0.682±0.000 | 0.058±0.000 | 2.469±0.000 | 1.516±0.000 | 0.250±0.000 | 2 |
| train_mean | 2.177±0.000 | 0.069±0.000 | 2.669±0.000 | 1.513±0.000 | 0.458±0.000 | 2 |

Verdict (benchmark): **FAIL** — best control `freefm_replay`, paired ΔED = 0.154 [0.037, 0.272] (n=2), uses_time = False, noise floor = -0.006, null q95 = 0.016, matches target = False

