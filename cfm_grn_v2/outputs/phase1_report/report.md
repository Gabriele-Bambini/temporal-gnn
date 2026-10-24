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

## toy_bimodal

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| freefm_replay | -0.000±0.002 | -0.000±0.000 | 0.441±0.267 | 0.080±0.020 | 0.064±0.014 | 5 |
| noise_floor | 0.002±0.000 | 0.000±0.000 | 0.392±0.000 | 0.097±0.000 | 0.074±0.000 | 5 |
| flow_matching | 0.002±0.007 | 0.001±0.002 | 0.481±0.229 | 0.072±0.083 | 0.066±0.048 | 5 |
| mlp | 0.007±0.003 | 0.002±0.001 | 0.414±0.158 | 0.092±0.034 | 0.092±0.014 | 5 |
| linear | 0.312±0.013 | 0.096±0.006 | 0.983±0.027 | 0.076±0.074 | 0.582±0.006 | 5 |
| mean_shift | 1.029±0.007 | 0.374±0.003 | 1.549±0.010 | 0.040±0.031 | 0.975±0.003 | 5 |
| flow_matching_shuffled_time | 1.034±0.012 | 0.376±0.004 | 1.552±0.015 | 0.120±0.072 | 0.976±0.007 | 5 |
| persistence | 1.037±0.009 | 0.377±0.004 | 1.553±0.011 | 0.111±0.042 | 0.978±0.003 | 5 |
| train_mean | 1.693±0.006 | 0.466±0.003 | 1.895±0.010 | 0.036±0.024 | 1.320±0.003 | 5 |

Verdict (sanity): **PASS** — best control `freefm_replay`, paired ΔED = 0.002 [-0.005, 0.009] (n=5), uses_time = True, noise floor = 0.002, null q95 = 0.008, matches target = True

## toy_condition_shift

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| mean_shift | -0.000±0.001 | -0.000±0.000 | 0.202±0.023 | 0.045±0.008 | 0.045±0.006 | 5 |
| noise_floor | -0.000±0.000 | -0.000±0.000 | 0.202±0.000 | 0.056±0.000 | 0.052±0.000 | 5 |
| linear | 0.000±0.002 | 0.000±0.000 | 0.207±0.023 | 0.039±0.019 | 0.046±0.011 | 5 |
| freefm_replay | 0.000±0.001 | 0.000±0.000 | 0.230±0.019 | 0.040±0.016 | 0.053±0.010 | 5 |
| mlp | 0.000±0.002 | 0.000±0.001 | 0.213±0.025 | 0.039±0.019 | 0.051±0.015 | 5 |
| flow_matching | 0.001±0.001 | 0.000±0.000 | 0.208±0.021 | 0.050±0.018 | 0.049±0.012 | 5 |
| train_mean | 0.604±0.007 | 0.067±0.002 | 1.181±0.015 | 0.026±0.013 | 0.643±0.005 | 5 |
| persistence | 0.764±0.011 | 0.233±0.003 | 1.172±0.019 | 1.147±0.009 | 0.605±0.006 | 5 |
| flow_matching_shuffled_time | 0.779±0.044 | 0.237±0.013 | 1.187±0.047 | 1.162±0.040 | 0.616±0.024 | 5 |

Verdict (sanity): **PASS** — best control `mean_shift`, paired ΔED = 0.001 [-0.000, 0.002] (n=5), uses_time = True, noise floor = -0.000, null q95 = 0.003, matches target = True

## toy_translation

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| noise_floor | 0.001±0.000 | 0.000±0.000 | 0.166±0.000 | 0.064±0.000 | 0.048±0.000 | 5 |
| mean_shift | 0.001±0.002 | 0.000±0.001 | 0.178±0.015 | 0.059±0.023 | 0.044±0.011 | 5 |
| linear | 0.002±0.002 | 0.001±0.001 | 0.181±0.016 | 0.063±0.024 | 0.047±0.013 | 5 |
| mlp | 0.002±0.002 | 0.001±0.000 | 0.184±0.016 | 0.066±0.015 | 0.049±0.008 | 5 |
| freefm_replay | 0.002±0.002 | 0.001±0.001 | 0.197±0.028 | 0.065±0.024 | 0.051±0.008 | 5 |
| flow_matching | 0.003±0.004 | 0.001±0.001 | 0.199±0.051 | 0.057±0.023 | 0.057±0.024 | 5 |
| train_mean | 0.457±0.004 | 0.029±0.001 | 0.871±0.014 | 0.033±0.009 | 0.493±0.006 | 5 |
| flow_matching_shuffled_time | 2.535±0.079 | 0.770±0.022 | 2.203±0.081 | 2.182±0.043 | 1.540±0.032 | 5 |
| persistence | 2.560±0.041 | 0.777±0.011 | 2.217±0.055 | 2.195±0.023 | 1.549±0.017 | 5 |

Verdict (sanity): **FAIL** — best control `mean_shift`, paired ΔED = 0.002 [-0.003, 0.007] (n=5), uses_time = True, noise floor = 0.001, null q95 = 0.003, matches target = False

## toy_variance

| model | energy_distance | mmd | w2 | mean_error | per_gene_w1 | n seeds |
|---|---|---|---|---|---|---|
| linear | 0.001±0.003 | 0.000±0.001 | 0.251±0.034 | 0.076±0.035 | 0.062±0.019 | 5 |
| mlp | 0.001±0.002 | 0.000±0.001 | 0.255±0.031 | 0.070±0.034 | 0.063±0.014 | 5 |
| noise_floor | 0.001±0.000 | 0.000±0.000 | 0.233±0.000 | 0.092±0.000 | 0.070±0.000 | 5 |
| freefm_replay | 0.001±0.002 | 0.000±0.001 | 0.273±0.045 | 0.076±0.030 | 0.069±0.011 | 5 |
| flow_matching | 0.001±0.002 | 0.000±0.001 | 0.270±0.027 | 0.086±0.027 | 0.073±0.018 | 5 |
| flow_matching_shuffled_time | 0.089±0.008 | 0.039±0.004 | 0.932±0.084 | 0.062±0.037 | 0.503±0.019 | 5 |
| persistence | 0.091±0.002 | 0.040±0.001 | 0.939±0.049 | 0.054±0.023 | 0.511±0.008 | 5 |
| mean_shift | 0.093±0.002 | 0.041±0.001 | 0.943±0.049 | 0.083±0.027 | 0.511±0.008 | 5 |
| train_mean | 0.702±0.011 | 0.110±0.005 | 1.386±0.043 | 0.042±0.017 | 0.742±0.011 | 5 |

Verdict (sanity): **PASS** — best control `linear`, paired ΔED = 0.001 [-0.001, 0.003] (n=5), uses_time = True, noise floor = 0.001, null q95 = 0.004, matches target = True


## Runs merged

- outputs/phase1_v1_toys: toys (v0.1.4 protocol, refit); code version 0.1.4; finished 2026-09-18 15:56:14; best configs: {"toy_translation": {"hidden": 256, "depth": 3, "lr": 0.0003, "weight_decay": 0.0, "batch_size": 256, "n_iters": 8000, "sigma_min": 0.01, "coupling": "independent", "n_ode_steps": 20, "ode_method": "rk4", "time_embed_dim": 32, "cond_dim": 8, "time_conditioning": "delta", "eval_every": 250, "patience": 8, "seed": 0, "shuffle_biological_time": false, "ema_decay": 0.999, "grad_clip": 1.0, "pair_mode": "consecutive", "clock": "flow"}, "toy_variance": {"hidden": 128, "depth": 3, "lr": 0.001, "weight_decay": 0.0, "batch_size": 256, "n_iters": 8000, "sigma_min": 0.01, "coupling": "independent", "n_ode_steps": 20, "ode_method": "rk4", "time_embed_dim": 32, "cond_dim": 8, "time_conditioning": "delta", "eval_every": 250, "patience": 8, "seed": 0, "shuffle_biological_time": false, "ema_decay": 0.999, "grad_clip": 1.0, "pair_mode": "consecutive", "clock": "flow"}, "toy_bimodal": {"hidden": 256, "depth": 3, "lr": 0.0003, "weight_decay": 0.0, "batch_size": 256, "n_iters": 8000, "sigma_min": 0.01, "coupling": "independent", "n_ode_steps": 20, "ode_method": "rk4", "time_embed_dim": 32, "cond_dim": 8, "time_conditioning": "delta", "eval_every": 250, "patience": 8, "seed": 0, "shuffle_biological_time": false, "ema_decay": 0.999, "grad_clip": 1.0, "pair_mode": "consecutive", "clock": "flow"}, "toy_condition_shift": {"hidden": 256, "depth": 3, "lr": 0.0003, "weight_decay": 0.0, "batch_size": 256, "n_iters": 8000, "sigma_min": 0.01, "coupling": "independent", "n_ode_steps": 20, "ode_method": "rk4", "time_embed_dim": 32, "cond_dim": 8, "time_conditioning": "delta", "eval_every": 250, "patience": 8, "seed": 0, "shuffle_biological_time": false, "ema_decay": 0.999, "grad_clip": 1.0, "pair_mode": "consecutive", "clock": "flow"}}
- outputs/phase1_v1_grn: GRN forecast, diagnostic (v0.1.4 protocol, refit); code version 0.1.4; finished 2026-09-18 15:21:21; best configs: {"synthetic_grn": {"hidden": 256, "depth": 3, "lr": 0.0003, "weight_decay": 0.0, "batch_size": 256, "n_iters": 8000, "sigma_min": 0.001, "coupling": "independent", "n_ode_steps": 20, "ode_method": "rk4", "time_embed_dim": 32, "cond_dim": 8, "time_conditioning": "delta", "eval_every": 250, "patience": 8, "seed": 0, "shuffle_biological_time": false, "ema_decay": 0.999, "grad_clip": 1.0, "pair_mode": "all", "clock": "flow"}}
- outputs/phase1_v2_grn_interp: GRN interpolation t=3, gate G1 (v0.1.7 protocol, no refit); code version 0.1.7; finished 2026-09-18 16:03:34; best configs: {"synthetic_grn_interp_t3": {"hidden": 128, "depth": 3, "lr": 0.001, "weight_decay": 0.0, "batch_size": 256, "n_iters": 8000, "sigma_min": 0.01, "coupling": "ot", "n_ode_steps": 20, "ode_method": "rk4", "time_embed_dim": 32, "cond_dim": 8, "time_conditioning": "delta", "eval_every": 250, "patience": 8, "seed": 0, "shuffle_biological_time": false, "ema_decay": 0.999, "grad_clip": 1.0, "pair_mode": "consecutive", "clock": "duration", "path": "spline"}}
