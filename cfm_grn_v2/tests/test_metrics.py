import numpy as np

from cfm_grn_v2.evaluation.metrics import (EvalProtocol, energy_distance, mean_error, median_bandwidth, mmd_rbf,
                                           noise_floor, per_gene_w1, w2_assignment)


def test_energy_distance_orders_shifts():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 3))
    b = rng.normal(size=(500, 3))
    near = energy_distance(a, b)
    far = energy_distance(a, b + 2.0)
    assert abs(near) < 0.05 and far > near + 1.0


def test_energy_distance_is_unbiased_at_small_n():
    rng = np.random.default_rng(3)
    vals = [energy_distance(rng.normal(size=(40, 3)), rng.normal(size=(40, 3))) for _ in range(200)]
    assert abs(np.mean(vals)) < 0.02


def test_w2_of_pure_shift_equals_shift_norm():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 2))
    shift = np.array([3.0, 4.0])
    assert abs(w2_assignment(a, a + shift, np.random.default_rng(1), n_max=300) - 5.0) < 1e-6


def test_mmd_and_mean_error():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 2))
    b = rng.normal(size=(300, 2))
    bw = median_bandwidth(a)
    assert mmd_rbf(a, b, bw) < mmd_rbf(a, b + 3.0, bw)
    assert abs(mean_error(a, a + 1.0) - np.sqrt(2.0)) < 1e-6


def test_permuted_genes_worsen_metrics():
    rng = np.random.default_rng(0)
    scale = np.array([1.0, 3.0, 0.2])
    a = rng.normal(size=(400, 3)) * scale
    b = rng.normal(size=(400, 3)) * scale
    b_perm = b[:, [2, 0, 1]]
    assert per_gene_w1(a, b_perm, np.random.default_rng(1)) > per_gene_w1(a, b, np.random.default_rng(1))
    assert energy_distance(a, b_perm) > energy_distance(a, b)


def test_protocol_and_noise_floor():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(2000, 4))
    p = EvalProtocol(n_samples=500, seed=0, bandwidth=1.0)
    m = p.evaluate(t + 0.5, t)
    assert set(m) == {"energy_distance", "mmd", "w2", "mean_error", "per_gene_w1", "null_q95", "var_ratio",
                      "ed_approx", "ed_mean_part", "ed_trace_part"}
    assert m["energy_distance"] > m["null_q95"]
    same = p.evaluate(t[:1000], t[1000:])
    assert same["energy_distance"] <= same["null_q95"]
    floor = noise_floor(t, p, n_rep=3)
    assert floor["energy_distance"] < m["energy_distance"]


def test_var_ratio_and_mean_trace_split():
    from cfm_grn_v2.evaluation.metrics import ed_mean_trace_split, var_ratio
    rng = np.random.default_rng(0)
    a = rng.normal(size=(800, 200))
    b = rng.normal(size=(800, 200))
    assert abs(var_ratio(a, b) - 1.0) < 0.05 and abs(var_ratio(2.0 * a, b) - 4.0) < 0.2
    shifted = ed_mean_trace_split(a + 1.0, b)                 # same covariance, mean moved by sqrt(200)
    assert abs(shifted["ed_trace_part"]) < 0.01 * shifted["ed_mean_part"] and shifted["ed_mean_part"] > 1.0
    assert abs(shifted["ed_approx"] - energy_distance(a + 1.0, b)) < 0.1 * energy_distance(a + 1.0, b)
    spread = ed_mean_trace_split(1.5 * a, b)                   # same mean, wider cloud
    assert spread["ed_trace_part"] > 0 and abs(spread["ed_mean_part"]) < 0.05
    keys = EvalProtocol(n_samples=200, n_perm=5).evaluate(a, b)
    assert {"var_ratio", "ed_approx", "ed_mean_part", "ed_trace_part"} <= set(keys)
