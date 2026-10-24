import numpy as np
import pytest
import torch

from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.evaluation.metrics import EvalProtocol, energy_distance
from cfm_grn_v2.models.conditioning import UnknownConditionError
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import FlowModel, WeightEMA, build_transitions, replicate_task, train_flow


def _setup():
    s = make_toy("translation", n_per_group=400, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    test = replicate_task(s, "r3", 0.0, 1.0)
    return s, fit, val, test


def test_build_transitions_consecutive_pairs():
    s, fit, _, _ = _setup()
    tr = build_transitions(s.subset(fit), [0.0, 1.0])
    assert len(tr) == 1 and tr[0].bio_t0 == 0.0 and tr[0].bio_t1 == 1.0 and tr[0].condition == "ctrl"
    assert tr[0].x0.shape == (400, 2) and tr[0].x1.shape == (400, 2)


def test_build_transitions_all_pairs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    t = np.repeat([0.0, 1.0, 2.0, 3.0], 10)
    s = SnapshotSet(x, t, np.array(["ctrl"] * 40), np.array(["r1"] * 40), ["a", "b"])
    cons = build_transitions(s, [0.0, 1.0, 2.0, 3.0])
    allp = build_transitions(s, [0.0, 1.0, 2.0, 3.0], pair_mode="all")
    assert [(tr.bio_t0, tr.bio_t1) for tr in cons] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert sorted((tr.bio_t0, tr.bio_t1) for tr in allp) == [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (1.0, 2.0), (1.0, 3.0), (2.0, 3.0)]
    assert all(tr.bio_t0 < tr.bio_t1 and len(tr.x0) == 10 for tr in allp)
    with pytest.raises(ValueError):
        build_transitions(s, [0.0, 1.0], pair_mode="every")


def test_weight_ema_matches_manual_average():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 2)
    ema = WeightEMA({"lin": lin}, decay=0.5)
    w0 = lin.weight.detach().clone()
    with torch.no_grad():
        lin.weight.fill_(1.0)
    ema.update()                                   # n=0: d = min(0.5, 1/10) = 0.1
    expected = 0.1 * w0 + 0.9 * torch.ones_like(w0)
    assert torch.allclose(ema.state()["lin"]["weight"], expected)
    with torch.no_grad():
        lin.weight.fill_(3.0)
    ema.update()                                   # n=1: d = min(0.5, 2/11)
    d = 2.0 / 11.0
    expected = d * expected + (1.0 - d) * 3.0 * torch.ones_like(w0)
    assert torch.allclose(ema.state()["lin"]["weight"], expected)
    assert not torch.allclose(lin.weight, ema.state()["lin"]["weight"])   # the shadow never aliases the module
    with pytest.raises(ValueError):
        WeightEMA({"lin": lin}, decay=1.0)


def test_train_flow_ema_and_grad_clip_options():
    s, fit, val, _ = _setup()
    for ema in (0.0, 0.9):
        cfg = TrainConfig(hidden=16, depth=1, n_iters=30, batch_size=64, eval_every=10, patience=10, seed=0,
                          ema_decay=ema, grad_clip=0.5)
        model, hist = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val, protocol=EvalProtocol(n_samples=100, n_perm=5))
        pred = model.predict(s.x[val.source_mask][:20], "ctrl", 0.0, 1.0)
        assert np.isfinite(pred).all() and len(hist) == 3 and model.best_iter in (10, 20, 30)


def test_duration_clock_trains_an_autonomous_field(tmp_path):
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=32, depth=2, lr=3e-3, n_iters=200, batch_size=128, eval_every=100, patience=5, seed=0,
                      clock="duration", ema_decay=0.0)
    model, hist = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val, protocol=EvalProtocol(n_samples=200, n_perm=5))
    assert model.vector_field.use_time is False and model.context_encoder.time_conditioning == "none"
    x0, x1 = s.x[test.source_mask], s.x[test.target_mask]
    pred1 = model.predict(x0, "ctrl", 0.0, 1.0)
    assert energy_distance(pred1, x1) < energy_distance(x0, x1)
    pred2 = model.predict(x0, "ctrl", 0.0, 2.0)                    # longer horizon: integrates further
    assert np.isfinite(pred2).all() and not np.allclose(pred1, pred2)
    assert np.allclose(model.predict(x0[:10], "ctrl", 3.0, 4.0), pred1[:10], atol=1e-5)   # only the horizon matters
    with pytest.raises(ValueError):
        model.predict(x0[:5], "ctrl", 1.0, 1.0)
    p = tmp_path / "m.pt"
    model.save(p)
    assert np.allclose(FlowModel.load(p).predict(x0[:10], "ctrl", 0.0, 1.0), pred1[:10], atol=1e-5)
    with pytest.raises(ValueError):
        train_flow(s, fit, [0.0, 1.0], cfg.replace(clock="sundial"))


def test_spline_path_learns_curvature_between_snapshots():
    rng = np.random.default_rng(0)
    n, times = 150, [0.0, 1.0, 2.0, 3.0]
    base = rng.normal(size=(n, 2)) * 0.05
    xs, ts = [], []
    for t in times:
        xs.append(base + np.array([t, 0.5 * t * t]) + rng.normal(size=(n, 2)) * 0.02)
        ts.append(np.full(n, t))
    s = SnapshotSet(np.concatenate(xs).astype(np.float32), np.concatenate(ts), np.array(["ctrl"] * (4 * n)),
                    np.array(["r1"] * (4 * n)), ["a", "b"])
    cfg = TrainConfig(hidden=32, depth=2, lr=3e-3, n_iters=300, batch_size=128, eval_every=100, patience=5, seed=0,
                      clock="duration", path="spline", ema_decay=0.0)
    model, _ = train_flow(s, s.mask(times=[0.0, 1.0, 3.0]), [0.0, 1.0, 3.0], cfg)
    x1, x2 = s.x[s.mask(times=[1.0])], s.x[s.mask(times=[2.0])]
    pred = model.predict(x1, "ctrl", 1.0, 2.0)
    assert energy_distance(pred, x2) < energy_distance(x1, x2)
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], cfg.replace(clock="flow"))
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], cfg.replace(path="bezier"))


def test_train_flow_beats_persistence_on_translation(tmp_path):
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=64, depth=2, lr=3e-3, n_iters=400, batch_size=128, eval_every=100, patience=3, seed=0)
    model, hist = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val, protocol=EvalProtocol(n_samples=300))
    assert "val_energy_distance" in hist[0] and model.best_iter is not None
    x0 = s.x[test.source_mask]
    x1 = s.x[test.target_mask]
    pred = model.predict(x0, "ctrl", 0.0, 1.0)
    assert energy_distance(pred, x1) < energy_distance(x0, x1)
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = FlowModel.load(path)
    assert np.allclose(loaded.predict(x0[:10], "ctrl", 0.0, 1.0), pred[:10], atol=1e-5)
    with pytest.raises(UnknownConditionError):
        model.predict(x0[:5], "never_seen", 0.0, 1.0)


def test_checkpoint_gene_mismatch_raises(tmp_path):
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=16, depth=1, n_iters=20, batch_size=64, eval_every=10, seed=0)
    model, _ = train_flow(s, fit, [0.0, 1.0], cfg)
    p = tmp_path / "m.pt"
    model.save(p)
    with pytest.raises(ValueError):
        FlowModel.load(p, expected_gene_names=["x", "y"])


def test_validation_cells_in_fit_mask_raise():
    s, fit, val, test = _setup()
    cfg = TrainConfig(hidden=16, depth=1, n_iters=5, batch_size=32, eval_every=5)
    with pytest.raises(ValueError):
        train_flow(s, fit | val.target_mask, [0.0, 1.0], cfg, val_task=val)


def test_field_time_bio_learns_a_reversal_the_autonomous_field_cannot():
    """field_time="bio": the duration-clock field sees the biological time. Cells go out and come back
    (x = 0 -> 1 -> 0 at t = 0, 1, 2). At the turning point the path velocity is zero and afterwards it points
    back, so an autonomous field v(x) started from the cells at t = 1 has no way to know it must return; a
    field v(x, t) does. The model is also non-autonomous (the start time matters) and survives save/load."""
    rng = np.random.default_rng(0)
    n, times, pos = 150, [0.0, 1.0, 2.0], [0.0, 1.0, 0.0]
    all_t, all_p = times + [0.5], pos + [0.5]                  # t = 0.5 is held out (the fit mask must hide cells)
    xs = [np.column_stack([np.full(n, p), np.zeros(n)]) + rng.normal(size=(n, 2)) * 0.05 for p in all_p]
    s = SnapshotSet(np.concatenate(xs).astype(np.float32), np.repeat(all_t, n), np.array(["ctrl"] * (4 * n)),
                    np.array(["r1"] * (4 * n)), ["a", "b"])
    cfg = TrainConfig(hidden=32, depth=2, lr=3e-3, n_iters=600, batch_size=128, eval_every=200, patience=5, seed=0,
                      clock="duration", path="spline", ema_decay=0.0)
    x1, x2 = s.x[s.mask(times=[1.0])], s.x[s.mask(times=[2.0])]
    auto, _ = train_flow(s, s.mask(times=times), times, cfg)
    bio, _ = train_flow(s, s.mask(times=times), times, cfg.replace(field_time="bio"))
    e_auto = energy_distance(auto.predict(x1, "ctrl", 1.0, 2.0), x2)
    e_bio = energy_distance(bio.predict(x1, "ctrl", 1.0, 2.0), x2)
    assert e_bio < 0.5 * e_auto and e_bio < energy_distance(x1, x2)
    z = np.zeros((4, 2), dtype=np.float32)
    assert not np.allclose(bio.predict_standardized(z, "ctrl", 0.0, 1.0), bio.predict_standardized(z, "ctrl", 1.0, 2.0))
    assert np.allclose(auto.predict_standardized(z, "ctrl", 0.0, 1.0), auto.predict_standardized(z, "ctrl", 1.0, 2.0),
                       atol=1e-6)


def test_field_time_bio_save_load_and_validation(tmp_path):
    rng = np.random.default_rng(1)
    n, times = 60, [0.0, 1.0, 2.0]
    xs = [rng.normal(size=(n, 2)) * 0.1 + np.array([t, 0.0]) for t in times]
    s = SnapshotSet(np.concatenate(xs).astype(np.float32), np.repeat(times, n), np.array(["ctrl"] * (3 * n)),
                    np.array(["r1"] * (3 * n)), ["a", "b"])
    cfg = TrainConfig(hidden=16, depth=2, n_iters=40, batch_size=32, eval_every=20, seed=0, clock="duration",
                      path="spline", field_time="bio", marginal_iters=4, marginal_eval_every=2, marginal_batch=16,
                      marginal_steps=2)
    model, _ = train_flow(s, s.mask(times=[0.0, 2.0]), [0.0, 2.0], cfg, replicate_task(s, "r1", 0.0, 1.0),
                          EvalProtocol(n_samples=40))
    p = tmp_path / "bio.pt"
    model.save(p)
    z = s.x[:5]
    assert np.allclose(FlowModel.load(p).predict(z, "ctrl", 0.5, 1.5), model.predict(z, "ctrl", 0.5, 1.5), atol=1e-5)
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], TrainConfig(n_iters=2, field_time="bio"))
    with pytest.raises(ValueError):
        train_flow(s, s.mask(times=[0.0, 1.0]), [0.0, 1.0], cfg.replace(field_time="clockwork"))


def _spline_setup():
    rng = np.random.default_rng(2)
    n, times = 80, [0.0, 1.0, 2.0, 3.0]
    xs = [rng.normal(size=(n, 2)) * 0.1 + np.array([t, 0.3 * t * t]) for t in times]
    s = SnapshotSet(np.concatenate(xs).astype(np.float32), np.repeat(times, n), np.array(["ctrl"] * (4 * n)),
                    np.array(["r1"] * (4 * n)), ["a", "b"])
    cfg = TrainConfig(hidden=16, depth=2, n_iters=40, batch_size=32, eval_every=20, seed=0, clock="duration",
                      path="spline", marginal_iters=20, marginal_eval_every=10, marginal_batch=16, marginal_steps=2)
    return s, cfg


def test_stage1_checkpoint_is_bit_identical_to_a_run_without_second_stage():
    """The refit (no validation task) keeps the weights at the end of stage 1 and a stage-2 snapshot at every
    marginal_eval_every iterations, without touching the random streams: its stage-1 prediction equals a separate
    run with marginal_iters=0, the last snapshot equals the final weights, and predicting with another stage
    leaves the final weights in place."""
    s, cfg = _spline_setup()
    fit = s.mask(times=[0.0, 1.0, 3.0])
    model, _ = train_flow(s, fit, [0.0, 1.0, 3.0], cfg, keep_snapshots=True)
    noft, _ = train_flow(s, fit, [0.0, 1.0, 3.0], cfg.replace(marginal_iters=0))
    x = s.x[s.mask(times=[1.0])][:20]
    final = model.predict(x, "ctrl", 1.0, 2.0)
    assert np.array_equal(model.predict(x, "ctrl", 1.0, 2.0, stage="stage1"), noft.predict(x, "ctrl", 1.0, 2.0))
    assert np.array_equal(model.predict(x, "ctrl", 1.0, 2.0, stage=0), noft.predict(x, "ctrl", 1.0, 2.0))
    assert sorted(model.marginal_snapshots) == [10, 20]
    assert np.array_equal(model.predict(x, "ctrl", 1.0, 2.0, stage=20), final)
    assert not np.allclose(model.predict(x, "ctrl", 1.0, 2.0, stage=10), final)
    assert np.array_equal(model.predict(x, "ctrl", 1.0, 2.0), final)            # the final weights were restored
    assert model.marginal_guarded is False and model.marginal_iter == 20
    with pytest.raises(ValueError):
        model.predict(x, "ctrl", 1.0, 2.0, stage=15)
    plain, _ = train_flow(s, fit, [0.0, 1.0, 3.0], cfg)                        # snapshots do not change the result
    assert np.array_equal(plain.predict(x, "ctrl", 1.0, 2.0), final) and plain.marginal_snapshots == {}


def test_guarded_second_stage_records_its_verdict(tmp_path):
    s, cfg = _spline_setup()
    fit = s.mask(times=[0.0, 1.0, 3.0])
    val = replicate_task(s, "r1", 1.0, 2.0)
    model, _ = train_flow(s, fit, [0.0, 1.0, 3.0], cfg, val, EvalProtocol(n_samples=60, n_perm=5))
    assert model.marginal_guarded is True and model.marginal_iter in (0, 10, 20)
    assert model.stage1_best_val is not None and model.best_val <= model.stage1_best_val
    p = tmp_path / "g.pt"
    model.save(p)
    loaded = FlowModel.load(p)
    x = s.x[:6]
    assert loaded.marginal_iter == model.marginal_iter and loaded.marginal_guarded is True
    assert np.allclose(loaded.predict(x, "ctrl", 0.0, 1.0, stage="stage1"),
                       model.predict(x, "ctrl", 0.0, 1.0, stage="stage1"), atol=1e-6)
