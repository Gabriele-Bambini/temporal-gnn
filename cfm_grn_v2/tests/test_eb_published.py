"""Job C of pre-registration v0.2 (Step 4, scripts_dev/eb_published_protocol.py) on a tiny synthetic five-window npz with
the keys of TrajectoryNet's eb_velocity_v5.npz: loading, fold designs, what the held-out window may touch (training,
the two scalers), the exact W1, the OT-CFM mapping, the rules on hand-made numbers and a --smoke run end to end."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_dev"))
import eb_published_protocol as E  # noqa: E402
from cfm_grn_v2.training.trainer import train_flow  # noqa: E402

TINY_MLP = {"hidden": 16, "n_iters": 30}


def _write_npz(path, sizes=(40, 40, 40, 40, 40), dim=8, seed=0):
    """Five windows drifting along a fixed direction, 8 columns (only the first 5 are used), an ignored extra key."""
    rng = np.random.default_rng(seed)
    direction = np.linspace(1.0, -0.5, dim)
    pcs = np.concatenate([rng.normal(size=(n, dim)) * (1.0 + 0.2 * t) + 1.5 * t * direction
                          for t, n in enumerate(sizes)])
    labels = np.repeat(np.arange(5), sizes)
    np.savez(path, pcs=pcs, sample_labels=labels, delta_embedding=rng.normal(size=(len(labels), 2)))
    return pcs, labels


def _load(folder, **kw):
    _write_npz(folder / "eb.npz", **kw)
    return E.load_eb(folder / "eb.npz")


def _tiny():
    cfg = E.frozen_config().replace(**E.SMOKE)
    return cfg, dict(E.SMOKE_OTCFM)


# ---------------------------------------------------------------- data, designs, scalers

def test_load_keeps_five_pcs_every_cell_and_the_labels(tmp_path):
    pcs, labels = _write_npz(tmp_path / "eb.npz", sizes=(30, 31, 32, 33, 34))
    x, t = E.load_eb(tmp_path / "eb.npz")
    assert x.shape == (160, 5) and np.array_equal(x, pcs[:, :5]) and t.dtype == np.float64
    assert sorted(set(t.tolist())) == E.TIMES and np.array_equal(t, labels.astype(float))
    np.savez(tmp_path / "bad.npz", pcs=pcs, sample_labels=np.where(labels == 4, 5, labels))
    with pytest.raises(ValueError):
        E.load_eb(tmp_path / "bad.npz")


def test_data_card_records_the_file_and_what_is_used(tmp_path):
    pcs, labels = _write_npz(tmp_path / "eb.npz", sizes=(30, 31, 32, 33, 34))
    card = E.data_card(tmp_path / "eb.npz", labels.astype(float), "abc")
    assert card["sha256"] == "abc" and not card["sha256_matches_prereg"]
    assert card["keys"] == {"pcs": [160, 8], "sample_labels": [160], "delta_embedding": [160, 2]}
    assert card["keys_used"] == ["pcs[:, :5]", "sample_labels"] and card["n_cells"] == 160
    assert card["window_counts"] == {"0": 30, "1": 31, "2": 32, "3": 33, "4": 34}
    assert card["windows"]["2"] == "days 12-15" and card["conditions"] == [E.COND] and card["replicates"] == [E.REP]


def test_frozen_config_is_the_measured_v0110():
    c = E.frozen_config().to_dict()
    want = dict(hidden=128, depth=3, lr=1e-3, sigma_min=1e-2, batch_size=256, n_iters=8000, eval_every=250, patience=8,
                clock="duration", coupling="ot", pair_mode="consecutive", path="spline", chain="ot",
                path_center="linear", field_time="none", marginal_iters=1000, marginal_batch=256, marginal_steps=8,
                marginal_eval_every=25, marginal_patience=10, marginal_prior=1.0, marginal_lr=3e-4,
                marginal_window=1.0, ema_decay=0.999, grad_clip=1.0, n_ode_steps=20, ode_method="rk4")
    assert {k: c[k] for k in want} == want


def test_fold_designs_are_the_nested_interpolation_designs():
    want = {1.0: (0.0, 2.0, 3.0, 2.0, [0.0, 2.0, 4.0]), 2.0: (1.0, 3.0, 3.0, 1.0, [0.0, 1.0, 4.0]),
            3.0: (2.0, 4.0, 1.0, 0.0, [0.0, 2.0, 4.0])}
    for fold, (t_prev, t_next, val, v_prev, fit) in want.items():
        d = E.fold_design(fold)
        assert (d["t_prev"], d["t_next"], d["val_time"], d["v_prev"], d["fit_times"]) == (t_prev, t_next, val, v_prev, fit)
        assert d["obs_times"] == [t for t in E.TIMES if t != fold] and fold not in d["fit_times"]
        assert d["horizon"] == 1.0 and d["frac"] == 0.5
    assert E.FOLDS == (1.0, 2.0, 3.0)


def test_primary_scaler_uses_every_cell_and_secondary_only_the_observed_windows(tmp_path):
    pcs, labels = _load(tmp_path)
    for fold in E.FOLDS:
        xp, sp_ = E.scaler_space(pcs, labels, "primary", fold)
        xs, ss = E.scaler_space(pcs, labels, "secondary", fold)
        obs = labels != fold
        assert np.allclose(sp_.mean_, pcs.mean(0)) and np.allclose(sp_.scale_, pcs.std(0))
        assert np.allclose(ss.mean_, pcs[obs].mean(0)) and np.allclose(ss.scale_, pcs[obs].std(0))
        assert np.allclose(xs[obs].mean(0), 0.0, atol=1e-5) and np.allclose(xp.mean(0), 0.0, atol=1e-5)
        moved = pcs.copy()
        moved[~obs] = moved[~obs] * 3.0 + 7.0
        assert np.array_equal(E.scaler_space(moved, labels, "secondary", fold)[0][obs], xs[obs])
        assert not np.allclose(E.scaler_space(moved, labels, "primary", fold)[0][obs], xp[obs])


def test_held_out_window_never_enters_training_and_secondary_predictions_ignore_it(tmp_path, monkeypatch):
    pcs, labels = _load(tmp_path)
    calls = []

    def spy(snapshots, fit_mask, fit_times, config, val_task=None, protocol=None, **kw):
        calls.append((snapshots.mask(times=[fold]), np.asarray(fit_mask), list(fit_times), val_task, config))
        return train_flow(snapshots, fit_mask, fit_times, config, val_task, protocol, **kw)

    monkeypatch.setattr(E, "train_flow", spy)
    cfg, oc_over = _tiny()
    fold = 2.0
    F = E.fold_predictions(pcs, labels, fold, "secondary", [0], cfg, TINY_MLP, otcfm_over=oc_over, verbose=False)
    assert len(calls) == 5                                  # flow and twin (nested + refit each), OT-CFM equivalent
    for test_mask, fit_mask, fit_times, val_task, config in calls:
        assert not (fit_mask & test_mask).any() and fold not in fit_times
        if val_task is not None:
            assert not ((val_task.source_mask | val_task.target_mask) & test_mask).any()
    assert sum(c[3] is not None for c in calls) == 2 and calls[-1][4].marginal_iters == 0
    models = {p["model"] for p in F.preds}
    assert models == {E.FLOW, E.SHUFFLED, E.OTCFM, "next", *E.CONTROLS}
    assert all(len(p["pred"]) for p in F.preds) and np.allclose(F.targets["EB|eb1"], F.s.x[labels == fold])
    moved = pcs.copy()
    moved[labels == fold] = moved[labels == fold][::-1] * -2.0 + 5.0
    G = E.fold_predictions(moved, labels, fold, "secondary", [0], cfg, TINY_MLP, otcfm_over=oc_over, verbose=False)
    assert [(p["model"], p["seed"]) for p in F.preds] == [(p["model"], p["seed"]) for p in G.preds]
    for p, q in zip(F.preds, G.preds):
        assert np.array_equal(p["pred"], q["pred"]), p["model"]
    assert not np.allclose(F.targets["EB|eb1"], G.targets["EB|eb1"])
    persist = next(p["pred"] for p in F.preds if p["model"] == "persistence")
    nxt = next(p["pred"] for p in F.preds if p["model"] == "next")
    assert np.array_equal(persist, F.s.x[labels == 1.0]) and np.array_equal(nxt, F.s.x[labels == 3.0])


# ---------------------------------------------------------------- exact W1

def test_w1_of_a_pure_shift_is_the_shift_norm():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 5))
    a = np.array([0.3, -1.2, 0.5, 2.0, -0.1])
    assert E.w1_exact(x, x)["w1"] == pytest.approx(0.0, abs=1e-12)
    r = E.w1_exact(x, x + a)
    assert r["w1"] == pytest.approx(np.linalg.norm(a), rel=1e-9) and r["w1_converged"]
    sub = E.w1_subsample(x, x + a, seed=3, n=40)
    assert sub["w1_sub2000"] == pytest.approx(np.linalg.norm(a), abs=0.8)


def test_w1_without_pot_is_the_preregistered_assignment_fallback(monkeypatch):
    """Step 1(e): without POT, exact assignment on equal-size subsamples, 2000 per side, 10 repeats (mean)."""
    monkeypatch.setattr(E, "have_pot", lambda: False)
    shapes = []
    real = E.linear_sum_assignment

    def spy(M):
        shapes.append(M.shape)
        return real(M)

    monkeypatch.setattr(E, "linear_sum_assignment", spy)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 5))
    a = np.array([1.0, 0.0, -2.0, 0.5, 0.0])
    r = E.w1_exact(x, x[::-1] + a)                      # equal sizes <= 2000: every draw is the whole set, exact
    assert r["w1"] == pytest.approx(np.linalg.norm(a), rel=1e-9) and r["w1_backend"] == "assignment"
    assert shapes == [(30, 30)] and r["w1_converged"]
    shapes.clear()
    y = rng.normal(size=(50, 5))
    u = E.w1_exact(x, y)
    assert shapes == [(30, 30)] * 10 and u["w1_backend"] == "assignment_sub30x10"
    assert E.w1_exact(y, x)["w1"] == pytest.approx(u["w1"], abs=1e-12)          # symmetric, as W1 is
    shapes.clear()
    big = E.w1_exact(x, x + a, sub_n=20, repeats=10)
    assert shapes == [(20, 20)] * 10 and big["w1_backend"] == "assignment_sub20x10"
    assert E.w1_exact(x, x + a, sub_n=20)["w1"] == big["w1"]            # seeded: deterministic
    assert (E.SUB_N, E.FALLBACK_REPEATS) == (2000, 10)


def test_w1_uses_pot_emd2_with_uniform_weights_euclidean_cost_and_records_convergence(monkeypatch):
    seen, calls = {}, []

    def emd2(a, b, M, numItermax=None, log=False):
        seen.update(a=a, b=b, M=M, numItermax=numItermax, log=log)
        calls.append(numItermax)
        stopped = numItermax < seen.get("enough", 0)                 # the simplex stops below `enough` iterations
        return (1.5 if stopped else 1.25), {"warning": "numItermax reached before optimality." if stopped else None,
                                            "result_code": 3 if stopped else 1}

    monkeypatch.setitem(sys.modules, "ot", SimpleNamespace(emd2=emd2))
    x, y = np.zeros((3, 2)), np.array([[3.0, 4.0], [0.0, 1.0]])
    r = E.w1_exact(x, y)
    assert r == {"w1": 1.25, "w1_converged": True, "w1_backend": "pot", "w1_retried": False}
    assert calls == [10_000_000] and seen["log"] is True
    assert np.allclose(seen["a"], 1 / 3) and np.allclose(seen["b"], 0.5) and np.allclose(seen["M"][0], [5.0, 1.0])
    calls.clear()
    seen["enough"] = 50_000_000                                     # stops at 1e7, optimal at the 1e8 retry
    assert E.w1_exact(x, y) == {"w1": 1.25, "w1_converged": True, "w1_backend": "pot", "w1_retried": True}
    assert calls == [10_000_000, 100_000_000]
    seen["enough"] = 10 ** 9                                        # still stopped at 1e8: recorded, not hidden
    r = E.w1_exact(x, y)
    assert r["w1_converged"] is False and r["w1_retried"] and r["w1"] == 1.5
    assert E.have_pot()


def test_have_pot_treats_any_import_failure_as_absent(monkeypatch):
    import builtins
    real = builtins.__import__

    def broken(name, *args, **kw):
        if name == "ot":
            raise OSError("cannot load the compiled network simplex")
        return real(name, *args, **kw)

    monkeypatch.delitem(sys.modules, "ot", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken)
    assert E.have_pot() is False


def test_pot_preflight_checks_a_translation_before_any_training(monkeypatch):
    from scipy.optimize import linear_sum_assignment

    def exact(a, b, M, numItermax=None, log=False):                 # equal sizes, uniform weights: an assignment
        r, c = linear_sum_assignment(M)
        return float(M[r, c].mean()), {"warning": None, "result_code": 1}

    monkeypatch.setitem(sys.modules, "ot", SimpleNamespace(emd2=exact, __version__="0.9.5"))
    pf = E.pot_preflight()
    assert pf["passes"] and pf["pot_version"] == "0.9.5" and pf["w1"] == pytest.approx(pf["expected"], abs=1e-9)
    monkeypatch.setitem(sys.modules, "ot", SimpleNamespace(emd2=lambda a, b, M, **kw: (0.5, {"warning": None,
                                                                                              "result_code": 1})))
    with pytest.raises(SystemExit, match="preflight"):
        E.pot_preflight()

    def boom(*a, **kw):
        raise TypeError("emd2() got an unexpected keyword argument")

    monkeypatch.setitem(sys.modules, "ot", SimpleNamespace(emd2=boom))
    with pytest.raises(SystemExit, match="preflight"):
        E.pot_preflight()


# ---------------------------------------------------------------- the OT-CFM equivalent

def test_otcfm_mapping_from_the_published_configs():
    import torch
    from torch.utils.data import random_split
    for n in (2381, 1000, 37, 4163, 11):
        assert E.split_lengths(n) == [len(s) for s in random_split(range(n), [0.8, 0.1, 0.1],
                                                                   generator=torch.Generator().manual_seed(0))]
    it = E.published_iters([2381, 3278, 3665, 3332])
    assert it == {"batches_per_epoch": 3, "n_iters": 3000, "batch_size": 512, "max_epochs": 1000}
    assert E.published_iters([5000, 600])["n_iters"] == 0
    c = E.otcfm_config(3000)
    assert (c.hidden, c.depth, c.lr, c.weight_decay, c.batch_size, c.n_iters, c.sigma_min) == (64, 3, 1e-3, 1e-5, 512,
                                                                                                3000, 0.1)
    assert (c.coupling, c.clock, c.time_conditioning, c.path, c.pair_mode) == ("ot", "flow", "absolute", "linear",
                                                                              "consecutive")
    assert (c.marginal_iters, c.ema_decay, c.grad_clip, c.ode_method, c.n_ode_steps) == (0, 0.0, 0.0, "euler", 100)
    assert E.TORCHCFM["read"] is True and E.TORCHCFM["hidden_dims"] == [64, 64, 64]


def test_otcfm_prediction_integrates_the_bracket_to_the_held_out_time(tmp_path):
    pcs, labels = _load(tmp_path)
    x, _ = E.scaler_space(pcs, labels, "primary", 2.0)
    s = E.make_snapshots(x, labels)
    obs = [0.0, 1.0, 3.0, 4.0]
    model, _ = train_flow(s, s.mask(times=obs), obs, E.otcfm_config(30).replace(hidden=16, batch_size=32))
    x0 = s.x[labels == 1.0]
    full = E.otcfm_predict(model, x0, "EB", 1.0, 3.0, 3.0, n_steps=7)
    assert np.allclose(full, model.predict(x0, "EB", 1.0, 3.0, n_steps=7, method="euler"), atol=1e-5)
    half = E.otcfm_predict(model, x0, "EB", 1.0, 3.0, 2.0)
    assert half.shape == x0.shape and not np.allclose(half, full, atol=1e-3)


# ---------------------------------------------------------------- rules on hand-made numbers

def _cal(shift=0.0, drop=None):
    w = {(m, f): v + shift for m, ref in E.TABLE3.items() for f, v in ref.items()}
    if drop:
        w.pop(drop)
    return w


def test_calibration_gate():
    assert E.calibration_gate(_cal())["passes"] and E.calibration_gate(_cal(0.049))["passes"]
    assert E.calibration_gate(_cal(-0.049))["passes"]
    w = _cal()
    w[("next", 2.0)] = 0.814 + 0.051
    g = E.calibration_gate(w)
    assert not g["passes"] and not g["checks"]["next|t=2"]["within_0.05"] and g["checks"]["persistence|t=2"]["within_0.05"]
    assert not E.calibration_gate(_cal(drop=("persistence", 3.0)))["passes"]


def test_model_check():
    assert E.model_check(pd.Series([0.80, 0.78, 0.82, 0.79, 0.81]))["passes"]
    assert E.model_check(pd.Series([0.857] * 5))["passes"] and E.model_check(pd.Series([0.723] * 5))["passes"]
    r = E.model_check(pd.Series([0.87, 0.86, 0.88, 0.87, 0.87]))
    assert not r["passes"] and r["mean"] == pytest.approx(0.87)
    assert not E.model_check(pd.Series([np.nan] * 5))["passes"]


def _ps(flow, **over):
    base = {"persistence": 1.3, "next": 1.30, "mean_interp": 0.95, "ot_interp": 0.9, "linear": 1.0, "mlp": 1.0,
            "freefm_replay": 1.1, E.OTCFM: 0.85, E.SHUFFLED: 1.2}
    cols = {k: [v] * 5 for k, v in base.items()}
    cols.update({k: v for k, v in over.items()})
    cols[E.FLOW] = flow
    return pd.DataFrame(cols, index=pd.Index(range(5), name="seed"))


def test_p2a():
    flow = [0.70, 0.71, 0.69, 0.70, 0.72]
    r = E.p2a(_ps(flow))
    assert r["passes"] and r["best_control"] == E.OTCFM and r["below_every_control"]
    assert r["vs_best_control"]["ci_high"] < 0 and r["vs_shuffled"]["ci_high"] < 0
    assert not E.p2a(_ps(flow, mean_interp=[0.69] * 5))["passes"]                    # a control below the flow
    assert r["below_every_control_in_every_seed"] and r["passes_seed_mean"]
    noisy = E.p2a(_ps([0.60, 0.95, 0.60, 0.95, 0.60], **{E.OTCFM: [0.76] * 5}))       # mean below, CI crosses 0
    assert noisy["below_every_control"] and noisy["vs_best_control"]["ci_high"] > 0 and not noisy["passes"]
    assert not noisy["below_every_control_in_every_seed"] and not noisy["passes_seed_mean"]
    # the literal per-seed rule decides: seed 4 above the OT-CFM control fails P2a although the seed means and both
    # upper CIs pass (the looser seed-mean reading is reported, never deciding)
    one = E.p2a(_ps([0.70, 0.70, 0.70, 0.70, 0.86]))
    assert one["below_every_control"] and one["vs_best_control"]["ci_high"] < 0 and one["vs_shuffled"]["ci_high"] < 0
    assert not one["below_every_control_in_every_seed"] and not one["passes"] and one["passes_seed_mean"]
    nan_seed = _ps(flow, mlp=[1.0, 1.0, np.nan, 1.0, 1.0])                           # a failed control seed
    assert not E.p2a(nan_seed)["below_every_control_in_every_seed"] and not E.p2a(nan_seed)["passes"]
    shuf = E.p2a(_ps(flow, **{E.SHUFFLED: [0.50, 0.95, 0.50, 0.95, 0.60]}))
    assert shuf["vs_shuffled"]["ci_high"] > 0 and not shuf["passes"]
    missing = _ps(flow).drop(columns=["next"])
    assert not E.p2a(missing)["passes"] and E.p2a(missing)["missing_controls"] == ["next"]


def test_p2b():
    assert E.p2b(pd.Series([0.70] * 5), True, False)["statement"] is None
    assert not E.p2b(pd.Series([0.70] * 5), False, True)["applies"]
    assert E.p2b(pd.Series([0.70, 0.71, 0.69, 0.70, 0.70]), True, True)["statement"] == "below published OT-CFM"
    wide = pd.Series([0.60, 0.95, 0.60, 0.95, 0.80])                                  # mean 0.78, upper > 0.790
    assert E.p2b(wide, True, True)["statement"] == "within the published range"
    assert E.p2b(pd.Series([0.80] * 5), True, True)["statement"] == "within the published range"
    assert E.p2b(pd.Series([0.848] * 5), True, True)["statement"] == "above"
    r = E.p2b(pd.Series([0.70, 0.72]), True, True)
    assert r["upper"] == pytest.approx(0.71 + 1.96 * np.std([0.70, 0.72], ddof=1) / np.sqrt(2))
    # a binding run states nothing unless the 5 pre-registered seeds all have a finite 3-fold mean
    assert E.p2b(pd.Series([0.70] * 5), True, True, n_required=5)["statement"] == "below published OT-CFM"
    four = E.p2b(pd.Series([0.70, 0.70, np.nan, 0.70, 0.70]), True, True, n_required=5)
    assert four["statement"] is None and four["n"] == 4 and "5 pre-registered seeds" in four["reason"]
    assert E.p2b(pd.Series([0.70] * 3), True, True, n_required=5)["statement"] is None


def _table(n_seeds=3, scalers=E.SCALERS):
    rows = []
    for scaler in scalers:
        for fold in E.FOLDS:
            for seed in range(n_seeds):
                vals = {E.FLOW: 0.70 + 0.01 * seed, E.SHUFFLED: 1.2, E.OTCFM: 0.79, "mean_interp": 0.9,
                        "ot_interp": 0.9, "linear": 1.0, "mlp": 1.0, "freefm_replay": 1.1,
                        "persistence": E.TABLE3["persistence"][fold], "next": E.TABLE3["next"][fold]}
                rows += [{"scaler": scaler, "fold": fold, "seed": seed, "model": m, "w1": v, "energy_distance": v / 2,
                          "w1_converged": True} for m, v in vals.items()]
    return rows


def test_decide_on_a_hand_made_table():
    rows = _table()
    out = E.decide(pd.DataFrame(rows), smoke=True)
    pr = out["primary"]
    assert pr["calibration_gate"]["passes"] and pr["model_check"]["passes"] and pr["p2a"]["passes"]
    assert all(pr[k]["verdict"] == "pass" and pr[k]["decidable"] for k in ("calibration_gate", "model_check", "p2a"))
    assert pr["p2b"]["statement"] == "below published OT-CFM" and pr["reading"].startswith("published comparison")
    assert out["secondary"]["p2a"]["passes"] and "calibration_gate" not in out["secondary"]
    assert out["per_fold"]["primary"]["w1"]["t=2"]["best_control"] == E.OTCFM
    assert not out["binding"] and not out["seeds_match_prereg"]
    full = E.decide(pd.DataFrame(_table(5)))
    assert full["binding"] and full["primary"]["p2b"]["n_required"] == 5
    assert full["primary"]["p2b"]["statement"] == "below published OT-CFM"
    # 3 seeds in a non-smoke table: the rules run, but nothing binding and no P2b statement (needs the 5 seeds)
    three = E.decide(pd.DataFrame(rows))
    assert not three["binding"] and three["primary"]["p2b"]["statement"] is None
    part = E.decide(pd.DataFrame(_table(5, scalers=("primary",))))              # the partial summary after primary
    assert "secondary" not in part and part["primary"]["p2a"]["passes"] and set(part["per_fold"]) == {"primary"}
    df = pd.DataFrame(rows)
    df.loc[(df["model"] == "persistence") & (df["fold"] == 2.0), "w1"] = np.nan         # a failed control
    assert E.per_fold(df, "primary", "w1")["t=2"]["best_control"] == E.OTCFM
    assert E.best_control(pd.Series({"a": np.nan, "b": 0.5, "c": 0.4}), ["a", "b", "c"]) == "c"


def test_a_rule_fed_by_a_non_converged_w1_is_not_decidable():
    df = pd.DataFrame(_table(5))
    bad = (df["scaler"] == "primary") & (df["model"] == E.OTCFM) & (df["fold"] == 2.0) & (df["seed"] == 3)
    df.loc[bad, "w1_converged"] = False
    pr = E.decide(df)["primary"]
    assert pr["model_check"]["passes"] is None and pr["model_check"]["verdict"] == "not decidable"
    assert pr["model_check"]["unconverged_rows"] == [{"model": E.OTCFM, "fold": 2.0, "seed": 3}]
    assert pr["p2a"]["verdict"] == "not decidable"                          # OT-CFM is one of P2a's controls
    assert pr["calibration_gate"]["verdict"] == "pass"                      # fed by prev and next only
    assert not pr["p2b"]["applies"] and pr["p2b"]["statement"] is None and pr["reading"].startswith("only within-run")
    assert E.verdict(pr["model_check"]) == "NOT DECIDABLE" and E.verdict(pr["calibration_gate"]) == "PASS"
    df = pd.DataFrame(_table(5))
    df.loc[(df["scaler"] == "primary") & (df["model"] == E.FLOW) & (df["seed"] == 0) & (df["fold"] == 1.0),
           "w1_converged"] = False
    pr = E.decide(df)["primary"]
    assert pr["model_check"]["verdict"] == "pass" and pr["p2a"]["verdict"] == "not decidable"
    assert pr["p2b"]["applies"] and pr["p2b"]["statement"] is None and pr["p2b"]["reason"].startswith("not decidable")
    df.loc[(df["scaler"] == "secondary") & (df["model"] == "persistence"), "w1_converged"] = False
    assert E.decide(df)["primary"]["calibration_gate"]["verdict"] == "pass"   # the secondary scaler never decides


# ---------------------------------------------------------------- C_b (--choice)

def test_choice_written_by_hand_is_refused(tmp_path):
    """C_b re-derives the winner (freshfold_select.verify_choice): a choice.json that names no Job-D outputs, i.e. one
    --decide did not write, stops the run before anything is trained."""
    (tmp_path / "c.json").write_text(json.dumps({"winner": "noft", "smoke": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not written by --decide"):
        E.load_choice(tmp_path / "c.json", smoke=False)


def test_choice_winner_semantics_and_refusals(tmp_path, monkeypatch):
    import freshfold_select as fs
    monkeypatch.setattr(fs, "verify_choice", lambda c: c["winner"])     # tested in test_freshfold_rules
    ns = SimpleNamespace
    assert E.arm_weights(None, 1.0, ns(marginal_iter=25), ns(marginal_iter=1000)) == (None, "final")
    assert E.arm_weights("noft", 1.0, ns(marginal_iter=25), ns(marginal_iter=1000)) == ("stage1", "stage1")
    assert E.arm_weights("gated", 1.0, ns(marginal_iter=25), ns(marginal_iter=1000)) == ("stage1", "stage1")
    assert E.arm_weights("transfer", 1.0, ns(marginal_iter=25), ns(marginal_iter=1000)) == (25, "stage2@25")
    assert E.arm_weights("transfer", 1.0, ns(marginal_iter=0), ns(marginal_iter=1000)) == ("stage1", "stage1")
    for winner, smoke in (("base", False), ("mystery", False), ("noft", True)):
        (tmp_path / "c.json").write_text(json.dumps({"winner": winner, "smoke": smoke}), encoding="utf-8")
        with pytest.raises(SystemExit):
            E.load_choice(tmp_path / "c.json", smoke=False)
    (tmp_path / "c.json").write_text(json.dumps({"winner": "spline_noft", "smoke": False}), encoding="utf-8")
    winner, recipe, _ = E.load_choice(tmp_path / "c.json", smoke=False)
    assert winner == "spline_noft" and recipe == {"path_center": "spline", "marginal_iters": 0}


def test_c_b_gets_its_own_folder_and_the_step8_label(tmp_path):
    assert E.config_labels(None) == ("C_a", None)
    assert E.config_labels("noft") == ("C_b:noft", E.STEP8_LABEL)
    assert E.STEP8_LABEL == ("rule adopted on fresh folds, then measured once on stages inspected earlier by the "
                             "diagnosis")
    assert E.resolve_out(None, None) == Path(E.OUT_CA) and E.resolve_out(None, "c.json") == Path(E.OUT_CB)
    assert E.resolve_out(str(tmp_path), "c.json") == tmp_path
    ca = tmp_path / "ca"
    E.guard_out(ca, "C_b:noft")                                             # nothing there yet
    ca.mkdir()
    pd.DataFrame({"config": ["C_a"] * 3, "w1": [1.0, 2.0, 3.0]}).to_csv(ca / "results.csv", index=False)
    with pytest.raises(SystemExit, match="C_a"):                            # C_a's first fold must survive
        E.guard_out(ca, "C_b:noft")
    E.guard_out(ca, "C_a")                                                  # a rerun of the same configuration
    E.guard_out(ca, "C_b:noft", force=True)
    (ca / "summary.json").write_text(json.dumps({"config_label": "C_a"}), encoding="utf-8")
    (ca / "results.csv").unlink()
    with pytest.raises(SystemExit, match="C_a"):
        E.guard_out(ca, "C_b:noft")
    (ca / "summary.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot be read"):
        E.guard_out(ca, "C_a")
    (ca / "summary.json").unlink()
    pd.DataFrame({"w1": [1.0]}).to_csv(ca / "results.csv", index=False)     # someone else's table
    with pytest.raises(SystemExit, match="cannot be read"):
        E.guard_out(ca, "C_a")


def test_transfer_winner_predicts_from_the_kept_checkpoint(tmp_path):
    pcs, labels = _load(tmp_path)
    cfg, oc_over = _tiny()
    F = E.fold_predictions(pcs, labels, 1.0, "primary", [0], cfg, TINY_MLP, winner="transfer", otcfm_over=oc_over,
                           verbose=False)
    flows = [p for p in F.preds if p["family"] in ("flow", "twin")]
    assert len(flows) == 2
    for p in flows:
        k = int(p["nested_marginal_iter"] or 0)
        assert p["weights"] == ("stage1" if k == 0 else ("final" if k == p["refit_marginal_iter"] else f"stage2@{k}"))


# ---------------------------------------------------------------- end to end

def test_smoke_run_end_to_end(tmp_path):
    _write_npz(tmp_path / "eb.npz", sizes=(36, 40, 38, 40, 42))
    out = tmp_path / "out"
    import torch
    threads = torch.get_num_threads()
    E.main(["--npz", str(tmp_path / "eb.npz"), "--out", str(out), "--smoke"])
    assert torch.get_num_threads() == threads
    df = pd.read_csv(out / "results.csv")
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert set(df["scaler"]) == set(E.SCALERS) and set(df["fold"]) == set(E.FOLDS) and set(df["seed"]) == {0, 1}
    assert set(df["model"]) == {E.FLOW, E.SHUFFLED, E.OTCFM, "next", *E.CONTROLS}
    for col in ("w1", "w1_converged", "w1_sub2000", "energy_distance", "mean_error", "var_ratio", "n_pred", "n_target"):
        assert col in df.columns and df[col].notna().all(), col
    assert df["w1_converged"].all() and (df["config"] == "C_a").all()
    pr = df[df["scaler"] == "primary"].groupby(["model", "fold"])["w1"]
    assert pr.std().fillna(0).loc[["persistence", "next", "mean_interp"]].max() < 1e-12
    w = pr.mean()
    assert w[("persistence", 2.0)] == pytest.approx(w[("next", 1.0)], abs=1e-9)          # W1(X1, X2) both ways
    assert (df[df["model"] == E.FLOW]["weights"] == "final").all()
    assert summ["smoke"] and summ["config_label"] == "C_a" and summ["otcfm"]["torchcfm"]["read"]
    for k in ("calibration_gate", "model_check", "p2a", "p2b"):
        assert k in summ["rules"]["primary"]
    assert set(summ["rules"]["per_fold"]["primary"]["w1"]) == {"t=1", "t=2", "t=3"}
    assert summ["window_counts"] == {"0": 36, "1": 40, "2": 38, "3": 40, "4": 42}
    assert summ["w1"]["all_converged"] and not summ["npz_matches_prereg"]
    assert summ["data_card"]["window_counts"] == summ["window_counts"] and summ["deviations"] == E.DEVIATIONS
    assert summ["readings"] == E.READINGS and summ["complete"] and summ["scalers_done"] == list(E.SCALERS)
    assert summ["step8_label"] is None and summ["torch_threads"] == 1 and summ["rules"]["seeds"] == [0, 1]
    assert not summ["rules"]["binding"] and summ["rules"]["primary"]["p2b"]["n_required"] is None
    assert (df[df["model"] == E.SHUFFLED]["weights"] == "final").all()
    assert (out / "per_fold.csv").exists()
    # Step 1(c): flow and twin rows also carry the 8-steps-per-unit prediction (non-voting); controls do not
    fam = df["family"].isin(["flow", "twin"])
    for col in ("w1_8steps", "energy_distance_8steps", "w1_8steps_converged"):
        assert df.loc[fam, col].notna().all() and df.loc[~fam, col].isna().all(), col
    for col in ("nested_best_iter", "nested_marginal_iter", "refit_iters"):
        assert df.loc[fam, col].notna().all(), col
    for k in ("calibration_gate", "model_check", "p2a"):
        assert summ["rules"]["primary"][k]["verdict"] in ("pass", "fail")


def test_the_deciding_summary_is_written_before_the_secondary_scaler(tmp_path, monkeypatch):
    """A job stopped (24 h limit) during the secondary scaler still leaves the primary scaler's rules on disk."""
    _write_npz(tmp_path / "eb.npz", sizes=(36, 40, 38, 40, 42))
    real = E.fold_predictions

    def stop_at_secondary(pcs, labels, fold, scaler, *a, **kw):
        if scaler == "secondary":
            raise RuntimeError("time limit")
        return real(pcs, labels, fold, scaler, *a, **kw)

    monkeypatch.setattr(E, "fold_predictions", stop_at_secondary)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="time limit"):
        E.main(["--npz", str(tmp_path / "eb.npz"), "--out", str(out), "--smoke"])
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert not summ["complete"] and summ["scalers_done"] == ["primary"]
    assert "p2a" in summ["rules"]["primary"] and "secondary" not in summ["rules"]
    assert set(pd.read_csv(out / "results.csv")["scaler"]) == {"primary"}


def test_a_full_run_refuses_a_file_that_is_not_the_preregistered_one(tmp_path):
    _write_npz(tmp_path / "eb.npz")
    with pytest.raises(SystemExit, match="SHA-256"):
        E.main(["--npz", str(tmp_path / "eb.npz"), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out" / "results.csv").exists()


def test_a_full_run_refuses_any_seed_set_but_the_preregistered_five(tmp_path):
    _write_npz(tmp_path / "eb.npz")
    for n in (3, 6):
        with pytest.raises(SystemExit, match="pre-registered seeds"):
            E.main(["--npz", str(tmp_path / "eb.npz"), "--out", str(tmp_path / "out"), "--seeds", str(n)])
    assert not (tmp_path / "out").exists()
