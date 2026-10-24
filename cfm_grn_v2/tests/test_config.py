import json

import pytest

from cfm_grn_v2.training.config import TrainConfig, expand_grid, random_subset


def test_config_roundtrip_and_replace(tmp_path):
    cfg = TrainConfig(hidden=64, n_iters=10)
    d = cfg.to_dict()
    assert TrainConfig.from_dict(d) == cfg
    assert cfg.replace(lr=5e-4).lr == 5e-4 and cfg.lr == 1e-3
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    assert TrainConfig.from_json(p) == cfg


def test_unknown_key_raises():
    with pytest.raises(TypeError):
        TrainConfig.from_dict({"hidden": 64, "not_a_field": 1})


def test_v013_training_fields_have_defaults():
    cfg = TrainConfig()
    assert cfg.ema_decay == 0.999 and cfg.grad_clip == 1.0 and cfg.pair_mode == "consecutive" and cfg.clock == "flow" and cfg.path == "linear" and cfg.chain == "ot"
    assert TrainConfig.from_dict({"pair_mode": "all", "ema_decay": 0.0}).pair_mode == "all"


def test_expand_grid_and_subset():
    grid = expand_grid({"hidden": [64, 128], "lr": [1e-3, 1e-4], "coupling": ["ot"]})
    assert len(grid) == 4 and all(g["coupling"] == "ot" for g in grid)
    sub = random_subset(grid, 2, seed=0)
    assert len(sub) == 2 and all(s in grid for s in sub)
