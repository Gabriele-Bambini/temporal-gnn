import numpy as np
import torch

from cfm_grn_v2.data.schema import SnapshotSet
from cfm_grn_v2.data.synthetic_toys import make_toy
from cfm_grn_v2.training.config import TrainConfig
from cfm_grn_v2.training.trainer import replicate_task, train_flow


def _params(model):
    return torch.cat([p.detach().flatten() for p in model.vector_field.parameters()]).numpy()


def test_test_cells_do_not_influence_training():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    val = replicate_task(s, "r2", 0.0, 1.0)
    cfg = TrainConfig(hidden=16, depth=1, n_iters=30, batch_size=64, eval_every=10, seed=0)
    m1, _ = train_flow(s, fit, [0.0, 1.0], cfg, val_task=val)
    x2 = s.x.copy()
    x2[s.mask(replicates=["r3"])] += 50.0
    s2 = SnapshotSet(x2, s.biological_time, s.condition, s.replicate, s.gene_names, name=s.name)
    m2, _ = train_flow(s2, fit, [0.0, 1.0], cfg, val_task=val)
    assert np.allclose(_params(m1), _params(m2))
    assert np.allclose(m1.standardizer.mean_, m2.standardizer.mean_)


def test_shuffled_time_control_trains():
    s = make_toy("translation", n_per_group=200, seed=0)
    fit = s.mask(replicates=["r1"])
    cfg = TrainConfig(hidden=16, depth=1, n_iters=20, batch_size=64, eval_every=10, seed=0, shuffle_biological_time=True)
    model, hist = train_flow(s, fit, [0.0, 1.0], cfg)
    assert model.config.shuffle_biological_time and len(hist) >= 1
