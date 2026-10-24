import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    yield
