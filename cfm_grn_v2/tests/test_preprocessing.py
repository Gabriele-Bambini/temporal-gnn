import numpy as np
import pytest

from cfm_grn_v2.data.preprocessing import NotFittedError, Standardizer, fit_on_train


def test_standardizer_roundtrip_and_serialization():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=(100, 4))
    st = Standardizer().fit(x)
    z = st.transform(x)
    assert np.allclose(z.mean(0), 0, atol=1e-5) and np.allclose(z.std(0), 1, atol=1e-5)
    assert np.allclose(st.inverse_transform(z), x, atol=1e-4)
    st2 = Standardizer.from_dict(st.to_dict())
    assert np.allclose(st2.transform(x), z)


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        Standardizer().transform(np.zeros((2, 2)))


def test_fit_on_train_ignores_test_cells():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    mask = np.zeros(50, dtype=bool)
    mask[:30] = True
    st = fit_on_train(x, mask)
    x2 = x.copy()
    x2[~mask] += 100.0
    st2 = fit_on_train(x2, mask)
    assert np.allclose(st.mean_, st2.mean_) and np.allclose(st.std_, st2.std_)
    with pytest.raises(ValueError):
        fit_on_train(x, np.ones(50, dtype=bool))
