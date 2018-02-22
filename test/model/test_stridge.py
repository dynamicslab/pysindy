import numpy as np
import pytest

from sparsereg.model import STRidge

test_settings = (
    dict(threshold=0.5, copy_x=True, normalize=True, fit_intercept=True),
    dict(threshold=0.1, copy_x=True, normalize=False, fit_intercept=True),
    dict(threshold=0.1, alpha=0, copy_X=True, normalize=True, fit_intercept=True),
)


@pytest.mark.parametrize("kw", test_settings)
def test_stridge_normalize(data, kw):
    x, y = data
    s = STRidge(**kw).fit(x, y)
    np.testing.assert_allclose(s.coef_, np.array([0, 2]), atol=1e-7)
    np.testing.assert_allclose(s.predict(x), y)
    assert s.complexity == 2
    assert len(s.history_) == 3 # 1 initial guess, 1 saturation, 1 unbias


def test_stridge_iterations_on_full_rank_data(data_full_rank):
    x, y = data_full_rank
    s = STRidge().fit(x, y)
    assert len(s.history_) == 2

def test_stridge_knob(data):
    x, y = data
    s = STRidge(normalize=False).fit(x, y)
    assert all(c > s.threshold or c == 0 for c in s.coef_)


def test_all_zero(data):
    x, y = data
    s = STRidge(threshold=10000).fit(x, y)
    assert not any(s.coef_)
    assert len(s.history_) == 1 # initial guess wipes everything

def test_all_nonzero(data):
    x, y = data
    s = STRidge(threshold=0).fit(x, y)
    assert s.complexity == 3
    assert len(s.history_) == 1 # initial guess is final guess
