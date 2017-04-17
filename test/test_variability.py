import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from sparsereg.variability import *
from sparsereg.sindy import SINDy


def test_exclude_by_variability():
    c1 = np.ones(10)
    c2 = np.random.random(10)
    c3 = np.zeros(10)
    coefs = np.array([c1, c2, c3]).T

    new_coefs = exclude_by_variability(coefs, 0)
    assert new_coefs[0] == 1
    assert new_coefs[1] == 0
    assert new_coefs[2] == 0


@pytest.mark.parametrize("lmc", [LinearRegression, Ridge, SINDy])
@pytest.mark.parametrize("sigma", [0.1, 0.05, 0.01])
def test_fit_with_noise(data, lmc, sigma):
    x, y = data

    lm = fit_with_noise(x, y, lmc(), sigma=sigma)
    assert len(lm.coef_) == x.shape[1]
    assert lm.coef_[0] == 0
