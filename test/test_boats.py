import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from sparsereg.boats import *
from sparsereg.sindy import SINDy

np.random.seed(42)

@pytest.mark.parametrize("lmc", [LinearRegression, Ridge, SINDy])
@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.1])
def test_fit_with_noise(data, lmc, sigma):
    x, y = data

    coef, intercept = fit_with_noise(x, y, sigma, 0.3, 200, lmc=lmc)
    assert len(coef) == x.shape[1]
    assert abs(coef[0]) <= 0.015


def test_boats(data):
    boat = BoATS(sigma=0.05, alpha=0.2, n=100).fit(*data)
    assert boat.coef_[0] <= 0.015
    assert abs(boat.intercept_ - 3) <= 1e-4


def test_boats_raise(data):
    with pytest.raises(FitFailedWarning):
        BoATS(sigma=0.05, alpha=2, n=100).fit(*data)