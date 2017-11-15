import numpy as np
import pytest

from sparsereg.model.sindy import *

@pytest.fixture
def data_linear():
    x = 2 * np.linspace(1, 100, 100).reshape(-1, 1)
    xdot = 2 * np.ones(100).reshape(-1, 1)
    return x, xdot

def test_sindy_derivative(data_linear):
    x, xdot = data_linear
    s = SINDy(dt=1.0).fit(x)
    np.testing.assert_allclose(s.predict(x), xdot)
