import numpy as np
import pytest
from sklearn.exceptions import FitFailedWarning

from sparsereg.sindy import *

@pytest.fixture
def data():
    x = 2*np.random.random(size=(10, 2)) + 5
    y = 2 * x[:, 1]
    return x, y


def test_sindy_normalize(data):
    x, y = data
    s = SINDy(knob=0.5).fit(x, y)
    np.testing.assert_allclose(s.coef_, np.array([0, 2]), atol=1e-7)

    np.testing.assert_allclose(s.predict(x), y)

def test_sindy_raise(data):
    x, y = data

    with pytest.raises(FitFailedWarning):
        s = SINDy(knob=10000).fit(x, y)
