import numpy as np
import pytest
from sklearn.exceptions import FitFailedWarning

from sparsereg.sindy import *


def test_sindy_normalize(data):
    x, y = data
    knob = 0.5
    s = SINDy(knob=knob).fit(x, y)
    
    np.testing.assert_allclose(s.coef_, np.array([0, 2]), atol=1e-7)
    np.testing.assert_allclose(s.predict(x), y)


def test_sindy_raise(data):
    x, y = data

    with pytest.raises(FitFailedWarning):
        s = SINDy(knob=10000).fit(x, y)
