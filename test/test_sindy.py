import numpy as np

from sparsereg.sindy import *


def test_sindy_normalize():
    x = 2*np.random.random(size=(10, 2)) + 5
    y = 2 * x[:, 1]

    s = SINDy(knob=0.5).fit(x, y)
    np.testing.assert_allclose(s.coef_, np.array([0, 2]), atol=1e-7)

    np.testing.assert_allclose(s.predict(x), y)
