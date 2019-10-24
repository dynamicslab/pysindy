import numpy as np
import pytest

from sindy.model.sindy import *


@pytest.fixture
def data_linear():
    x = 2 * np.linspace(1, 100, 100).reshape(-1, 1)
    xdot = 2 * np.ones(100).reshape(-1, 1)
    return x, xdot


def test_sindy_derivative(data_linear):
    x, xdot = data_linear
    s = SINDy(dt=1.0).fit(x)
    np.testing.assert_allclose(s.predict(x), xdot)


def test_sindy_score(data_linear):
    x, xdot = data_linear
    s = SINDy().fit(x, xdot)
    assert s.score(x, xdot) == 1.0


def test_sindy_n_features(data_linear):
    x, xdot = data_linear
    s = SINDy(degree=2, operators={"sin": np.sin, "cos": np.cos})
    s.fit(x)
    assert s.n_input_features_ == 1
    assert s.n_output_features_ == 15
