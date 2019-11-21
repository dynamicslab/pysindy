import sys
import os
import pytest
import numpy as np

from scipy.integrate import odeint
from sklearn.exceptions import NotFittedError

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

from sindy import SINDy
from sindy.differentiation import FiniteDifference


@pytest.fixture
def data_lorenz():

    def lorenz(z, t):
        return [
            10*(z[1] - z[0]),
            z[0]*(28 - z[2]) - z[1],
            z[0]*z[1] - 8/3*z[2]
        ]

    t = np.linspace(0, 5, 100)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)

    return x, t


@pytest.fixture
def data_1d():
    t = np.linspace(0, 5, 100)
    x = 2 * t.reshape(-1, 1)
    return x, t


@pytest.fixture
def data_1d_bad_shape():
    x = np.linspace(0, 5, 100)
    return x


def test_get_feature_names_len(data_lorenz):
    x, t = data_lorenz

    model = SINDy()
    model.fit(x, t)

    # Assumes default library is polynomial features of degree 2
    assert len(model.get_feature_names()) == 10


def test_predict_not_fitted(data_1d):
    x, t = data_1d
    model = SINDy()
    with pytest.raises(NotFittedError):
        model.predict(x)


def test_coefficient_not_fitted():
    model = SINDy()
    with pytest.raises(NotFittedError):
        model.coefficients()


def test_equation_not_fitted():
    model = SINDy()
    with pytest.raises(NotFittedError):
        model.equations()


def test_improper_shape_input(data_1d):
    x, t = data_1d
    model = SINDy()
    model.fit(x.flatten(), t)
    model.fit(x, t, x_dot=x.flatten())
    model.fit(x.flatten(), t, x_dot=x.flatten())


def test_xdot_input(data_1d):
    x, t = data_1d
    model = SINDy()
    model.fit(x, t, x_dot=x)


def test_nan_derivatives(data_lorenz):
    x, t = data_lorenz

    model = SINDy(
        differentiation_method=FiniteDifference(drop_endpoints=True)
    )
    model.fit(x, t)
