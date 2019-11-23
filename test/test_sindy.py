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


def test_nan_derivatives(data_lorenz):
    x, t = data_lorenz

    model = SINDy(
        differentiation_method=FiniteDifference(drop_endpoints=True)
    )
    model.fit(x, t)


@pytest.mark.parametrize(
    'data',
    [
        (data_1d()),
        (data_lorenz())
    ]
)
def test_mixed_inputs(data):
    x, t = data
    model = SINDy()

    # Scalar t
    model.fit(x, t=2)

    # x_dot is passed in
    model.fit(x, t, x_dot=x)


@pytest.mark.parametrize(
    'data',
    [
        (data_1d()),
        (data_lorenz())
    ]
)
def test_bad_t(data):
    x, t = data
    model = SINDy()

    # No t
    with pytest.raises(ValueError):
        model.fit(x, t=None)

    # Invalid value of t
    with pytest.raises(ValueError):
        model.fit(x, t=-1)

    # t is a list
    with pytest.raises(ValueError):
        model.fit(x, list(t))

    # Wrong number of time points
    with pytest.raises(ValueError):
        model.fit(x, t[:-1])

    # Two points in t out of order
    t[2], t[4] = t[4], t[2]
    with pytest.raises(ValueError):
        model.fit(x, t)
    t[2], t[4] = t[4], t[2]

    # Two matching times in t
    t[3] = t[5]
    with pytest.raises(ValueError):
        model.fit(x, t)


# TODO: add tests for manually specifying x_dot,
# especially for multiple trajectories
