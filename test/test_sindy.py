import sys
import os
import pytest
import numpy as np

from scipy.integrate import odeint
from sklearn.exceptions import NotFittedError

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + "/../")

from sindy import SINDy
from sindy.differentiation import FiniteDifference
from sindy.optimizers import STLSQ, SR3, LASSO, ElasticNet


@pytest.fixture
def data_1d():
    t = np.linspace(0, 5, 100)
    x = 2 * t.reshape(-1, 1)
    return x, t


@pytest.fixture
def data_1d_bad_shape():
    x = np.linspace(0, 5, 100)
    return x


@pytest.fixture
def data_lorenz():
    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]),
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    t = np.linspace(0, 5, 100)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)

    return x, t


@pytest.fixture
def data_multiple_trajctories():
    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]),
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    n_points = [10, 50, 100]
    initial_conditions = [[8, 27, -7], [9, 28, -8], [-1, 10, 1]]

    x_list = []
    t_list = []
    for n, x0 in zip(n_points, initial_conditions):
        t = np.linspace(0, 5, n)
        t_list.append(t)
        x_list.append(odeint(lorenz, x0, t))

    return x_list, t_list


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

    model = SINDy(differentiation_method=FiniteDifference(drop_endpoints=True))
    model.fit(x, t)


@pytest.mark.parametrize(
    'data',
    [
        pytest.lazy_fixture('data_1d'),
        pytest.lazy_fixture('data_lorenz')
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
        pytest.lazy_fixture('data_1d'),
        pytest.lazy_fixture('data_lorenz')
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


@pytest.mark.parametrize(
    "data, optimizer",
    [
        (pytest.lazy_fixture('data_1d'), STLSQ()),
        (pytest.lazy_fixture('data_lorenz'), STLSQ()),
        (pytest.lazy_fixture('data_1d'), SR3()),
        (pytest.lazy_fixture('data_lorenz'), SR3()),
        (pytest.lazy_fixture('data_1d'), LASSO()),
        (pytest.lazy_fixture('data_lorenz'), LASSO()),
        (pytest.lazy_fixture('data_1d'), ElasticNet()),
        (pytest.lazy_fixture('data_lorenz'), ElasticNet()),
    ]
)
def test_predict(data, optimizer):
    x, t = data
    model = SINDy(optimizer=optimizer)
    model.fit(x, t)
    x_dot = model.predict(x)

    assert x.shape == x_dot.shape


@pytest.mark.parametrize(
    'data',
    [
        pytest.lazy_fixture('data_1d'),
        pytest.lazy_fixture('data_lorenz')
    ]
)
def test_simulate(data):
    x, t = data
    model = SINDy()
    model.fit(x, t)
    x1 = model.simulate(x[0], t)

    assert len(x1) == len(t)


@pytest.mark.parametrize(
    'data',
    [
        pytest.lazy_fixture('data_1d'),
        pytest.lazy_fixture('data_lorenz')
    ]
)
def test_score(data):
    x, t = data
    model = SINDy()
    model.fit(x, t)

    model.score(x)
    model.score(x, t)
    model.score(x, x_dot=x)
    model.score(x, t, x_dot=x)


def test_parallel(data_lorenz):
    x, t = data_lorenz
    model = SINDy(n_jobs=4)
    model.fit(x, t)

    x_dot = model.predict(x)
    model.score(x, x_dot=x_dot)


def test_fit_multiple_trajectores(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    model = SINDy()

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.fit(x, t=t)

    model.fit(x, multiple_trajectories=True)
    model.fit(x, t=t, multiple_trajectories=True)
    model.fit(x, x_dot=x, multiple_trajectories=True)
    model.fit(x, t=t, x_dot=x, multiple_trajectories=True)


def test_predict_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    model = SINDy()
    model.fit(x, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x)

    model.predict(x, multiple_trajectories=True)


def test_score_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    model = SINDy()
    model.fit(x, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x)

    model.score(x, multiple_trajectories=True)
    model.score(x, t=t, multiple_trajectories=True)
    model.score(x, x_dot=x, multiple_trajectories=True)
    model.score(x, t=t, x_dot=x, multiple_trajectories=True)
