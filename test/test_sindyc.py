"""Unit tests for SINDyC: SINDy with control.

SINDyC inherits from SINDy, so we focus on testing methods with
new functionality.
"""
import numpy as np
import pytest
from scipy.integrate import odeint
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

from pysindy import SINDyC
from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ


@pytest.fixture
def data_lorenz_c_1d():
    def u(t):
        return np.sin(2 * t)

    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]) + u(t) ** 2,
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)
    u_eval = u(t)

    return x, t, u_eval, u


@pytest.fixture
def data_lorenz_c_2d():
    def u(t):
        return np.column_stack([np.sin(2 * t), t ** 2])

    def lorenz(z, t):
        u_eval = u(t)
        return [
            10 * (z[1] - z[0]) + u_eval[0, 0] ** 2,
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2] - u_eval[0, 1],
        ]

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)
    u_eval = u(t)

    return x, t, u_eval, u


@pytest.fixture
def data_discrete_time_c():
    def logistic_map(x, mu, ui):
        return mu * x * (1 - x) + ui

    n_steps = 100
    mu = 3.6

    u = 0.01 * np.random.randn(n_steps)
    x = np.zeros((n_steps))
    x[0] = 0.5
    for i in range(1, n_steps):
        x[i] = logistic_map(x[i - 1], mu, u[i - 1])

    return x, u


@pytest.fixture
def data_discrete_time_multiple_trajectories_c():
    def logistic_map(x, mu, ui):
        return mu * x * (1 - x) + ui

    n_steps = 100
    mus = [1, 2.3, 3.6]
    u = [0.001 * np.random.randn(n_steps) for mu in mus]
    x = [np.zeros((n_steps)) for mu in mus]
    for i, mu in enumerate(mus):
        x[i][0] = 0.5
        for k in range(1, n_steps):
            x[i][k] = logistic_map(x[i][k - 1], mu, u[i][k - 1])

    return x, u


def test_get_feature_names_len(data_lorenz_c_1d):
    x, t, u_eval, _ = data_lorenz_c_1d
    model = SINDyC()

    model.fit(x, u_eval, t=t)

    # Assumes default library is polynomial features of degree 2
    assert len(model.get_feature_names()) == 15


def test_not_fitted(data_lorenz_c_1d):
    x, t, u_eval, u = data_lorenz_c_1d
    model = SINDyC()

    with pytest.raises(NotFittedError):
        model.predict(x, u_eval)
    with pytest.raises(NotFittedError):
        model.simulate(x[0], u, t)


def test_improper_shape_input(data_1d):
    x, t = data_1d
    u_eval = np.ones_like(x)

    # Ensure model successfully handles different data shapes
    model = SINDyC()
    model.fit(x.flatten(), u_eval, t=t)
    check_is_fitted(model)

    model = SINDyC()
    model.fit(x.flatten(), u_eval, t=t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDyC()
    model.fit(x, u_eval, t=t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDyC()
    model.fit(x.flatten(), u_eval.reshape(-1, 1), t=t)
    check_is_fitted(model)

    model = SINDyC()
    model.fit(x.flatten(), u_eval.reshape(-1, 1), t=t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDyC()
    model.fit(x, u_eval.reshape(-1, 1), t=t, x_dot=x.flatten())
    check_is_fitted(model)


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_mixed_inputs(data):
    x, t, u_eval, _ = data

    # Scalar t
    model = SINDyC()
    model.fit(x, u_eval, t=2)
    check_is_fitted(model)

    # x_dot is passed in
    model = SINDyC()
    model.fit(x, u_eval, x_dot=x)
    check_is_fitted(model)

    model = SINDyC()
    model.fit(x, u_eval, t, x_dot=x)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_bad_t(data):
    x, t, u_eval, _ = data
    model = SINDyC()

    # No t
    with pytest.raises(ValueError):
        model.fit(x, u_eval, t=None)

    # Invalid value of t
    with pytest.raises(ValueError):
        model.fit(x, u_eval, t=-1)

    # t is a list
    with pytest.raises(ValueError):
        model.fit(x, u_eval, list(t))

    # Wrong number of time points
    with pytest.raises(ValueError):
        model.fit(x, u_eval, t[:-1])

    # Two points in t out of order
    t[2], t[4] = t[4], t[2]
    with pytest.raises(ValueError):
        model.fit(x, u_eval, t)
    t[2], t[4] = t[4], t[2]

    # Two matching times in t
    t[3] = t[5]
    with pytest.raises(ValueError):
        model.fit(x, u_eval, t)


@pytest.mark.parametrize(
    "data, optimizer",
    [
        (pytest.lazy_fixture("data_lorenz_c_1d"), STLSQ()),
        (pytest.lazy_fixture("data_lorenz_c_2d"), STLSQ()),
        (pytest.lazy_fixture("data_lorenz_c_1d"), SR3()),
        (pytest.lazy_fixture("data_lorenz_c_2d"), SR3()),
        (pytest.lazy_fixture("data_lorenz_c_1d"), Lasso(fit_intercept=False)),
        (pytest.lazy_fixture("data_lorenz_c_2d"), Lasso(fit_intercept=False)),
        (pytest.lazy_fixture("data_lorenz_c_1d"), ElasticNet(fit_intercept=False)),
        (pytest.lazy_fixture("data_lorenz_c_2d"), ElasticNet(fit_intercept=False)),
    ],
)
def test_predict(data, optimizer):
    x, t, u_eval, _ = data
    model = SINDyC(optimizer=optimizer)
    model.fit(x, u_eval, t)
    x_dot = model.predict(x, u_eval)

    assert x.shape == x_dot.shape


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_simulate(data):
    x, t, u_eval, u = data
    model = SINDyC()
    model.fit(x, u_eval, t)
    x1 = model.simulate(x[0], u, t)

    assert len(x1) == len(t)


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_score(data):
    x, t, u_eval, _ = data
    model = SINDyC()
    model.fit(x, u_eval, t)

    assert model.score(x, u_eval) <= 1

    assert model.score(x, u_eval, t=t) <= 1

    assert model.score(x, u_eval, x_dot=x) <= 1

    assert model.score(x, u_eval, t=t, x_dot=x) <= 1


def test_parallel(data_lorenz_c_1d):
    x, t, u_eval, _ = data_lorenz_c_1d
    model = SINDyC(n_jobs=4)
    model.fit(x, u_eval, t)

    x_dot = model.predict(x, u_eval)
    s = model.score(x, u_eval, x_dot=x_dot)
    assert s >= 0.95


def test_fit_multiple_trajectores(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    u_eval = [np.ones((xi.shape[0], 2)) for xi in x]

    model = SINDyC()

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.fit(x, u_eval, t=t)

    model.fit(x, u_eval, multiple_trajectories=True)
    check_is_fitted(model)

    model.fit(x, u_eval, t=t, multiple_trajectories=True)
    assert model.score(x, u_eval, t=t, multiple_trajectories=True) > 0.8

    model = SINDyC()
    model.fit(x, u_eval, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)

    model = SINDyC()
    model.fit(x, u_eval, t=t, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)


def test_predict_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    u_eval = [np.ones((xi.shape[0], 2)) for xi in x]

    model = SINDyC()
    model.fit(x, u_eval, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x, u_eval)

    p = model.predict(x, u_eval, multiple_trajectories=True)
    assert len(p) == len(x)


def test_score_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    u_eval = [np.ones((xi.shape[0], 2)) for xi in x]

    model = SINDyC()
    model.fit(x, u_eval, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x, u_eval)

    s = model.score(x, u_eval, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, u_eval, t=t, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, u_eval, x_dot=x, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, u_eval, t=t, x_dot=x, multiple_trajectories=True)
    assert s <= 1


def test_fit_discrete_time(data_discrete_time_c):
    x, u_eval = data_discrete_time_c

    model = SINDyC(discrete_time=True)
    model.fit(x, u_eval)
    check_is_fitted(model)

    model = SINDyC(discrete_time=True)
    model.fit(x[:-1], u_eval[:-1], x_dot=x[1:])
    check_is_fitted(model)


def test_simulate_discrete_time(data_discrete_time_c):
    x, u_eval = data_discrete_time_c
    model = SINDyC(discrete_time=True)
    model.fit(x, u_eval)
    n_steps = x.shape[0]
    x1 = model.simulate(x[0], u_eval, n_steps)

    assert len(x1) == n_steps

    # TODO: implement test using the stop_condition option


def test_predict_discrete_time(data_discrete_time_c):
    x, u_eval = data_discrete_time_c
    model = SINDyC(discrete_time=True)
    model.fit(x, u_eval)
    assert len(model.predict(x, u_eval)) == len(x)


def test_score_discrete_time(data_discrete_time_c):
    x, u_eval = data_discrete_time_c
    model = SINDyC(discrete_time=True)
    model.fit(x, u_eval)
    assert model.score(x, u_eval) > 0.75
    assert model.score(x, u_eval, x_dot=x) < 1


def test_fit_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories_c,
):
    x, u_eval = data_discrete_time_multiple_trajectories_c

    # Should fail if multiple_trajectories flag is not set
    model = SINDyC(discrete_time=True)
    with pytest.raises(ValueError):
        model.fit(x, u_eval)

    model.fit(x, u_eval, multiple_trajectories=True)
    check_is_fitted(model)

    model = SINDyC(discrete_time=True)
    model.fit(x, u_eval, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)


def test_predict_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories_c,
):
    x, u_eval = data_discrete_time_multiple_trajectories_c
    model = SINDyC(discrete_time=True)
    model.fit(x, u_eval, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x, u_eval)

    y = model.predict(x, u_eval, multiple_trajectories=True)
    assert len(y) == len(x)


def test_score_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories_c,
):
    x, u_eval = data_discrete_time_multiple_trajectories_c
    model = SINDyC(discrete_time=True)
    model.fit(x, u_eval, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x, u_eval)

    s = model.score(x, u_eval, multiple_trajectories=True)
    assert s > 0.75

    # x is not its own derivative, so we expect bad performance here
    s = model.score(x, u_eval, x_dot=x, multiple_trajectories=True)
    assert s < 1


def test_simulate_errors(data_lorenz_c_1d):
    x, t, u_eval, u = data_lorenz_c_1d
    model = SINDyC()
    model.fit(x, u_eval, t)

    with pytest.raises(ValueError):
        model.simulate(x[0], u, t=1)

    model = SINDyC(discrete_time=True)
    with pytest.raises(ValueError):
        model.simulate(x[0], u, t=[1, 2])


@pytest.mark.parametrize(
    "params, warning",
    [({"threshold": 100}, UserWarning), ({"max_iter": 1}, ConvergenceWarning)],
)
def test_fit_warn(data_lorenz_c_1d, params, warning):
    x, t, u_eval, _ = data_lorenz_c_1d
    model = SINDyC(optimizer=STLSQ(**params))

    with pytest.warns(warning):
        model.fit(x, u_eval, t)

    with pytest.warns(None) as warn_record:
        model.fit(x, u_eval, t, quiet=True)

    assert len(warn_record) == 0
