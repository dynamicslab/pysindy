"""Unit tests for SINDy with control."""
import numpy as np
import pytest
from scipy.integrate import odeint
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

from pysindy import SINDy
from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ


@pytest.fixture
def data_lorenz_c_1d():
    def u_fun(t):
        return np.sin(2 * t)

    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]) + u_fun(t) ** 2,
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)
    u = u_fun(t)

    return x, t, u, u_fun


@pytest.fixture
def data_lorenz_c_2d():
    def u_fun(t):
        return np.column_stack([np.sin(2 * t), t ** 2])

    def lorenz(z, t):
        u = u_fun(t)
        return [
            10 * (z[1] - z[0]) + u[0, 0] ** 2,
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2] - u[0, 1],
        ]

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)
    u = u_fun(t)

    return x, t, u, u_fun


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
    x, t, u, _ = data_lorenz_c_1d
    model = SINDy()

    model.fit(x, u=u, t=t)

    # Assumes default library is polynomial features of degree 2
    assert len(model.get_feature_names()) == 15


def test_not_fitted(data_lorenz_c_1d):
    x, t, u, u_fun = data_lorenz_c_1d
    model = SINDy()

    with pytest.raises(NotFittedError):
        model.predict(x, u=u)
    with pytest.raises(NotFittedError):
        model.simulate(x[0], t=t, u=u_fun)


def test_improper_shape_input(data_1d):
    x, t = data_1d
    u = np.ones_like(x)

    # Ensure model successfully handles different data shapes
    model = SINDy()
    model.fit(x.flatten(), u=u, t=t)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x.flatten(), u=u, t=t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, u=u, t=t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDy()
    model.fit(x.flatten(), u=u.reshape(-1, 1), t=t)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x.flatten(), u=u.reshape(-1, 1), t=t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, u=u.reshape(-1, 1), t=t, x_dot=x.flatten())
    check_is_fitted(model)

    # Should fail if x and u have incompatible numbers of rows
    with pytest.raises(ValueError):
        model.fit(x[:-1, :], u=u, t=t[:-1])


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_mixed_inputs(data):
    x, t, u, _ = data

    # Scalar t
    model = SINDy()
    model.fit(x, u=u, t=2)
    check_is_fitted(model)

    # x_dot is passed in
    model = SINDy()
    model.fit(x, u=u, x_dot=x)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, u=u, t=t, x_dot=x)
    check_is_fitted(model)


def test_bad_control_input(data_lorenz_c_1d):
    x, t, u, _ = data_lorenz_c_1d
    model = SINDy()

    with pytest.raises(TypeError):
        model.fit(x, u=set(u), t=t)


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_bad_t(data):
    x, t, u, _ = data
    model = SINDy()

    # No t
    with pytest.raises(ValueError):
        model.fit(x, u=u, t=None)

    # Invalid value of t
    with pytest.raises(ValueError):
        model.fit(x, u=u, t=-1)

    # t is a list
    with pytest.raises(ValueError):
        model.fit(x, u=u, t=list(t))

    # Wrong number of time points
    with pytest.raises(ValueError):
        model.fit(x, u=u, t=t[:-1])

    # Two points in t out of order
    t[2], t[4] = t[4], t[2]
    with pytest.raises(ValueError):
        model.fit(x, u=u, t=t)
    t[2], t[4] = t[4], t[2]

    # Two matching times in t
    t[3] = t[5]
    with pytest.raises(ValueError):
        model.fit(x, u=u, t=t)


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
    x, t, u, _ = data
    model = SINDy(optimizer=optimizer)
    model.fit(x, u=u, t=t)
    x_dot = model.predict(x, u=u)

    assert x.shape == x_dot.shape


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_simulate(data):
    x, t, u, u_fun = data
    model = SINDy()
    model.fit(x, u=u, t=t)
    x1 = model.simulate(x[0], t=t, u=u_fun)

    assert len(x1) == len(t)


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_lorenz_c_1d"), pytest.lazy_fixture("data_lorenz_c_2d")],
)
def test_score(data):
    x, t, u, _ = data
    model = SINDy()
    model.fit(x, u=u, t=t)

    assert model.score(x, u=u) <= 1

    assert model.score(x, u=u, t=t) <= 1

    assert model.score(x, u=u, x_dot=x) <= 1

    assert model.score(x, u=u, t=t, x_dot=x) <= 1


def test_parallel(data_lorenz_c_1d):
    x, t, u, _ = data_lorenz_c_1d
    model = SINDy(n_jobs=4)
    model.fit(x, u=u, t=t)

    x_dot = model.predict(x, u=u)
    s = model.score(x, u=u, x_dot=x_dot)
    assert s >= 0.95


def test_fit_multiple_trajectores(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    u = [np.ones((xi.shape[0], 2)) for xi in x]

    model = SINDy()

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.fit(x, u=u, t=t)

    # Should fail if either x or u is not a list
    with pytest.raises(ValueError):
        model.fit(x, u=u[0], multiple_trajectories=True)

    with pytest.raises(ValueError):
        model.fit(x[0], u=u, multiple_trajectories=True)

    # x and u should be lists of the same length
    with pytest.raises(ValueError):
        model.fit([x[:-1]], u=u, multiple_trajectories=True)

    model.fit(x, u=u, multiple_trajectories=True)
    check_is_fitted(model)

    model.fit(x, u=u, t=t, multiple_trajectories=True)
    assert model.score(x, u=u, t=t, multiple_trajectories=True) > 0.8

    model = SINDy()
    model.fit(x, u=u, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, u=u, t=t, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)


def test_predict_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    u = [np.ones((xi.shape[0], 2)) for xi in x]

    model = SINDy()
    model.fit(x, u=u, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x, u=u)

    p = model.predict(x, u=u, multiple_trajectories=True)
    assert len(p) == len(x)


def test_score_multiple_trajectories(data_multiple_trajctories):
    x, t = data_multiple_trajctories
    u = [np.ones((xi.shape[0], 2)) for xi in x]

    model = SINDy()
    model.fit(x, u=u, t=t, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x, u=u)

    s = model.score(x, u=u, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, u=u, t=t, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, u=u, x_dot=x, multiple_trajectories=True)
    assert s <= 1

    s = model.score(x, u=u, t=t, x_dot=x, multiple_trajectories=True)
    assert s <= 1


def test_fit_discrete_time(data_discrete_time_c):
    x, u = data_discrete_time_c

    model = SINDy(discrete_time=True)
    model.fit(x, u=u)
    check_is_fitted(model)

    model = SINDy(discrete_time=True)
    model.fit(x[:-1], u=u[:-1], x_dot=x[1:])
    check_is_fitted(model)


def test_simulate_discrete_time(data_discrete_time_c):
    x, u = data_discrete_time_c
    model = SINDy(discrete_time=True)
    model.fit(x, u=u)
    n_steps = x.shape[0]
    x1 = model.simulate(x[0], t=n_steps, u=u)

    assert len(x1) == n_steps

    # TODO: implement test using the stop_condition option


def test_predict_discrete_time(data_discrete_time_c):
    x, u = data_discrete_time_c
    model = SINDy(discrete_time=True)
    model.fit(x, u=u)
    assert len(model.predict(x, u=u)) == len(x)


def test_score_discrete_time(data_discrete_time_c):
    x, u = data_discrete_time_c
    model = SINDy(discrete_time=True)
    model.fit(x, u=u)
    assert model.score(x, u=u) > 0.75
    assert model.score(x, u=u, x_dot=x) < 1


def test_fit_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories_c,
):
    x, u = data_discrete_time_multiple_trajectories_c

    # Should fail if multiple_trajectories flag is not set
    model = SINDy(discrete_time=True)
    with pytest.raises(ValueError):
        model.fit(x, u=u)

    model.fit(x, u=u, multiple_trajectories=True)
    check_is_fitted(model)

    model = SINDy(discrete_time=True)
    model.fit(x, u=u, x_dot=x, multiple_trajectories=True)
    check_is_fitted(model)


def test_predict_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories_c,
):
    x, u = data_discrete_time_multiple_trajectories_c
    model = SINDy(discrete_time=True)
    model.fit(x, u=u, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x, u=u)

    y = model.predict(x, u=u, multiple_trajectories=True)
    assert len(y) == len(x)


def test_score_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories_c,
):
    x, u = data_discrete_time_multiple_trajectories_c
    model = SINDy(discrete_time=True)
    model.fit(x, u=u, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x, u=u)

    s = model.score(x, u=u, multiple_trajectories=True)
    assert s > 0.75

    # x is not its own derivative, so we expect bad performance here
    s = model.score(x, u=u, x_dot=x, multiple_trajectories=True)
    assert s < 1


def test_simulate_errors(data_lorenz_c_1d):
    x, t, u, u_fun = data_lorenz_c_1d
    model = SINDy()
    model.fit(x, u=u, t=t)

    with pytest.raises(ValueError):
        model.simulate(x[0], t=1, u=u)

    model = SINDy(discrete_time=True)
    with pytest.raises(ValueError):
        model.simulate(x[0], t=[1, 2], u=u)


@pytest.mark.parametrize(
    "params, warning",
    [({"threshold": 100}, UserWarning), ({"max_iter": 1}, ConvergenceWarning)],
)
def test_fit_warn(data_lorenz_c_1d, params, warning):
    x, t, u, _ = data_lorenz_c_1d
    model = SINDy(optimizer=STLSQ(**params))

    with pytest.warns(warning):
        model.fit(x, u=u, t=t)

    with pytest.warns(None) as warn_record:
        model.fit(x, u=u, t=t, quiet=True)

    assert len(warn_record) == 0


def test_u_omitted(data_lorenz_c_1d):
    x, t, u, _ = data_lorenz_c_1d
    model = SINDy()

    model.fit(x, u=u, t=t)

    with pytest.raises(TypeError):
        model.predict(x)

    with pytest.raises(TypeError):
        model.score(x)

    with pytest.raises(TypeError):
        model.simulate(x[0], t=t)


def test_extra_u_warn(data_lorenz_c_1d):
    x, t, u, _ = data_lorenz_c_1d
    model = SINDy()
    model.fit(x, t=t)

    with pytest.warns(UserWarning):
        model.predict(x, u=u)

    with pytest.warns(UserWarning):
        model.score(x, u=u)

    with pytest.warns(UserWarning):
        model.simulate(x[0], t=t, u=u)


def test_extra_u_warn_discrete(data_discrete_time_c):
    x, u = data_discrete_time_c
    model = SINDy(discrete_time=True)
    model.fit(x)

    with pytest.warns(UserWarning):
        model.predict(x, u=u)

    with pytest.warns(UserWarning):
        model.score(x, u=u)

    with pytest.warns(UserWarning):
        model.simulate(x[0], u=u, t=10)
