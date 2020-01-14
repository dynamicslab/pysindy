"""
Unit tests for SINDy class.

Note: all tests should be encapsulated in functions whose
names start with "test_"

To run all tests for this package, navigate to the top-level
directory and execute the following command:
pytest

To run tests for just one file, run
pytest file_to_test.py

"""

import pytest

from sklearn.exceptions import NotFittedError

from pysindy import SINDy
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ, SR3, LASSO, ElasticNet
from pysindy.feature_library import PolynomialLibrary, FourierLibrary


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
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
)
def test_mixed_inputs(data):
    x, t = data
    model = SINDy()

    # Scalar t
    model.fit(x, t=2)

    # x_dot is passed in
    model.fit(x, x_dot=x)
    model.fit(x, t, x_dot=x)


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")],
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
        (pytest.lazy_fixture("data_1d"), STLSQ()),
        (pytest.lazy_fixture("data_lorenz"), STLSQ()),
        (pytest.lazy_fixture("data_1d"), SR3()),
        (pytest.lazy_fixture("data_lorenz"), SR3()),
        (pytest.lazy_fixture("data_1d"), LASSO()),
        (pytest.lazy_fixture("data_lorenz"), LASSO()),
        (pytest.lazy_fixture("data_1d"), ElasticNet()),
        (pytest.lazy_fixture("data_lorenz"), ElasticNet()),
    ],
)
def test_predict(data, optimizer):
    x, t = data
    model = SINDy(optimizer=optimizer)
    model.fit(x, t)
    x_dot = model.predict(x)

    assert x.shape == x_dot.shape


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
)
def test_simulate(data):
    x, t = data
    model = SINDy()
    model.fit(x, t)
    x1 = model.simulate(x[0], t)

    assert len(x1) == len(t)


@pytest.mark.parametrize(
    "library",
    [
        PolynomialLibrary(degree=3),
        FourierLibrary(n_frequencies=3),
        pytest.lazy_fixture("data_custom_library"),
    ],
)
def test_libraries(data_lorenz, library):
    x, t = data_lorenz
    model = SINDy(feature_library=library)
    model.fit(x, t)

    model.score(x, t)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
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


def test_fit_discrete_time(data_discrete_time):
    x = data_discrete_time

    model = SINDy(discrete_time=True)

    model.fit(x)
    model.fit(x[:-1], x_dot=x[1:])


def test_simulate_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    n_steps = x.shape[0]
    x1 = model.simulate(x[0], n_steps)

    assert len(x1) == n_steps


def test_predict_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    model.predict(x)


def test_score_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    model.score(x)
    model.score(x, x_dot=x)


def test_fit_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.fit(x)

    model.fit(x, multiple_trajectories=True)
    model.fit(x, x_dot=x, multiple_trajectories=True)


def test_predict_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.predict(x)

    model.predict(x, multiple_trajectories=True)


def test_score_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x, multiple_trajectories=True)

    # Should fail if multiple_trajectories flag is not set
    with pytest.raises(ValueError):
        model.score(x)

    model.score(x, multiple_trajectories=True)
    model.score(x, x_dot=x, multiple_trajectories=True)
