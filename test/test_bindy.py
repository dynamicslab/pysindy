"""
Unit tests for SINDy class.

Note: all tests should be encapsulated in functions with
names starting with "test_"

To run all tests for this package, navigate to the top-level
directory and execute the following command:
pytest

To run tests for just one file, run
pytest file_to_test.py

"""
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted

from pysindy import _core
from pysindy import SINDy
from pysindy import BINDy
from pysindy.differentiation import SINDyDerivative
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ
from pysindy.optimizers import EvidenceGreedy


def test_get_feature_names_len(data_lorenz):
    x, t = data_lorenz
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)

    with pytest.raises(NotFittedError):
        model.get_feature_names()

    model.fit(x, t)

    # Assumes default library is polynomial features of degree 2
    assert len(model.get_feature_names()) == 10


def test_not_fitted(data_1d):
    x, t = data_1d
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)

    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.get_feature_names()
    with pytest.raises(NotFittedError):
        model.coefficients()
    with pytest.raises(NotFittedError):
        model.equations()
    with pytest.raises(NotFittedError):
        model.simulate(x[0], t)


def test_improper_shape_input(data_1d):
    x, t = data_1d
    x = x + 1e-2 * np.random.randn(*x.shape)

    # Ensure model successfully handles different data shapes
    model = BINDy(1e-2)
    model.fit(x.flatten(), t)
    check_is_fitted(model)

    model = BINDy(1e-2)
    model.fit(x.flatten(), t, x_dot=x.flatten())
    check_is_fitted(model)

    model = BINDy(1e-2)
    model.fit(x, t, x_dot=x.flatten())
    check_is_fitted(model)


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
    x = x + 1e-2 * np.random.randn(*x.shape)

    # Scalar t
    model = BINDy(1e-2)
    model.fit(x, t=2)
    check_is_fitted(model)

    model = BINDy(1e-2)
    model.fit(x, t, x_dot=x)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
def test_bad_t(data):
    x, t = data
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)

    # Wrong type
    with pytest.raises(ValueError):
        model.fit(x, t="1")

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
    t_new = np.copy(t)
    t_new[2], t_new[4] = t_new[4], t_new[2]
    with pytest.raises(ValueError):
        model.fit(x, t_new)
    t_new[2], t_new[4] = t_new[4], t_new[2]

    # Two matching times in t
    t_new[3] = t_new[5]
    with pytest.raises(ValueError):
        model.fit(x, t_new)


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
@pytest.mark.parametrize(
    "optimizer",
    [
        EvidenceGreedy(),
    ],
)
def test_predict(data, optimizer):
    x, t = data
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2, optimizer=optimizer)
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
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2, feature_library=PolynomialLibrary(degree=1))
    model.fit(x, t)
    x1 = model.simulate(np.ravel(x[0]), t, integrator_kws={"rtol": 0.1})
    assert len(x1) == len(t)
    x1 = model.simulate(
        np.ravel(x[0]), t, integrator="odeint", integrator_kws={"rtol": 0.1}
    )
    assert len(x1) == len(t)
    with pytest.raises(ValueError):
        x1 = model.simulate(np.ravel(x[0]), t, integrator="None")


@pytest.mark.parametrize(
    "library",
    [
        PolynomialLibrary(degree=3),
        FourierLibrary(n_frequencies=3),
        pytest.lazy_fixture("sindypi_library"),
        PolynomialLibrary() + FourierLibrary(),
    ],
)
def test_libraries(data_lorenz, library):
    x, t = data_lorenz
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2, feature_library=library)
    model.fit(x, t)

    s = model.score(x, t)
    assert s <= 1


def test_integration_smoothed_finite_difference(data_lorenz):
    x, t = data_lorenz
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2, differentiation_method=SmoothedFiniteDifference())

    model.fit(x, t=t)

    check_is_fitted(model)


@pytest.mark.parametrize(
    "derivative_kws",
    [
        dict(kind="finite_difference", k=1),
        dict(kind="savitzky_golay", order=3, left=1, right=1),
    ],
)
def test_integration_derivative_methods(data_lorenz, derivative_kws):
    x, t = data_lorenz
    x = x + 1e-2 * np.random.randn(*x.shape)
    fd = SINDyDerivative(**derivative_kws)
    
    sigma2 = EvidenceGreedy.TemporalNoisePropagation(fd, t, 1e-2)
    model = BINDy(1e-2, optimizer=EvidenceGreedy(_sigma2=sigma2), differentiation_method=fd)
    model.fit(x, t=t)

    check_is_fitted(model)


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
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)
    model.fit(x, t)

    assert model.score(x, t) <= 1

    assert model.score(x, t, x_dot=x) <= 1


def test_fit_multiple_trajectories(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)

    model.fit(x, t=t)
    check_is_fitted(model)
    assert model.score(x, t=t) > 0.8

    model = BINDy(1e-2)
    model.fit(x, t=t, x_dot=x)
    check_is_fitted(model)

    # Test validate_input
    t[0] = None
    with pytest.raises(ValueError):
        model.fit(x, t=t)


def test_predict_multiple_trajectories(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)
    model.fit(x, t=t)

    p = model.predict(x)
    assert len(p) == len(x)


def test_score_multiple_trajectories(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)
    model.fit(x, t=t)

    s = model.score(x, t=t)
    assert s <= 1

    s = model.score(x, t=t, x_dot=x)
    assert s <= 1


def test_bad_multiple_trajectories(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    with pytest.raises(TypeError):
        _core._check_multiple_trajectories(x, x_dot=x[0], u=None)
    with pytest.raises(ValueError):
        _core._check_multiple_trajectories(x, x_dot=x[:-1], u=None)

@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_1d"),
        pytest.lazy_fixture("data_lorenz"),
        pytest.lazy_fixture("data_1d_bad_shape"),
    ],
)
def test_equations(data, capsys):
    x, t = data
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)
    model.fit(x, t)

    out, _ = capsys.readouterr()
    assert len(out) == 0

    model.print(precision=2)

    out, _ = capsys.readouterr()

    assert len(out) > 0
    assert "(x0)' = " in out



def test_coefficients_equals_complexity(data_lorenz):
    x, t = data_lorenz
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)
    model.fit(x, t)
    c = model.coefficients()
    assert model.complexity == np.count_nonzero(c)


def test_simulate_errors(data_lorenz):
    x, t = data_lorenz
    x = x + 1e-2 * np.random.randn(*x.shape)
    model = BINDy(1e-2)
    model.fit(x, t)

    with pytest.raises(ValueError):
        model.simulate(x[0], t=1)


def test_data_shapes():
    model = BINDy(1e-2)
    n = 10
    x = np.ones(n)
    t = 1
    model.fit(x, t)
    x = np.ones((n, 2))
    model.fit(x, t)
    x = np.ones((n, n, 2))
    model.fit(x, t)
    x = np.ones((n, n, n, 2))
    model.fit(x, t)
    x = np.ones((n, n, n, n, 2))
    model.fit(x, t)



