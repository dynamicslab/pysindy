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
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted

from pysindy import pysindy
from pysindy import SINDy
from pysindy.differentiation import SINDyDerivative
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import PDELibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import WeakPDELibrary
from pysindy.optimizers import ConstrainedSR3
from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ
from pysindy.optimizers import WrappedOptimizer


def test_get_feature_names_len(data_lorenz):
    x, t = data_lorenz
    model = SINDy()

    with pytest.raises(NotFittedError):
        model.get_feature_names()

    model.fit(x, t)

    # Assumes default library is polynomial features of degree 2
    assert len(model.get_feature_names()) == 10


def test_not_fitted(data_1d):
    x, t = data_1d
    model = SINDy()

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

    # Ensure model successfully handles different data shapes
    model = SINDy()
    model.fit(x.flatten(), t)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x.flatten(), t, x_dot=x.flatten())
    check_is_fitted(model)

    model = SINDy()
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

    # Scalar t
    model = SINDy()
    model.fit(x, t=2)
    check_is_fitted(model)

    # x_dot is passed in
    model = SINDy()
    model.fit(x, x_dot=x)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, t, x_dot=x)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
def test_bad_t(data):
    x, t = data
    model = SINDy()

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
def test_t_default(data):
    x, t = data
    dt = t[1] - t[0]

    with pytest.raises(ValueError):
        model = SINDy(t_default=0)
    with pytest.raises(ValueError):
        model = SINDy(t_default="1")

    model = SINDy()
    model.fit(x, dt)

    model_t_default = SINDy(t_default=dt)
    model_t_default.fit(x)

    np.testing.assert_allclose(model.coefficients(), model_t_default.coefficients())
    np.testing.assert_almost_equal(model.score(x, t=dt), model_t_default.score(x))
    np.testing.assert_almost_equal(
        model.differentiate(x, t=dt), model_t_default.differentiate(x)
    )


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
def test_differentiate_returns_compatible_data_type(data):
    x, t = data
    x_dot = SINDy().differentiate(x)
    assert isinstance(x_dot, type(x))


@pytest.mark.parametrize(
    "data", [pytest.lazy_fixture("data_1d"), pytest.lazy_fixture("data_lorenz")]
)
@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ(),
        SR3(),
        ConstrainedSR3(),
        WrappedOptimizer(Lasso(fit_intercept=False)),
        WrappedOptimizer(ElasticNet(fit_intercept=False)),
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
    model = SINDy(feature_library=PolynomialLibrary(degree=1))
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
        pytest.lazy_fixture("custom_library"),
        pytest.lazy_fixture("sindypi_library"),
        PolynomialLibrary() + FourierLibrary(),
    ],
)
def test_libraries(data_lorenz, library):
    x, t = data_lorenz
    model = SINDy(feature_library=library)
    model.fit(x, t)

    s = model.score(x, t)
    assert s <= 1


def test_integration_smoothed_finite_difference(data_lorenz):
    x, t = data_lorenz
    model = SINDy(differentiation_method=SmoothedFiniteDifference())

    model.fit(x, t=t)

    check_is_fitted(model)


@pytest.mark.parametrize(
    "derivative_kws",
    [
        dict(kind="finite_difference", k=1),
        dict(kind="spectral"),
        dict(kind="spline", s=1e-2),
        dict(kind="trend_filtered", order=0, alpha=1e-2),
        dict(kind="savitzky_golay", order=3, left=1, right=1),
    ],
)
def test_integration_derivative_methods(data_lorenz, derivative_kws):
    x, t = data_lorenz
    model = SINDy(differentiation_method=SINDyDerivative(**derivative_kws))
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
    model = SINDy()
    model.fit(x, t)

    assert model.score(x) <= 1

    assert model.score(x, t) <= 1

    assert model.score(x, x_dot=x) <= 1

    assert model.score(x, t, x_dot=x) <= 1


def test_score_pde(data_1d_random_pde):
    t, x, u, u_dot = data_1d_random_pde
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=x,
        include_bias=True,
    )
    model = SINDy(feature_library=pde_lib).fit(
        u,
        t,
    )

    assert model.score(u) <= 1

    assert model.score(u, t) <= 1

    assert model.score(u, x_dot=u) <= 1

    assert model.score(u, t, x_dot=u) <= 1

    X, T = np.meshgrid(x, t)
    XT = np.array([X, T]).T
    weak_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatiotemporal_grid=XT,
        include_bias=True,
    )
    model = SINDy(feature_library=weak_lib).fit(
        u,
        t=t,
    )

    assert model.score(u) <= 1

    assert model.score(u, t) <= 1


def test_fit_multiple_trajectores(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    model = SINDy()

    model.fit(x)
    check_is_fitted(model)

    model.fit(x, t=t)
    assert model.score(x, t=t) > 0.8

    model = SINDy()
    model.fit(x, x_dot=x)
    check_is_fitted(model)

    model = SINDy()
    model.fit(x, t=t, x_dot=x)
    check_is_fitted(model)

    # Test validate_input
    t[0] = None
    with pytest.raises(ValueError):
        model.fit(x, t=t)


def test_predict_multiple_trajectories(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    model = SINDy()
    model.fit(x, t=t)

    p = model.predict(x)
    assert len(p) == len(x)


def test_score_multiple_trajectories(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    model = SINDy()
    model.fit(x, t=t)

    s = model.score(x)
    assert s <= 1

    s = model.score(x, t=t)
    assert s <= 1

    s = model.score(x, x_dot=x)
    assert s <= 1

    s = model.score(x, t=t, x_dot=x)
    assert s <= 1


def test_fit_discrete_time(data_discrete_time):
    x = data_discrete_time

    model = SINDy(discrete_time=True)
    model.fit(x)
    check_is_fitted(model)

    model = SINDy(discrete_time=True)
    model.fit(x[:-1], x_dot=x[1:])
    check_is_fitted(model)


def test_simulate_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    n_steps = x.shape[0]
    x1 = model.simulate(x[0], n_steps)

    assert len(x1) == n_steps

    def stop_func(xi):
        # check if we are at the 2nd to last element
        return np.isclose(xi[0], 0.874363)

    x2 = model.simulate(x[0], n_steps, stop_condition=stop_func)
    assert len(x2) == n_steps - 2


def test_predict_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    assert len(model.predict(x)) == len(x)


def test_score_discrete_time(data_discrete_time):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    assert model.score(x) > 0.75
    assert model.score(x, x_dot=x) < 1


def test_bad_multiple_trajectories(data_multiple_trajectories):
    x, t = data_multiple_trajectories
    with pytest.raises(TypeError):
        pysindy._check_multiple_trajectories(x, x_dot=x[0], u=None)
    with pytest.raises(ValueError):
        pysindy._check_multiple_trajectories(x, x_dot=x[:-1], u=None)


def test_fit_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x)
    check_is_fitted(model)

    model = SINDy(discrete_time=True)
    model.fit(x, x_dot=x)
    check_is_fitted(model)


def test_predict_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x)

    y = model.predict(x)
    assert len(y) == len(x)


def test_score_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories,
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x)

    s = model.score(x)
    assert s > 0.75

    # x is not its own derivative, so we expect bad performance here
    s = model.score(x, x_dot=x)
    assert s < 1


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
    model = SINDy()
    model.fit(x, t)

    out, _ = capsys.readouterr()
    assert len(out) == 0

    model.print(precision=2)

    out, _ = capsys.readouterr()
    assert len(out) > 0


def test_print_discrete_time(data_discrete_time, capsys):
    x = data_discrete_time
    model = SINDy(discrete_time=True)
    model.fit(x)
    model.print()

    out, _ = capsys.readouterr()
    assert len(out) > 0


def test_print_discrete_time_multiple_trajectories(
    data_discrete_time_multiple_trajectories, capsys
):
    x = data_discrete_time_multiple_trajectories
    model = SINDy(discrete_time=True)
    model.fit(x)

    model.print()

    out, _ = capsys.readouterr()
    assert len(out) > 1


def test_differentiate(data_lorenz, data_multiple_trajectories):
    x, t = data_lorenz

    model = SINDy()
    model.differentiate(x, t)

    x, t = data_multiple_trajectories
    model.differentiate(x, t)

    model = SINDy(discrete_time=True)
    with pytest.raises(RuntimeError):
        model.differentiate(x)


def test_coefficients_equals_complexity(data_lorenz):
    x, t = data_lorenz
    model = SINDy()
    model.fit(x, t)
    c = model.coefficients()
    assert model.complexity == np.count_nonzero(c)
    assert model.complexity < 30


def test_simulate_errors(data_lorenz):
    x, t = data_lorenz
    model = SINDy()
    model.fit(x, t)

    with pytest.raises(ValueError):
        model.simulate(x[0], t=1)

    model = SINDy(discrete_time=True)
    with pytest.raises(ValueError):
        model.simulate(x[0], t=[1, 2])

    model = SINDy(discrete_time=True)
    with pytest.raises(ValueError):
        model.simulate(x[0], t=-1)

    model = SINDy(discrete_time=True)
    with pytest.raises(ValueError):
        model.simulate(x[0], t=0.5)


@pytest.mark.parametrize(
    "params, warning",
    [({"threshold": 100}, UserWarning), ({"max_iter": 1}, ConvergenceWarning)],
)
def test_fit_warn(data_lorenz, params, warning):
    x, t = data_lorenz
    model = SINDy(optimizer=STLSQ(**params))

    with pytest.warns(warning):
        model.fit(x, t=t)


def test_cross_validation(data_lorenz):
    x, t = data_lorenz
    dt = t[1] - t[0]

    model = SINDy(
        t_default=dt, differentiation_method=SINDyDerivative(kind="spline", s=1e-2)
    )

    param_grid = {
        "optimizer__threshold": [0.01, 0.1],
        "differentiation_method__kwargs": [
            {"kind": "spline", "s": 1e-2},
            {"kind": "finite_difference", "k": 1},
        ],
        "feature_library__degree": [1, 2],
    }

    search = RandomizedSearchCV(
        model, param_grid, cv=TimeSeriesSplit(n_splits=3), n_iter=5
    )
    search.fit(x)
    check_is_fitted(search)


def test_linear_constraints(data_lorenz):
    x, t = data_lorenz

    library = PolynomialLibrary().fit(x)

    constraint_rhs = np.ones(2)
    constraint_lhs = np.zeros((2, x.shape[1] * library.n_output_features_))

    target_1, target_2 = 1, 3
    constraint_lhs[0, 3] = target_1
    constraint_lhs[1, library.n_output_features_] = target_2

    optimizer = ConstrainedSR3(
        constraint_lhs=constraint_lhs, constraint_rhs=constraint_rhs
    )
    model = SINDy(feature_library=library, optimizer=optimizer).fit(x, t)

    coeffs = model.coefficients()

    np.testing.assert_allclose(
        np.array([coeffs[0, 3], coeffs[1, 0]]), np.array([1 / target_1, 1 / target_2])
    )


def test_data_shapes():
    model = SINDy()
    n = 10
    x = np.ones(n)
    model.fit(x)
    x = np.ones((n, 2))
    model.fit(x)
    x = np.ones((n, n, 2))
    model.fit(x)
    x = np.ones((n, n, n, 2))
    model.fit(x)
    x = np.ones((n, n, n, n, 2))
    model.fit(x)


def test_diffusion_pde(diffuse_multiple_trajectories):
    t, x, u = diffuse_multiple_trajectories
    library_functions = [lambda x: x]
    library_function_names = [lambda x: x]

    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=x,
    )

    X, T = np.meshgrid(x, t, indexing="ij")
    XT = np.transpose([X, T], [1, 2, 0])

    weak_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatiotemporal_grid=XT,
        K=100,
    )

    optimizer = STLSQ(threshold=0.2, alpha=1e-5, normalize_columns=False)
    model = SINDy(feature_library=pde_lib, optimizer=optimizer, feature_names=["u"])
    model.fit(u, t=t)
    assert abs(model.coefficients()[0, -1] - 1) < 1e-1
    assert np.all(model.coefficients()[0, :-1] == 0)

    model = SINDy(feature_library=weak_lib, optimizer=optimizer, feature_names=["u"])
    model.fit(u, t=t)
    assert abs(model.coefficients()[0, -1] - 1) < 1e-1
    assert np.all(model.coefficients()[0, :-1] == 0)
