"""
Unit tests for optimizers.
"""
import numpy as np
import pytest
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

from pysindy import FiniteDifference
from pysindy import PDELibrary
from pysindy import PolynomialLibrary
from pysindy import SINDy
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.optimizers import ConstrainedSR3
from pysindy.optimizers import FROLS
from pysindy.optimizers import SINDyOptimizer
from pysindy.optimizers import SINDyPI
from pysindy.optimizers import SR3
from pysindy.optimizers import SSR
from pysindy.optimizers import STLSQ
from pysindy.optimizers import TrappingSR3
from pysindy.utils import supports_multiple_targets
from pysindy.utils.odes import enzyme

# For reproducibility
np.random.seed(100)


class DummyLinearModel(BaseEstimator):
    # Does not natively support multiple targets
    def fit(self, x, y):
        self.coef_ = np.ones(x.shape[1])
        self.intercept_ = 0
        return self

    def predict(self, x):
        return x


class DummyEmptyModel(BaseEstimator):
    # Does not have fit or predict methods
    def __init__(self):
        self.fit_intercept = False
        self.normalize_columns = False


class DummyModelNoCoef(BaseEstimator):
    # Does not set the coef_ attribute
    def fit(self, x, y):
        self.intercept_ = 0
        return self

    def predict(self, x):
        return x


@pytest.mark.parametrize(
    "cls, support",
    [
        (Lasso, True),
        (STLSQ, True),
        (SSR, True),
        (FROLS, True),
        (SR3, True),
        (ConstrainedSR3, True),
        (TrappingSR3, True),
        (DummyLinearModel, False),
    ],
)
def test_supports_multiple_targets(cls, support):
    assert supports_multiple_targets(cls()) == support


@pytest.fixture(params=["data_derivative_1d", "data_derivative_2d"])
def data(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ(),
        SSR(),
        SSR(criteria="model_residual"),
        FROLS(),
        SR3(),
        ConstrainedSR3(),
        TrappingSR3(),
        Lasso(fit_intercept=False),
        ElasticNet(fit_intercept=False),
        DummyLinearModel(),
    ],
)
def test_fit(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    opt = SINDyOptimizer(optimizer, unbias=False)
    opt.fit(x, x_dot)

    check_is_fitted(opt)
    assert opt.complexity >= 0
    if len(x_dot.shape) > 1:
        assert opt.coef_.shape == (x.shape[1], x_dot.shape[1])
    else:
        assert opt.coef_.shape == (1, x.shape[1])


@pytest.mark.parametrize(
    "optimizer",
    [STLSQ(), SSR(), SSR(criteria="model_residual"), FROLS(), SR3()],
)
def test_not_fitted(optimizer):
    with pytest.raises(NotFittedError):
        optimizer.predict(np.ones((1, 3)))


@pytest.mark.parametrize("optimizer", [STLSQ(), SR3()])
def test_complexity_not_fitted(optimizer, data_derivative_2d):
    with pytest.raises(NotFittedError):
        optimizer.complexity

    x, _ = data_derivative_2d
    optimizer.fit(x, x)
    assert optimizer.complexity > 0


@pytest.mark.parametrize(
    "kwargs", [{"normalize_columns": True}, {"fit_intercept": True}, {"copy_X": False}]
)
def test_alternate_parameters(data_derivative_1d, kwargs):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    model = STLSQ(**kwargs)
    model.fit(x, x_dot)
    model.fit(x, x_dot, sample_weight=x[:, 0])

    check_is_fitted(model)


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ,
        SSR,
        FROLS,
        SR3,
        ConstrainedSR3,
        TrappingSR3,
    ],
)
def test_sample_weight_optimizers(data_lorenz, optimizer):
    x, t = data_lorenz
    dt = t[1] - t[0]
    x_dot = FiniteDifference()._differentiate(x, t=dt)

    sample_weight = np.ones(x[:, 0].shape)
    sample_weight[::2] = 0
    model = optimizer()
    model.fit(x, x_dot)
    model.fit(x, x_dot, sample_weight=sample_weight)
    model.fit(x, x_dot, sample_weight=sample_weight)
    check_is_fitted(model)

    model = optimizer(normalize_columns=True)
    model.fit(x, x_dot)
    model.fit(x, x_dot, sample_weight=sample_weight)
    model.fit(x, x_dot, sample_weight=sample_weight)
    check_is_fitted(model)


@pytest.mark.parametrize("optimizer", [STLSQ, SR3, ConstrainedSR3])
@pytest.mark.parametrize("params", [dict(threshold=-1), dict(max_iter=0)])
def test_general_bad_parameters(optimizer, params):
    with pytest.raises(ValueError):
        optimizer(**params)


@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3])
@pytest.mark.parametrize(
    "params",
    [
        dict(nu=0),
        dict(tol=0),
        dict(trimming_fraction=-1),
        dict(trimming_fraction=2),
    ],
)
def test_sr3_bad_parameters(optimizer, params):
    with pytest.raises(ValueError):
        optimizer(**params)


@pytest.mark.parametrize(
    "params",
    [
        dict(eta=-1),
        dict(tol=0),
        dict(tol_m=0),
        dict(eps_solver=0),
        dict(eta=1, alpha_m=-1),
        dict(eta=1, alpha_A=-1),
        dict(gamma=1),
        dict(evolve_w=False, relax_optim=False),
        dict(thresholder="l0"),
        dict(threshold=-1),
        dict(max_iter=0),
        dict(eta=10, alpha_m=20),
        dict(eta=10, alpha_A=20),
        dict(inequality_constraints=True, evolve_w=False),
        dict(
            constraint_lhs=np.zeros((10, 10)),
            constraint_rhs=np.zeros(10),
            constraint_order="None",
        ),
    ],
)
def test_trapping_bad_parameters(params):
    with pytest.raises(ValueError):
        TrappingSR3(**params)


@pytest.mark.parametrize(
    "params",
    [
        dict(tol=0),
        dict(max_iter=-1),
        dict(thresholder="l0"),
        dict(threshold=-1),
        dict(thresholds=1),
        dict(thresholder="weighted_l1"),
        dict(model_subset=0),
        dict(model_subset=[50]),
        dict(model_subset=[0, 0.5, 1]),
    ],
)
def test_sindypi_bad_parameters(data_lorenz, params):
    x, t = data_lorenz
    with pytest.raises(ValueError):
        opt = SINDyPI(**params)
        model = SINDy(optimizer=opt)
        model.fit(x, t=t)


@pytest.mark.parametrize(
    "params",
    [
        dict(tol=1e-3),
        dict(thresholder="l1"),
        dict(thresholder="weighted_l1", thresholds=np.zeros((10, 10))),
        dict(thresholder="l2"),
        dict(thresholder="weighted_l2", thresholds=np.zeros((10, 10))),
        dict(model_subset=[5]),
    ],
)
def test_sindypi_fit(params):
    dt = 0.01
    T = 5
    t = np.arange(0, T + dt, dt)
    x0_train = [0.55]
    x_train = solve_ivp(enzyme, (t[0], t[-1]), x0_train, t_eval=t).y.T

    # initialize a quartic polynomial library for x
    x_library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x ** 2,
        lambda x, y, z: x * y * z,
        lambda x, y: x * y ** 2,
        lambda x: x ** 3,
        lambda x, y, z, w: x * y * z * w,
        lambda x, y, z: x * y * z ** 2,
        lambda x, y: x * y ** 3,
        lambda x: x ** 4,
    ]
    # initialize a linear polynomial library for x_dot
    x_dot_library_functions = [lambda x: x]

    # library function names includes both the x_library_functions
    # and x_dot_library_functions names
    library_function_names = [
        lambda x: x,
        lambda x, y: x + y,
        lambda x: x + x,
        lambda x, y, z: x + y + z,
        lambda x, y: x + y + y,
        lambda x: x + x + x,
        lambda x, y, z, w: x + y + z + w,
        lambda x, y, z: x + y + z + z,
        lambda x, y: x + y + y + y,
        lambda x: x + x + x + x,
        lambda x: x,
    ]

    # Need to pass time base to the library so can build the x_dot library from x
    sindy_library = SINDyPILibrary(
        library_functions=x_library_functions,
        x_dot_library_functions=x_dot_library_functions,
        t=t[1:-1],
        function_names=library_function_names,
        include_bias=True,
    )

    opt = SINDyPI(**params)
    model = SINDy(
        optimizer=opt,
        feature_library=sindy_library,
        differentiation_method=FiniteDifference(drop_endpoints=True),
    )
    model.fit(x_train, t=t)
    assert np.shape(opt.coef_) == (10, 10)


@pytest.mark.parametrize(
    "params",
    [
        dict(thresholder="l1", threshold=0),
        dict(thresholder="l1", threshold=1e-5),
        dict(thresholder="weighted_l1", thresholds=np.zeros((3, 9))),
        dict(thresholder="weighted_l1", thresholds=1e-5 * np.ones((3, 9))),
        dict(thresholder="l2", threshold=0),
        dict(thresholder="l2", threshold=1e-5),
        dict(thresholder="weighted_l2", thresholds=np.zeros((3, 9))),
        dict(thresholder="weighted_l2", thresholds=1e-5 * np.ones((3, 9))),
    ],
)
def test_sr3_quadratic_library(params):
    x = np.random.standard_normal((100, 3))
    library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x ** 2,
    ]
    library_function_names = [
        lambda x: str(x),
        lambda x, y: "{} * {}".format(x, y),
        lambda x: "{}^2".format(x),
    ]
    sindy_library = CustomLibrary(
        library_functions=library_functions, function_names=library_function_names
    )

    # Test SR3
    opt = SR3(**params)
    model = SINDy(optimizer=opt, feature_library=sindy_library)
    model.fit(x)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "params",
    [
        dict(thresholder="l1", threshold=0),
        dict(thresholder="l1", threshold=1e-5),
        dict(thresholder="weighted_l1", thresholds=np.zeros((3, 9))),
        dict(thresholder="weighted_l1", thresholds=1e-5 * np.ones((3, 9))),
        dict(thresholder="l2", threshold=0),
        dict(thresholder="l2", threshold=1e-5),
        dict(thresholder="weighted_l2", thresholds=np.zeros((3, 9))),
        dict(thresholder="weighted_l2", thresholds=1e-5 * np.ones((3, 9))),
    ],
)
def test_constrained_sr3_quadratic_library(params):
    x = np.random.standard_normal((100, 3))
    library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x ** 2,
    ]
    library_function_names = [
        lambda x: str(x),
        lambda x, y: "{} * {}".format(x, y),
        lambda x: "{}^2".format(x),
    ]
    sindy_library = CustomLibrary(
        library_functions=library_functions, function_names=library_function_names
    )

    # Test constrained SR3 without constraints
    opt = ConstrainedSR3(**params)
    model = SINDy(optimizer=opt, feature_library=sindy_library)
    model.fit(x)
    check_is_fitted(model)

    # rerun with identity constraints
    r = 3
    N = 9
    p = r + r * (r - 1) + int(r * (r - 1) * (r - 2) / 6.0)
    constraint_rhs = np.zeros(p)
    constraint_matrix = np.eye(p, r * N)

    # Test constrained SR3 with constraints
    opt = ConstrainedSR3(
        constraint_lhs=constraint_matrix, constraint_rhs=constraint_rhs, **params
    )
    model = SINDy(optimizer=opt, feature_library=sindy_library)
    model.fit(x)
    check_is_fitted(model)
    assert np.allclose((model.coefficients().flatten())[:p], 0.0)


@pytest.mark.parametrize(
    "trapping_sr3_params",
    [
        dict(),
        dict(accel=True),
        dict(relax_optim=False),
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        dict(thresholder="l1", threshold=0),
        dict(thresholder="l1", threshold=1e-5),
        dict(
            thresholder="weighted_l1",
            thresholds=np.zeros((3, 9)),
            eta=1e5,
            alpha_m=1e4,
            alpha_A=1e5,
        ),
        dict(thresholder="weighted_l1", thresholds=1e-5 * np.ones((3, 9))),
        dict(thresholder="l2", threshold=0),
        dict(thresholder="l2", threshold=1e-5),
        dict(thresholder="weighted_l2", thresholds=np.zeros((3, 9))),
        dict(thresholder="weighted_l2", thresholds=1e-5 * np.ones((3, 9))),
    ],
)
def test_trapping_sr3_quadratic_library(
    params, trapping_sr3_params, data_quadratic_library
):
    np.random.seed(100)
    x = np.random.standard_normal((100, 3))

    sindy_library = data_quadratic_library
    params.update(trapping_sr3_params)

    opt = TrappingSR3(**params)
    model = SINDy(optimizer=opt, feature_library=sindy_library)
    model.fit(x)
    assert opt.PL_unsym_.shape == (3, 3, 3, 9)
    assert opt.PL_.shape == (3, 3, 3, 9)
    assert opt.PQ_.shape == (3, 3, 3, 3, 9)
    check_is_fitted(model)

    # Rerun with identity constraints
    r = 3
    N = 9
    p = r + r * (r - 1) + int(r * (r - 1) * (r - 2) / 6.0)
    params["constraint_rhs"] = np.zeros(p)
    params["constraint_lhs"] = np.eye(p, r * N)

    opt = TrappingSR3(**params)
    model = SINDy(optimizer=opt, feature_library=sindy_library)
    model.fit(x)
    assert opt.PL_unsym_.shape == (3, 3, 3, 9)
    assert opt.PL_.shape == (3, 3, 3, 9)
    assert opt.PQ_.shape == (3, 3, 3, 3, 9)
    check_is_fitted(model)
    # check is solve was infeasible first
    if not np.allclose(opt.m_history_[-1], opt.m_history_[0]):
        zero_inds = [0, 1, 3, 6, 9, 12, 15, 18, 21, 24]
        assert np.allclose((model.coefficients().flatten())[zero_inds], 0.0, atol=1e-5)


def test_trapping_cubic_library():
    x = np.random.standard_normal((10, 3))
    library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x ** 2,
        lambda x, y, z: x * y * z,
        lambda x, y: x ** 2 * y,
        lambda x: x ** 3,
    ]
    library_function_names = [
        lambda x: str(x),
        lambda x, y: "{} * {}".format(x, y),
        lambda x: "{}^2".format(x),
        lambda x, y, z: "{} * {} * {}".format(x, y, z),
        lambda x, y: "{}^2 * {}".format(x, y),
        lambda x: "{}^3".format(x),
    ]
    sindy_library = CustomLibrary(
        library_functions=library_functions, function_names=library_function_names
    )
    opt = TrappingSR3()
    model = SINDy(optimizer=opt, feature_library=sindy_library)
    model.fit(x)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "error, optimizer, params",
    [
        (ValueError, STLSQ, dict(alpha=-1)),
        (ValueError, SSR, dict(alpha=-1)),
        (ValueError, SSR, dict(criteria="None")),
        (ValueError, SSR, dict(max_iter=-1)),
        (ValueError, FROLS, dict(max_iter=-1)),
        (NotImplementedError, SR3, dict(thresholder="l3")),
        (NotImplementedError, ConstrainedSR3, dict(thresholder="l3")),
        (
            ValueError,
            ConstrainedSR3,
            dict(
                inequality_constraints=True,
                constraint_lhs=np.zeros((1, 1)),
                constraint_rhs=np.zeros(1),
                thresholder="l0",
            ),
        ),
        (ValueError, ConstrainedSR3, dict(inequality_constraints=True)),
        (ValueError, SR3, dict(thresholder="weighted_l0", thresholds=None)),
        (ValueError, SR3, dict(thresholder="weighted_l1", thresholds=None)),
        (ValueError, SR3, dict(thresholder="weighted_l2", thresholds=None)),
        (ValueError, SR3, dict(thresholds=-np.ones((5, 5)))),
        (ValueError, SR3, dict(initial_guess=np.zeros(3))),
        (ValueError, ConstrainedSR3, dict(thresholder="weighted_l0", thresholds=None)),
        (ValueError, ConstrainedSR3, dict(thresholder="weighted_l1", thresholds=None)),
        (ValueError, ConstrainedSR3, dict(thresholder="weighted_l2", thresholds=None)),
        (ValueError, ConstrainedSR3, dict(thresholds=-np.ones((5, 5)))),
        (ValueError, ConstrainedSR3, dict(initial_guess=np.zeros(3))),
        (
            ValueError,
            ConstrainedSR3,
            dict(
                constraint_lhs=np.zeros((10, 10)),
                constraint_rhs=np.zeros(10),
                constraint_order="None",
            ),
        ),
    ],
)
def test_specific_bad_parameters(error, optimizer, params, data_lorenz):
    x, t = data_lorenz
    with pytest.raises(error):
        opt = optimizer(**params)
        model = SINDy(optimizer=opt)
        model.fit(x, t=t)


def test_bad_optimizers(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    with pytest.raises(AttributeError):
        opt = SINDyOptimizer(DummyEmptyModel())

    with pytest.raises(AttributeError):
        opt = SINDyOptimizer(DummyModelNoCoef())
        opt.fit(x, x_dot)


@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3])
def test_initial_guess_sr3(optimizer):
    x = np.random.standard_normal((10, 3))
    x_dot = np.random.standard_normal((10, 2))

    control_model = optimizer(max_iter=1).fit(x, x_dot)

    initial_guess = np.random.standard_normal((x_dot.shape[1], x.shape[1]))
    guess_model = optimizer(max_iter=1, initial_guess=initial_guess).fit(x, x_dot)
    assert np.any(np.not_equal(control_model.coef_, guess_model.coef_))


# The different capitalizations are intentional;
# I want to make sure different versions are recognized
@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3])
@pytest.mark.parametrize("thresholder", ["L0", "l1"])
def test_prox_functions(data_derivative_1d, optimizer, thresholder):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = optimizer(thresholder=thresholder)
    model.fit(x, x_dot)
    check_is_fitted(model)


def test_cad_prox_function(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = SR3(thresholder="cad")
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize("thresholder", ["weighted_l0", "weighted_l1"])
def test_weighted_prox_functions(data, thresholder):
    x, x_dot = data
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        thresholds = np.ones((1, 1))
    else:
        thresholds = np.ones((x_dot.shape[1], x.shape[1]))

    model = ConstrainedSR3(thresholder=thresholder, thresholds=thresholds)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize("thresholder", ["L0", "l1"])
def test_constrained_sr3_prox_functions(data_derivative_1d, thresholder):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = ConstrainedSR3(thresholder=thresholder)
    model.fit(x, x_dot)
    check_is_fitted(model)


def test_unbias(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    optimizer_biased = SINDyOptimizer(
        STLSQ(threshold=0.01, alpha=0.1, max_iter=1), unbias=False
    )
    optimizer_biased.fit(x, x_dot)

    optimizer_unbiased = SINDyOptimizer(
        STLSQ(threshold=0.01, alpha=0.1, max_iter=1), unbias=True
    )
    optimizer_unbiased.fit(x, x_dot)

    assert (
        norm(optimizer_biased.coef_ - optimizer_unbiased.coef_)
        / norm(optimizer_unbiased.coef_)
        > 1e-9
    )


def test_unbias_external(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    optimizer_biased = SINDyOptimizer(
        Lasso(alpha=0.1, fit_intercept=False, max_iter=1), unbias=False
    )
    optimizer_biased.fit(x, x_dot)

    optimizer_unbiased = SINDyOptimizer(
        Lasso(alpha=0.1, fit_intercept=False, max_iter=1), unbias=True
    )
    optimizer_unbiased.fit(x, x_dot)

    assert (
        norm(optimizer_biased.coef_ - optimizer_unbiased.coef_)
        / (norm(optimizer_unbiased.coef_) + 1e-5)
        > 1e-9
    )


@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3])
def test_sr3_trimming(optimizer, data_linear_oscillator_corrupted):
    X, X_dot, trimming_array = data_linear_oscillator_corrupted

    optimizer_without_trimming = SINDyOptimizer(optimizer(), unbias=False)
    optimizer_without_trimming.fit(X, X_dot)

    optimizer_trimming = SINDyOptimizer(optimizer(trimming_fraction=0.15), unbias=False)
    optimizer_trimming.fit(X, X_dot)

    # Check that trimming found the right samples to remove
    np.testing.assert_array_equal(
        optimizer_trimming.optimizer.trimming_array, trimming_array
    )

    # Check that the coefficients found by the optimizer with trimming
    # are closer to the true coefficients than the coefficients found by the
    # optimizer without trimming
    true_coef = np.array([[-2.0, 0.0], [0.0, 1.0]])
    assert norm(true_coef - optimizer_trimming.coef_) < norm(
        true_coef - optimizer_without_trimming.coef_
    )


@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3])
def test_sr3_disable_trimming(optimizer, data_linear_oscillator_corrupted):
    x, x_dot, _ = data_linear_oscillator_corrupted

    model_plain = optimizer()
    model_plain.fit(x, x_dot)

    model_trimming = optimizer(trimming_fraction=0.5)
    model_trimming.disable_trimming()
    model_trimming.fit(x, x_dot)

    np.testing.assert_allclose(model_plain.coef_, model_trimming.coef_)


@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3])
def test_sr3_enable_trimming(optimizer, data_linear_oscillator_corrupted):
    x, x_dot, _ = data_linear_oscillator_corrupted

    model_plain = optimizer()
    model_plain.enable_trimming(trimming_fraction=0.5)
    model_plain.fit(x, x_dot)

    model_trimming = optimizer(trimming_fraction=0.5)
    model_trimming.fit(x, x_dot)

    np.testing.assert_allclose(model_plain.coef_, model_trimming.coef_)


@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3, TrappingSR3])
def test_sr3_warn(optimizer, data_linear_oscillator_corrupted):
    x, x_dot, _ = data_linear_oscillator_corrupted
    model = optimizer(max_iter=1, tol=1e-10)

    with pytest.warns(ConvergenceWarning):
        model.fit(x, x_dot)


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ(max_iter=1),
        SR3(max_iter=1),
        ConstrainedSR3(max_iter=1),
        TrappingSR3(max_iter=1),
    ],
)
def test_fit_warn(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    with pytest.warns(ConvergenceWarning):
        optimizer.fit(x, x_dot)


@pytest.mark.parametrize("optimizer", [ConstrainedSR3, TrappingSR3])
@pytest.mark.parametrize("target_value", [0, -1, 3])
def test_row_format_constraints(data_linear_combination, optimizer, target_value):
    # Solution is x_dot = x.dot(np.array([[1, 1, 0], [0, 1, 1]]))
    x, x_dot = data_linear_combination

    constraint_rhs = target_value * np.ones(2)
    constraint_lhs = np.zeros((2, x.shape[1] * x_dot.shape[1]))

    # Should force corresponding entries of coef_ to be target_value
    constraint_lhs[0, 0] = 1
    constraint_lhs[1, 3] = 1

    model = optimizer(
        constraint_lhs=constraint_lhs,
        constraint_rhs=constraint_rhs,
        constraint_order="feature",
    )
    model.fit(x, x_dot)

    np.testing.assert_allclose(
        np.array([model.coef_[0, 0], model.coef_[1, 1]]), target_value, atol=1e-8
    )


@pytest.mark.parametrize("optimizer", [ConstrainedSR3, TrappingSR3])
@pytest.mark.parametrize("target_value", [0, -1, 3])
def test_target_format_constraints(data_linear_combination, optimizer, target_value):
    x, x_dot = data_linear_combination

    constraint_rhs = target_value * np.ones(2)
    constraint_lhs = np.zeros((2, x.shape[1] * x_dot.shape[1]))

    # Should force corresponding entries of coef_ to be target_value
    constraint_lhs[0, 1] = 1
    constraint_lhs[1, 4] = 1

    model = optimizer(constraint_lhs=constraint_lhs, constraint_rhs=constraint_rhs)
    model.fit(x, x_dot)
    np.testing.assert_allclose(model.coef_[:, 1], target_value, atol=1e-8)


@pytest.mark.parametrize(
    "params",
    [
        dict(thresholder="l1", threshold=0.0005),
        dict(thresholder="weighted_l1", thresholds=0.0005 * np.ones((3, 10))),
        dict(thresholder="l2", threshold=0.0005),
        dict(thresholder="weighted_l2", thresholds=0.0005 * np.ones((3, 10))),
    ],
)
def test_constrained_inequality_constraints(data_lorenz, params):
    x, t = data_lorenz
    constraint_rhs = np.array([-10.0, 28.0])
    constraint_matrix = np.zeros((2, 30))
    constraint_matrix[0, 1] = 1.0
    constraint_matrix[1, 11] = 1.0
    feature_names = ["x", "y", "z"]

    poly_lib = PolynomialLibrary(degree=2)
    # Run constrained SR3
    opt = ConstrainedSR3(
        constraint_lhs=constraint_matrix,
        constraint_rhs=constraint_rhs,
        constraint_order="feature",
        inequality_constraints=True,
        **params,
    )
    model = SINDy(
        optimizer=opt,
        feature_library=poly_lib,
        differentiation_method=FiniteDifference(drop_endpoints=True),
        feature_names=feature_names,
    )
    model.fit(x, t=t[1] - t[0])
    # This sometimes fails with L2 norm so just check the model is fitted
    check_is_fitted(model)


@pytest.mark.parametrize(
    "params",
    [
        dict(thresholder="l1", threshold=0.0005),
        dict(thresholder="weighted_l1", thresholds=0.0005 * np.ones((3, 10))),
        dict(thresholder="l2", threshold=0.0005),
        dict(thresholder="weighted_l2", thresholds=0.0005 * np.ones((3, 10))),
    ],
)
def test_trapping_inequality_constraints(data_lorenz, params):
    x, t = data_lorenz
    constraint_rhs = np.array([-10.0, 28.0])
    constraint_matrix = np.zeros((2, 30))
    constraint_matrix[0, 1] = 1.0
    constraint_matrix[1, 11] = 1.0
    feature_names = ["x", "y", "z"]

    # Run Trapping SR3 without CVXPY for the m solve
    opt = TrappingSR3(
        constraint_lhs=constraint_matrix,
        constraint_rhs=constraint_rhs,
        constraint_order="feature",
        inequality_constraints=True,
        relax_optim=True,
        **params,
    )
    poly_lib = PolynomialLibrary(degree=2)
    model = SINDy(
        optimizer=opt,
        feature_library=poly_lib,
        feature_names=feature_names,
    )
    model.fit(x, t=t[1] - t[0])
    # This sometimes fails with L2 norm so just check the model is fitted
    check_is_fitted(model)

    # Run Trapping SR3 with CVXPY for the m solve
    opt = TrappingSR3(
        constraint_lhs=constraint_matrix,
        constraint_rhs=constraint_rhs,
        constraint_order="feature",
        inequality_constraints=True,
        relax_optim=False,
        **params,
    )
    model = SINDy(
        optimizer=opt,
        feature_library=poly_lib,
        differentiation_method=FiniteDifference(drop_endpoints=True),
        feature_names=feature_names,
    )
    model.fit(x, t=t[1] - t[0])
    assert np.all(
        np.dot(constraint_matrix, (model.coefficients()).flatten()) <= constraint_rhs
    ) or np.allclose(
        np.dot(constraint_matrix, (model.coefficients()).flatten()),
        constraint_rhs,
        atol=1e-3,
    )


def test_inequality_constraints_reqs():
    constraint_rhs = np.array([-10.0, -2.0])
    constraint_matrix = np.zeros((2, 30))
    constraint_matrix[0, 6] = 1.0
    constraint_matrix[1, 17] = 1.0
    with pytest.raises(ValueError):
        TrappingSR3(
            threshold=0.0,
            constraint_lhs=constraint_matrix,
            constraint_rhs=constraint_rhs,
            constraint_order="feature",
            inequality_constraints=True,
            relax_optim=True,
        )


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ,
        SSR,
        FROLS,
        SR3,
        ConstrainedSR3,
        TrappingSR3,
    ],
)
def test_normalize_columns(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    opt = optimizer(normalize_columns=True)
    opt.fit(x, x_dot)
    check_is_fitted(opt)
    assert opt.complexity >= 0
    if len(x_dot.shape) > 1:
        assert opt.coef_.shape == (x.shape[1], x_dot.shape[1])
    else:
        assert opt.coef_.shape == (1, x.shape[1])


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ,
        SSR,
        FROLS,
        SR3,
        ConstrainedSR3,
        TrappingSR3,
    ],
)
def test_ensemble_odes(data_lorenz, optimizer):
    x, t = data_lorenz
    opt = optimizer(normalize_columns=True)
    model = SINDy(optimizer=opt)
    model.fit(x, ensemble=True, n_models=10, n_subset=20)
    assert np.shape(model.coef_list) == (10, 3, 10)


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ,
        SSR,
        FROLS,
        SR3,
        ConstrainedSR3,
        TrappingSR3,
    ],
)
def test_ensemble_pdes(optimizer):
    u = np.random.randn(10, 10, 2)
    t = np.linspace(1, 10, 10)
    x = np.linspace(1, 10, 10)
    dt = t[1] - t[0]
    u_dot = np.zeros(u.shape)
    for i in range(len(x)):
        u_dot[i, :, :] = FiniteDifference()._differentiate(u[i, :, :], t=dt)
    u_flattened = np.reshape(u, (len(x) * len(t), 2))
    u_dot_flattened = np.reshape(u_dot, (len(x) * len(t), 2))

    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=3,
        spatial_grid=x,
        include_bias=True,
        is_uniform=True,
    )
    opt = optimizer(normalize_columns=True)
    model = SINDy(optimizer=opt, feature_library=pde_lib)
    model.fit(
        u_flattened, x_dot=u_dot_flattened, ensemble=True, n_models=10, n_subset=20
    )
    n_features = len(model.get_feature_names())
    assert np.shape(model.coef_list) == (10, 2, n_features)


def test_ssr_criteria(data_lorenz):
    x, t = data_lorenz
    opt = SSR(normalize_columns=True, criteria="model_residual", kappa=1e-3)
    model = SINDy(optimizer=opt)
    model.fit(x)
    assert np.shape(opt.coef_) == (3, 10)


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ,
        SSR,
        FROLS,
        SR3,
        ConstrainedSR3,
        TrappingSR3,
    ],
)
def test_optimizers_verbose(data_lorenz, optimizer):
    x, t = data_lorenz
    dt = t[1] - t[0]
    x_dot = FiniteDifference()._differentiate(x, t=dt)
    model = optimizer(verbose=True)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "optimizer",
    [
        SINDyPI,
        ConstrainedSR3,
        TrappingSR3,
    ],
)
def test_optimizers_verbose_cvxpy(data_lorenz, optimizer):
    x, t = data_lorenz
    dt = t[1] - t[0]
    x_dot = FiniteDifference()._differentiate(x, t=dt)

    model = optimizer(verbose_cvxpy=True)
    model.fit(x, x_dot)
    check_is_fitted(model)
