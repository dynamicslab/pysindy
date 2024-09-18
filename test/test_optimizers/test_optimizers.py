"""
Unit tests for optimizers.
"""
import pickle

import cvxpy as cp
import numpy as np
import pytest
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.validation import check_is_fitted

from pysindy import FiniteDifference
from pysindy import PolynomialLibrary
from pysindy import SINDy
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.optimizers import BaseOptimizer
from pysindy.optimizers import ConstrainedSR3
from pysindy.optimizers import EnsembleOptimizer
from pysindy.optimizers import FROLS
from pysindy.optimizers import MIOSR
from pysindy.optimizers import SBR
from pysindy.optimizers import SINDyPI
from pysindy.optimizers import SR3
from pysindy.optimizers import SSR
from pysindy.optimizers import StableLinearSR3
from pysindy.optimizers import STLSQ
from pysindy.optimizers import TrappingSR3
from pysindy.optimizers import WrappedOptimizer
from pysindy.optimizers.stlsq import _remove_and_decrement
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


def _align_optimizer_and_1dfeatures(
    opt: BaseOptimizer, features: NDArray
) -> tuple[BaseOptimizer, NDArray]:
    # This is a hack until constraints are moved from init to fit
    if isinstance(opt, TrappingSR3):
        opt = TrappingSR3(_n_tgts=1, _include_bias=False)
        features = np.hstack([features, features])
    else:
        features = features
    return opt, features


@pytest.mark.parametrize(
    "cls, support",
    [
        (Lasso, True),
        (STLSQ, True),
        (SSR, True),
        (FROLS, True),
        (SR3, True),
        (ConstrainedSR3, True),
        (StableLinearSR3, True),
        (TrappingSR3, True),
        (DummyLinearModel, False),
        (SBR, True),
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
        StableLinearSR3(),
        TrappingSR3(_n_tgts=1),
        Lasso(fit_intercept=False),
        ElasticNet(fit_intercept=False),
        DummyLinearModel(),
        MIOSR(),
        SBR(),
    ],
    ids=lambda param: type(param),
)
def test_fit(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    optimizer, x = _align_optimizer_and_1dfeatures(optimizer, x)
    opt = WrappedOptimizer(optimizer, unbias=False)
    opt.fit(x, x_dot)

    check_is_fitted(opt)
    assert opt.complexity >= 0
    if len(x_dot.shape) > 1:
        assert opt.coef_.shape == (x.shape[1], x_dot.shape[1])
    else:
        assert opt.coef_.shape == (1, x.shape[1])


@pytest.mark.parametrize(
    "optimizer",
    [STLSQ(), SSR(), SSR(criteria="model_residual"), FROLS(), SR3(), MIOSR(), SBR()],
)
def test_not_fitted(optimizer):
    with pytest.raises(NotFittedError):
        optimizer.predict(np.ones((1, 3)))


@pytest.mark.parametrize("optimizer", [STLSQ(), SR3(), SBR()])
def test_complexity_not_fitted(optimizer, data_derivative_2d):
    with pytest.raises(NotFittedError):
        optimizer.complexity

    x, _ = data_derivative_2d
    optimizer.fit(x, x)
    assert optimizer.complexity > 0


@pytest.mark.parametrize("kwargs", [{"normalize_columns": True}, {"copy_X": False}])
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
        StableLinearSR3,
        TrappingSR3,
        MIOSR,
    ],
)
def test_sample_weight_optimizers(data_1d, optimizer):
    y, t = data_1d
    opt = optimizer()
    opt, x = _align_optimizer_and_1dfeatures(opt, y)
    sample_weight = np.ones(x[:, 0].shape)
    sample_weight[::2] = 0
    opt.fit(x, y, sample_weight=sample_weight)
    check_is_fitted(opt)


@pytest.mark.parametrize("params", [dict(threshold=-1), dict(max_iter=0)])
def test_STLSQ_bad_parameters(params):
    with pytest.raises(ValueError):
        STLSQ(**params)


@pytest.mark.parametrize(
    "optimizer", [SR3, ConstrainedSR3, StableLinearSR3, TrappingSR3]
)
@pytest.mark.parametrize("params", [dict(reg_weight_lam=-1), dict(max_iter=0)])
def test_general_bad_parameters(optimizer, params):
    with pytest.raises(ValueError):
        optimizer(**params)


@pytest.mark.parametrize("optimizer", [SR3, ConstrainedSR3, StableLinearSR3])
@pytest.mark.parametrize(
    "params",
    [
        dict(relax_coeff_nu=0),
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
        dict(regularizer="l0"),
        dict(max_iter=0),
        dict(eta=10, alpha_m=20),
        dict(eta=10, alpha_A=20),
        dict(
            constraint_lhs=np.zeros((10, 10)),
            constraint_rhs=np.zeros(10),
            constraint_order="None",
        ),
    ],
)
def test_trapping_bad_parameters(params):
    with pytest.raises(ValueError):
        TrappingSR3(_n_tgts=1, **params)


def test_trapping_objective_print():
    # test error in verbose print logic when max_iter < 10
    opt = TrappingSR3(_n_tgts=1, max_iter=2, verbose=True)
    arr = np.ones((1, 1))
    opt.PQ_ = np.ones((1, 1, 1, 1, 1))
    opt._objective(arr, arr, arr, arr, arr, 1)


@pytest.mark.parametrize(
    "params",
    [
        dict(tol=0),
        dict(max_iter=-1),
        dict(regularizer="l0"),
        dict(reg_weight_lam=-1),
        dict(regularizer="weighted_l1", reg_weight_lam=1),
        dict(regularizer="weighted_l1"),
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
        dict(regularizer="l1"),
        dict(regularizer="weighted_l1", reg_weight_lam=np.zeros((10, 10))),
        dict(regularizer="l2"),
        dict(regularizer="weighted_l2", reg_weight_lam=np.zeros((10, 10))),
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
        lambda x: x**2,
        lambda x, y, z: x * y * z,
        lambda x, y: x * y**2,
        lambda x: x**3,
        lambda x, y, z, w: x * y * z * w,
        lambda x, y, z: x * y * z**2,
        lambda x, y: x * y**3,
        lambda x: x**4,
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
        t=t,
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
        dict(sparsity_coef_tau0=0),
        dict(sparsity_coef_tau0=-1),
        dict(slab_shape_nu=0),
        dict(slab_shape_nu=-1),
        dict(slab_shape_s=0),
        dict(slab_shape_s=-1),
        dict(noise_hyper_lambda=0),
        dict(noise_hyper_lambda=-1),
        dict(num_warmup=0.5),
        dict(num_warmup=-1),
        dict(num_samples=0.5),
        dict(num_samples=-1),
    ],
)
def test_sbr_bad_parameters(params):
    with pytest.raises(ValueError):
        SBR(**params)


def test_sbr_accurate():
    # It's really hard to tune SBR to get desired shrinkage
    # This just tests that SBR fits "close" to unregularized regression
    x = np.tile(np.eye(2), 4).reshape((-1, 2))
    y = np.tile([[1], [1e-1]], 4).reshape((-1, 1))
    opt = SBR(num_warmup=50, num_samples=50).fit(x, y)
    result = opt.coef_
    unregularized = np.array([[1, 1e-1]])
    np.testing.assert_allclose(result, unregularized, atol=1e-3)
    assert hasattr(opt, "mcmc_")
    expected_names = ["beta", "c_sq", "lambda", "sigma", "tau"]
    result_names = opt.mcmc_.get_samples().keys()
    assert all(expected in result_names for expected in expected_names)


@pytest.mark.parametrize(
    "params",
    [
        dict(regularizer="l1", reg_weight_lam=0),
        dict(regularizer="l1", reg_weight_lam=1e-5),
        dict(regularizer="weighted_l1", reg_weight_lam=np.zeros((3, 9))),
        dict(regularizer="weighted_l1", reg_weight_lam=1e-5 * np.ones((3, 9))),
        dict(regularizer="l2", reg_weight_lam=0),
        dict(regularizer="l2", reg_weight_lam=1e-5),
        dict(regularizer="weighted_l2", reg_weight_lam=np.zeros((3, 9))),
        dict(regularizer="weighted_l2", reg_weight_lam=1e-5 * np.ones((3, 9))),
    ],
)
def test_sr3_quadratic_library(params):
    x = np.random.standard_normal((100, 3))
    library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x**2,
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
        dict(regularizer="l1", reg_weight_lam=0),
        dict(regularizer="l1", reg_weight_lam=1e-5),
        dict(regularizer="weighted_l1", reg_weight_lam=np.zeros((3, 9))),
        dict(regularizer="weighted_l1", reg_weight_lam=1e-5 * np.ones((3, 9))),
        dict(regularizer="l2", reg_weight_lam=0),
        dict(regularizer="l2", reg_weight_lam=1e-5),
        dict(regularizer="weighted_l2", reg_weight_lam=np.zeros((3, 9))),
        dict(regularizer="weighted_l2", reg_weight_lam=1e-5 * np.ones((3, 9))),
    ],
)
def test_constrained_sr3_quadratic_library(params):
    x = np.random.standard_normal((100, 3))
    library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x**2,
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
    ["regularizer", "lam", "expected"],
    [
        ("l1", np.array([[2]]), 20),
        ("weighted_l1", np.array([[3, 2, 0.5]]).T, 14.5),
        ("l2", np.array([[2]]), 76),
        ("weighted_l2", np.array([[3, 2, 0.5]]).T, 42.5),
    ],
)
def test_constrained_sr3_penalty_term(regularizer, lam, expected):
    xi = cp.Variable(3)
    penalty = ConstrainedSR3._calculate_penalty(regularizer, np.ravel(lam), xi)
    xi.value = np.array([-2, 3, 5])
    np.testing.assert_allclose(penalty.value, expected)


def test_stable_linear_sr3_linear_library():
    x = np.ones((2, 1))
    opt = StableLinearSR3()
    opt.fit(x, x)
    check_is_fitted(opt)

    constraint_rhs = np.zeros((1, 1))
    constraint_matrix = np.eye(1)
    opt = StableLinearSR3(
        constraint_lhs=constraint_matrix, constraint_rhs=constraint_rhs
    )
    opt.fit(x, x)
    check_is_fitted(opt)
    assert np.allclose(opt.coef_.flatten(), 0.0)


@pytest.mark.parametrize("bias", (True, False))
@pytest.mark.parametrize("method", ("global", "local"))
@pytest.mark.parametrize("reg_weight", (0.0, 1e-1))
def test_trapping_sr3_quadratic_library(bias, method, reg_weight):
    t = np.arange(0, 1, 0.1)
    x = np.exp(-t).reshape((-1, 1))
    x_dot = -x
    features = np.hstack([x, x**2])
    if bias:
        features = np.hstack([np.ones_like(x), features])

    params = {
        "_n_tgts": 1,
        "_include_bias": bias,
        "method": method,
        "reg_weight_lam": reg_weight,
    }

    opt = TrappingSR3(**params)
    opt.fit(features, x_dot)
    check_is_fitted(opt)

    # Rerun with identity constraints
    r = x.shape[1]
    N = 2 + bias
    params["constraint_rhs"] = np.zeros(r * N)
    params["constraint_lhs"] = np.eye(r * N, r * N)

    opt = TrappingSR3(**params)
    opt.fit(features, x_dot)
    check_is_fitted(opt)
    # check if solve was infeasible first
    if not np.allclose(opt.m_history_[-1], opt.m_history_[0]):
        assert np.allclose((opt.coef_.flatten())[0], 0.0, atol=1e-5)


@pytest.mark.parametrize(
    "error, optimizer, params",
    [
        (ValueError, STLSQ, dict(alpha=-1)),
        (ValueError, SSR, dict(alpha=-1)),
        (ValueError, SSR, dict(criteria="None")),
        (ValueError, SSR, dict(max_iter=-1)),
        (ValueError, FROLS, dict(max_iter=-1)),
        (NotImplementedError, SR3, dict(regularizer="l3")),
        (NotImplementedError, ConstrainedSR3, dict(regularizer="l3")),
        (NotImplementedError, StableLinearSR3, dict(regularizer="l3")),
        (
            ValueError,
            ConstrainedSR3,
            dict(
                inequality_constraints=True,
                constraint_lhs=np.zeros((1, 1)),
                constraint_rhs=np.zeros(1),
                regularizer="l0",
            ),
        ),
        (ValueError, ConstrainedSR3, dict(inequality_constraints=True)),
        (ValueError, SR3, dict(regularizer="weighted_l0", reg_weight_lam=None)),
        (ValueError, SR3, dict(regularizer="weighted_l1", reg_weight_lam=None)),
        (ValueError, SR3, dict(regularizer="weighted_l2", reg_weight_lam=None)),
        (ValueError, SR3, dict(reg_weight_lam=-np.ones((5, 5)))),
        (ValueError, SR3, dict(initial_guess=np.zeros(3))),
        (
            ValueError,
            ConstrainedSR3,
            dict(regularizer="weighted_l0", reg_weight_lam=None),
        ),
        (
            ValueError,
            ConstrainedSR3,
            dict(regularizer="weighted_l1", reg_weight_lam=None),
        ),
        (
            ValueError,
            ConstrainedSR3,
            dict(regularizer="weighted_l2", reg_weight_lam=None),
        ),
        (ValueError, ConstrainedSR3, dict(reg_weight_lam=-np.ones((5, 5)))),
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
    x_dot = x_dot.reshape(-1, 1)
    with pytest.raises(InvalidParameterError):
        # Error: optimizer does not have a callable fit method
        opt = WrappedOptimizer(DummyEmptyModel())
        opt.fit(x, x_dot)

    with pytest.raises(AttributeError):
        # Error: object has no attribute 'coef_'
        opt = WrappedOptimizer(DummyModelNoCoef())
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
@pytest.mark.parametrize("regularizer", ["L0", "l1"])
def test_prox_functions(data_derivative_1d, optimizer, regularizer):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = optimizer(regularizer=regularizer)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize("regularizer", ["weighted_l0", "weighted_l1"])
def test_weighted_prox_functions(data, regularizer):
    x, x_dot = data
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        reg_weight_lam = np.ones((1, 1))
    else:
        reg_weight_lam = np.ones((x_dot.shape[1], x.shape[1]))

    model = ConstrainedSR3(regularizer=regularizer, reg_weight_lam=reg_weight_lam)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize("regularizer", ["L0", "l1"])
def test_constrained_sr3_prox_functions(data_derivative_1d, regularizer):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = ConstrainedSR3(regularizer=regularizer)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize(
    ("opt_cls", "opt_args"),
    (
        (SR3, {"trimming_fraction": 0.1}),
        (ConstrainedSR3, {"constraint_lhs": [1], "constraint_rhs": [1]}),
        (ConstrainedSR3, {"trimming_fraction": 0.1}),
        (TrappingSR3, {"_n_tgts": 1, "constraint_lhs": [1], "constraint_rhs": [1]}),
        (StableLinearSR3, {"constraint_lhs": [1], "constraint_rhs": [1]}),
        (StableLinearSR3, {"trimming_fraction": 0.1}),
        (SINDyPI, {}),
        (MIOSR, {"constraint_lhs": [1]}),
        (SBR, {}),
    ),
)
def test_illegal_unbias(data_derivative_1d, opt_cls, opt_args):
    x, x_dot = data_derivative_1d
    with pytest.raises(ValueError):
        opt_cls(unbias=True, **opt_args).fit(x, x_dot)


def test_unbias(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    x_dot = x_dot.reshape(-1, 1)

    optimizer_biased = STLSQ(threshold=0.01, alpha=0.1, max_iter=1, unbias=False)
    optimizer_biased.fit(x, x_dot)

    optimizer_unbiased = STLSQ(threshold=0.01, alpha=0.1, max_iter=1, unbias=True)
    optimizer_unbiased.fit(x, x_dot)

    assert (
        norm(optimizer_biased.coef_ - optimizer_unbiased.coef_)
        / norm(optimizer_unbiased.coef_)
        > 1e-9
    )


def test_unbias_external(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    x_dot = x_dot.reshape(-1, 1)

    optimizer_biased = WrappedOptimizer(
        Lasso(alpha=0.1, fit_intercept=False, max_iter=1), unbias=False
    )
    optimizer_biased.fit(x, x_dot)

    optimizer_unbiased = WrappedOptimizer(
        Lasso(alpha=0.1, fit_intercept=False, max_iter=1), unbias=True
    )
    optimizer_unbiased.fit(x, x_dot)

    assert (
        norm(optimizer_biased.coef_ - optimizer_unbiased.coef_)
        / (norm(optimizer_unbiased.coef_) + 1e-5)
        > 1e-9
    )


@pytest.mark.parametrize("OptCls", [SR3, ConstrainedSR3])
def test_sr3_trimming(OptCls, data_linear_oscillator_corrupted):
    X, X_dot, trimming_array = data_linear_oscillator_corrupted

    optimizer_without_trimming = OptCls(unbias=False)
    optimizer_without_trimming.fit(X, X_dot)

    optimizer_trimming = OptCls(trimming_fraction=0.15, unbias=False)
    optimizer_trimming.fit(X, X_dot)

    # Check that trimming found the right samples to remove
    np.testing.assert_array_equal(optimizer_trimming.trimming_array, trimming_array)

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


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ(max_iter=1),
        SR3(max_iter=1),
        ConstrainedSR3(max_iter=1),
        StableLinearSR3(max_iter=1),
        TrappingSR3(_n_tgts=1, max_iter=1),
    ],
)
def test_fit_warn(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    optimizer.max_iter = 0  # normally prohibited in constructor

    with pytest.warns(ConvergenceWarning):
        optimizer.fit(x, x_dot)


@pytest.mark.parametrize(
    "optimizer",
    [
        (ConstrainedSR3, {"max_iter": 80}),
        (TrappingSR3, {"_n_tgts": 3, "max_iter": 100, "eps_solver": 1e-5}),
        (MIOSR, {}),
    ],
    ids=lambda param: param[0].__name__ + " " + ",".join([key for key in param[1]]),
)
@pytest.mark.parametrize("target_value", [0, -1, 3])
def test_feature_format_constraints(data_linear_combination, optimizer, target_value):
    # Solution is x_dot = x.dot(np.array([[1, 1, 0], [0, 1, 1]]))
    x, y = data_linear_combination

    constraint_rhs = target_value * np.ones(2)
    constraint_lhs = np.zeros((2, x.shape[1], y.shape[1]))

    # Should force corresponding entries of coef_ to be target_value
    constraint_lhs[0, 1, 1] = 1
    constraint_lhs[1, 2, 2] = 1
    # reshape to "feature" order
    constraint_lhs = np.reshape(constraint_lhs, (constraint_lhs.shape[0], -1))

    model = optimizer[0](
        constraint_lhs=constraint_lhs,
        constraint_rhs=constraint_rhs,
        constraint_order="feature",
        **optimizer[1],
    )
    model.fit(x, y)

    np.testing.assert_allclose(
        np.array([model.coef_[1, 1], model.coef_[2, 2]]), target_value, atol=1e-7
    )


@pytest.mark.parametrize(
    "optimizer",
    [
        (ConstrainedSR3, {"max_iter": 80}),
        (StableLinearSR3, {}),
        (TrappingSR3, {"_n_tgts": 3, "max_iter": 200, "eps_solver": 1e-5}),
        (MIOSR, {}),
    ],
    ids=lambda param: param[0].__name__ + " " + ",".join([key for key in param[1]]),
)
@pytest.mark.parametrize("target_value", [0, -1, 3])
def test_target_format_constraints(data_linear_combination, optimizer, target_value):
    x, y = data_linear_combination

    constraint_rhs = target_value * np.ones(2)
    constraint_lhs = np.zeros((2, x.shape[1], y.shape[1]))

    # Should force corresponding entries of coef_ to be target_value
    constraint_lhs[0, 2, 1] = 1
    constraint_lhs[1, 1, 2] = 1
    # reshape to "target" order
    constraint_lhs = np.reshape(
        np.transpose(constraint_lhs, [0, 2, 1]), (constraint_lhs.shape[0], -1)
    )

    model = optimizer[0](
        constraint_lhs=constraint_lhs,
        constraint_rhs=constraint_rhs,
        constraint_order="target",
        **optimizer[1],
    )
    model.fit(x, y)

    np.testing.assert_allclose(
        np.array([model.coef_[1, 2], model.coef_[2, 1]]), target_value, atol=1e-7
    )


@pytest.mark.parametrize(
    "params",
    [
        dict(regularizer="l1", reg_weight_lam=0.0005),
        dict(regularizer="weighted_l1", reg_weight_lam=0.0005 * np.ones((3, 10))),
        dict(regularizer="l2", reg_weight_lam=0.0005),
        dict(regularizer="weighted_l2", reg_weight_lam=0.0005 * np.ones((3, 10))),
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
        dict(regularizer="l1", reg_weight_lam=2, expected=2.5),
        dict(
            regularizer="weighted_l1",
            reg_weight_lam=0.5 * np.ones((1, 2)),
            expected=1.0,
        ),
        dict(regularizer="l2", reg_weight_lam=2, expected=1.5),
        dict(
            regularizer="weighted_l2",
            reg_weight_lam=0.5 * np.ones((1, 2)),
            expected=0.75,
        ),
        dict(regularizer="l1", reg_weight_lam=0, expected=0.5),
        dict(
            regularizer="weighted_l1",
            reg_weight_lam=0.0 * np.ones((1, 2)),
            expected=0.5,
        ),
        dict(regularizer="l2", reg_weight_lam=0.0, expected=0.5),
        dict(
            regularizer="weighted_l2",
            reg_weight_lam=0.0 * np.ones((1, 2)),
            expected=0.5,
        ),
    ],
    ids=lambda d: d["regularizer"],
)
def test_trapping_cost_function(params):
    # TODO: are all these parameters necessary?  What are we testing?
    expected = params.pop("expected")
    opt = TrappingSR3(**params)
    x = np.eye(2)
    y = np.ones(2)
    xi, cost = opt._create_var_and_part_cost(2, x, y)
    xi.value = np.array([0.5, 0.5])
    np.testing.assert_allclose(cost.value, expected)


def test_trapping_inequality_constraints():
    t = np.arange(0, 1, 0.1)
    x = np.stack((t, t**2)).T
    y = x[:, 0] + 0.1 * x[:, 1]
    constraint_rhs = np.array([0.1])
    constraint_matrix = np.zeros((1, 2))
    constraint_matrix[0, 1] = 0.1

    # Run Trapping SR3
    opt = TrappingSR3(
        constraint_lhs=constraint_matrix,
        constraint_rhs=constraint_rhs,
        constraint_order="feature",
        inequality_constraints=True,
    )
    opt.fit(x, y)
    assert np.all(np.dot(constraint_matrix, (opt.coef_).flatten()) <= constraint_rhs)


@pytest.mark.parametrize(
    "params",
    [
        dict(target_sparsity=2),
        dict(target_sparsity=7),
    ],
)
def test_miosr_equality_constraints(data_lorenz, params):
    x, t = data_lorenz
    constraint_rhs = np.array([-10.0, 28.0])
    constraint_matrix = np.zeros((2, 30))
    constraint_matrix[0, 1] = 1.0
    constraint_matrix[1, 11] = 1.0
    feature_names = ["x", "y", "z"]

    opt = MIOSR(
        constraint_lhs=constraint_matrix,
        constraint_rhs=constraint_rhs,
        constraint_order="target",
        **params,
    )
    poly_lib = PolynomialLibrary(degree=2)
    model = SINDy(
        optimizer=opt,
        feature_library=poly_lib,
        feature_names=feature_names,
    )
    model.fit(x, t=t[1] - t[0])
    assert np.allclose(
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
            reg_weight_lam=0.0,
            constraint_lhs=constraint_matrix,
            constraint_rhs=constraint_rhs,
            constraint_order="feature",
            inequality_constraints=True,
        )


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ,
        SSR,
        FROLS,
        SR3,
        ConstrainedSR3,
        StableLinearSR3,
        TrappingSR3,
        MIOSR,
        SBR,
    ],
)
def test_normalize_columns(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    opt = optimizer(normalize_columns=True)
    opt, x = _align_optimizer_and_1dfeatures(opt, x)
    opt.fit(x, x_dot)
    check_is_fitted(opt)
    assert opt.complexity >= 0
    if len(x_dot.shape) > 1:
        assert opt.coef_.shape == (x.shape[1], x_dot.shape[1])
    else:
        assert opt.coef_.shape == (1, x.shape[1])


@pytest.mark.parametrize(
    "optimizer_params",
    (
        {"library_ensemble": True, "n_models": 2},
        {"bagging": True, "n_models": 2, "n_subset": 2},
        {"library_ensemble": True, "bagging": True, "n_models": 2, "n_subset": 2},
    ),
)
def test_ensemble_optimizer(data_lorenz, optimizer_params):
    x, t = data_lorenz
    optimizer = EnsembleOptimizer(STLSQ(), **optimizer_params)
    optimizer.fit(x, x)
    assert optimizer.coef_.shape == (3, 3)
    assert len(optimizer.coef_list) == 2


@pytest.mark.parametrize(
    "params",
    [
        dict(),
        dict(bagging=True, n_models=0),
        dict(bagging=True, n_subset=0),
        dict(library_ensemble=True, n_candidates_to_drop=0),
    ],
)
def test_bad_ensemble_params(data_lorenz, params):
    with pytest.raises(ValueError):
        EnsembleOptimizer(opt=STLSQ(), **params)


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
        StableLinearSR3,
        TrappingSR3,
        MIOSR,
    ],
)
def test_optimizers_verbose(data_1d, optimizer):
    y, _ = data_1d
    opt = optimizer(verbose=True)
    opt, x = _align_optimizer_and_1dfeatures(opt, y)
    opt.verbose = True
    opt.fit(x, y)
    check_is_fitted(opt)


@pytest.mark.parametrize(
    "optimizer",
    [
        SINDyPI,
        ConstrainedSR3,
        StableLinearSR3,
        TrappingSR3,
    ],
)
def test_optimizers_verbose_cvxpy(data_1d, optimizer):
    y, _ = data_1d
    opt = optimizer(verbose_cvxpy=True)
    opt, x = _align_optimizer_and_1dfeatures(opt, y)
    opt.fit(x, y)
    check_is_fitted(opt)


def test_frols_error_linear_dependence():
    opt = FROLS(normalize_columns=True)
    x = np.array([[1.0, 1.0]])
    y = np.array([[1.0, 1.0]])
    with pytest.raises(ValueError):
        opt.fit(x, y)


def test_sparse_subset_multitarget():
    A = np.eye(4)
    b = np.array([[1, 1, 0.5, 1], [1, 1, 1, 0.5]]).T
    opt = STLSQ(unbias=False, threshold=0.5, alpha=0.1, sparse_ind=[2, 3])
    opt.fit(A, b)
    X = opt.coef_
    Y = opt.optvar_non_sparse_
    assert X[0, 0] == 0.0
    assert 0.0 < X[0, 1] < 1.0
    np.testing.assert_equal(Y[:, :2], np.ones((2, 2)))
    assert X[1, 1] == 0.0
    assert 0.0 < X[1, 0] < 1.0


def test_sparse_subset_off_diagonal():
    A = np.array([[1, 1], [0, 1]])
    b = np.array([1, 1])
    opt = STLSQ(unbias=False, threshold=0.1, alpha=0.1, sparse_ind=[1])
    opt.fit(A, b)
    X = opt.coef_
    Y = opt.optvar_non_sparse_
    assert Y[0, 0] > 0.0 and Y[0, 0] < 0.5
    assert X[0, 0] > 0.5 and X[0, 0] < 1.0


def test_sparse_subset_unbias():
    A = np.array([[1, 1], [0, 1]])
    b = np.array([1, 1])
    opt = STLSQ(unbias=True, threshold=0.1, alpha=0.1, sparse_ind=[1])
    opt.fit(A, b)
    X = opt.coef_
    Y = opt.optvar_non_sparse_
    assert np.abs(Y[0, 0]) < 2e-16
    assert np.abs(X[0, 0] - 1.0) < 2e-16


def test_remove_and_decrement():
    existing_vals = np.array([2, 3, 4, 5])
    vals_to_remove = np.array([3, 5])
    expected = np.array([2, 3])
    result = _remove_and_decrement(
        existing_vals=existing_vals, vals_to_remove=vals_to_remove
    )
    np.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize(
    ("opt_cls", "opt_args"),
    (
        (MIOSR, {"target_sparsity": 7}),
        (SBR, {"num_warmup": 10, "num_samples": 10}),
    ),
)
def test_pickle(data_lorenz, opt_cls, opt_args):
    x, t = data_lorenz
    y = PolynomialLibrary(degree=2).fit_transform(x)
    opt = opt_cls(**opt_args).fit(x, y)
    expected = opt.coef_
    new_opt = pickle.loads(pickle.dumps(opt))
    result = new_opt.coef_
    np.testing.assert_array_equal(result, expected)
