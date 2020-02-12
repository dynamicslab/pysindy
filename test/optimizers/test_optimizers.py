"""
Unit tests for optimizers.
"""
import pytest
from numpy.linalg import norm
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted

from pysindy.optimizers import SINDyOptimizer
from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ


@pytest.mark.parametrize(
    "optimizer",
    [STLSQ(), SR3(), Lasso(fit_intercept=False), ElasticNet(fit_intercept=False)],
)
def test_fit(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    opt = SINDyOptimizer(optimizer, unbias=False)
    opt.fit(x, x_dot)

    check_is_fitted(opt)
    assert opt.complexity >= 0


@pytest.mark.parametrize(
    "kwargs", [{"normalize": True}, {"fit_intercept": True}, {"copy_X": False}]
)
def test_alternate_parameters(data_derivative_1d, kwargs):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    model = STLSQ(**kwargs)
    model.fit(x, x_dot)
    model.fit(x, x_dot, sample_weight=x[:, 0])

    check_is_fitted(model)


def test_bad_parameters(data_derivative_1d):
    x, x_dot = data_derivative_1d

    with pytest.raises(ValueError):
        STLSQ(threshold=-1)

    with pytest.raises(ValueError):
        STLSQ(alpha=-1)

    with pytest.raises(ValueError):
        STLSQ(max_iter=0)

    with pytest.raises(ValueError):
        SR3(threshold=-1)

    with pytest.raises(ValueError):
        SR3(nu=0)

    with pytest.raises(ValueError):
        SR3(tol=0)

    with pytest.raises(NotImplementedError):
        SR3(thresholder="l2")

    with pytest.raises(ValueError):
        SR3(max_iter=0)


# The different capitalizations are intentional;
# I want to make sure different versions are recognized
@pytest.mark.parametrize("thresholder", ["L0", "l1", "CAD"])
def test_sr3_prox_functions(data_derivative_1d, thresholder):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    model = SR3(thresholder=thresholder)
    model.fit(x, x_dot)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "optimizer",
    [
        STLSQ(threshold=0.01, alpha=0.1, max_iter=1),
        Lasso(alpha=0.1, fit_intercept=False, max_iter=1),
    ],
)
def test_unbias(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)

    optimizer_biased = SINDyOptimizer(optimizer, unbias=False)
    optimizer_biased.fit(x, x_dot)

    optimizer_unbiased = SINDyOptimizer(optimizer, unbias=True)
    optimizer_unbiased.fit(x, x_dot)

    assert (
        norm(optimizer_biased.coef_ - optimizer_unbiased.coef_)
        / norm(optimizer_unbiased.coef_)
        > 1e-9
    )
