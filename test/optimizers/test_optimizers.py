"""
Unit tests for optimizers.
"""

import pytest

from pysindy.optimizers import STLSQ, SR3, LASSO, ElasticNet


@pytest.mark.parametrize("optimizer", [STLSQ(), SR3(), LASSO(), ElasticNet()])
def test_fit(data_derivative_1d, optimizer):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    x_dot = x_dot.reshape(-1)
    optimizer.fit(x, x_dot)


def test_alternate_parameters(data_derivative_1d):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    x_dot = x_dot.reshape(-1)

    model = STLSQ(normalize=True)
    model.fit(x, x_dot)

    model = STLSQ(fit_intercept=True)
    model.fit(x, x_dot)

    model = STLSQ(copy_X=False)
    model.fit(x, x_dot)


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

    with pytest.raises(ValueError):
        LASSO(alpha=-1)

    with pytest.raises(ValueError):
        LASSO(max_iter=0)

    with pytest.raises(ValueError):
        ElasticNet(alpha=-1)

    with pytest.raises(ValueError):
        ElasticNet(max_iter=0)

    with pytest.raises(ValueError):
        ElasticNet(l1_ratio=-0.1)


# The different capitalizations are intentional;
# I want to make sure different versions are recognized
@pytest.mark.parametrize("thresholder", ["L0", "l1", "CAD"])
def test_sr3_prox_functions(data_derivative_1d, thresholder):
    x, x_dot = data_derivative_1d
    x = x.reshape(-1, 1)
    x_dot = x_dot.reshape(-1)
    model = SR3(thresholder=thresholder)
    model.fit(x, x_dot)
