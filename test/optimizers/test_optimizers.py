import sys
import os
import pytest
import numpy as np

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../../')
from sindy.optimizers import STLSQ, SR3, LASSO, ElasticNet


@pytest.fixture
def data_1d():
    x = 2 * np.linspace(1, 100, 100).reshape(-1, 1)
    x_dot = 2 * np.ones(100)
    return x, x_dot


@pytest.mark.parametrize(
    'optimizer',
    [
        STLSQ(),
        SR3(),
        LASSO(),
        ElasticNet(),
    ]
)
def test_fit(data_1d, optimizer):
    x, x_dot = data_1d
    optimizer.fit(x, x_dot)


def test_alternate_parameters(data_1d):
    x, x_dot = data_1d

    model = STLSQ(normalize=True)
    model.fit(x, x_dot)

    model = STLSQ(fit_intercept=True)
    model.fit(x, x_dot)

    model = STLSQ(copy_X=False)
    model.fit(x, x_dot)


def test_bad_parameters(data_1d):
    x, x_dot = data_1d

    with pytest.raises(ValueError):
        model = STLSQ(threshold=-1)

    with pytest.raises(ValueError):
        model = STLSQ(alpha=-1)

    with pytest.raises(ValueError):
        model = STLSQ(max_iter=0)

    with pytest.raises(ValueError):
        model = SR3(threshold=-1)

    with pytest.raises(ValueError):
        model = SR3(nu=0)

    with pytest.raises(ValueError):
        model = SR3(tol=0)

    with pytest.raises(NotImplementedError):
        model = SR3(thresholder='l2')

    with pytest.raises(ValueError):
        model = SR3(max_iter=0)

    with pytest.raises(ValueError):
        model = LASSO(alpha=-1)

    with pytest.raises(ValueError):
        model = LASSO(max_iter=0)

    with pytest.raises(ValueError):
        model = ElasticNet(alpha=-1)

    with pytest.raises(ValueError):
        model = ElasticNet(max_iter=0)

    with pytest.raises(ValueError):
        model = ElasticNet(l1_ratio=-0.1)


# The different captilizations are intentional;
# I want to make sure different versions are recognized
@pytest.mark.parametrize(
    'thresholder',
    [
        'L0',
        'l1',
        'CAD'
    ]
)
def test_sr3_prox_functions(data_1d, thresholder):
    x, x_dot = data_1d
    model = SR3(thresholder=thresholder)
    model.fit(x, x_dot)
