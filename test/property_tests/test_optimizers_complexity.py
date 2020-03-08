from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import integers
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from pysindy.optimizers import SINDyOptimizer
from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ


@given(
    n_samples=integers(min_value=100, max_value=10000),
    n_features=integers(min_value=3, max_value=30),
    n_informative=integers(min_value=1, max_value=10),
    random_state=integers(min_value=0, max_value=2 ** 32 - 1),
)
@settings(max_examples=10)
def test_complexity(n_samples, n_features, n_informative, random_state):
    """Behaviour test for complexity.

    We assume that more regularized optimizers are less complex on the same dataset.
    """
    assume(n_informative <= n_features)

    x, y = make_regression(
        n_samples, n_features, n_informative, 1, 0, noise=0.1, random_state=random_state
    )
    y = y.reshape(-1, 1)

    opt_kwargs = dict(fit_intercept=True, normalize=False)
    optimizers = [
        SR3(thresholder="l0", **opt_kwargs),
        SR3(thresholder="l1", **opt_kwargs),
        Lasso(**opt_kwargs),
        STLSQ(**opt_kwargs),
        ElasticNet(**opt_kwargs),
        Ridge(**opt_kwargs),
        LinearRegression(**opt_kwargs),
    ]

    optimizers = [SINDyOptimizer(o, unbias=True) for o in optimizers]

    for opt in optimizers:
        opt.fit(x, y)

    for less_complex, more_complex in zip(optimizers, optimizers[1:]):
        assert less_complex.complexity <= more_complex.complexity
