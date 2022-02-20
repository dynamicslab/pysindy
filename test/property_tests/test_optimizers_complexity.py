import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import integers
from numpy.random import randint
from numpy.random import seed
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
    n_features=integers(min_value=10, max_value=30),
    n_informative=integers(min_value=3, max_value=9),
    random_state=integers(min_value=0, max_value=2 ** 32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_complexity(n_samples, n_features, n_informative, random_state):
    """Behaviour test for complexity.

    We assume that more regularized optimizers are less complex on the same dataset.
    """
    assume(n_informative < n_features)

    # Average complexity over multiple datasets
    n_datasets = 5
    complexities = [0] * 7

    seed(random_state)
    for rs in randint(low=0, high=2 ** 32 - 1, size=n_datasets):

        x, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_targets=1,
            bias=0,
            noise=0.1,
            random_state=rs,
        )
        y = y.reshape(-1, 1)

        opt_kwargs = dict(fit_intercept=True)
        optimizers = [
            SR3(thresholder="l0", threshold=0.1, **opt_kwargs),
            SR3(thresholder="l1", threshold=0.1, **opt_kwargs),
            Lasso(**opt_kwargs),
            STLSQ(**opt_kwargs),
            ElasticNet(**opt_kwargs),
            Ridge(**opt_kwargs),
            LinearRegression(**opt_kwargs),
        ]

        optimizers = [SINDyOptimizer(o, unbias=True) for o in optimizers]

        for k, opt in enumerate(optimizers):
            opt.fit(x, y)
            complexities[k] += opt.complexity

    for less_complex, more_complex in zip(complexities, complexities[1:]):
        # relax the condition to account for
        # noise and non-normalized threshold parameters
        assert less_complex <= more_complex + 5


@pytest.mark.parametrize(
    "opt_cls, reg_name",
    [[Lasso, "alpha"], [Ridge, "alpha"], [STLSQ, "threshold"], [SR3, "threshold"]],
)
@given(
    n_samples=integers(min_value=100, max_value=10000),
    n_features=integers(min_value=10, max_value=30),
    n_informative=integers(min_value=3, max_value=9),
    random_state=integers(min_value=0, max_value=2 ** 32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_complexity_parameter(
    opt_cls, reg_name, n_samples, n_features, n_informative, random_state
):
    """Behaviour test for complexity 2.

    We assume that a model with a bigger regularization parameter is less complex.
    """
    assume(n_informative <= n_features)

    x, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=1,
        bias=0,
        noise=0.1,
        random_state=random_state,
    )
    y = y.reshape(-1, 1)

    optimizers = [
        SINDyOptimizer(opt_cls(**{reg_name: reg_value}), unbias=True)
        for reg_value in [3, 1, 0.3, 0.1, 0.01]
    ]

    for opt in optimizers:
        opt.fit(x, y)

    for less_complex, more_complex in zip(optimizers, optimizers[1:]):
        assert less_complex.complexity <= more_complex.complexity
