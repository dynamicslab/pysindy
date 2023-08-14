import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import integers
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ
from pysindy.optimizers import WrappedOptimizer


@pytest.mark.parametrize(
    "opt_cls, reg_name",
    [[Lasso, "alpha"], [Ridge, "alpha"], [STLSQ, "threshold"], [SR3, "threshold"]],
)
@given(
    n_samples=integers(min_value=30, max_value=50),
    n_features=integers(min_value=6, max_value=10),
    n_informative=integers(min_value=2, max_value=5),
    random_state=integers(min_value=0, max_value=2**32 - 1),
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
        WrappedOptimizer(opt_cls(**{reg_name: reg_value}), normalize_columns=True)
        for reg_value in [10, 1, 0.1, 0.01]
    ]

    for opt in optimizers:
        opt.fit(x, y)

    for less_complex, more_complex in zip(optimizers, optimizers[1:]):
        assert less_complex.complexity <= more_complex.complexity
