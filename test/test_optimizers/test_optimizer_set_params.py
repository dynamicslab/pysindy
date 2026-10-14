"""Optimizer parameter updates must preserve sklearn's estimator protocol."""
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from pysindy.optimizers import STLSQ
from pysindy.optimizers import TrappingSR3


def _optimizer_and_update(kind):
    if kind == "stlsq":
        return STLSQ(alpha=0.0, unbias=False), {"threshold": 0.2}
    return (
        TrappingSR3(
            _n_tgts=1,
            method="local",
            eta=1.0,
            alpha=1.0,
            beta=1.0,
            m0=np.zeros(1),
            A0=-0.1 * np.eye(1),
        ),
        {"eta": 2.0},
    )


@pytest.mark.parametrize("kind", ["stlsq", "trapping"])
@pytest.mark.parametrize("update", [False, True], ids=["empty", "changed"])
def test_set_params_returns_same_optimizer(kind, update):
    optimizer, parameters = _optimizer_and_update(kind)
    result = optimizer.set_params(**parameters if update else {})
    assert result is optimizer
    if update:
        for name, expected in parameters.items():
            assert optimizer.get_params()[name] == expected


def _linear_problem():
    # Each KFold train/test partition has Gram matrix 2 I. The OLS
    # coefficients are unique and both folds have the same exact prediction.
    x = np.tile(np.eye(2), (4, 1))
    coefficients = np.array([0.1, 1.0])
    return x, x @ coefficients, coefficients


def _assert_search_matches_linear_reference(search, coefficients):
    # Removing the first coefficient leaves squared residual 1/100 on
    # half the samples, hence MSE 1/200 in each fold. Threshold zero retains
    # the unique OLS solution and has zero residual.
    expected = np.array([0.0, -float(Fraction(1, 200))])
    for split in ["split0_test_score", "split1_test_score", "mean_test_score"]:
        assert_allclose(search.cv_results_[split], expected, rtol=1e-12, atol=1e-12)
    assert_allclose(search.best_score_, 0.0, rtol=0.0, atol=1e-12)
    assert search.best_index_ == 0
    predictions = search.predict(np.tile(np.eye(2), (4, 1))).ravel()
    assert_allclose(predictions, np.tile(coefficients, 4), rtol=1e-12, atol=1e-12)


def test_grid_search_uses_optimizer_return_value():
    x, y, coefficients = _linear_problem()
    search = GridSearchCV(
        STLSQ(alpha=0.0, unbias=False),
        {"threshold": [0.0, 0.2]},
        cv=KFold(n_splits=2, shuffle=False),
        scoring="neg_mean_squared_error",
        error_score="raise",
        n_jobs=1,
    )
    search.fit(x, y)
    _assert_search_matches_linear_reference(search, coefficients)
    assert search.best_params_ == {"threshold": 0.0}
    assert_allclose(
        search.best_estimator_.coef_, coefficients[None, :], rtol=1e-12, atol=1e-12
    )


def test_nested_pipeline_search_remains_supported():
    x, y, coefficients = _linear_problem()
    search = GridSearchCV(
        Pipeline([("optimizer", STLSQ(alpha=0.0, unbias=False))]),
        {"optimizer__threshold": [0.0, 0.2]},
        cv=KFold(n_splits=2, shuffle=False),
        scoring="neg_mean_squared_error",
        error_score="raise",
        n_jobs=1,
    )
    search.fit(x, y)
    _assert_search_matches_linear_reference(search, coefficients)
    assert search.best_params_ == {"optimizer__threshold": 0.0}
    assert_allclose(
        search.best_estimator_.named_steps["optimizer"].coef_,
        coefficients[None, :],
        rtol=1e-12,
        atol=1e-12,
    )
