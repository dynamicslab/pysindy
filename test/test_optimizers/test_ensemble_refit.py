import numpy as np
import pytest

from pysindy import SINDy
from pysindy.feature_library import IdentityLibrary
from pysindy.optimizers import EnsembleOptimizer
from pysindy.optimizers import STLSQ


def mean_members(members):
    return np.mean(members, axis=0)


def full_sample_ensemble(n_samples, **kwargs):
    return EnsembleOptimizer(
        STLSQ(threshold=0, alpha=0),
        bagging=True,
        n_models=3,
        n_subset=n_samples,
        replace=False,
        **kwargs,
    )


@pytest.mark.parametrize("aggregator", [None, mean_members], ids=["median", "mean"])
def test_refit_uses_only_latest_training_data(aggregator):
    # Every bag contains every row. Permuting the rows cannot change the
    # unique least-squares solution for these full-rank designs.
    x = np.tile(np.eye(2), (3, 1))
    initial = np.array([[1.0, -2.0]])
    current = np.array([[3.0, 4.0]])
    optimizer = full_sample_ensemble(len(x), ensemble_aggregator=aggregator)

    optimizer.fit(x, x @ initial.T)
    np.testing.assert_allclose(optimizer.coef_, initial, rtol=1e-12, atol=1e-12)
    optimizer.fit(x, x @ current.T)

    np.testing.assert_allclose(optimizer.coef_, current, rtol=1e-12, atol=1e-12)
    assert len(optimizer.coef_list) == 3
    for member in optimizer.coef_list:
        np.testing.assert_allclose(member, current, rtol=1e-12, atol=1e-12)


def test_refit_allows_different_coefficient_shapes():
    initial_x = np.tile(np.eye(2), (3, 1))
    current_x = np.tile(np.eye(3), (2, 1))
    initial = np.array([[1.0, -2.0]])
    current = np.array([[3.0, 0.0, -1.0], [0.0, 4.0, 2.0]])
    optimizer = full_sample_ensemble(len(initial_x))

    optimizer.fit(initial_x, initial_x @ initial.T)
    optimizer.fit(current_x, current_x @ current.T)

    assert optimizer.coef_.shape == current.shape
    np.testing.assert_allclose(optimizer.coef_, current, rtol=1e-12, atol=1e-12)
    assert len(optimizer.coef_list) == 3
    for member in optimizer.coef_list:
        assert member.shape == current.shape
        np.testing.assert_allclose(member, current, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "modes",
    [
        {"bagging": True},
        {"library_ensemble": True},
        {"bagging": True, "library_ensemble": True},
    ],
    ids=["bagging", "library", "both"],
)
def test_refit_updates_member_count_and_preserves_previous_snapshot(modes):
    # The zero response has zero coefficients for every possible library
    # subset, so this tests member ownership without depending on random draws.
    x = np.tile(np.eye(3), (2, 1))
    y = np.zeros((len(x), 1))
    optimizer = EnsembleOptimizer(
        STLSQ(threshold=0, alpha=0),
        n_models=3,
        n_subset=len(x),
        replace=False,
        **modes,
    )
    optimizer.fit(x, y)
    previous_members = optimizer.coef_list
    previous_values = [member.copy() for member in previous_members]
    assert len(previous_members) == 3

    optimizer.set_params(n_models=2)
    optimizer.fit(x, y)

    assert len(optimizer.coef_list) == 2
    assert optimizer.coef_list is not previous_members
    np.testing.assert_array_equal(optimizer.coef_, np.zeros((1, 3)))
    assert len(previous_members) == 3
    for retained, expected in zip(previous_members, previous_values):
        np.testing.assert_array_equal(retained, expected)


def test_repeated_sindy_fit_uses_latest_coefficients():
    x = np.tile(np.eye(2), (3, 1))
    initial = np.array([[1.0, 0.0], [0.0, -2.0]])
    current = np.array([[3.0, -1.0], [2.0, 4.0]])
    optimizer = full_sample_ensemble(len(x))
    model = SINDy(feature_library=IdentityLibrary(), optimizer=optimizer)

    model.fit(x, t=1.0, x_dot=x @ initial.T)
    np.testing.assert_allclose(model.coefficients(), initial, rtol=1e-12, atol=1e-12)
    model.fit(x, t=1.0, x_dot=x @ current.T)

    np.testing.assert_allclose(model.coefficients(), current, rtol=1e-12, atol=1e-12)
    assert len(optimizer.coef_list) == 3
