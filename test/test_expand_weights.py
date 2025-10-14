import warnings

import numpy as np
import pytest

from pysindy._core import _expand_sample_weights


class Trajectory:
    """Minimal trajectory object with attributes matching SINDy expectations."""

    def __init__(self, n_time, n_variables):
        self.n_time = n_time
        self.n_coord = n_variables


class WeakLibrary:
    """Mock weak-form feature library with K test functions."""

    def __init__(self, K=2):
        self.K = K


# ---------------------------------------------------------------------
# STANDARD MODE TESTS
# ---------------------------------------------------------------------


def test_standard_mode_with_scalar_weights():
    """
    Each trajectory's scalar weight should expand to one per sample.
    """
    trajectories = [Trajectory(4, 2), Trajectory(6, 2)]
    weights = [1.0, 2.0]

    expanded = _expand_sample_weights(weights, trajectories, mode="standard")

    assert expanded.shape == (10,)  # 4 + 6 samples
    assert np.allclose(expanded[:4], 1.0)
    assert np.allclose(expanded[4:], 2.0)


def test_standard_mode_with_per_sample_weights():
    """
    Per-sample weights should concatenate into one long 1D array.
    """
    trajectories = [Trajectory(3, 2), Trajectory(2, 2)]
    weights = [np.arange(3), np.arange(10, 12)]

    expanded = _expand_sample_weights(weights, trajectories, mode="standard")

    expected = np.concatenate([np.arange(3), np.arange(10, 12)])
    assert np.allclose(expanded, expected)
    assert expanded.ndim == 1


def test_standard_mode_flattens_column_vectors():
    """Column vectors (n, 1) should flatten correctly."""
    trajectories = [Trajectory(5, 3)]
    weights = [np.arange(5).reshape(-1, 1)]

    expanded = _expand_sample_weights(weights, trajectories, mode="standard")

    assert expanded.shape == (5,)
    assert np.allclose(expanded, np.arange(5))


def test_standard_mode_rejects_mismatched_lengths():
    """A weight array must match the number of samples in its trajectory."""
    trajectories = [Trajectory(4, 2)]
    weights = [np.ones(3)]

    with pytest.raises(ValueError, match="trajectory length"):
        _expand_sample_weights(weights, trajectories, mode="standard")


# ---------------------------------------------------------------------
# WEAK FORM MODE TESTS
# ---------------------------------------------------------------------


def test_weak_mode_expands_one_weight_per_trajectory():
    """
    Weak mode: one scalar weight per trajectory is repeated by K test functions.
    The total length is n_trajectories * K.
    """
    num_trajectories = 3
    trajectories = [Trajectory(5, 2) for _ in range(num_trajectories)]
    library = WeakLibrary(K=4)

    weights = [1.0, 2.0, 3.0]

    expanded = _expand_sample_weights(
        weights, trajectories, feature_library=library, mode="weak"
    )

    assert expanded.shape == (num_trajectories * library.K,)
    expected = np.repeat([1.0, 2.0, 3.0], library.K)
    assert np.allclose(expanded, expected)


def test_weak_mode_rejects_per_sample_weights():
    """
    Weak mode should not accept per-sample weights.
    Each trajectory must have exactly one scalar weight.
    """
    trajectories = [Trajectory(5, 1)]
    library = WeakLibrary(K=2)
    weights = [np.ones(5)]  # Invalid: multiple samples

    with pytest.raises(ValueError, match="one weight per trajectory"):
        _expand_sample_weights(
            weights, trajectories, feature_library=library, mode="weak"
        )


def test_weak_mode_warns_if_library_missing_K():
    """Warn and assume K=1 if the feature library has no K attribute."""
    trajectories = [Trajectory(2, 2)]

    class LibraryWithoutK:
        pass

    library = LibraryWithoutK()
    weights = [1.0]

    with warnings.catch_warnings(record=True) as captured:
        expanded = _expand_sample_weights(
            weights, trajectories, feature_library=library, mode="weak"
        )

    assert expanded.shape == (1 * len(trajectories),)
    assert any("missing 'K'" in str(warning.message) for warning in captured)


# ---------------------------------------------------------------------
# VALIDATION TESTS
# ---------------------------------------------------------------------


def test_rejects_non_list_weights():
    """sample_weight must be a list or tuple, not a numpy array."""
    trajectories = [Trajectory(3, 1)]
    with pytest.raises(ValueError, match="list or tuple"):
        _expand_sample_weights(np.ones(3), trajectories)


def test_rejects_mismatched_number_of_trajectories():
    """The number of weight entries must match the number of trajectories."""
    trajectories = [Trajectory(3, 1), Trajectory(3, 1)]
    weights = [np.ones(3)]
    with pytest.raises(ValueError, match="length must match"):
        _expand_sample_weights(weights, trajectories)


# ---------------------------------------------------------------------
# INTEGRATION TEST
# ---------------------------------------------------------------------


def test_expanded_weights_work_with_rescale_data():
    """
    Expanded weights should work seamlessly with optimizer._rescale_data.
    """
    from pysindy.optimizers.base import _rescale_data

    num_samples, num_features = 8, 3
    X = np.random.randn(num_samples, num_features)
    y = np.random.randn(num_samples, 1)

    trajectories = [Trajectory(8, 3)]
    weights = [np.linspace(1.0, 2.0, 8)]

    expanded = _expand_sample_weights(weights, trajectories, mode="standard")

    assert expanded.shape == (8,)
    X_scaled, y_scaled = _rescale_data(X, y, expanded)

    assert X_scaled.shape == X.shape
    assert y_scaled.shape == y.shape
    # Check scaling for first sample
    assert np.allclose(X_scaled[0], X[0] * np.sqrt(expanded[0]))
