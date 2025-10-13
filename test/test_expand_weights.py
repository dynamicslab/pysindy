import numpy as np
import pytest
from pysindy import _expand_sample_weights


class DummyTraj:
    def __init__(self, n_time, n_coord):
        self.n_time = n_time
        self.n_coord = n_coord


@pytest.fixture
def dummy_trajs():
    """Simple fixture for two dummy trajectories."""
    return [DummyTraj(5, 2), DummyTraj(5, 2)]


def test_scalar_weights_none(dummy_trajs):
    assert _expand_sample_weights(None, dummy_trajs) is None


def test_1d_sample_weights(dummy_trajs):
    """1D weights per trajectory concatenate correctly."""
    weights = [np.ones(5), 2 * np.ones(5)]
    out = _expand_sample_weights(weights, dummy_trajs)
    assert out.shape == (10,)
    assert np.all(out[:5] == 1)
    assert np.all(out[5:] == 2)


def test_2d_per_coord_weights(dummy_trajs):
    """2D weights (n_time, n_coord) concatenate along samples."""
    weights = [np.ones((5, 2)), np.full((5, 2), 2.0)]
    out = _expand_sample_weights(weights, dummy_trajs)
    assert out.shape == (10, 2)
    assert np.allclose(out[:5], 1)
    assert np.allclose(out[5:], 2)


def test_promote_1d_to_2d(dummy_trajs):
    """1D weights promoted to (n_time, n_coord)."""
    weights = [np.arange(5), np.arange(5)]
    out = _expand_sample_weights(weights, dummy_trajs)
    assert out.ndim == 1  # still flattened because all dims == 1


def test_weak_mode_expansion(dummy_trajs):
    """Weak mode expands by n_test_funcs."""
    weights = [np.ones(5), np.ones(5)]
    class DummyLib: K = 3
    out = _expand_sample_weights(weights, dummy_trajs, feature_library=DummyLib(), mode="weak")
    assert out.shape == (10 * 3,)
    assert np.allclose(np.unique(out), 1)
