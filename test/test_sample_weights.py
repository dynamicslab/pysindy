import numpy as np
from typing import Sequence
import pytest
from pysindy._core import _assert_sample_weights, _expand_sample_weights

# Minimal Trajectory stub used for testing. The real project Trajectory objects
# expose attributes `n_time` and `n_coord` which we need for validation.
class _TrajStub:
    def __init__(self, n_time, n_coord):
        self.n_time = int(n_time)
        self.n_coord = int(n_coord)


def make_traj(n_time, n_coord):
    """Create a lightweight trajectory object for tests."""
    return _TrajStub(n_time, n_coord)


def test_per_trajectory_1d_concat():
    t1 = make_traj(3, 2)
    t2 = make_traj(2, 2)
    sw = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]

    validated = _assert_sample_weights(sw, [t1, t2])
    assert isinstance(validated, list)

    expanded = _expand_sample_weights(sw, [t1, t2])
    assert expanded.shape == (5,)
    assert np.allclose(expanded, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_per_trajectory_2d_broadcast_to_1d():
    t1 = make_traj(3, 2)
    t2 = make_traj(2, 2)
    sw = [np.array([[1.0], [2.0], [3.0]]), np.array([[4.0], [5.0]])]

    validated = _assert_sample_weights(sw, [t1, t2])
    assert isinstance(validated, list)

    expanded = _expand_sample_weights(sw, [t1, t2])
    assert expanded.shape == (5,)
    assert np.allclose(expanded, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_per_trajectory_2d_full():
    t1 = make_traj(3, 2)
    t2 = make_traj(2, 2)
    sw = [
        np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        np.array([[4.0, 40.0], [5.0, 50.0]]),
    ]

    validated = _assert_sample_weights(sw, [t1, t2])
    assert isinstance(validated, list)

    expanded = _expand_sample_weights(sw, [t1, t2])
    assert expanded.shape == (5, 2)
    assert np.allclose(expanded[0], [1.0, 10.0])
    assert np.allclose(expanded[-1], [5.0, 50.0])


def test_scalar_rejected():
    t1 = make_traj(3, 2)
    t2 = make_traj(2, 2)
    with pytest.raises(ValueError):
        _assert_sample_weights(1.0, [t1, t2])


def test_list_length_mismatch():
    t1 = make_traj(3, 2)
    t2 = make_traj(2, 2)
    sw = [np.array([1.0, 2.0, 3.0])]
    with pytest.raises(ValueError):
        _assert_sample_weights(sw, [t1, t2])


def test_element_length_mismatch():
    t1 = make_traj(3, 2)
    t2 = make_traj(2, 2)
    sw = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])]
    with pytest.raises(ValueError):
        _assert_sample_weights(sw, [t1, t2])
