import numpy as np
import pytest
from numpy.testing import assert_
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from pysindy import AxesArray
from pysindy.utils import axes
from pysindy.utils.axes import _AxisMapping
from pysindy.utils.axes import AxesWarning


def test_reduce_mean_noinf_recursion():
    arr = AxesArray(np.array([[1]]), {"ax_a": [0, 1]})
    np.mean(arr, axis=0)


def test_repr():
    a = AxesArray(np.arange(5.0), {"ax_time": 0})
    result = a.__repr__()
    expected = "AxesArray([0., 1., 2., 3., 4.])"
    assert result == expected


def test_ufunc_override():
    # This is largely a clone of test_ufunc_override_with_super() from
    # numpy/core/tests/test_umath.py

    class B:
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            if any(isinstance(input_, AxesArray) for input_ in inputs):
                return "A!"
            else:
                return NotImplemented

    d = np.arange(5.0)
    # 1 input, 1 output
    a = AxesArray(d, {"ax_time": 0})
    b = np.sin(a)
    check = np.sin(d)
    assert_(np.all(check == b))
    b = np.sin(d, out=(a,))
    assert_(np.all(check == b))
    assert_(b is a)
    a = AxesArray(np.arange(5.0), {"ax_time": 0})
    b = np.sin(a, out=a)
    assert_(np.all(check == b))

    # 1 input, 2 outputs
    a = AxesArray(np.arange(5.0), {"ax_time": 0})
    b1, b2 = np.modf(a)
    b1, b2 = np.modf(d, out=(None, a))
    assert_(b2 is a)
    a = AxesArray(np.arange(5.0), {"ax_time": 0})
    b = AxesArray(np.arange(5.0), {"ax_time": 0})
    c1, c2 = np.modf(a, out=(a, b))
    assert_(c1 is a)
    assert_(c2 is b)

    # 2 input, 1 output
    a = AxesArray(np.arange(5.0), {"ax_time": 0})
    b = AxesArray(np.arange(5.0), {"ax_time": 0})
    c = np.add(a, b, out=a)
    assert_(c is a)
    # some tests with a non-ndarray subclass
    a = np.arange(5.0)
    b = B()
    assert_(a.__array_ufunc__(np.add, "__call__", a, b) is NotImplemented)
    assert_(b.__array_ufunc__(np.add, "__call__", a, b) is NotImplemented)
    assert_raises(TypeError, np.add, a, b)
    a = AxesArray(a, {"ax_time": 0})
    assert_(a.__array_ufunc__(np.add, "__call__", a, b) is NotImplemented)
    assert_(b.__array_ufunc__(np.add, "__call__", a, b) == "A!")
    assert_(np.add(a, b) == "A!")
    # regression check for gh-9102 -- tests ufunc.reduce implicitly.
    d = np.array([[1, 2, 3], [1, 2, 3]])
    a = AxesArray(d, {"ax_time": [0, 1]})
    c = a.any()
    check = d.any()
    assert_equal(c, check)
    c = a.max()
    check = d.max()
    assert_equal(c, check)
    b = np.array(0).view(AxesArray)
    c = a.max(out=b)
    assert_equal(c, check)
    assert_(c is b)
    check = a.max(axis=0)
    b = np.zeros_like(check).view(AxesArray)
    c = a.max(axis=0, out=b)
    assert_equal(c, check)
    assert_(c is b)
    # simple explicit tests of reduce, accumulate, reduceat
    check = np.add.reduce(d, axis=1)
    c = np.add.reduce(a, axis=1)
    assert_equal(c, check)
    b = np.zeros_like(c)
    c = np.add.reduce(a, 1, None, b)
    assert_equal(c, check)
    assert_(c is b)


# @pytest.mark.skip("Expected error")
def test_ufunc_override_accumulate():
    d = np.array([[1, 2, 3], [1, 2, 3]])
    a = AxesArray(d, {"ax_time": [0, 1]})
    check = np.add.accumulate(d, axis=0)
    c = np.add.accumulate(a, axis=0)
    assert_equal(c, check)
    b = np.zeros_like(c)
    c = np.add.accumulate(a, 0, None, b)
    assert_equal(c, check)
    assert_(c is b)
    indices = [0, 2, 1]
    check = np.add.reduceat(d, indices, axis=1)
    c = np.add.reduceat(a, indices, axis=1)
    assert_equal(c, check)
    b = np.zeros_like(c)
    c = np.add.reduceat(a, indices, 1, None, b)
    assert_equal(c, check)
    assert_(c is b)
    # and a few tests for at
    d = np.array([[1, 2, 3], [1, 2, 3]])
    check = d.copy()
    a = d.copy().view(AxesArray)
    np.add.at(check, ([0, 1], [0, 2]), 1.0)
    np.add.at(a, ([0, 1], [0, 2]), 1.0)
    assert_equal(a, check)
    b = np.array(1.0).view(AxesArray)
    a = d.copy().view(AxesArray)
    np.add.at(a, ([0, 1], [0, 2]), b)
    assert_equal(a, check)


def test_n_elements():
    arr = np.empty(np.arange(1, 5))
    arr = AxesArray(arr, {"ax_spatial": [0, 1], "ax_time": 2, "ax_coord": 3})
    assert arr.n_spatial == (1, 2)
    assert arr.n_time == 3
    assert arr.n_coord == 4

    arr2 = np.concatenate((arr, arr), axis=arr.ax_time)
    assert arr2.n_spatial == (1, 2)
    assert arr2.n_time == 6
    assert arr2.n_coord == 4


@pytest.mark.skip("Expected error")
def test_limited_slice():
    arr = np.empty(np.arange(1, 5))
    arr = AxesArray(arr, {"ax_spatial": [0, 1], "ax_time": 2, "ax_coord": 3})
    arr3 = arr[..., :2, 0]
    assert arr3.n_spatial == (1, 2)
    assert arr3.n_time == 2
    # No way to intercept slicing and remove ax_coord
    with pytest.raises(IndexError):
        assert arr3.n_coord == 1
    assert arr3.n_sample == 1


def test_warn_toofew_axes():
    axes = {"ax_time": 0, "ax_coord": 1}
    with pytest.warns(AxesWarning):
        AxesArray(np.ones(8).reshape((2, 2, 2)), axes)


def test_toomany_axes():
    axes = {"ax_time": 0, "ax_coord": 2}
    with pytest.raises(ValueError):
        AxesArray(np.ones(4).reshape((2, 2)), axes)


def test_conflicting_axes_defn():
    axes = {"ax_time": 0, "ax_coord": 0}
    with pytest.raises(ValueError):
        AxesArray(np.ones(4), axes)


def test_basic_indexing_modifies_axes():
    axes = {"ax_time": 0, "ax_coord": 1}
    arr = AxesArray(np.ones(4).reshape((2, 2)), axes)
    slim = arr[1, :, None]
    with pytest.raises(KeyError):
        slim.ax_time
    assert slim.ax_unk == 1
    assert slim.ax_coord == 0


def test_fancy_indexing_modifies_axes():
    axes = {"ax_time": 0, "ax_coord": 1}
    arr = AxesArray(np.arange(4).reshape((2, 2)), axes)
    flat = arr[[0, 1], [0, 1]]
    same = arr[[[0], [1]], [0, 1]]
    tpose = arr[[0, 1], [[0], [1]]]
    assert flat.shape == (2,)
    np.testing.assert_array_equal(np.asarray(flat), np.array([0, 3]))

    assert flat.ax__timecoord == 0
    with pytest.raises(AttributeError):
        flat.ax_coord
    with pytest.raises(AttributeError):
        flat.ax_time

    assert same.shape == arr.shape
    np.testing.assert_equal(same, arr)
    assert same.ax_time == 0
    assert same.ax_coord == 1

    assert tpose.shape == arr.shape
    np.testing.assert_equal(same, arr.T)
    assert same.ax_time == 1
    assert same.ax_coord == 0

    fat = arr[[[0, 1], [0, 1]]]
    assert fat.shape == (2, 2, 2)
    assert fat.ax_time == [0, 1]
    assert fat.ax_coord == 2


def test_standardize_basic_indexer():
    arr = np.arange(6).reshape(2, 3)
    result_indexer, result_fancy = axes._standardize_indexer(arr, Ellipsis)
    assert result_indexer == (slice(None), slice(None))
    assert result_fancy == ()

    result_indexer, result_fancy = axes._standardize_indexer(
        arr, (np.newaxis, 1, 1, Ellipsis)
    )
    assert result_indexer == (None, 1, 1)
    assert result_fancy == ()


def test_standardize_fancy_indexer():
    arr = np.arange(6).reshape(2, 3)
    result_indexer, result_fancy = axes._standardize_indexer(arr, [1])
    assert result_indexer == (np.ones(1), slice(None))
    assert result_fancy == (0,)

    result_indexer, result_fancy = axes._standardize_indexer(
        arr, (np.newaxis, [1], 1, Ellipsis)
    )
    assert result_indexer == (None, np.ones(1), 1)
    assert result_fancy == (1,)


def test_reduce_AxisMapping():
    ax_map = _AxisMapping(
        {
            "ax_a": [0, 1],
            "ax_b": 2,
            "ax_c": 3,
            "ax_d": 4,
            "ax_e": [5, 6],
        },
        7,
    )
    result = ax_map.remove_axis(3)
    expected = {
        "ax_a": [0, 1],
        "ax_b": 2,
        "ax_d": 3,
        "ax_e": [4, 5],
    }
    assert result == expected


def test_reduce_multiple_AxisMapping():
    ax_map = _AxisMapping(
        {
            "ax_a": [0, 1],
            "ax_b": 2,
            "ax_c": 3,
            "ax_d": 4,
            "ax_e": [5, 6],
        },
        7,
    )
    result = ax_map.remove_axis([3, 4])
    expected = {
        "ax_a": [0, 1],
        "ax_b": 2,
        "ax_e": [3, 4],
    }
    assert result == expected


def test_reduce_twisted_AxisMapping():
    ax_map = _AxisMapping(
        {
            "ax_a": [0, 6],
            "ax_b": 2,
            "ax_c": 3,
            "ax_d": 4,
            "ax_e": [1, 5],
        },
        7,
    )
    result = ax_map.remove_axis([3, 4])
    expected = {
        "ax_a": [0, 4],
        "ax_b": 2,
        "ax_e": [1, 3],
    }
    assert result == expected


def test_insert_AxisMapping():
    ax_map = _AxisMapping(
        {
            "ax_a": [0, 1],
            "ax_b": 2,
            "ax_c": 3,
            "ax_d": [4, 5],
        },
        6,
    )
    result = ax_map.insert_axis(3)
    expected = {
        "ax_a": [0, 1],
        "ax_b": 2,
        "ax_unk": 3,
        "ax_c": 4,
        "ax_d": [5, 6],
    }
    assert result == expected


def test_insert_multiple_AxisMapping():
    ax_map = _AxisMapping(
        {
            "ax_a": [0, 1],
            "ax_b": 2,
            "ax_c": 3,
            "ax_d": [4, 5],
        },
        6,
    )
    result = ax_map.insert_axis([1, 4])
    expected = {
        "ax_a": [0, 2],
        "ax_unk": [1, 4],
        "ax_b": 3,
        "ax_c": 5,
        "ax_d": [6, 7],
    }
    assert result == expected
