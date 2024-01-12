import numpy as np
import pytest
from numpy.testing import assert_
from numpy.testing import assert_array_equal
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
    assert_equal(np.asarray(a), np.asarray(check))  # modified
    b = np.array(1.0).view(AxesArray)
    a = d.copy().view(AxesArray)
    np.add.at(a, ([0, 1], [0, 2]), b)
    assert_equal(np.asarray(a), np.asarray(check))  # modified


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


def test_missing_axis_errors():
    axes = {"ax_time": 0}
    arr = AxesArray(np.arange(3), axes)
    with pytest.raises(AttributeError):
        arr.ax_spatial
    with pytest.raises(AttributeError):
        arr.n_spatial


def test_simple_slice():
    arr = AxesArray(np.ones(2), {"ax_coord": 0})
    assert_array_equal(arr[:], arr)
    assert_array_equal(arr[slice(None)], arr)
    assert arr[0] == 1


def test_basic_indexing_modifies_axes():
    axes = {"ax_time": 0, "ax_coord": 1}
    arr = AxesArray(np.ones(4).reshape((2, 2)), axes)
    slim = arr[1, :, None]
    with pytest.raises(AttributeError):
        slim.ax_time
    assert slim.ax_unk == 1
    assert slim.ax_coord == 0
    reverse_slim = arr[None, :, 1]
    with pytest.raises(AttributeError):
        reverse_slim.ax_coord
    assert reverse_slim.ax_unk == 0
    assert reverse_slim.ax_time == 1
    almost_new = arr[None, None, 1, :, None, None]
    with pytest.raises(AttributeError):
        almost_new.ax_time
    assert almost_new.ax_coord == 2
    assert set(almost_new.ax_unk) == {0, 1, 3, 4}


def test_insert_named_axis():
    arr = AxesArray(np.ones(1), axes={"ax_time": 0})
    expanded = arr["time", :]
    result = expanded.axes
    expected = {"ax_time": [0, 1]}
    assert result == expected


def test_adv_indexing_modifies_axes():
    axes = {"ax_time": 0, "ax_coord": 1}
    arr = AxesArray(np.arange(4).reshape((2, 2)), axes)
    flat = arr[[0, 1], [0, 1]]
    same = arr[[[0], [1]], [0, 1]]
    tpose = arr[[0, 1], [[0], [1]]]
    assert flat.shape == (2,)
    np.testing.assert_array_equal(np.asarray(flat), np.array([0, 3]))

    assert flat.ax_time_coord == 0
    with pytest.raises(AttributeError):
        flat.ax_coord
    with pytest.raises(AttributeError):
        flat.ax_time

    assert same.shape == arr.shape
    np.testing.assert_equal(np.asarray(same), np.asarray(arr))
    assert same.ax_time_coord == [0, 1]
    with pytest.raises(AttributeError):
        same.ax_coord

    assert tpose.shape == arr.shape
    np.testing.assert_equal(np.asarray(tpose), np.asarray(arr.T))
    assert tpose.ax_time_coord == [0, 1]
    with pytest.raises(AttributeError):
        tpose.ax_coord


def test_adv_indexing_adds_axes():
    axes = {"ax_time": 0, "ax_coord": 1}
    arr = AxesArray(np.arange(4).reshape((2, 2)), axes)
    fat = arr[[[0, 1], [0, 1]]]
    assert fat.shape == (2, 2, 2)
    assert fat.ax_time == [0, 1]
    assert fat.ax_coord == 2


def test_standardize_basic_indexer():
    arr = np.arange(6).reshape(2, 3)
    result_indexer, result_fancy = axes.standardize_indexer(arr, Ellipsis)
    assert result_indexer == [slice(None), slice(None)]
    assert result_fancy == ()

    result_indexer, result_fancy = axes.standardize_indexer(
        arr, (np.newaxis, 1, 1, Ellipsis)
    )
    assert result_indexer == [None, 1, 1]
    assert result_fancy == ()


def test_standardize_advanced_indexer():
    arr = np.arange(6).reshape(2, 3)
    result_indexer, result_fancy = axes.standardize_indexer(arr, [1])
    assert result_indexer == [np.ones(1), slice(None)]
    assert result_fancy == (0,)

    result_indexer, result_fancy = axes.standardize_indexer(
        arr, (np.newaxis, [1], 1, Ellipsis)
    )
    assert result_indexer == [None, np.ones(1), 1]
    assert result_fancy == (1,)


def test_standardize_bool_indexer():
    arr = np.ones((1, 2))
    result, result_adv = axes.standardize_indexer(arr, [[True, True]])
    assert_equal(result, [[0, 0], [0, 1]])
    assert result_adv == (0, 1)


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


def test_reduce_misordered_AxisMapping():
    ax_map = _AxisMapping({"ax_a": [0, 1], "ax_b": 2, "ax_c": 3}, 7)
    result = ax_map.remove_axis([2, 1])
    expected = {"ax_a": 0, "ax_c": 1}
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
    result = ax_map.insert_axis(3, "ax_unk")
    expected = {
        "ax_a": [0, 1],
        "ax_b": 2,
        "ax_unk": 3,
        "ax_c": 4,
        "ax_d": [5, 6],
    }
    assert result == expected


def test_insert_existing_AxisMapping():
    ax_map = _AxisMapping(
        {
            "ax_a": [0, 1],
            "ax_b": 2,
            "ax_c": 3,
            "ax_d": [4, 5],
        },
        6,
    )
    result = ax_map.insert_axis(3, "ax_b")
    expected = {
        "ax_a": [0, 1],
        "ax_b": [2, 3],
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
    result = ax_map.insert_axis([1, 4], new_name="ax_unk")
    expected = {
        "ax_a": [0, 2],
        "ax_unk": [1, 4],
        "ax_b": 3,
        "ax_c": 5,
        "ax_d": [6, 7],
    }
    assert result == expected


def test_insert_misordered_AxisMapping():
    ax_map = _AxisMapping(
        {
            "ax_a": [0, 1],
            "ax_b": 2,
            "ax_c": 3,
            "ax_d": [4, 5],
        },
        6,
    )
    result = ax_map.insert_axis([4, 1], new_name="ax_unk")
    expected = {
        "ax_a": [0, 2],
        "ax_unk": [1, 4],
        "ax_b": 3,
        "ax_c": 5,
        "ax_d": [6, 7],
    }
    assert result == expected


def test_determine_adv_broadcasting():
    indexers = (1, np.ones(1), np.ones((4, 1)), np.ones(3))
    res_nd, res_start = axes._determine_adv_broadcasting(indexers, [1, 2, 3])
    assert res_nd == 2
    assert res_start == 1

    indexers = (None, np.ones(1), 2, np.ones(3))
    res_nd, res_start = axes._determine_adv_broadcasting(indexers, [1, 3])
    assert res_nd == 1
    assert res_start == 0

    res_nd, res_start = axes._determine_adv_broadcasting(indexers, [])
    assert res_nd == 0
    assert res_start is None
