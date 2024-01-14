import numpy as np
import pytest
from numpy.testing import assert_
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from pysindy import AxesArray


def test_concat_out():
    arr = AxesArray(np.arange(3).reshape(1, 3), {"ax_a": 0, "ax_b": 1})
    arr_out = np.empty((2, 3)).view(AxesArray)
    result = np.concatenate((arr, arr), axis=0, out=arr_out)
    assert_equal(result, arr_out)


def test_reduce_mean_noinf_recursion():
    arr = AxesArray(np.array([[1]]), {})
    np.mean(arr, axis=0)


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
    a = AxesArray(d, {})
    b = np.sin(a)
    check = np.sin(d)
    assert_(np.all(check == b))
    b = np.sin(d, out=(a,))
    assert_(np.all(check == b))
    assert_(b is a)
    a = AxesArray(np.arange(5.0), {})
    b = np.sin(a, out=a)
    assert_(np.all(check == b))

    # 1 input, 2 outputs
    a = AxesArray(np.arange(5.0), {})
    b1, b2 = np.modf(a)
    b1, b2 = np.modf(d, out=(None, a))
    assert_(b2 is a)
    a = AxesArray(np.arange(5.0), {})
    b = AxesArray(np.arange(5.0), {})
    c1, c2 = np.modf(a, out=(a, b))
    assert_(c1 is a)
    assert_(c2 is b)

    # 2 input, 1 output
    a = AxesArray(np.arange(5.0), {})
    b = AxesArray(np.arange(5.0), {})
    c = np.add(a, b, out=a)
    assert_(c is a)
    # some tests with a non-ndarray subclass
    a = np.arange(5.0)
    b = B()
    assert_(a.__array_ufunc__(np.add, "__call__", a, b) is NotImplemented)
    assert_(b.__array_ufunc__(np.add, "__call__", a, b) is NotImplemented)
    assert_raises(TypeError, np.add, a, b)
    a = AxesArray(a, {})
    assert_(a.__array_ufunc__(np.add, "__call__", a, b) is NotImplemented)
    assert_(b.__array_ufunc__(np.add, "__call__", a, b) == "A!")
    assert_(np.add(a, b) == "A!")
    # regression check for gh-9102 -- tests ufunc.reduce implicitly.
    d = np.array([[1, 2, 3], [1, 2, 3]])
    a = AxesArray(d, {})
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
    assert arr.n_sample == 1

    arr2 = np.concatenate((arr, arr), axis=arr.ax_time)
    assert arr2.n_spatial == (1, 2)
    assert arr2.n_time == 6
    assert arr2.n_coord == 4
    assert arr2.n_sample == 1

    arr3 = arr[..., :2, 0]
    assert arr3.n_spatial == (1, 2)
    assert arr3.n_time == 2
    # No way to intercept slicing and remove ax_coord
    with pytest.raises(IndexError):
        assert arr3.n_coord == 1
    assert arr3.n_sample == 1
