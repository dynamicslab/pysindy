"""
Unit tests for differentiation methods.
"""
import numpy as np
import pytest

from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.differentiation.base import BaseDifferentiation


# Simplest example: just use an assert statement
def test_forward_difference_length():
    x = 2 * np.linspace(1, 100, 100)
    forward_difference = FiniteDifference(order=1)
    assert len(forward_difference(x)) == len(x)

    forward_difference_nans = FiniteDifference(order=1, drop_endpoints=True)
    assert len(forward_difference_nans(x)) == len(x)


def test_forward_difference_variable_timestep_length():
    t = np.linspace(1, 10, 100) ** 2
    x = 2 * t
    forward_difference = FiniteDifference(order=1)
    assert len(forward_difference(x, t) == len(x))


def test_centered_difference_length():
    x = 2 * np.linspace(1, 100, 100)
    centered_difference = FiniteDifference(order=2)
    assert len(centered_difference(x)) == len(x)

    centered_difference_nans = FiniteDifference(order=2, drop_endpoints=True)
    assert len(centered_difference_nans(x)) == len(x)


def test_centered_difference_variable_timestep_length():
    t = np.linspace(1, 10, 100) ** 2
    x = 2 * t
    centered_difference = FiniteDifference(order=2)
    assert len(centered_difference(x, t) == len(x))


# Fixtures: data sets to be re-used in multiple tests
# data_derivative_1d and data_derivative_2d are defined
# in ../conftest.py


def test_forward_difference_1d(data_derivative_1d):
    x, x_dot = data_derivative_1d
    forward_difference = FiniteDifference(order=1)
    np.testing.assert_allclose(forward_difference(x), x_dot)


def test_forward_difference_2d(data_derivative_2d):
    x, x_dot = data_derivative_2d
    forward_difference = FiniteDifference(order=1)
    np.testing.assert_allclose(forward_difference(x), x_dot)


def test_centered_difference_1d(data_derivative_1d):
    x, x_dot = data_derivative_1d
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x), x_dot)


def test_centered_difference_2d(data_derivative_2d):
    x, x_dot = data_derivative_2d
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x), x_dot)


# Alternative implementation of the four tests above using parametrization
@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_derivative_1d"),
        pytest.lazy_fixture("data_derivative_2d"),
    ],
)
def test_forward_difference(data):
    x, x_dot = data
    forward_difference = FiniteDifference(order=1)
    np.testing.assert_allclose(forward_difference(x), x_dot)


@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_derivative_1d"),
        pytest.lazy_fixture("data_derivative_2d"),
    ],
)
def test_centered_difference(data):
    x, x_dot = data
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x), x_dot)


# pytest can also check that methods throw errors when appropriate
def test_forward_difference_dim():
    x = np.ones((5, 5, 5))
    forward_difference = FiniteDifference(order=1)
    with pytest.raises(ValueError):
        forward_difference(x)


def test_centered_difference_dim():
    x = np.ones((5, 5, 5))
    centered_difference = FiniteDifference(order=2)
    with pytest.raises(ValueError):
        centered_difference(x)


def test_order_error():
    with pytest.raises(NotImplementedError):
        FiniteDifference(order=3)
    with pytest.raises(ValueError):
        FiniteDifference(order=-1)


def test_base_class(data_derivative_1d):
    x, x_dot = data_derivative_1d
    with pytest.raises(NotImplementedError):
        BaseDifferentiation()._differentiate(x)


# Test smoothed finite difference method
@pytest.mark.parametrize(
    "data",
    [
        pytest.lazy_fixture("data_derivative_1d"),
        pytest.lazy_fixture("data_derivative_2d"),
    ],
)
def test_smoothed_finite_difference(data):
    x, x_dot = data
    smoothed_centered_difference = SmoothedFiniteDifference()
    np.testing.assert_allclose(smoothed_centered_difference(x), x_dot)
