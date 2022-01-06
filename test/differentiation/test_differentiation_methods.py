"""
Unit tests for differentiation methods.
"""
import numpy as np
import pytest
from derivative import dxdt

from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SINDyDerivative
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.differentiation import SpectralDerivative
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


def test_spectral_derivative_1d(data_derivative_quasiperiodic_1d):
    t, x, x_dot = data_derivative_quasiperiodic_1d
    spectral_derivative = SpectralDerivative()
    np.testing.assert_allclose(spectral_derivative(x, t), x_dot, atol=1e-12)
    centered_difference = FiniteDifference(order=2, periodic=True)._differentiate(x, t)
    np.testing.assert_allclose(
        centered_difference[0], centered_difference[-1], rtol=1e-4
    )


def test_centered_difference_2d(data_derivative_2d):
    x, x_dot = data_derivative_2d
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x), x_dot)


def test_spectral_derivative_2d(data_derivative_quasiperiodic_2d):
    t, x, x_dot = data_derivative_quasiperiodic_2d
    spectral_derivative = SpectralDerivative()
    np.testing.assert_allclose(spectral_derivative(x, t), x_dot, atol=1e-12)
    centered_difference = FiniteDifference(order=2, periodic=True)._differentiate(x, t)
    np.testing.assert_allclose(
        centered_difference[0, 0], centered_difference[-1, 0], rtol=1e-4
    )


def test_centered_difference_2d_uniform(data_derivative_2d):
    x, x_dot = data_derivative_2d
    centered_difference = FiniteDifference(order=2, is_uniform=True)
    np.testing.assert_allclose(centered_difference(x), x_dot)


def test_centered_difference_2d_uniform_time(data_derivative_2d):
    x, x_dot = data_derivative_2d
    t = np.linspace(0, x.shape[0] - 1, x.shape[0])
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x, t), x_dot)


def test_centered_difference_2d_nonuniform_time(data_derivative_2d):
    x, x_dot = data_derivative_2d
    t = np.linspace(0, x.shape[0] - 1, x.shape[0])
    centered_difference = FiniteDifference(order=2)
    np.testing.assert_allclose(centered_difference(x, t), x_dot, atol=1e-8)


def test_centered_difference_xy_yx(data_2dspatial):
    x_grid, y_grid, u = data_2dspatial
    u_xy = np.zeros(u.shape)
    u_yx = np.zeros(u.shape)
    u_y = FiniteDifference(order=2, d=1, axis=1)._differentiate(u, y_grid)
    u_x = FiniteDifference(order=2, d=1, axis=0)._differentiate(u, x_grid)
    u_xy = FiniteDifference(order=2, d=1, axis=1)._differentiate(u_x, y_grid)
    u_yx = FiniteDifference(order=2, d=1, axis=0)._differentiate(u_y, x_grid)
    np.testing.assert_allclose(u_xy, u_yx)
    u_y = FiniteDifference(order=1, d=1, axis=1)._differentiate(u, y_grid)
    u_x = FiniteDifference(order=1, d=1, axis=0)._differentiate(u, x_grid)
    u_xy = FiniteDifference(order=1, d=1, axis=1)._differentiate(u_x, y_grid)
    u_yx = FiniteDifference(order=1, d=1, axis=0)._differentiate(u_y, x_grid)
    np.testing.assert_allclose(u_xy, u_yx)


def test_centered_difference_hot(data_derivative_2d):
    x, _ = data_derivative_2d
    t = np.linspace(0, x.shape[0], x.shape[0])
    dt = t[1] - t[0]
    atol = 1e-8
    for d in range(1, 2):
        forward_difference = FiniteDifference(order=1, d=d)._differentiate
        np.testing.assert_allclose(
            forward_difference(x, t=dt),
            forward_difference(x, t=t),
            atol=atol,
        )
    for d in range(1, 6):
        centered_difference = FiniteDifference(order=2, d=d)._differentiate
        np.testing.assert_allclose(
            centered_difference(x, t=dt),
            centered_difference(x, t=t),
            atol=atol,
        )
    for d in range(1, 6):
        spectral_deriv = SpectralDerivative(d=d)._differentiate
        np.testing.assert_allclose(
            spectral_deriv(x, t=dt),
            spectral_deriv(x, t=t),
            atol=atol,
        )


# Alternative implementation of the four tests above using parametrization
@pytest.mark.parametrize(
    "data, order",
    [
        (pytest.lazy_fixture("data_derivative_1d"), 2),
        (pytest.lazy_fixture("data_derivative_2d"), 2),
        (pytest.lazy_fixture("data_derivative_1d"), 4),
        (pytest.lazy_fixture("data_derivative_2d"), 4),
        (pytest.lazy_fixture("data_derivative_1d"), 8),
        (pytest.lazy_fixture("data_derivative_2d"), 8),
    ],
)
def test_finite_difference(data, order):
    x, x_dot = data
    method = FiniteDifference(order=order)
    np.testing.assert_allclose(method(x), x_dot)
    method = SmoothedFiniteDifference()
    np.testing.assert_allclose(method(x), x_dot)


def test_order_error():
    with pytest.raises(ValueError):
        FiniteDifference(order=-1)
    with pytest.raises(ValueError):
        FiniteDifference(d=-1)
    with pytest.raises(ValueError):
        FiniteDifference(d=2, order=1)
    with pytest.raises(ValueError):
        FiniteDifference(d=1, order=0.5)
    with pytest.raises(ValueError):
        FiniteDifference(d=1, order=0)
    with pytest.raises(ValueError):
        FiniteDifference(d=0, order=1)


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


@pytest.mark.parametrize(
    "data, derivative_kws",
    [
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spline", s=1e-2)),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spline", s=1e-2)),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
    ],
)
def test_wrapper_equivalence_with_dxdt(data, derivative_kws):
    x, _ = data
    t = np.arange(x.shape[0])

    if np.ndim(x) == 1:
        np.testing.assert_allclose(
            dxdt(x.reshape(-1, 1), t, axis=0, **derivative_kws),
            SINDyDerivative(**derivative_kws)(x, t),
        )
    else:
        np.testing.assert_allclose(
            dxdt(x, t, axis=0, **derivative_kws),
            SINDyDerivative(**derivative_kws)(x, t),
        )


@pytest.mark.parametrize(
    "data, derivative_kws",
    [
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spectral")),
        (pytest.lazy_fixture("data_derivative_1d"), dict(kind="spline", s=1e-2)),
        (pytest.lazy_fixture("data_derivative_2d"), dict(kind="spline", s=1e-2)),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="trend_filtered", order=0, alpha=1e-2),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="finite_difference", k=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_1d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
        (
            pytest.lazy_fixture("data_derivative_2d"),
            dict(kind="savitzky_golay", order=3, left=1, right=1),
        ),
    ],
)
def test_derivative_output_shape(data, derivative_kws):
    x, x_dot = data
    t = np.arange(x.shape[0])

    method = SINDyDerivative(**derivative_kws)

    assert x_dot.shape == method(x).shape
    assert x_dot.shape == method(x, t).shape


def test_bad_t_values(data_derivative_1d):
    x, x_dot = data_derivative_1d

    method = SINDyDerivative(kind="finite_difference", k=1)

    with pytest.raises(ValueError):
        method(x, t=-1)

    with pytest.raises(ValueError):
        method._differentiate(x, t=-1)


def test_centered_difference_hot_axis(data_2d_resolved_pde):
    _, u_flat, u_dot_flat = data_2d_resolved_pde
    x = np.reshape(u_flat, (8, 8, 1000, 2))
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    atol = 1e-8
    for d in range(1, 2):
        forward_difference = FiniteDifference(order=1, d=d, axis=-2)._differentiate
        np.testing.assert_allclose(
            forward_difference(x, t=dt),
            forward_difference(x, t=t),
            atol=atol,
        )
    for d in range(1, 6):
        centered_difference = FiniteDifference(order=2, d=d, axis=-2)._differentiate
        np.testing.assert_allclose(
            centered_difference(x, t=dt),
            centered_difference(x, t=t),
            atol=atol,
        )
    for d in range(1, 6):
        spectral_deriv = SpectralDerivative(d=d, axis=-2)._differentiate
        np.testing.assert_allclose(
            spectral_deriv(x, t=dt),
            spectral_deriv(x, t=t),
            atol=atol,
        )


def test_centered_difference_noaxis_vs_axis(data_2d_resolved_pde):
    _, u_flat, u_dot_flat = data_2d_resolved_pde
    n = 8
    x = np.reshape(u_flat, (n, n, 1000, 2))
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    atol = 1e-10
    for d in range(1, 6):
        centered_difference = FiniteDifference(order=2, d=d, axis=-2)._differentiate
        slow_differences = np.zeros(x.shape)
        slow_differences_t = np.zeros(x.shape)
        for i in range(n):
            for j in range(n):
                slow_differences[i, j, :, :] = FiniteDifference(
                    order=2, d=d
                )._differentiate(x[i, j, :, :], t=dt)
                slow_differences_t[i, j, :, :] = FiniteDifference(
                    order=2, d=d
                )._differentiate(x[i, j, :, :], t=t)
        np.testing.assert_allclose(
            centered_difference(x, t=dt),
            slow_differences,
            atol=atol,
        )
        np.testing.assert_allclose(
            centered_difference(x, t=t),
            slow_differences_t,
            atol=atol,
        )
