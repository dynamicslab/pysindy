import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.integrate import quad

from pysindy._weak import _get_spatial_endpoints
from pysindy._weak import _left_weights
from pysindy._weak import _linear_weights
from pysindy._weak import _phi
from pysindy._weak import _phi_int
from pysindy._weak import _right_weights
from pysindy._weak import _xphi_int


def test_get_spatial_endpoints():
    expected = ((-1, 3, 0), (4, 10, 1.5))
    x = np.linspace(expected[0][0], expected[1][0])
    y = np.linspace(expected[0][1], expected[1][1])
    z = np.linspace(expected[0][2], expected[1][2])
    st_grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    result = _get_spatial_endpoints(st_grid)
    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])


@pytest.mark.parametrize("p", [2, 3, 4])
def test_test_function_phi(p):
    # Convince yourself that these are the correct derivatives
    # by differentiating (1-x^2)^p by hand.
    # Built-in phi uses vectorized operations, generic by derivative,
    # So these are easy to read but slower and manually defined by
    # derivative order.
    def d0(x):
        return (1 - x**2) ** p

    def d1(x):
        return -2 * p * x * (1 - x**2) ** (p - 1)

    def d2(x):
        return -2 * p * (1 - x**2) ** (p - 1) + 4 * p * (p - 1) * x**2 * (
            1 - x**2
        ) ** (p - 2)

    rt2 = np.sqrt(2)
    x = np.array([0, rt2 / 2, 1])
    expected = np.array([d0(0), d0(rt2 / 2), d0(1)])
    result = _phi(x, 0, p)
    assert_allclose(result, expected)
    expected = np.array([d1(0), d1(rt2 / 2), d1(1)])
    result = _phi(x, 1, p)
    assert_allclose(result, expected)
    # Second derivative doesn't die at boundary if p=2
    expected = np.array([d2(0), d2(rt2 / 2), d2(1)])
    result = _phi(x, 2, p)
    assert_allclose(result, expected)


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("d", [0, 1, 2])
@pytest.mark.parametrize("n_grid", [10, 100, 1000])
def test_weak_derivative(p, d, n_grid):
    def true_f(x):
        return x**2

    expected = quad(lambda x: true_f(x) * _phi(np.array([x]), d, p), -1, 1)[0]
    x_i = np.linspace(-1, 1, n_grid)
    f_i = true_f(x_i)
    weights = _linear_weights(x_i, d, p)
    assert_allclose(
        weights[0], _left_weights(np.array([x_i[0]]), np.array([x_i[1]]), d, p)
    )
    assert_allclose(
        weights[-1], _right_weights(np.array([x_i[-2]]), np.array([x_i[-1]]), d, p)
    )
    result = sum(f_i * weights)
    assert_allclose(result, expected, atol=1 / n_grid)
