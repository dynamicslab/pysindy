import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.integrate import dblquad
from scipy.integrate import quad

from pysindy import AxesArray
from pysindy import PDELibrary
from pysindy import PolynomialLibrary
from pysindy import STLSQ
from pysindy._weak import _derivative_weights
from pysindy._weak import _flatten_libraries
from pysindy._weak import _get_spatial_endpoints
from pysindy._weak import _integrate_by_parts
from pysindy._weak import _integrate_product_by_parts
from pysindy._weak import _linear_weights
from pysindy._weak import _phi
from pysindy._weak import WeakSINDy
from pysindy.feature_library import ConcatLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import TensoredLibrary


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
def test_integral_weights(p, d, n_grid):
    r"""Tests the 1D fast piecewise-linear integration weights.

    .. math::

        \int_{-1}^{1} f(x) * phi^{(d)}(x) dx

    Evaluates the integration directly using scipy's default quadrature,
    then compares to our integration-by-parts that moves derivatives onto
    the test function :math:`phi(x)` and approximates f as piecewise linear.
    Comparison should be increasingly accurate as number of grid points
    increases.

    """

    def true_f(x):
        return x**2

    expected = quad(lambda x: true_f(x) * _phi(np.array([x]), d, p), -1, 1)[0]
    x_i = np.linspace(-1, 1, n_grid)
    f_i = true_f(x_i)
    weights = _linear_weights(x_i, d, p)
    result = sum(f_i * weights)
    assert_allclose(result, expected, atol=1 / n_grid)


@pytest.mark.parametrize(
    "true_f", [lambda x: np.sin(x), lambda x: np.ones_like(x)], ids=["sin", "const"]
)
@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("deriv_op", [(0,), (1,), (2,)], ids=("D0", "D1", "D2"))
def test_integrate_domain1d(true_f, p, deriv_op):
    grid1d = np.linspace(2, 5, 30)

    xl, xu = grid1d[0], grid1d[-1]
    y_of_x = lambda x: -1 + 2 * (x - xl) / (xu - xl)
    dy_dx = 2 / (xu - xl)

    def integrand(x):
        return (
            true_f(x)
            * _phi(np.array([y_of_x(x)]), deriv_op[0], p)
            * dy_dx ** deriv_op[0]
        )

    expected, _ = quad(integrand, xl, xu)  # type: ignore
    half_dims = AxesArray(np.array([(xu - xl) / 2]), axes={"ax_coord": 0})
    grid_shape = (len(grid1d),)
    scaled_subgrid = [np.linspace(-1, 1, grid_shape[0])]
    x_mesh = grid1d[..., None]
    f_i = true_f(x_mesh[..., 0])
    weights = _derivative_weights(scaled_subgrid, half_dims, grid_shape, deriv_op, p)
    result = f_i.flatten() @ np.asarray(weights).flatten()
    trap_err_est = max(half_dims**2 / np.array(grid_shape) ** 2)
    # If expected is zero, can't use rtol
    if np.linalg.norm(expected) < 1e-5:
        assert_allclose(result, expected, atol=trap_err_est)
    else:
        assert_allclose(result, expected, rtol=trap_err_est)


@pytest.mark.parametrize(
    "true_f",
    [lambda x, y: np.sin(x) + np.sin(y), lambda x, y: np.ones_like(x)],
    ids=["sin", "const"],
)
@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("deriv_op", [(0, 0), (1, 1), (2, 0)], ids=("D0", "D1", "D2"))
def test_integrate_domain2d(true_f, p, deriv_op):
    grid1d = [np.linspace(-1, 1, 10), np.linspace(2, 5, 30)]

    xl, xu, yl, yu = grid1d[0][0], grid1d[0][-1], grid1d[1][0], grid1d[1][-1]
    u_of_x = lambda x: -1 + 2 * (x - xl) / (xu - xl)
    u_of_y = lambda y: -1 + 2 * (y - yl) / (yu - yl)
    du_dx = 2 / (xu - xl)
    du_dy = 2 / (yu - yl)

    def integrand(y, x):  # yes... y, then x
        return (
            true_f(x, y)
            * _phi(np.array([u_of_x(x)]), deriv_op[0], p)
            * _phi(np.array([u_of_y(y)]), deriv_op[1], p)
            * du_dx ** deriv_op[0]
            * du_dy ** deriv_op[1]
        )

    expected, _ = dblquad(integrand, xl, xu, yl, yu)  # type: ignore
    half_dims = AxesArray(
        np.array([(xu - xl) / 2, (yu - yl) / 2]), axes={"ax_coord": 0}
    )
    grid_shape = (len(grid1d[0]), len(grid1d[1]))
    scaled_subgrid = [
        np.linspace(-1, 1, grid_shape[0]),
        np.linspace(-1, 1, grid_shape[1]),
    ]
    xy_i = np.stack(np.meshgrid(grid1d[0], grid1d[1], indexing="ij"), axis=-1)
    f_i = true_f(xy_i[..., 0], xy_i[..., 1])
    weights = _derivative_weights(scaled_subgrid, half_dims, grid_shape, deriv_op, p)
    result = f_i.flatten() @ np.asarray(weights).flatten()
    trap_err_est = max(half_dims**2 / np.array(grid_shape) ** 2)
    # If expected is zero, can't use rtol
    if np.linalg.norm(expected) < 1e-5:
        assert_allclose(result, expected, atol=trap_err_est)
    else:
        assert_allclose(result, expected, rtol=trap_err_est)


def test_flatten_libraries():
    lib = PolynomialLibrary(1) * (FourierLibrary() + PolynomialLibrary(3))
    result = _flatten_libraries(lib)
    expected = PolynomialLibrary(1) * FourierLibrary() + PolynomialLibrary(
        1
    ) * PolynomialLibrary(4)
    assert result == expected


def test_weak_class(data_1d_random_pde):
    model = WeakSINDy(PolynomialLibrary(), STLSQ())
    t, x, u, u_dot = data_1d_random_pde
    st_grid = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1)

    model.fit(x=[u], st_grids=[st_grid])


def test_integrate_by_parts():
    spatial_grid = np.array([[[0]], [[0]]])
    features = PDELibrary(derivative_order=2, spatial_grid=spatial_grid)
    inputs = [np.ones((1, 2))]
    features.fit(inputs)
    result = _integrate_by_parts(features)
    # Current ordering is
    # u_y, u_yy, u_x, u_xy, u_xx
    expected = [
        ((0, 0), None, -1, (0, 1)),
        ((0, 0), None, 1, (0, 2)),
        ((0, 0), None, -1, (1, 0)),
        ((0, 0), None, 1, (1, 1)),
        ((0, 0), None, 1, (2, 0)),
    ]
    assert result == expected


def test_integrate_product_by_parts():
    f_lib = PolynomialLibrary()
    d_lib = PDELibrary(derivative_order=4)
    features = f_lib * d_lib
    inputs = [np.ones((1, 3))]
    features.fit(inputs)
    result = _integrate_product_by_parts(f_lib, d_lib)
    expected = []
    assert result == expected
