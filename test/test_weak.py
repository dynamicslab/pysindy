import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.integrate import dblquad
from scipy.integrate import quad

from pysindy._typing import Float1D
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
from pysindy._weak import UniformEvenBump, TestFunctionPhi
from pysindy._weak import WeakSINDy
from pysindy._weak import SubdomainSpecs
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
def test_uniform_even_bump(p):
    # Convince yourself that these are the correct derivatives
    # by differentiating (1-x^2)^p by hand.
    # Built-in phi uses vectorized operations, generic by derivative,
    # So these are easy to read but slower and manually defined by
    # derivative order.
    test_func = UniformEvenBump(p)
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
    result = test_func.phi(x, 0)
    assert_allclose(result, expected)
    expected = np.array([d1(0), d1(rt2 / 2), d1(1)])
    result = test_func.phi(x, 1)
    assert_allclose(result, expected)
    # Second derivative doesn't die at boundary if p=2
    expected = np.array([d2(0), d2(rt2 / 2), d2(1)])
    result = test_func.phi(x, 2)
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

    test_func = UniformEvenBump(p)
    expected = quad(lambda x: true_f(x) * test_func.phi(np.array([x]), d), -1, 1)[0]
    x_i = np.linspace(-1, 1, n_grid)
    f_i = true_f(x_i)
    weights = _linear_weights(x_i, d, test_func)
    result = sum(f_i * weights)
    assert_allclose(result, expected, atol=1 / n_grid)


@pytest.mark.parametrize(
    "true_f", [lambda x: np.sin(x), lambda x: np.ones_like(x)], ids=["sin", "const"]
)
@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("deriv_op", [(0,), (1,), (2,)], ids=("D0", "D1", "D2"))
def test_integrate_domain1d(true_f, p, deriv_op):
    grid1d = np.linspace(2, 5, 30)
    test_func = UniformEvenBump(p)
    xl, xu = grid1d[0], grid1d[-1]
    y_of_x = lambda x: -1 + 2 * (x - xl) / (xu - xl)
    dy_dx = 2 / (xu - xl)

    def integrand(x):
        return (
            true_f(x)
            * test_func.phi(np.array([y_of_x(x)]), deriv_op[0])
            * dy_dx ** deriv_op[0]
        )

    expected, _ = quad(integrand, xl, xu)  # type: ignore
    half_dims = AxesArray(np.array([(xu - xl) / 2]), axes={"ax_coord": 0})
    grid_shape = (len(grid1d),)
    scaled_subgrid = [np.linspace(-1, 1, grid_shape[0])]
    x_mesh = grid1d[..., None]
    f_i = true_f(x_mesh[..., 0])
    weights = _derivative_weights(scaled_subgrid, half_dims, grid_shape, deriv_op, test_func)
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
    test_func = UniformEvenBump(p)

    xl, xu, yl, yu = grid1d[0][0], grid1d[0][-1], grid1d[1][0], grid1d[1][-1]
    u_of_x = lambda x: -1 + 2 * (x - xl) / (xu - xl)
    u_of_y = lambda y: -1 + 2 * (y - yl) / (yu - yl)
    du_dx = 2 / (xu - xl)
    du_dy = 2 / (yu - yl)

    def integrand(y, x):  # yes... y, then x
        return (
            true_f(x, y)
            * test_func.phi(np.array([u_of_x(x)]), deriv_op[0])
            * test_func.phi(np.array([u_of_y(y)]), deriv_op[1])
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
    weights = _derivative_weights(scaled_subgrid, half_dims, grid_shape, deriv_op, test_func)
    result = f_i.flatten() @ np.asarray(weights).flatten()
    trap_err_est = max(half_dims**2 / np.array(grid_shape) ** 2)
    # If expected is zero, can't use rtol
    if np.linalg.norm(expected) < 1e-5:
        assert_allclose(result, expected, atol=trap_err_est)
    else:
        assert_allclose(result, expected, rtol=trap_err_est)


def test_flatten_libraries():
    fake_spatial_grid = np.array([[[0]], [[0]]])
    pde_lib = PDELibrary(derivative_order=2, spatial_grid=fake_spatial_grid)
    lib = PolynomialLibrary(1) * (FourierLibrary() + pde_lib)
    result = _flatten_libraries(lib)

    assert isinstance(result, ConcatLibrary)
    assert len(result.libraries) == 2
    assert isinstance(result.libraries[0], TensoredLibrary)
    assert isinstance(result.libraries[1], TensoredLibrary)
    assert type(result.libraries[0].libraries[0]) == PolynomialLibrary
    assert type(result.libraries[0].libraries[1]) == FourierLibrary
    assert type(result.libraries[1].libraries[0]) == PolynomialLibrary
    assert type(result.libraries[1].libraries[1]) == PDELibrary


def test_weak_class(data_1d_random_pde):
    t, x, u, u_dot = data_1d_random_pde
    mesh = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1)
    f_lib = PolynomialLibrary()
    u_lib = PDELibrary(derivative_order=2, spatial_grid=x)
    lib = f_lib + u_lib + f_lib * u_lib
    model = WeakSINDy(lib, STLSQ())

    model.fit(x=[u], st_grids=[mesh])
    model.print()


def test_integrate_by_parts():
    spatial_grid = np.array([[[0]], [[0]]])
    features = PDELibrary(derivative_order=2, spatial_grid=spatial_grid)
    inputs = [np.ones((1, 2))]
    features.fit(inputs)
    result = _integrate_by_parts(features.multiindices)
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


# need to parametrize this for 1d high order and 2d
def test_integrate_product_by_parts():
    f_lib = PolynomialLibrary()
    spatial_grid = np.array([[[0]], [[0]]])
    d_lib = PDELibrary(derivative_order=2, spatial_grid=spatial_grid)
    features = f_lib * d_lib
    inputs = [np.ones((1, 2))]
    features.fit(inputs)
    result = _integrate_product_by_parts(f_lib, d_lib.multiindices)
    expected = [
        [((0, 1), (f_lib, (0, 0)), 1, (0, 0))],
        [((0, 1), (f_lib, (0, 0)), -1, (0, 1)), ((0, 1), (f_lib, (0, 1)), -1, (0, 0))],
        [((1, 0), (f_lib, (0, 0)), 1, (0, 0))],
        [((1, 1), (f_lib, (0, 0)), 1, (0, 0))],
        [((1, 0), (f_lib, (0, 0)), -1, (1, 0)), ((1, 0), (f_lib, (1, 0)), -1, (0, 0))],
    ]
    assert result == expected


def test_weak_feature_ordering(fake_domains):
    st_grid = fake_domains.domain
    # 3 quick commits needed:
    # DONE: Fake subdomains,
    # Fake test function
    # DONE: linear_weights takes test function
    f_lib = PolynomialLibrary()
    d_lib = PDELibrary(derivative_order=2, spatial_grid=st_grid)
    features = f_lib * d_lib

    sorted_lib = WeakSINDy._sort_feature_library(features)
    # The observed field is a saddle point of quadratic monomials
    # we map expected terms to their feature_names;
    # generate SemiTerms for the weak integrals,
    # and compare the sorted library's feature names to the resulting transformation.
    u = st_grid[..., 0] ** 2 - st_grid[..., 1] ** 2
    v = - st_grid[..., 0] ** 2 + st_grid[..., 1] ** 2
    # We need a fake test function that returns constants for all derivatives.
    # Or something like a dirac delta.
    # We also need a subdomain specs where each subdomain is a single point
    # and then reassemble the features in the input shape from the subdomains.
    expected_features = {
        "u_11": 2 * np.ones_like(u),
        "u_22": -2 * np.ones_like(u),
        "u_12": np.zeros_like(u),
        "u_1": 2 * st_grid[..., 0],
        "u_2": -2 * st_grid[..., 1],
        "v_11": -2 * np.ones_like(u),
        "v_22": 2 * np.ones_like(u),
        "v_12": np.zeros_like(u),
        "v_1": -2 * st_grid[..., 0],
        "v_2": 2 * st_grid[..., 1],
        "1": np.ones_like(u),
        "u v": u * v,
        "u^2": u**2,
        "v^2": v**2
    }


@pytest.fixture(scope="session")
def fake_domains() -> SubdomainSpecs:
    # A fake subdomain spec where each subdomain maps to a single point,
    # so that a dirac test function will sift the weak value to the true value
    # Does scaling need to be adjusted so integral loses its scale?
    ordinates = np.arange(0, 1, .1)
    times = ordinates
    st_grid = np.stack(np.meshgrid(ordinates, times, indexing="ij"), axis=-1)
    st_grid = AxesArray(st_grid, axes={"ax_spatial": 0, "ax_time": 1, "ax_coord": 2})
    sorted_inds = np.ndindex(st_grid.shape[:-1])
    axis_inds_per_subdom = []
    subgrid_dims = []
    subgrid_shapes = []
    subgrids_scaled = []
    for ind in sorted_inds:
        axis_inds_per_subdom.append([np.array([ind[0]]), np.array([ind[1]])])
        subgrid_dims.append((.05, .05))
        subgrid_shapes.append((1,1))
        subgrids_scaled.append(np.array([[[0]], [[0]]]))
    subgrid_dims = np.array(subgrid_dims)
    subgrid_shapes = np.array(subgrid_shapes)
    return SubdomainSpecs(
        domain=st_grid,
        axis_inds_per_subdom=axis_inds_per_subdom,
        subgrid_dims=subgrid_dims,
        subgrid_shapes=subgrid_shapes,
        subgrids_scaled=subgrids_scaled
    )

class MockSiftingTestFunction(TestFunctionPhi):
    max_order = 999  # type: ignore

    def phi(self, x: Float1D, d: int) -> Float1D:
        return np.ones_like(x)
    
    def phi_int(self, x: Float1D, d: int) -> Float1D:
        return self.phi(x, d)
    
    def xphi_int(self, x: Float1D, d: int) -> Float1D:
        return x * self.phi(x, d)
