from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional

import numpy as np
from scipy.special import binom
from scipy.special import perm
from typing_extensions import Self

from ._core import _adapt_to_multiple_trajectories
from ._core import _BaseSINDy
from ._core import _check_multiple_trajectories
from ._core import _validate_inputs
from ._core import standardize_shape
from ._core import validate_control_variables
from ._typing import Float1D
from ._typing import Float2D
from ._typing import FloatND
from ._typing import Int1D
from .differentiation import FiniteDifference
from .differentiation.base import BaseDifferentiation
from .feature_library import ConcatLibrary
from .feature_library import PDELibrary
from .feature_library import TensoredLibrary
from .feature_library.base import BaseFeatureLibrary
from .optimizers.base import BaseOptimizer
from .utils import comprehend_axes
from .utils._axes import AxesArray


@dataclass
class SubdomainSpecs:
    """The schematics for subdomains

    Attributes:
        domain: The overall spatiotemporal domain, as a meshgrid of (optional)
            spatial axes, followed by a time axis, stacked along a final
            coordinate axis
        axis_inds_per_subdom: An open mesh of indices along each axis that
            belong in each subdomain.  Outer list indexes the subdomains,
            inner list indexes the spatiotemporal axes
        subgrid_dims: The dimensions of each subdomain in original spatiotemporal
            units, divided by two (i.e. half the length of each axis in a subodmain).
            Rows index subdomain, columns index spatiotemporal axis
        subgrid_shapes: The shape of each supdomain meshgrid
        subgrids_scaled: subdomain meshgrid in same format as ``domain`` attr,
            rescaled along each axis from [-1, 1]
    """

    domain: AxesArray
    axis_inds_per_subdom: list[list[AxesArray]]
    subgrid_dims: Float2D
    subgrid_shapes: np.ndarray
    subgrids_scaled: list[AxesArray]

    def __post_init__(self):
        n_st_axes0 = self.domain.ndim - 1
        n_subdomains0 = len(self.axis_inds_per_subdom)
        n_st_axes1 = min(*[len(subdom) for subdom in self.axis_inds_per_subdom])
        n_st_axes2 = max(*[len(subdom) for subdom in self.axis_inds_per_subdom])
        n_subdomains1, n_st_axes3 = self.subgrid_dims.shape
        n_subdomains2, n_st_axes4 = self.subgrid_shapes.shape
        n_subdomains3 = len(self.subgrids_scaled)
        n_st_axes5 = min(*[subdom.ndim - 1 for subdom in self.subgrids_scaled])
        n_st_axes6 = max(*[subdom.ndim - 1 for subdom in self.subgrids_scaled])
        err_msg = "Inconsistent shape of subdomains/number of subdomains"

        if not (
            (n_st_axes0 == n_st_axes1 == n_st_axes2 == n_st_axes3)
            and (n_st_axes3 == n_st_axes4 == n_st_axes5 == n_st_axes6)
            and (n_subdomains0 == n_subdomains1 == n_subdomains2 == n_subdomains3)
        ):
            raise ValueError(err_msg)
        self.n_subdomains = n_subdomains0
        self.grid_ndim = n_st_axes0

    def __getitem__(self, i: int):
        return (
            self.axis_inds_per_subdom[i],
            self.subgrid_dims[i],
            self.subgrid_shapes[i],
            self.subgrids_scaled[i],
        )

    def __iter__(self):
        for i in range(self.n_subdomains):
            yield self[i]


@dataclass
class TestFunctionPhi:
    _phi: Callable
    _phi_int: Callable
    _xphi_int: Callable


class WeakSINDy(_BaseSINDy):
    """For test function basis, see Section 2.4 of Messenger and Bortz, "Weak SINDy:
    Galerkin-Based Data-Driven Model Selection".  Here, the polynomial exponent
    decaying at the left and right boundary are the same: p==q in the manuscript's
    notation.
    """

    def __init__(
        self,
        feature_library: Optional[BaseFeatureLibrary],
        optimizer: Optional[BaseOptimizer],
        n_subdomains: int = 100,
        test_fn_order=4,
        spatial_diff_method: type[BaseDifferentiation] = FiniteDifference,
        diff_kwargs: Optional[dict] = None,
    ):
        super().__init__(feature_library=feature_library, optimizer=optimizer)
        self.n_subdomains = n_subdomains
        self.test_fn_order = test_fn_order
        self.spatial_diff_method = spatial_diff_method
        self.diff_kwargs = diff_kwargs
        self.__post_init()

    def __post_init(self):
        if self.diff_kwargs is None:
            self.diff_kwargs = {}
        if self.n_subdomains <= 0:
            raise ValueError("The number of subdomains must be > 0")
        if self.test_fn_order <= 0:
            raise ValueError("Poly degree of the spatial weights must be > 0")

        def max_derivative_order(library: BaseFeatureLibrary):
            if isinstance(library, PDELibrary):
                return library.derivative_order
            elif isinstance(library, (ConcatLibrary, TensoredLibrary)):
                return max(*[max_derivative_order(lib) for lib in library.libraries])
            return 0

        derivative_order = max_derivative_order(self.feature_library)
        if self.test_fn_order < derivative_order:
            self.test_fn_order = derivative_order

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init()

    def fit(
        self,
        x: np.ndarray | Sequence[FloatND],
        st_grids: np.ndarray | Sequence[np.ndarray],
        *,
        H_xt: Optional[float | FloatND | Sequence[FloatND]] = None,
        u: Optional[np.ndarray | Sequence[FloatND]] = None,
        feature_names: Optional[list[str]] = None
    ) -> Self:
        if not _check_multiple_trajectories(x, st_grids, None, u):
            x = cast(np.ndarray, x)
            st_grids = cast(np.ndarray, st_grids)
            u = cast(Optional[np.ndarray], u)
            x, st_grids, _, u = _adapt_to_multiple_trajectories(x, st_grids, None, u)
        x = [standardize_shape(xi) for xi in x]
        st_grids = [standardize_shape(grid) for grid in st_grids]
        if u is not None:
            u = [standardize_shape(ui) for ui in u]
        _validate_inputs(x, st_grids, None, u)

        self.feature_names_ = feature_names

        if u is None:
            self.n_control_features_ = 0
        else:
            u = validate_control_variables(x, u)
            self.n_control_features_ = u[0].n_coord  # type: ignore

        # Here's where we do the fancy stuff!
        subdomain_dims = _normalize_subdomain_dims(st_grids, H_xt)
        subdomain_specs = [
            _make_domains(grid, sub_dim, self.n_subdomains)
            for grid, sub_dim in zip(st_grids, subdomain_dims)
        ]
        # Weak should be subclass of PDEFIND?  Or PDELib.transform gives placeholders?
        def get_PDElib(lib):
            if isinstance(lib, PDELibrary):
                return lib
            elif isinstance(lib, (ConcatLibrary, TensoredLibrary)):
                libs = [get_PDElib(sublib) for sublib in lib.libraries]
                return next((sublib for sublib in libs if sublib), False)
            else:
                return False

        pde_lib = get_PDElib(self.feature_library)
        if pde_lib:
            multiindices = pde_lib.multiindices  # type: ignore
            time_zeros = np.zeros((len(multiindices), 1), dtype=int)
            time_deriv = np.zeros((1, multiindices.shape[1] + 1), dtype=int)
            time_deriv[0, -1] = 1
            multiindices = np.hstack((multiindices, time_zeros))
            multiindices = np.vstack((multiindices, time_deriv))
        else:
            multiindices = np.zeros((1, subdomain_specs[0].grid_ndim), dtype=int)
            multiindices[0, -1] = 1

        multiindices = [tuple(deriv_op) for deriv_op in multiindices]

        weights_per_deriv_op = [
            _set_up_weights(spec, self.test_fn_order, multiindices)
            for spec in subdomain_specs
        ]

        # Need feature names and set n_output_features_ (this is same with PDE library)
        u_dot_wk = [
            convert_u_dot_integral(xi, weights[0], spec.axis_inds_per_subdom)
            for xi, weights, spec in zip(x, weights_per_stgrid, subdomain_specs)
        ]

        self._fit_shape()
        return self

    def predict(self):
        pass

    def simulate(self):
        pass

    def score(self):
        pass


def _normalize_subdomain_dims(
    st_grids: list[AxesArray],
    subdomain_dims: Optional[float | FloatND | Sequence[FloatND]],
) -> list[AxesArray]:
    """Establish domain-dependent functionality"""
    n_traj = len(st_grids)
    grid_ndim = st_grids[0].ndim - 1
    xts = [_get_spatial_endpoints(grid) for grid in st_grids]
    L_xts = [xt2 - xt1 for xt1, xt2 in xts]
    if subdomain_dims is not None:
        if np.isscalar(subdomain_dims):
            subdomain_dims = [
                AxesArray(np.array(grid_ndim * [subdomain_dims]), {"ax_coord": 0})
                for _ in range(n_traj)
            ]
        elif isinstance(subdomain_dims, np.ndarray):
            subdomain_dims = [
                AxesArray(subdomain_dims, {"ax_coord": 0}) for _ in range(n_traj)
            ]
        else:
            subdomain_dims = cast(Sequence[FloatND], subdomain_dims)
            subdomain_dims = [AxesArray(dim, {"ax_coord": 0}) for dim in subdomain_dims]
        if any([grid_ndim != len(dims) for dims in subdomain_dims]):
            raise ValueError(
                "The user-defined grid (spatiotemporal_grid) and "
                "the user-defined sizes of the subdomains for the "
                "weak form do not have the same # of spatiotemporal "
                "dimensions. For instance, if spatiotemporal_grid is 4D, "
                "then subdomain_dims should contain a list of four "
                "subdomain lengths."
            )
        if any([(dim < 0).any() for dim in subdomain_dims]):
            raise ValueError("subdomain dimensions must be positive numbers.")
        elif any(
            [(dim >= L_xt / 2.0).any() for dim, L_xt in zip(subdomain_dims, L_xts)]
        ):
            raise ValueError(
                "Some subdomain dimension is larger than half the "
                "corresponding grid dimension."
            )
    else:
        subdomain_dims = [L_xt / 20.0 for L_xt in L_xts]
    return subdomain_dims


def convert_u_dot_integral(
    u: AxesArray, fulltweights, axis_inds_per_subdom: list[list[AxesArray]]
) -> AxesArray:
    """
    Takes a full set of spatiotemporal fields u(x, t) and finds the weak
    form of u_dot.
    """
    n_subdomains = len(fulltweights)
    grid_dim = u.ndim - 1
    u_dot_integral = np.zeros((n_subdomains, u.shape[-1]))
    u_subdomains = [u[np.ix_(*ax_inds)] for ax_inds in axis_inds_per_subdom]

    u_dot_integral = np.array(
        [
            np.tensordot(weights, values, axis=range(u.ndim - 1))
            for weights, values in zip(fulltweights, u_subdomains)
        ]
    )

    return AxesArray(u_dot_integral, {"ax_time": 0, "ax_coord": 1})


def _get_spatial_endpoints(st_grid: FloatND) -> tuple[Float1D, Float1D]:
    """Retrieve the min/max coordinates for each axis of a meshgrid"""
    grid_ndim = st_grid.ndim - 1
    min_inds = (0,) * grid_ndim
    max_inds = (-1,) * grid_ndim
    # After python 3.10 EOL, can write st_grid[*min_inds, :]
    return st_grid[(*min_inds, Ellipsis)], st_grid[(*max_inds, Ellipsis)]


def _set_up_weights(
    subdoms: SubdomainSpecs, p: int, multiindices: list[tuple[int, ...]]
) -> dict[tuple[int, ...], list[AxesArray]]:
    """
    Sets up weights needed for the weak library. Integrals over domain cells are
    approximated as dot products of weights and the input data.
    """
    weight_lookup: dict[tuple[int, ...], list[AxesArray]] = defaultdict(list)
    for deriv_op in multiindices:
        for _, dims, shape, scaled_coord in iter(subdoms):
            weights = _derivative_weights(
                _dense_to_open_mesh(scaled_coord), dims, shape, deriv_op, p
            )
            weight_lookup[deriv_op].append(weights)

    return weight_lookup


def _make_domains(
    st_grid: AxesArray, subdomain_dims: AxesArray, n_subdomains: int
) -> SubdomainSpecs:
    xt1, xt2 = _get_spatial_endpoints(st_grid)
    grid_ndim = st_grid.ndim - 1
    domain_centers = AxesArray(
        np.zeros((n_subdomains, grid_ndim)), {"ax_sample": 0, "ax_coord": 1}
    )
    for ax_ind in range(grid_ndim):
        domain_centers[:, ax_ind] = np.random.uniform(
            xt1[ax_ind] + subdomain_dims[ax_ind],
            xt2[ax_ind] - subdomain_dims[ax_ind],
            size=n_subdomains,
        )

    # Indices for space-time points that lie in the domain cells
    axis_inds_per_subdom: list[list[AxesArray]] = []
    ind_subdom = 0
    attempts = 0
    while ind_subdom < n_subdomains:
        inds = []
        for ax_ind in range(grid_ndim):
            s: list[int | slice] = [0] * (grid_ndim + 1)
            s[ax_ind] = slice(None)
            s[-1] = ax_ind
            ax_vals = st_grid[tuple(s)]
            cell_left = domain_centers[ind_subdom][ax_ind] - subdomain_dims[ax_ind]
            cell_right = domain_centers[ind_subdom][ax_ind] + subdomain_dims[ax_ind]
            newinds = AxesArray(
                ((ax_vals > cell_left) & (ax_vals < cell_right)).nonzero()[0],
                ax_vals.axes,
            )
            # If less than two indices along any axis, resample
            if len(newinds) < 2:
                for ax_ind in range(grid_ndim):
                    domain_centers[ind_subdom, ax_ind] = np.random.uniform(
                        xt1[ax_ind] + subdomain_dims[ax_ind],
                        xt2[ax_ind] - subdomain_dims[ax_ind],
                        size=1,
                    )
                include = False
                break
            else:
                include = True
                inds = inds + [newinds]
        attempts += 1
        if attempts > 2 * n_subdomains:
            raise RuntimeError(
                "Having trouble generating random subdomains. "
                "There may not be enough data"
            )
        if include:
            axis_inds_per_subdom = axis_inds_per_subdom + [inds]
            ind_subdom = ind_subdom + 1

    # TODO: fix meaning of axes in XT_k
    # Values of the spatiotemporal grid on the domain cells
    subgrids = [
        st_grid[np.ix_(*axis_inds_per_subdom[ind_subdom])]
        for ind_subdom in range(n_subdomains)
    ]

    # Recenter and shrink the domain cells so that grid points lie at the boundary
    # and calculate the new size
    subgrid_dims = np.zeros((n_subdomains, grid_ndim))
    for ind_subdom in range(n_subdomains):
        for axis in range(grid_ndim):
            s = [0] * (grid_ndim + 1)
            s[axis] = slice(None)
            s[-1] = axis
            subgrid_dims[ind_subdom, axis] = (
                subgrids[ind_subdom][tuple(s)][-1] - subgrids[ind_subdom][tuple(s)][0]
            ) / 2
            domain_centers[ind_subdom][axis] = (
                subgrids[ind_subdom][tuple(s)][-1] + subgrids[ind_subdom][tuple(s)][0]
            ) / 2
    # Rescaled space-time values for integration weights
    subgrids_scaled = [
        (subgrids[ind_subdom] - domain_centers[ind_subdom]) / subgrid_dims[ind_subdom]
        for ind_subdom in range(n_subdomains)
    ]

    # Shapes of the grid restricted to each cell
    subgrid_shapes = np.array(
        [
            [len(axis_inds_per_subdom[ind_subdom][i]) for i in range(grid_ndim)]
            for ind_subdom in range(n_subdomains)
        ]
    )

    return SubdomainSpecs(
        st_grid, axis_inds_per_subdom, subgrid_dims, subgrid_shapes, subgrids_scaled
    )


def _derivative_weights(
    scaled_subgrid_open: list[AxesArray],
    subgrid_dims: AxesArray,
    subgrid_shape: tuple[int, ...],
    deriv: tuple[int, ...],
    p: int,
) -> AxesArray:
    r"""Calculate integral weights for differential operator on basis function

    .. math::

        \int_\Omega f(x) * phi^{(d)}(y(x)) dx

    Where \Omega is a rectangular domain, x are coordinates in the original domain,
    and y are coordinates in the rescaled domain where each axis ranges from [-1, 1].

    Parameters:
        scaled_subgrid_open: Open mesh of rescaled subdomain coordinates
        subgrid_dims: 1D array of subdomain half-dimensions
        subgrid_shape: Shape of the subdomain
        deriv: 1D array of derivative orders along each axis
        p: Polynomial exponent for test function

    Returns:
        Weights for the integral, shaped as the subdomain grid with an added
        axis to support stacking with other differential operators.
        Return matrix should solve the integral as the dot product of
        ``np.dot(weights_nd.flatten(), f.flatten())``
    """

    weights_nd = np.ones(subgrid_shape)
    for it in enumerate(zip(deriv, scaled_subgrid_open, strict=True)):
        st_axis, (deriv_1d, scaled_coords_1d) = it
        # AxesArray errors in next few lines
        scaled_coords_1d = np.asarray(scaled_coords_1d)
        weights_1d = _linear_weights(scaled_coords_1d, int(deriv_1d), p)
        weights_nd = np.apply_along_axis(lambda x: x * weights_1d, st_axis, weights_nd)

    weights_nd *= np.prod(subgrid_dims ** (1.0 - np.array(deriv)))
    weights_nd = np.reshape(weights_nd, (*subgrid_shape, 1))
    ax_labels = comprehend_axes(weights_nd)
    ax_labels["ax_diffop"] = ax_labels.pop("ax_coord")

    return AxesArray(weights_nd, ax_labels)


def _time_derivative_weights(
    subdoms: SubdomainSpecs,
    grids: list[Float1D],
    left_inds: Float2D,
    right_inds: Float2D,
    p: int,
) -> AxesArray:
    # Weights for the time integrals along each axis
    weights1d = []
    deriv = np.zeros(subdoms.grid_ndim)
    deriv[-1] = 1
    # Weights for the integrals along each axis
    for i in range(subdoms.grid_ndim):
        weights1d = weights1d + [_linear_weights(grids[i], deriv[i], p)]
    # TODO: get rest of code to work with AxesArray.  Too unsure of
    # which axis labels to use at this point to continue
    weights1d = [np.asarray(arr) for arr in weights1d]
    # Product weights over the axes, shaped as inds_k
    fullweights = []
    for k in range(subdoms.n_subdomains):
        ret = np.ones(subdoms.subgrid_shapes[k])
        for i in range(subdoms.grid_ndim):
            dims = subdoms.subgrid_shapes[k][i]
            ret = ret * np.reshape(
                weights1d[i][left_inds[k][i] : right_inds[k][i] + 1], dims
            )

        fullweights = fullweights + [
            ret * np.prod(subdoms.subgrid_dims[k] ** (1.0 - deriv))
        ]
    # Not 100% sure about these axes:
    fullweights = np.stack(fullweights, axis=-1)
    return AxesArray(fullweights, comprehend_axes(fullweights))


def _pure_derivative_weights(
    subdoms: SubdomainSpecs,
    grids: list[Float1D],
    left_inds: Float2D,
    right_inds: Float2D,
    p: int,
) -> AxesArray:
    weights1d = []
    deriv = np.zeros(subdoms.grid_ndim)
    # Weights for the integrals along each axis
    for i in range(subdoms.grid_ndim):
        weights1d = weights1d + [_linear_weights(grids[i], deriv[i], p)]
    # TODO: get rest of code to work with AxesArray.  Too unsure of
    # which axis labels to use at this point to continue
    weights1d = [np.asarray(arr) for arr in weights1d]
    # Product weights over the axes, shaped as inds_k
    fullweights = []
    for k in range(subdoms.n_subdomains):
        ret = np.ones(subdoms.subgrid_shapes[k])
        for i in range(subdoms.grid_ndim):
            dims = subdoms.subgrid_shapes[k][i]
            ret = ret * np.reshape(
                weights1d[i][left_inds[k][i] : right_inds[k][i] + 1], dims
            )

        fullweights = fullweights + [ret * np.prod(subdoms.subgrid_dims[k])]
    fullweights = np.stack(fullweights, axis=-1)
    return AxesArray(fullweights, comprehend_axes(fullweights))


def _mixed_derivative_weights(
    subdoms: SubdomainSpecs,
    grids: list[Float1D],
    left_inds: Float2D,
    right_inds: Float2D,
    p: int,
    multiindices: list[Float1D],
) -> AxesArray:
    # Weights for the mixed library derivative terms along each axis
    weights1 = []
    for deriv in multiindices:
        weights2 = []
        # Add in time index
        deriv = np.concatenate([deriv, [0]])
        for i in range(subdoms.grid_ndim):
            weights2 = weights2 + [_linear_weights(grids[i], deriv[i], p)]
        weights1 = weights1 + [weights2]
    # TODO: get rest of code to work with AxesArray.  Too unsure of
    # which axis labels to use at this point to continue
    weights1 = [[np.asarray(arr) for arr in sublist] for sublist in weights1]
    # Product weights over the axes for mixed derivative terms, shaped as inds_k
    fullweights1 = []
    for k in range(subdoms.n_subdomains):
        weights2 = []
        for j, deriv in enumerate(multiindices):
            ret = np.ones(subdoms.subgrid_shapes[k])
            for i in range(subdoms.grid_ndim):
                dims = np.ones(subdoms.grid_ndim, dtype=int)
                dims[i] = subdoms.subgrid_shapes[k][i]
                ret = ret * np.reshape(
                    weights1[j][i][left_inds[i][k] : right_inds[i][k] + 1],
                    dims,
                )

            weights2 = weights2 + [
                ret * np.prod(subdoms.subgrid_dims[k] ** (1.0 - deriv))
            ]
        fullweights1 = fullweights1 + [weights2]
    fullweights1 = np.stack(fullweights1, axis=-1)
    return AxesArray(fullweights1, comprehend_axes(fullweights1))


def _phi(x: Float1D, d: int, p: int) -> Float1D:
    """
    One-dimensional polynomial test function (1-x**2)**p,
    differentiated d times, calculated term-wise in the binomial
    expansion.
    """
    ks = np.arange(p + 1)
    ks = ks[np.where(2 * (p - ks) - d >= 0)][:, np.newaxis]
    poly_coef = binom(p, ks)
    sign = (-1) ** (p - ks)
    x_pows = 2 * (p - ks) - d
    deriv_coef = perm(2 * (p - ks), d)
    monomials = sign * poly_coef * deriv_coef * x[np.newaxis, :] ** x_pows
    return np.sum(monomials, axis=0)


def _phi_int(x: Float1D, d: int, p: int) -> Float1D:
    """
    Indefinite integral of one-dimensional polynomial test
    function (1-x**2)**p, differentiated d times, calculated
    term-wise in the binomial expansion.
    """
    ks = np.arange(p + 1)
    ks = ks[np.where(2 * (p - ks) - d >= 0)][:, np.newaxis]
    poly_coef = binom(p, ks)
    sign = (-1) ** (p - ks)
    x_pows = 2 * (p - ks) - d + 1
    deriv_coef = perm(2 * (p - ks), d) / (2 * (p - ks) - d + 1)
    monomials = sign * poly_coef * deriv_coef * x[np.newaxis, :] ** x_pows
    return np.sum(monomials, axis=0)


def _xphi_int(x: Float1D, d: int, p: int) -> Float1D:
    """
    Indefinite integral of one-dimensional polynomial test function
    x*(1-x**2)**p, differentiated d times, calculated term-wise in the
    binomial expansion.
    """
    ks = np.arange(p + 1)
    ks = ks[np.where(2 * (p - ks) - d >= 0)][:, np.newaxis]
    poly_coef = binom(p, ks)
    sign = (-1) ** (p - ks)
    x_pows = 2 * (p - ks) - d + 2
    deriv_coef = perm(2 * (p - ks), d) / (2 * (p - ks) - d + 2)
    monomials = sign * poly_coef * deriv_coef * x[np.newaxis, :] ** x_pows
    return np.sum(monomials, axis=0)


def _linear_weights(x: Float1D, d: int, p: int) -> Float1D:
    """
    One-dimensional weights for integration against the dth derivative
    of the polynomial test function (1-x**2)**p. This is derived
    assuming the function to integrate is linear between grid points:
    f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
    so that f(x)*dphi(x) is a piecewise polynomial.
    The piecewise components are computed analytically, and the integral is
    expressed as a dot product of weights against the f_i.
    """
    ws = _phi_int(x, d, p)
    zs = _xphi_int(x, d, p)
    x_i = x[:-1]
    x_plus = x[1:]
    w_i = ws[:-1]
    w_plus = ws[1:]
    z_i = zs[:-1]
    z_plus = zs[1:]

    # if any(np.abs(x) > 1.0):
    #     raise ValueError("Extraneous calculation; truncate to domain [-1, 1]")

    weights = np.zeros_like(x)
    const_term1 = x_plus / (x_plus - x_i) * (w_plus - w_i)
    const_term2 = -x_i / (x_plus - x_i) * (w_plus - w_i)
    lin_term1 = -1 / (x_plus - x_i) * (z_plus - z_i)
    lin_term2 = 1 / (x_plus - x_i) * (z_plus - z_i)

    weights[:-1] += const_term1 + lin_term1
    weights[1:] += const_term2 + lin_term2

    return weights


def _dense_to_open_mesh(meshgrid: AxesArray) -> list[AxesArray]:
    """Convert a dense meshgrid, stacked along last coordinate, to an open mesh"""
    openmesh: list[AxesArray] = []
    for axis_ind in range(meshgrid.ndim - 1):
        indexer: list[int | slice] = [0] * (meshgrid.ndim)
        indexer[-1] = axis_ind
        indexer[axis_ind] = slice(None)
        openmesh.append(meshgrid[tuple(indexer)])

    return openmesh


def _calculate_weak_features(
    x: AxesArray,
    spec: SubdomainSpecs,
    lib: BaseFeatureLibrary,
    weights: tuple[list[AxesArray], list[AxesArray], list[AxesArray]],
    multiindices: list[Float1D],
    n_output_features_: int,
) -> AxesArray:
    n_features = x.n_coord
    xp = np.empty((spec.n_subdomains, n_output_features_), dtype=x.dtype)

    # Extract the input features on indices in each domain cell
    x_subdomains = [x[np.ix_(*ax_inds)] for ax_inds in spec.axis_inds_per_subdom]
    f_subdomains = [lib.fit(xi).transform(xi) for xi in x_subdomains]

    # Evaluate the functions on the indices of domain cells
    library_functions = np.array(
        [
            np.tensordot(weights, funcs, axis=spec.grid_ndim)
            for weights, funcs in zip(weights[0], f_subdomains)
        ]
    )

    if multiindices:
        # pure integral terms
        library_integrals = np.empty((K, n_features * num_derivatives), dtype=x.dtype)

        for weight_sub, x_sub in zip(
            weights[1], x_subdomains
        ):  # loop over domain cells
            library_idx = 0
            for j, diff_op in enumerate(multiindices):  # loop over derivatives
                # Calculate the integral feature by taking the dot product
                # of the weights and data x_k over each axis.
                # Integration by parts gives power of (-1).
                lib_int = -(1 ** np.sum(diff_op)) * np.tensordot(
                    weight_sub[j], x_sub, axes=range(spec.grid_ndim)
                )
                library_integrals[k, library_idx : library_idx + n_features] = lib_int
                library_idx += n_features

    library_idx = 0
    # library function terms
    xp[:, library_idx : library_idx + n_library_terms] = library_functions
    library_idx += n_library_terms

    if multiindices:
        # pure integral terms
        xp[
            :, library_idx : library_idx + num_derivatives * n_features
        ] = library_integrals
        library_idx += num_derivatives * n_features

        # mixed function integral terms
        if include_interaction:
            xp[
                :,
                library_idx : library_idx
                + n_library_terms * num_derivatives * n_features,
            ] = library_mixed_integrals
            library_idx += n_library_terms * num_derivatives * n_features
    return AxesArray(xp, {"ax_sample": 0, "ax_coord": 1})
