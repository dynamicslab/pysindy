r"""Learning via weak differential equations.

Developer notes:

There are several key types involved.

* A meshgrid is a spatiotemporal array that begins with spatial axes, then time,
  then a coordinate axis.  The (i, j, k)th grid coordinate of ``arr`` is
  ``arr[i, j, k, :]``.
* An open mesh is a list of the coordinates along each axis.  The (i, j, k)th
  grid coordinate of ``gridlist`` is
  ``[gridlist[0][i], gridlist[1][j], gridlist[2][k]]``.
* a derivative operation (``deriv_op``) is a tuple of values, each indicating
  the order of derivative in that coordinate.  E.g
  :math:`\partial_{x_0}^2\partial_{x_2}\partial_t^3` would be represented as
  ``(2, 0, 1, 3)``
* A ``SubdomainSpecs`` describes the subdomains of a larger domain of
  integration, including information such as the indexes of the subdomains
  in an original mesh, the size of the subdomains, and their rescaled values.
* A test function :math:`\phi` is currently not abstracted, but it is
  represented at several points in the code. ``_phi``, ``_phi_int`` and
  ``_xphi_int`` are needed to evaluate :math:`\int f\phi dx` on its support.
  Support is assumed to be [-1, 1], which is translated in
  ``_derivative_weights``.
* Library can be arbitrary trees of tensor/concat libraries.
  Libraries are flattened via distributive property, then integrated by parts
  to move derivatives onto test function.
* Between libraries of functions and individual terms in the differential
  equation, we have ``SemiTerm`` s.  A ``SemiTerm`` refers to an individual
  term of the PDELibrary, e.g. u_xy, multiplied by any tensored non-pde library.
  They exist because all the terms in a semiterm have the same integration by
  parts path.
  When a PDE library is multiplied by another library, derivatives are moved
  onto the product (f * phi), which results in multiple semiterms for each
  part of that product.

Research directions

* Implementing other test functions means collecting all calls to ``_phi``
  functions into a new type
* Choosing subdomains can be customized with ``_make_domains``
* sample_weights can be added to boost the importance of certain subdomains.
* Sarah, if you're doing DMD, you might be able to use ``_derivative_weights``,
  ``_make_domains``, and ``_convert_u_dot_integral``.  These handle 90% of the
  weak linear/nonlinear integral.  The really challenging part are terms like
  u*Du.
"""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
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
from .feature_library import GeneralizedLibrary
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


SemiTerm = tuple[
    tuple[int, ...] | None, # derivatives of u
    tuple[BaseFeatureLibrary, tuple[int, ...]] | None, # derivatives of f(u)
    float, # coefficient
    tuple[int, ...] # derivatives of phi
]

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

        if isinstance(self.feature_library, GeneralizedLibrary):
            raise TypeError("WeakSINDy not yet compatible with GeneralizedLibrary")

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
        n_system_vars = cast(int, x[0].n_coord)
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
            diff_method = pde_lib.differentiation_method
        else:
            time_deriv = np.zeros((1, subdomain_specs[0].grid_ndim), dtype=int)
            time_deriv[0, -1] = 1
            multiindices = time_deriv
            diff_method = FiniteDifference()

        multiindices = [tuple(deriv_op) for deriv_op in multiindices]
        zero_deriv = (0,) * n_system_vars

        weights_per_deriv_op = [
            _set_up_weights(spec, self.test_fn_order, multiindices)
            for spec in subdomain_specs
        ]

        # Need feature names and set n_output_features_ (this is same with PDE library)
        u_dot_wk = [
            convert_u_dot_integral(xi, weights[tuple(time_deriv)], spec.axis_inds_per_subdom)
            for xi, weights, spec in zip(x, weights_per_deriv_op, subdomain_specs)
        ]
        self.sorted_lib_ = _flatten_libraries(self.feature_library)
        terms: list[SemiTerm] = []
        for lib in self.sorted_lib_.libraries:
            # Populate the semiterm list. Libs are either PDELibrary, non-PDE, or tensor of those
            no_derivs = (
                isinstance(lib, TensoredLibrary)
                and not any(isinstance(lib, PDELibrary) for lib in lib.libraries)
                or not isinstance(lib, (PDELibrary, TensoredLibrary))
            )
            if no_derivs:
                terms.append((None, (lib, zero_deriv), 1, zero_deriv))
            elif isinstance(lib, PDELibrary):
                terms.extend(_integrate_by_parts(lib))
            else:
                # tensor library with product rule inside integration by parts
                pde_lib = lib.libraries[-1]
                non_pde_lib = TensoredLibrary(lib.libraries[:-1])
                terms.extend(_integrate_product_by_parts(non_pde_lib, pde_lib))


        for it in zip(x, subdomain_specs, weights_per_deriv_op, strict=True):
            x_i, sub_spec, weight_map = it
            weak_feats = []
            st_axes = tuple(range(sub_spec.grid_ndim))
            # Regular feature library gets distributed against terms in PDELibrary
            # into a set of SemiTerms.  To prevent re-evaluation, cache results
            eval_cache: dict[BaseFeatureLibrary, np.ndarray] = {}
            for diff1, prod_term, coeff, diff3 in terms:
                weights_per_subdom = weight_map[diff3]
                if not diff1 and not diff3:
                    # pure feature library
                    lib = prod_term[0]
                    if lib in eval_cache:
                        x_transform = eval_cache[lib]
                    else:
                        x_transform = lib.transform(x_i)
                        eval_cache[lib] = x_transform
                    x_subdoms = [x_transform[*inds] for inds in sub_spec.axis_inds_per_subdom]
                    weak_feats.append(
                        np.tensordot(x_subdom, weights, axis=[st_axes, st_axes])
                        for x_subdom, weights in zip(x_subdoms, weights_per_subdom)
                    )
                elif not prod_term:
                    # pure derivative term
                    x_subdoms = [x_i[*inds] for inds in sub_spec.axis_inds_per_subdom]
                    neg_coef = -1 ** sum()
                    weak_feats.append(
                        np.tensordot(x_subdom, weights, axis=[st_axes, st_axes])
                        for x_subdom, weights in zip(x_subdoms, weights_per_subdom)
                    )
                else:
                    # integration by parts term
                    lib = prod_term[0]
                    diff2 = prod_term[1]
                    if lib in eval_cache:
                        x_transform = eval_cache[lib]
                    else:
                        x_transform = lib.transform(x_i)
                        eval_cache[lib] = x_transform
                    x_transform = _differentiate(x_transform, diff2, diff_method)
                    x_diff = _differentiate(x_i, diff1, diff_method)
                    xt_subdoms = [
                        (x_diff * x_transform)[*inds]
                        for inds in sub_spec.axis_inds_per_subdom
                    ]
                    weak_feats.append(
                        np.tensordot(x_subdom, weights, axis=[st_axes, st_axes])
                        for x_subdom, weights in zip(xt_subdoms, weights_per_subdom)
                    )



        # Fit the optimizer!
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
    """Create the quadrature weights for each integral.

    Integrals over domain cells are approximated as dot products of weights and the input data.
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


def _integrate_by_parts(pde_lib: PDELibrary) -> list[SemiTerm]:
    """Move derivatives from each PDE term to test function"""
    terms = []
    for deriv_op in pde_lib.multiindices:
        zeros = tuple(np.zeros_like(deriv_op))
        coeff = (-1) ** deriv_op.sum()
        terms.append((zeros, None, coeff, tuple(deriv_op)))
    return terms


def _integrate_product_by_parts(
    f_lib: BaseFeatureLibrary, pde_lib: PDELibrary
) -> list[SemiTerm]:
    """Move derivatives from each PDE term to test function"""
    terms = []
    for deriv_op in pde_lib.multiindices:
        derivs_to_move = deriv_op // 2
        deriv_op_u = deriv_op - derivs_to_move
        for deriv_op_f in range(derivs_to_move):
            deriv_op_phi = derivs_to_move - deriv_op_f
            coeff = -1 ** derivs_to_move * binom(deriv_op_f, derivs_to_move)
            terms.append((deriv_op_u, (f_lib, deriv_op_f), coeff, deriv_op_phi))
    return terms



def _flatten_libraries(library: BaseFeatureLibrary) -> ConcatLibrary:
    """Flattens a tree of tensored/concat libraries, maintaining order

    Returns: A flat tree of libraries whose root is a concat library, and every
        leaf is either a non-iterable library or a tensored library of non-iterable
        libraries.  PDELibraries are at the end.

    .. todo::
        ensure that flat tree maintains identities of libraries and caches transforms
    """
    if isinstance(library, ConcatLibrary):
        root = library
    elif isinstance(library, TensoredLibrary):
        root = ConcatLibrary([])
    else:
        return ConcatLibrary([library])
    children = [_flatten_libraries(lib) for lib in library.libraries]
    root.libraries = children
    return root


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
