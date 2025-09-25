
from collections.abc import Sequence
from typing import cast
from typing import Optional
from typing_extensions import Self

import numpy as np
from scipy.special import binom
from scipy.special import perm

from ._core import _BaseSINDy
from ._core import _check_multiple_trajectories
from ._core import _adapt_to_multiple_trajectories
from ._core import _validate_inputs
from ._core import validate_control_variables
from ._core import standardize_shape
from ._typing import Float1D, FloatND
from .utils import AxesArray
from .feature_library.base import BaseFeatureLibrary
from .optimizers.base import BaseOptimizer
from .differentiation import FiniteDifference
from .differentiation.base import BaseDifferentiation


class WeakSINDy(_BaseSINDy):
    def __init__(
        self,
        feature_library: Optional[BaseFeatureLibrary],
        optimizer: Optional[BaseOptimizer],
        n_subdomains: int=100,
        test_fn_order=4,
        spatial_diff_method: type[BaseDifferentiation]=FiniteDifference,
        diff_kwargs: Optional[dict]=None,
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

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init()

    def fit(
        self,
        x: np.ndarray | Sequence[FloatND],
        st_grids: np.ndarray | Sequence[np.ndarray],
        *,
        H_xt: Optional[float | FloatND | Sequence[FloatND]]=None,
        u: Optional[np.ndarray | Sequence[FloatND]],
        feature_names: Optional[list[str]] = None) -> Self:
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
        self._weak_form_setup(st_grids, H_xt)
        self._set_up_weights()
        u_dot_wk = [convert_u_dot_integral(xi, K) for xi in x]

        self._fit_shape()
        return self

    def _weak_form_setup(self, st_grids: list[AxesArray], H_xt):
        """Establish domain-dependent functionality"""
        xts = [_get_spatial_endpoints(grid) for grid in st_grids]
        L_xt = xt2 - xt1
        if self.H_xt is not None:
            if np.isscalar(self.H_xt):
                self.H_xt = AxesArray(
                    np.array(self.grid_ndim * [self.H_xt]), {"ax_coord": 0}
                )
            if self.grid_ndim != len(self.H_xt):
                raise ValueError(
                    "The user-defined grid (spatiotemporal_grid) and "
                    "the user-defined sizes of the subdomains for the "
                    "weak form do not have the same # of spatiotemporal "
                    "dimensions. For instance, if spatiotemporal_grid is 4D, "
                    "then H_xt should be a 4D list of the subdomain lengths."
                )
            if any(self.H_xt <= np.zeros(len(self.H_xt))):
                raise ValueError("Values in H_xt must be a positive float.")
            elif any(self.H_xt >= L_xt / 2.0):
                raise ValueError(
                    "2 * H_xt in some dimension is larger than the "
                    "corresponding grid dimension."
                )
        else:
            self.H_xt = L_xt / 20.0

        if self.spatiotemporal_grid is not None:
            if self.p < 0:
                raise ValueError("Poly degree of the spatial weights must be > 0")
            if self.p < self.derivative_order:
                self.p = self.derivative_order
        if self.K <= 0:
            raise ValueError("The number of subdomains must be > 0")


def convert_u_dot_integral(u: AxesArray, fulltweights, subdomain_inds):
    """
    Takes a full set of spatiotemporal fields u(x, t) and finds the weak
    form of u_dot.
    """
    n_subdomains = len(fulltweights)
    grid_dim = u.ndim - 1
    u_dot_integral = np.zeros((n_subdomains, u.shape[-1]))

    for domain in range(n_subdomains):  # loop over domain cells
        # calculate the integral feature by taking the dot product
        # of the weights and functions over each axis
        u_dot_integral[domain] = np.tensordot(
            fulltweights[domain],
            -u[np.ix_(*subdomain_inds[domain])],
            axes=(
                tuple(np.arange(grid_dim)),
                tuple(np.arange(grid_dim)),
            ),
        )

    return u_dot_integral

def _get_spatial_endpoints(st_grid: FloatND) -> tuple[Float1D, Float1D]:
    """Retrieve the min/max coordinates for each axis of a meshgrid"""
    grid_ndim = st_grid.ndim - 1
    min_inds = (0,) * grid_ndim
    max_inds = (-1,) * grid_ndim
    # After python 3.10 EOL, can write st_grid[*min_inds, :]
    return st_grid[(*min_inds, Ellipsis)], st_grid[(*max_inds, Ellipsis)]


def _set_up_weights(st_grid):
    """
    Sets up weights needed for the weak library. Integrals over domain cells are
    approximated as dot products of weights and the input data.
    """
    xt1, xt2 = _get_spatial_endpoints(st_grid)
    domains = _make_domains(st_grid, n_subdomains)
    # Below we calculate the weights to convert integrals into dot products
    # To speed up evaluations, we proceed in several steps

    # Since the grid is a tensor product grid, we calculate weights along each axis
    # Later, we multiply the weights along each axis to produce the full weights

    # Within each domain cell, we calculate the interior weights
    # and the weights at the left and right boundaries separately,
    # since the  expression differ at the boundaries of the domains

    # Extract the space-time coordinates for each domain and the indices for
    # the left-most and right-most points for each domain.
    # We stack the values for each domain cell into a single vector to speed up
    grids = []  # the rescaled coordinates for each domain
    lefts = []  # the spatiotemporal indices at the left of each domain
    rights = []  # the spatiotemporal indices at the right of each domain
    for i in range(self.grid_ndim):
        s = [0] * (self.grid_ndim + 1)
        s[-1] = i
        s[i] = slice(None)
        # stacked coordinates for axis i over all domains
        grids = grids + [np.hstack([xtilde_k[k][tuple(s)] for k in range(self.K)])]
        # stacked indices for right-most point for axis i over all domains
        rights = rights + [np.cumsum(shapes_k[:, i]) - 1]
        # stacked indices for left-most point for axis i over all domains
        lefts = lefts + [np.concatenate([[0], np.cumsum(shapes_k[:, i])[:-1]])]

    # Weights for pure derivative terms along each axis
    weights0 = []
    deriv = np.zeros(self.grid_ndim)
    for i in range(self.grid_ndim):
        # weights for interior points
        weights0 = weights0 + [self._linear_weights(grids[i], deriv[i], self.p)]
        # correct the values for the left-most points
        weights0[i][lefts[i]] = self._left_weights(
            grids[i][lefts[i]],
            grids[i][lefts[i] + 1],
            deriv[i],
            self.p,
        )
        # correct the values for the right-most points
        weights0[i][rights[i]] = self._right_weights(
            grids[i][rights[i] - 1],
            grids[i][rights[i]],
            deriv[i],
            self.p,
        )

    # Weights for the mixed library derivative terms along each axis
    weights1 = []
    for j in range(self.num_derivatives):
        weights2 = []
        deriv = np.concatenate([self.multiindices[j], [0]])
        for i in range(self.grid_ndim):
            # weights for interior points
            weights2 = weights2 + [self._linear_weights(grids[i], deriv[i], self.p)]
            # correct the values for the left-most points
            weights2[i][lefts[i]] = self._left_weights(
                grids[i][lefts[i]],
                grids[i][lefts[i] + 1],
                deriv[i],
                self.p,
            )
            # correct the values for the right-most points
            weights2[i][rights[i]] = self._right_weights(
                grids[i][rights[i] - 1],
                grids[i][rights[i]],
                deriv[i],
                self.p,
            )
        weights1 = weights1 + [weights2]

    # TODO: get rest of code to work with AxesArray.  Too unsure of
    # which axis labels to use at this point to continue
    tweights = [np.asarray(arr) for arr in tweights]
    weights0 = [np.asarray(arr) for arr in weights0]
    weights1 = [[np.asarray(arr) for arr in sublist] for sublist in weights1]

    fulltweights = _time_derivative_weights()

    # Product weights over the axes for pure derivative terms, shaped as inds_k
    fullweights_pure = _pure_derivative_weights()
    # self.fullweights0 = []
    # for k in range(self.K):
    #     ret = np.ones(shapes_k[k])
    #     for i in range(self.grid_ndim):
    #         s = [0] * (self.grid_ndim + 1)
    #         s[i] = slice(None, None, None)
    #         s[-1] = i
    #         dims = np.ones(self.grid_ndim, dtype=int)
    #         dims[i] = shapes_k[k][i]
    #         ret = ret * np.reshape(
    #             weights0[i][lefts[i][k] : rights[i][k] + 1], dims
    #         )

    #     self.fullweights0 = self.fullweights0 + [ret * np.prod(H_xt_k[k])]

    # Product weights over the axes for mixed derivative terms, shaped as inds_k
    fullweights_mixed = _mixed_derivative_terms()
    # self.fullweights1 = []
    # for k in range(self.K):
    #     weights2 = []
    #     for j in range(self.num_derivatives):
    #         if not self.implicit_terms:
    #             deriv = np.concatenate([self.multiindices[j], [0]])
    #         else:
    #             deriv = self.multiindices[j]

    #         ret = np.ones(shapes_k[k])
    #         for i in range(self.grid_ndim):
    #             s = [0] * (self.grid_ndim + 1)
    #             s[i] = slice(None, None, None)
    #             s[-1] = i
    #             dims = np.ones(self.grid_ndim, dtype=int)
    #             dims[i] = shapes_k[k][i]
    #             ret = ret * np.reshape(
    #                 weights1[j][i][lefts[i][k] : rights[i][k] + 1],
    #                 dims,
    #             )

    #         weights2 = weights2 + [ret * np.prod(H_xt_k[k] ** (1.0 - deriv))]
    #     self.fullweights1 = self.fullweights1 + [weights2]
    return fulltweights, fullweights_pure, fullweights_mixed


def _make_domains(st_grid, n_subdomains):
    pass

def _time_derivative_weights():
    # Weights for the time integrals along each axis
    tweights = []
    deriv = np.zeros(self.grid_ndim)
    deriv[-1] = 1
    for i in range(self.grid_ndim):
        # weights for interior points
        tweights = tweights + [self._linear_weights(grids[i], deriv[i], self.p)]
        # correct the values for the left-most points
        tweights[i][lefts[i]] = self._left_weights(
            grids[i][lefts[i]],
            grids[i][lefts[i] + 1],
            deriv[i],
            self.p,
        )
        # correct the values for the right-most points
        tweights[i][rights[i]] = self._right_weights(
            grids[i][rights[i] - 1],
            grids[i][rights[i]],
            deriv[i],
            self.p,
        )
    # Product weights over the axes for time derivatives, shaped as inds_k
    self.fulltweights = []
    deriv = np.zeros(self.grid_ndim)
    deriv[-1] = 1
    for k in range(self.K):
        ret = np.ones(shapes_k[k])
        for i in range(self.grid_ndim):
            s = [0] * (self.grid_ndim + 1)
            s[i] = slice(None, None, None)
            s[-1] = i
            dims = np.ones(self.grid_ndim, dtype=int)
            dims[i] = shapes_k[k][i]
            ret = ret * np.reshape(
                tweights[i][lefts[i][k] : rights[i][k] + 1], dims
            )

        self.fulltweights = self.fulltweights + [
            ret * np.prod(H_xt_k[k] ** (1.0 - deriv))
        ]


def _phi(x, d, p):
    """
    One-dimensional polynomial test function (1-x**2)**p,
    differentiated d times, calculated term-wise in the binomial
    expansion.
    """
    ks = np.arange(p + 1)
    ks = ks[np.where(2 * (p - ks) - d >= 0)][:, np.newaxis]
    return np.sum(
        binom(p, ks)
        * (-1) ** ks
        * x[np.newaxis, :] ** (2 * (p - ks) - d)
        * perm(2 * (p - ks), d),
        axis=0,
    )

def _phi_int(x, d, p):
    """
    Indefinite integral of one-dimensional polynomial test
    function (1-x**2)**p, differentiated d times, calculated
    term-wise in the binomial expansion.
    """
    ks = np.arange(p + 1)
    ks = ks[np.where(2 * (p - ks) - d >= 0)][:, np.newaxis]
    return np.sum(
        binom(p, ks)
        * (-1) ** ks
        * x[np.newaxis, :] ** (2 * (p - ks) - d + 1)
        * perm(2 * (p - ks), d)
        / (2 * (p - ks) - d + 1),
        axis=0,
    )

def _xphi_int(x, d, p):
    """
    Indefinite integral of one-dimensional polynomial test function
    x*(1-x**2)**p, differentiated d times, calculated term-wise in the
    binomial expansion.
    """
    ks = np.arange(p + 1)
    ks = ks[np.where(2 * (p - ks) - d >= 0)][:, np.newaxis]
    return np.sum(
        binom(p, ks)
        * (-1) ** ks
        * x[np.newaxis, :] ** (2 * (p - ks) - d + 2)
        * perm(2 * (p - ks), d)
        / (2 * (p - ks) - d + 2),
        axis=0,
    )

def _linear_weights(x, d, p):
    """
    One-dimensioal weights for integration against the dth derivative
    of the polynomial test function (1-x**2)**p. This is derived
    assuming the function to integrate is linear between grid points:
    f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
    so that f(x)*dphi(x) is a piecewise polynomial.
    The piecewise components are computed analytically, and the integral is
    expressed as a dot product of weights against the f_i.
    """
    ws = _phi_int(x, d, p)
    zs = _xphi_int(x, d, p)
    return np.concatenate(
        [
            [
                x[1] / (x[1] - x[0]) * (ws[1] - ws[0])
                - 1 / (x[1] - x[0]) * (zs[1] - zs[0])
            ],
            x[2:] / (x[2:] - x[1:-1]) * (ws[2:] - ws[1:-1])
            - x[:-2] / (x[1:-1] - x[:-2]) * (ws[1:-1] - ws[:-2])
            + 1 / (x[1:-1] - x[:-2]) * (zs[1:-1] - zs[:-2])
            - 1 / (x[2:] - x[1:-1]) * (zs[2:] - zs[1:-1]),
            [
                -x[-2] / (x[-1] - x[-2]) * (ws[-1] - ws[-2])
                + 1 / (x[-1] - x[-2]) * (zs[-1] - zs[-2])
            ],
        ]
    )

def _left_weights(x1, x2, d, p):
    """
    One-dimensioal weight for left-most point in integration against the dth
    derivative of the polynomial test function (1-x**2)**p. This is derived
    assuming the function to integrate is linear between grid points:
    f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
    so that f(x)*dphi(x) is a piecewise polynomial.
    The piecewise components are computed analytically, and the integral is
    expressed as a dot product of weights against the f_i.
    """
    w1 = _phi_int(x1, d, p)
    w2 = _phi_int(x2, d, p)
    z1 = _xphi_int(x1, d, p)
    z2 = _xphi_int(x2, d, p)
    return x2 / (x2 - x1) * (w2 - w1) - 1 / (x2 - x1) * (z2 - z1)

def _right_weights(x1, x2, d, p):
    """
    One-dimensioal weight for right-most point in integration against the dth
    derivative of the polynomial test function (1-x**2)**p. This is derived
    assuming the function to integrate is linear between grid points:
    f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
    so that f(x)*dphi(x) is a piecewise polynomial.
    The piecewise components are computed analytically, and the integral is
    expressed as a dot product of weights against the f_i.
    """
    w1 = _phi_int(x1, d, p)
    w2 = _phi_int(x2, d, p)
    z1 = _xphi_int(x1, d, p)
    z2 = _xphi_int(x2, d, p)
    return -x1 / (x2 - x1) * (w2 - w1) + 1 / (x2 - x1) * (z2 - z1)

