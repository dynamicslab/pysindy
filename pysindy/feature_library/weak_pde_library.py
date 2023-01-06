import warnings
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r
from itertools import product as iproduct

import numpy as np
from scipy.special import binom
from scipy.special import perm
from sklearn import __version__
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item
from pysindy.differentiation import FiniteDifference


class WeakPDELibrary(BaseFeatureLibrary):
    """Generate a weak formulation library with custom functions and,
       optionally, any spatial derivatives in arbitrary dimensions.

       The features in the weak formulation are integrals of derivatives of input data
       multiplied by a test function phi, which are evaluated on K subdomains
       randomly sampled across the spatiotemporal grid. Each subdomain
       is initial generated with a size H_xt along each axis, and is then shrunk
       such that the left and right boundaries lie on spatiotemporal grid points.
       The expressions are integrated by parts to remove as many derivatives from the
       input data as possible and put the derivatives onto the test functions.

       The weak integral features are calculated assuming the function f(x) to
       integrate against derivatives of the test function dphi(x)
       is linear between grid points provided by the data:
       f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
       Thus f(x)*dphi(x) is approximated as a piecewise polynomial.
       The piecewise components are integrated analytically. To improve performance,
       the complete integral is expressed as a dot product of weights against the
       input data f_i, which enables vectorized evaulations.

    Parameters
    ----------
    library_functions : list of mathematical functions, optional (default None)
        Functions to include in the library. Each function will be
        applied to each input variable (but not their derivatives)

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable,
        can be arbitrary non-negative integer.

    spatiotemporal_grid : np.ndarray (default None)
        The spatiotemporal grid for computing derivatives.
        This variable must be specified with
        at least one dimension corresponding to a temporal grid, so that
        integration by parts can be done in the weak formulation.

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names list
        must be the same length as library_functions.
        If no list of function names is provided, defaults to using
        :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`.

    interaction_only : boolean, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaulations of the form :math:`f(x,x)`
        and :math:`f(x,y,x)` will be omitted, but those of the form
        :math:`f(x,y)` and :math:`f(x,y,z)` will be included.
        If False, all combinations will be included.

    include_bias : boolean, optional (default False)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
        This is hard to do with just lambda functions, because if the system
        is not 1D, lambdas will generate duplicates.

    include_interaction : boolean, optional (default True)
        This is a different than the use for the PolynomialLibrary. If true,
        it generates all the mixed derivative terms. If false, the library
        will consist of only pure no-derivative terms and pure derivative
        terms, with no mixed terms.

    K : int, optional (default 100)
        Number of domain centers, corresponding to subdomain squares of length
        Hxt. If K is not
        specified, defaults to 100.

    H_xt : array of floats, optional (default None)
        Half of the length of the square subdomains in each spatiotemporal
        direction. If H_xt is not specified, defaults to H_xt = L_xt / 20,
        where L_xt is the length of the full domain in each spatiotemporal
        direction. If H_xt is specified as a scalar, this value will be applied
        to all dimensions of the subdomains.

    p : int, optional (default 4)
        Positive integer to define the polynomial degree of the spatial weights
        used for weak/integral SINDy.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

    num_pts_per_domain : int, deprecated (default None)
        Included here to retain backwards compatibility with older code
        that uses this parameter. However, it merely raises a
        DeprecationWarning and then is ignored.

    implicit_terms : boolean
        Flag to indicate if SINDy-PI (temporal derivatives) is being used
        for the right-hand side of the SINDy fit.

    multiindices : list of integer arrays,  (default None)
        Overrides the derivative_order to customize the included derivative
        orders. Each integer array indicates the order of differentiation
        along the corresponding axis for each derivative term.

    differentiation_method : callable,  (default FiniteDifference)
        Spatial differentiation method.

     diff_kwargs: dictionary,  (default {})
        Keyword options to supply to differtiantion_method.


    Attributes
    ----------
    functions : list of functions
        Mathematical library functions to be applied to each input feature.

    function_names : list of functions
        Functions for generating string representations of each library
        function.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import WeakPDELibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = WeakPDELibrary(library_functions=functions).fit(x)
    >>> lib.transform(x)
    array([[ 1.        ,  0.36787944, -0.84147098],
           [ 2.71828183,  1.        ,  0.84147098],
           [ 7.3890561 ,  0.36787944,  0.84147098]])
    >>> lib.get_feature_names()
    ['f0(x0)', 'f0(x1)', 'f1(x0,x1)']
    """

    def __init__(
        self,
        library_functions=[],
        derivative_order=0,
        spatiotemporal_grid=None,
        function_names=None,
        interaction_only=True,
        include_bias=False,
        include_interaction=True,
        K=100,
        H_xt=None,
        p=4,
        library_ensemble=False,
        ensemble_indices=[0],
        num_pts_per_domain=None,
        implicit_terms=False,
        multiindices=None,
        differentiation_method=FiniteDifference,
        diff_kwargs={},
        is_uniform=None,
        periodic=None,
    ):
        super(WeakPDELibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.derivative_order = derivative_order
        self.function_names = function_names
        self.interaction_only = interaction_only
        self.implicit_terms = implicit_terms
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.K = K
        self.H_xt = H_xt
        self.p = p
        self.num_trajectories = 1
        self.differentiation_method = differentiation_method
        self.diff_kwargs = diff_kwargs

        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if library_functions is None and derivative_order == 0:
            raise ValueError(
                "No library functions were specified, and no "
                "derivatives were asked for. The library is empty."
            )
        if spatiotemporal_grid is None:
            raise ValueError(
                "Spatiotemporal grid was not passed, and at least a 1D"
                " grid is required, corresponding to the time base."
            )
        if num_pts_per_domain is not None:
            warnings.warn(
                "The parameter num_pts_per_domain is now deprecated. This "
                "value will be ignored by the library."
            )
        if is_uniform is not None or periodic is not None:
            # DeprecationWarning are ignored by default...
            warnings.warn(
                "is_uniform and periodic have been deprecated."
                "in favor of differetiation_method and diff_kwargs.",
                UserWarning,
            )

        # list of integrals
        indices = ()
        if np.array(spatiotemporal_grid).ndim == 1:
            spatiotemporal_grid = np.reshape(
                spatiotemporal_grid, (len(spatiotemporal_grid), 1)
            )
        dims = spatiotemporal_grid.shape[:-1]
        self.grid_dims = dims
        self.grid_ndim = len(dims)

        # if want to include temporal terms -> range(len(dims))
        if self.implicit_terms:
            self.ind_range = len(dims)
        else:
            self.ind_range = len(dims) - 1

        for i in range(self.ind_range):
            indices = indices + (range(derivative_order + 1),)

        if multiindices is None:
            multiindices = []
            for ind in iproduct(*indices):
                current = np.array(ind)
                if np.sum(ind) > 0 and np.sum(ind) <= derivative_order:
                    multiindices.append(current)
            multiindices = np.array(multiindices)
        num_derivatives = len(multiindices)
        if num_derivatives > 0:
            self.derivative_order = np.max(multiindices)

        self.num_derivatives = num_derivatives
        self.multiindices = multiindices
        self.spatiotemporal_grid = spatiotemporal_grid

        # Weak form checks and setup
        self._weak_form_setup()

    def _weak_form_setup(self):
        xt1, xt2 = self._get_spatial_endpoints()
        L_xt = xt2 - xt1
        if self.H_xt is not None:
            if np.isscalar(self.H_xt):
                self.H_xt = np.array(self.grid_ndim * [self.H_xt])
            if self.grid_ndim != len(self.H_xt):
                raise ValueError(
                    "The user-defined grid (spatiotemporal_grid) and "
                    "the user-defined sizes of the subdomains for the "
                    "weak form, do not have the same # of spatiotemporal "
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

        self._set_up_weights()

    def _get_spatial_endpoints(self):
        x1 = np.zeros(self.grid_ndim)
        x2 = np.zeros(self.grid_ndim)
        for i in range(self.grid_ndim):
            inds = [slice(None)] * (self.grid_ndim + 1)
            for j in range(self.grid_ndim):
                inds[j] = 0
            x1[i] = self.spatiotemporal_grid[tuple(inds)][i]
            inds[i] = -1
            x2[i] = self.spatiotemporal_grid[tuple(inds)][i]
        return x1, x2

    def _set_up_weights(self):
        """
        Sets up weights needed for the weak library. Integrals over domain cells are
        approximated as dot products of weights and the input data.
        """
        dims = self.spatiotemporal_grid.shape[:-1]
        self.grid_dims = dims

        # Sample the random domain centers
        xt1, xt2 = self._get_spatial_endpoints()
        domain_centers = np.zeros((self.K, self.grid_ndim))
        for i in range(self.grid_ndim):
            domain_centers[:, i] = np.random.uniform(
                xt1[i] + self.H_xt[i], xt2[i] - self.H_xt[i], size=self.K
            )

        # Indices for space-time points that lie in the domain cells
        self.inds_k = []
        k = 0
        while k < self.K:
            inds = []
            for i in range(self.grid_ndim):
                s = [0] * (self.grid_ndim + 1)
                s[i] = slice(None)
                s[-1] = i
                newinds = np.intersect1d(
                    np.where(
                        self.spatiotemporal_grid[tuple(s)]
                        >= domain_centers[k][i] - self.H_xt[i]
                    ),
                    np.where(
                        self.spatiotemporal_grid[tuple(s)]
                        <= domain_centers[k][i] + self.H_xt[i]
                    ),
                )
                # If less than two indices along any axis, resample
                if len(newinds) < 2:
                    for i in range(self.grid_ndim):
                        domain_centers[k, i] = np.random.uniform(
                            xt1[i] + self.H_xt[i], xt2[i] - self.H_xt[i], size=1
                        )
                    include = False
                    break
                else:
                    include = True
                    inds = inds + [newinds]
            if include:
                self.inds_k = self.inds_k + [inds]
                k = k + 1

        # Values of the spatiotemporal grid on the domain cells
        XT_k = [
            self.spatiotemporal_grid[np.ix_(*self.inds_k[k])] for k in range(self.K)
        ]

        # Recenter and shrink the domain cells so that grid points lie at the boundary
        # and calculate the new size
        H_xt_k = np.zeros((self.K, self.grid_ndim))
        for k in range(self.K):
            for axis in range(self.grid_ndim):
                s = [0] * (self.grid_ndim + 1)
                s[axis] = slice(None)
                s[-1] = axis
                H_xt_k[k, axis] = (XT_k[k][tuple(s)][-1] - XT_k[k][tuple(s)][0]) / 2
                domain_centers[k][axis] = (
                    XT_k[k][tuple(s)][-1] + XT_k[k][tuple(s)][0]
                ) / 2
        # Rescaled space-time values for integration weights
        xtilde_k = [(XT_k[k] - domain_centers[k]) / H_xt_k[k] for k in range(self.K)]

        # Shapes of the grid restricted to each cell
        shapes_k = np.array(
            [
                [len(self.inds_k[k][i]) for i in range(self.grid_ndim)]
                for k in range(self.K)
            ]
        )

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
                ret * np.product(H_xt_k[k] ** (1.0 - deriv))
            ]

        # Product weights over the axes for pure derivative terms, shaped as inds_k
        self.fullweights0 = []
        for k in range(self.K):

            ret = np.ones(shapes_k[k])
            for i in range(self.grid_ndim):
                s = [0] * (self.grid_ndim + 1)
                s[i] = slice(None, None, None)
                s[-1] = i
                dims = np.ones(self.grid_ndim, dtype=int)
                dims[i] = shapes_k[k][i]
                ret = ret * np.reshape(
                    weights0[i][lefts[i][k] : rights[i][k] + 1], dims
                )

            self.fullweights0 = self.fullweights0 + [ret * np.product(H_xt_k[k])]

        # Product weights over the axes for mixed derivative terms, shaped as inds_k
        self.fullweights1 = []
        for k in range(self.K):
            weights2 = []
            for j in range(self.num_derivatives):
                if not self.implicit_terms:
                    deriv = np.concatenate([self.multiindices[j], [0]])
                else:
                    deriv = self.multiindices[j]

                ret = np.ones(shapes_k[k])
                for i in range(self.grid_ndim):
                    s = [0] * (self.grid_ndim + 1)
                    s[i] = slice(None, None, None)
                    s[-1] = i
                    dims = np.ones(self.grid_ndim, dtype=int)
                    dims[i] = shapes_k[k][i]
                    ret = ret * np.reshape(
                        weights1[j][i][lefts[i][k] : rights[i][k] + 1],
                        dims,
                    )

                weights2 = weights2 + [ret * np.product(H_xt_k[k] ** (1.0 - deriv))]
            self.fullweights1 = self.fullweights1 + [weights2]

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """
        Get the combinations of features to be passed to a library function.
        """
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def _phi(self, x, d, p):
        """
        One-dimensional polynomial test function (1-x**2)**p,
        differentiated d times, calculated term-wise in the binomial
        expansion.
        """
        ks = np.arange(self.p + 1)
        ks = ks[np.where(2 * (self.p - ks) - d >= 0)][:, np.newaxis]
        return np.sum(
            binom(self.p, ks)
            * (-1) ** ks
            * x[np.newaxis, :] ** (2 * (self.p - ks) - d)
            * perm(2 * (self.p - ks), d),
            axis=0,
        )

    def _phi_int(self, x, d, p):
        """
        Indefinite integral of one-dimensional polynomial test
        function (1-x**2)**p, differentiated d times, calculated
        term-wise in the binomial expansion.
        """
        ks = np.arange(self.p + 1)
        ks = ks[np.where(2 * (self.p - ks) - d >= 0)][:, np.newaxis]
        return np.sum(
            binom(self.p, ks)
            * (-1) ** ks
            * x[np.newaxis, :] ** (2 * (self.p - ks) - d + 1)
            * perm(2 * (self.p - ks), d)
            / (2 * (self.p - ks) - d + 1),
            axis=0,
        )

    def _xphi_int(self, x, d, p):
        """
        Indefinite integral of one-dimensional polynomial test function
        x*(1-x**2)**p, differentiated d times, calculated term-wise in the
        binomial expansion.
        """
        ks = np.arange(self.p + 1)
        ks = ks[np.where(2 * (self.p - ks) - d >= 0)][:, np.newaxis]
        return np.sum(
            binom(self.p, ks)
            * (-1) ** ks
            * x[np.newaxis, :] ** (2 * (self.p - ks) - d + 2)
            * perm(2 * (self.p - ks), d)
            / (2 * (self.p - ks) - d + 2),
            axis=0,
        )

    def _linear_weights(self, x, d, p):
        """
        One-dimensioal weights for integration against the dth derivative
        of the polynomial test function (1-x**2)**p. This is derived
        assuming the function to integrate is linear between grid points:
        f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
        so that f(x)*dphi(x) is a piecewise polynomial.
        The piecewise components are computed analytically, and the integral is
        expressed as a dot product of weights against the f_i.
        """
        ws = self._phi_int(x, d, p)
        zs = self._xphi_int(x, d, p)
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

    def _left_weights(self, x1, x2, d, p):
        """
        One-dimensioal weight for left-most point in integration against the dth
        derivative of the polynomial test function (1-x**2)**p. This is derived
        assuming the function to integrate is linear between grid points:
        f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
        so that f(x)*dphi(x) is a piecewise polynomial.
        The piecewise components are computed analytically, and the integral is
        expressed as a dot product of weights against the f_i.
        """
        w1 = self._phi_int(x1, d, p)
        w2 = self._phi_int(x2, d, p)
        z1 = self._xphi_int(x1, d, p)
        z2 = self._xphi_int(x2, d, p)
        return x2 / (x2 - x1) * (w2 - w1) - 1 / (x2 - x1) * (z2 - z1)

    def _right_weights(self, x1, x2, d, p):
        """
        One-dimensioal weight for right-most point in integration against the dth
        derivative of the polynomial test function (1-x**2)**p. This is derived
        assuming the function to integrate is linear between grid points:
        f(x)=f_i+(x-x_i)/(x_{i+1}-x_i)*(f_{i+1}-f_i)
        so that f(x)*dphi(x) is a piecewise polynomial.
        The piecewise components are computed analytically, and the integral is
        expressed as a dot product of weights against the f_i.
        """
        w1 = self._phi_int(x1, d, p)
        w2 = self._phi_int(x2, d, p)
        z1 = self._xphi_int(x1, d, p)
        z2 = self._xphi_int(x2, d, p)
        return -x1 / (x2 - x1) * (w2 - w1) + 1 / (x2 - x1) * (z2 - z1)

    def convert_u_dot_integral(self, u):
        """
        Takes a full set of spatiotemporal fields u(x, t) and finds the weak
        form of u_dot.
        """
        K = self.K
        gdim = self.grid_ndim
        u_dot_integral = np.zeros((K, u.shape[-1]))
        deriv_orders = np.zeros(gdim)
        deriv_orders[-1] = 1

        # Extract the input features on indices in each domain cell
        dims = np.array(self.spatiotemporal_grid.shape)
        dims[-1] = u.shape[-1]

        for k in range(self.K):  # loop over domain cells
            # calculate the integral feature by taking the dot product
            # of the weights and functions over each axis
            u_dot_integral[k] = np.tensordot(
                self.fulltweights[k],
                -u[np.ix_(*self.inds_k[k])],
                axes=(
                    tuple(np.arange(self.grid_ndim)),
                    tuple(np.arange(self.grid_ndim)),
                ),
            )

        return u_dot_integral

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        if float(__version__[:3]) >= 1.0:
            n_features = self.n_features_in_
        else:
            n_features = self.n_input_features_
        if input_features is None:
            input_features = ["x%d" % i for i in range(n_features)]
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(n_features),
                )
            )
        feature_names = []

        # Include constant term
        if self.include_bias:
            feature_names.append("1")

        # Include any non-derivative terms
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )

        if self.grid_ndim != 0:

            def derivative_string(multiindex):
                ret = ""
                for axis in range(self.ind_range):
                    if (axis == self.ind_range - 1) and (
                        self.ind_range == self.grid_ndim
                    ):
                        str_deriv = "t"
                    else:
                        str_deriv = str(axis + 1)
                    for i in range(multiindex[axis]):
                        ret = ret + str_deriv
                return ret

            # Include integral terms
            for k in range(self.num_derivatives):
                for j in range(n_features):
                    feature_names.append(
                        input_features[j]
                        + "_"
                        + derivative_string(self.multiindices[k])
                    )
            # Include mixed non-derivative + integral terms
            if self.include_interaction:
                for k in range(self.num_derivatives):
                    for i, f in enumerate(self.functions):
                        for c in self._combinations(
                            n_features,
                            f.__code__.co_argcount,
                            self.interaction_only,
                        ):
                            for jj in range(n_features):
                                feature_names.append(
                                    self.function_names[i](
                                        *[input_features[j] for j in c]
                                    )
                                    + input_features[jj]
                                    + "_"
                                    + derivative_string(self.multiindices[k])
                                )
        return feature_names

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_features = x_full[0].shape[x_full[0].ax_coord]
        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        n_output_features = 0

        # Count the number of non-derivative terms
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )

        if self.grid_ndim != 0:
            # Add the mixed derivative library_terms
            if self.include_interaction:
                n_output_features += (
                    n_output_features * n_features * self.num_derivatives
                )
            # Add the pure derivative library terms
            n_output_features += n_features * self.num_derivatives

        # If there is a constant term, add 1 to n_output_features
        if self.include_bias:
            n_output_features += 1

        self.n_output_features_ = n_output_features

        # required to generate the function names
        self.get_feature_names()

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to custom features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of
            features generated from applying the custom functions
            to the inputs.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            n_features = x.shape[x.ax_coord]
            xp = np.empty((self.K, self.n_output_features_), dtype=x.dtype)

            # Extract the input features on indices in each domain cell
            self.x_k = [x[np.ix_(*self.inds_k[k])] for k in range(self.K)]

            # library function terms
            n_library_terms = 0
            for f in self.functions:
                for c in self._combinations(
                    n_features, f.__code__.co_argcount, self.interaction_only
                ):
                    n_library_terms += 1
            library_functions = np.empty((self.K, n_library_terms), dtype=x.dtype)

            # Evaluate the functions on the indices of domain cells
            funcs = np.zeros((*x.shape[:-1], n_library_terms))
            func_idx = 0
            for f in self.functions:
                for c in self._combinations(
                    n_features, f.__code__.co_argcount, self.interaction_only
                ):
                    funcs[..., func_idx] = f(*[x[..., j] for j in c])
                    func_idx += 1

            # library function terms
            for k in range(self.K):  # loop over domain cells
                # calculate the integral feature by taking the dot product
                # of the weights and functions over each axis
                library_functions[k] = np.tensordot(
                    self.fullweights0[k],
                    funcs[np.ix_(*self.inds_k[k])],
                    axes=(
                        tuple(np.arange(self.grid_ndim)),
                        tuple(np.arange(self.grid_ndim)),
                    ),
                )

            if self.derivative_order != 0:
                # pure integral terms
                library_integrals = np.empty(
                    (self.K, n_features * self.num_derivatives), dtype=x.dtype
                )

                for k in range(self.K):  # loop over domain cells
                    library_idx = 0
                    for j in range(self.num_derivatives):  # loop over derivatives
                        # Calculate the integral feature by taking the dot product
                        # of the weights and data x_k over each axis.
                        # Integration by parts gives power of (-1).
                        library_integrals[k, library_idx : library_idx + n_features] = (
                            -1
                        ) ** (np.sum(self.multiindices[j])) * np.tensordot(
                            self.fullweights1[k][j],
                            self.x_k[k],
                            axes=(
                                tuple(np.arange(self.grid_ndim)),
                                tuple(np.arange(self.grid_ndim)),
                            ),
                        )
                        library_idx += n_features

                # Mixed derivative/non-derivative terms
                if self.include_interaction:
                    library_mixed_integrals = np.empty(
                        (
                            self.K,
                            n_library_terms * n_features * self.num_derivatives,
                        ),
                        dtype=x.dtype,
                    )

                    # Below we integrate the product of function and feature
                    # derivatives against the derivatives of phi to calculate the weak
                    # features. We cannot remove all derivatives of data in this case,
                    # but we can reduce the derivative order by half.

                    # Calculate the necessary function and feature derivatives
                    funcs_derivs = np.zeros(
                        np.concatenate([[self.num_derivatives + 1], funcs.shape])
                    )
                    x_derivs = np.zeros(
                        np.concatenate([[self.num_derivatives + 1], x.shape])
                    )
                    funcs_derivs[0] = funcs
                    x_derivs[0] = x
                    self.dx_k_j = []
                    self.dfx_k_j = []

                    for j in range(self.num_derivatives):
                        for axis in range(self.ind_range):
                            s = [0] * (self.grid_ndim + 1)
                            s[axis] = slice(None, None, None)
                            s[-1] = axis
                            # Need derivatives of order less than half derivative_order
                            if self.multiindices[j][axis] > 0 and self.multiindices[j][
                                axis
                            ] <= (self.derivative_order // 2):
                                funcs_derivs[j + 1] = self.differentiation_method(
                                    d=self.multiindices[j][axis],
                                    axis=axis,
                                    **self.diff_kwargs,
                                )._differentiate(
                                    funcs, self.spatiotemporal_grid[tuple(s)]
                                )
                            if self.multiindices[j][axis] > 0 and self.multiindices[j][
                                axis
                            ] <= (self.derivative_order - (self.derivative_order // 2)):
                                x_derivs[j + 1] = self.differentiation_method(
                                    d=self.multiindices[j][axis],
                                    axis=axis,
                                    **self.diff_kwargs,
                                )._differentiate(x, self.spatiotemporal_grid[tuple(s)])

                    # Extract the function and feature derivatives on the domains
                    self.dx_k_j = [
                        [
                            x_derivs[j][np.ix_(*self.inds_k[k])]
                            for j in range(self.num_derivatives + 1)
                        ]
                        for k in range(self.K)
                    ]
                    self.dfx_k_j = [
                        [
                            funcs_derivs[j][np.ix_(*self.inds_k[k])]
                            for j in range(self.num_derivatives + 1)
                        ]
                        for k in range(self.K)
                    ]

                    # Calculate the mixed integrals
                    library_idx = 0
                    for j in range(self.num_derivatives):
                        integral = np.zeros((self.K, n_library_terms, n_features))
                        # Derivative orders after integration by parts
                        derivs_mixed = self.multiindices[j] // 2
                        derivs_pure = self.multiindices[j] - derivs_mixed
                        # Derivative orders for mixed derivatives product rule
                        derivs = np.concatenate(
                            [
                                [np.zeros(self.ind_range, dtype=int)],
                                self.multiindices,
                            ],
                            axis=0,
                        )
                        # Sum the terms in product rule
                        for deriv in derivs[
                            np.where(np.all(derivs <= derivs_mixed, axis=1))[0]
                        ]:
                            for k in range(self.K):
                                # Weights are either in fullweights0 or fullweights1
                                j0 = np.where(np.all(derivs == deriv, axis=1))[0][0]
                                if j0 == 0:
                                    weights = self.fullweights0[k]
                                else:
                                    weights = self.fullweights1[k][j0 - 1]

                                # indices for product rule terms
                                j1 = np.where(
                                    np.all(derivs == derivs_mixed - deriv, axis=1)
                                )[0][0]
                                j2 = np.where(np.all(derivs == derivs_pure, axis=1))[0][
                                    0
                                ]
                                # Calculate the integral by taking the dot product
                                # of the weights and data x_k over each axis.
                                # Integration by parts gives power of (-1).
                                # Binomial factor comes by product rule.
                                integral[k] = integral[k] + (-1) ** (
                                    np.sum(derivs_mixed)
                                ) * np.tensordot(
                                    weights,
                                    self.dfx_k_j[k][j1][..., np.newaxis]
                                    * self.dx_k_j[k][j2][..., np.newaxis, :],
                                    axes=(
                                        tuple(np.arange(self.grid_ndim)),
                                        tuple(np.arange(self.grid_ndim)),
                                    ),
                                ) * np.product(
                                    binom(derivs_mixed, deriv)
                                )
                        # collect the results
                        for n in range(n_features):
                            for m in range(n_library_terms):
                                library_mixed_integrals[:, library_idx] = integral[
                                    :, m, n
                                ]
                                library_idx += 1

            library_idx = 0
            # Constant term
            if self.include_bias:
                constants_final = np.zeros(self.K)
                for k in range(self.K):
                    constants_final[k] = np.sum(self.fullweights0[k])
                xp[:, library_idx] = constants_final
                library_idx += 1

            # library function terms
            xp[:, library_idx : library_idx + n_library_terms] = library_functions
            library_idx += n_library_terms

            if self.derivative_order != 0:
                # pure integral terms
                xp[
                    :, library_idx : library_idx + self.num_derivatives * n_features
                ] = library_integrals
                library_idx += self.num_derivatives * n_features

                # mixed function integral terms
                if self.include_interaction:
                    xp[
                        :,
                        library_idx : library_idx
                        + n_library_terms * self.num_derivatives * n_features,
                    ] = library_mixed_integrals
                    library_idx += n_library_terms * self.num_derivatives * n_features

            xp_full = xp_full + [AxesArray(xp, {"ax_sample": 0, "ax_coord": 1})]
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
        return xp_full

    def calc_trajectory(self, diff_method, x, t):
        x_dot = self.convert_u_dot_integral(x)
        return AxesArray(x_dot, {"ax_sample": 0, "ax_coord": 1})
