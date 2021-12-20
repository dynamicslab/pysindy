from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r
from itertools import product as iproduct

import numpy.random
from numpy import array
from numpy import concatenate
from numpy import empty
from numpy import linspace
from numpy import meshgrid
from numpy import newaxis
from numpy import prod
from numpy import reshape
from numpy import sum as np_sum
from numpy import take
from numpy import zeros
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.special import hyp2f1
from scipy.special import poch
from sklearn import __version__
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary


class WeakPDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions, optional (default None)
        Functions to include in the library. Each function will be
        applied to each input variable (but not their derivatives)

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable, max 4

    spatiotemporal_grid : np.ndarray, optional (default None)
        The spatial grid for computing derivatives. If using weak form
        terms with derivatives, must be specified with
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
        If True, function evaulations of the form :math:`f(x,x)` and :math:`f(x,y,x)`
        will be omitted, but those of the form :math:`f(x,y)` and :math:`f(x,y,z)`
        will be included.
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

    is_uniform : boolean, optional (default True)
        If True, assume the grid is uniform in all spatial directions, so
        can use uniform grid spacing for the derivative calculations.

    K : int, optional (default 100)
        Number of domain centers, corresponding to subdomain squares of length
        Hx and height Hy. If K is not
        specified, defaults to 100.

    H_xt : float, optional (default None)
        Half of the length of the square subdomains in each spatiotemporal
        direction. If H_xt is not specified, defaults to H_xt = L_xt / 20,
        where L_xt is the length of the full domain in each spatiotemporal
        direction.

    p : int, optional (default 4)
        Positive integer to define the polynomial degree of the spatial weights
        used for weak/integral SINDy.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

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
        is_uniform=False,
        K=100,
        num_pts_per_domain=100,
        H_xt=None,
        p=4,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(WeakPDELibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.derivative_order = derivative_order
        self.function_names = function_names
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.is_uniform = is_uniform
        self.K = K
        self.num_pts_per_domain = num_pts_per_domain
        self.H_xt = H_xt
        self.p = p

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
        if spatiotemporal_grid is None and derivative_order != 0:
            raise ValueError(
                "Spatiotemporal grid was not passed but derivative terms "
                "were asked for by the user."
            )
        if spatiotemporal_grid is None:
            raise ValueError(
                "Spatiotemporal grid was not passed, and at least a 1D"
                " grid is required, corresponding to the time base."
            )

        # list of integrals
        indices = ()
        if len(spatiotemporal_grid.shape) == 1:
            spatiotemporal_grid = reshape(
                spatiotemporal_grid, (len(spatiotemporal_grid), 1)
            )
        dims = spatiotemporal_grid.shape[:-1]
        self.grid_dims = dims
        self.grid_ndim = len(dims)

        for i in range(len(dims)):
            indices = indices + (range(derivative_order + 1),)

        multiindices = []
        for ind in iproduct(*indices):
            current = array(ind)
            if sum(ind) > 0 and sum(ind) <= derivative_order:
                multiindices.append(current)
        multiindices = array(multiindices)
        num_integrals = len(multiindices)

        self.num_derivatives = num_integrals
        self.multiindices = multiindices
        self.spatiotemporal_grid = spatiotemporal_grid

        # Weak form checks and setup
        self._weak_form_setup()

    def _weak_form_setup(self):
        xt1, xt2 = self._get_spatial_endpoints()
        L_xt = xt2 - xt1
        if self.H_xt is not None:
            if len(self.grid_ndim) != len(self.H_xt):
                raise ValueError(
                    "The user-defined grid (spatiotemporal_grid) and "
                    "the user-defined sizes of the subdomains for the "
                    "weak form, do not have the same # of spatiotemporal "
                    "dimensions. For instance, if spatiotemporal_grid is 4D, "
                    "then H_xt should be a 4D list of the subdomain lengths."
                )
            if any(self.H_xt <= 0):
                raise ValueError("Values in H_xt must be a positive float")
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

        if self.K <= 0:
            raise ValueError("The number of subdomains must be > 0")

        self._set_up_grids()

    def _get_spatial_endpoints(self):
        x1 = zeros(self.grid_ndim)
        x2 = zeros(self.grid_ndim)
        for i in range(self.grid_ndim):
            inds = [slice(None)] * (self.grid_ndim + 1)
            for j in range(self.grid_ndim):
                inds[j] = 0
            x1[i] = self.spatiotemporal_grid[inds][i]
            inds[i] = -1
            x2[i] = self.spatiotemporal_grid[inds][i]
        return x1, x2

    def _set_up_grids(self):
        xt1, xt2 = self._get_spatial_endpoints()
        domain_centers = zeros((self.K, self.grid_ndim))
        for i in range(self.grid_ndim):
            domain_centers[:, i] = numpy.random.uniform(
                xt1[i] + self.H_xt[i], xt2[i] - self.H_xt[i], size=self.K
            )
        xtgrid_k = zeros((self.K, self.num_pts_per_domain, self.grid_ndim))
        xt1_k = domain_centers - self.H_xt
        xt2_k = domain_centers + self.H_xt
        # mini_grids = []
        for k in range(self.K):
            for i in range(self.grid_ndim):
                xtgrid_k[k, :, i] = linspace(
                    xt1_k[k, i], xt2_k[k, i], self.num_pts_per_domain
                )
                # mini_grids.append(xtgrid_k[k, :, i])
        XT = zeros(
            (self.K, *(self.grid_ndim * [self.num_pts_per_domain]), self.grid_ndim)
        )
        inds_XT = [slice(None)] * XT.ndim
        for k in range(self.K):
            inds_XT[0] = k
            grid_k = array([xtgrid_k[k, :, i] for i in range(self.grid_ndim)])
            XT[inds_XT] = array(meshgrid(*grid_k, indexing="ij")).T
        self.xtgrid_k = xtgrid_k
        self.XT = XT

        grid_pts = []
        for i in range(self.grid_ndim):
            inds = [slice(None)] * (self.grid_ndim + 1)
            if self.grid_ndim > 1:
                for j in range(self.grid_ndim):
                    if j != i:
                        inds[j] = 0
            inds[-1] = i
            grid_pts.append(self.spatiotemporal_grid[inds])
        self.grid_pts = tuple(grid_pts)

        self.domain_centers_expanded = zeros(
            (self.K, *(self.grid_ndim * [self.num_pts_per_domain]), self.grid_ndim)
        )
        domain_inds = [slice(None)] * self.XT.ndim
        for i in range(self.K):
            for j in range(self.grid_ndim):
                domain_inds[0] = i
                domain_inds[-1] = j
                self.domain_centers_expanded[domain_inds] = domain_centers[i, j]

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def _poly_derivative(self, xt, d_xt):
        """Compute analytic derivatives instead of relying on finite diffs"""
        return prod(
            (2 * xt) ** d_xt
            * (xt ** 2 - 1) ** (self.p - d_xt)
            * hyp2f1((1 - d_xt) / 2.0, -d_xt / 2.0, self.p + 1 - d_xt, 1 - 1 / xt ** 2)
            * poch(self.p + 1 - d_xt, d_xt)
            / self.H_xt ** d_xt,
            axis=-1,
        )

    def _smooth_ppoly(self, d_xt):
        xt_tilde = (self.XT - self.domain_centers_expanded) / self.H_xt
        return self._poly_derivative(xt_tilde, d_xt)

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
            n_features = self.n_input_features
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
        # function_lib_len = 0
        # if self.functions is not None:
        #     function_lib_len = len(self.functions)
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
                for axis in range(len(self.spatiotemporal_grid.shape[:-1])):
                    for i in range(multiindex[axis]):
                        ret = ret + str(axis + 1)
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

    def fit(self, x, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(x).shape
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

    def transform(self, x):
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

        x = check_array(x)

        n_samples, n_features = x.shape
        if float(__version__[:3]) >= 1.0:
            if n_features != self.n_features_in_:
                raise ValueError("x shape does not match training shape")
        else:
            if n_features != self.n_input_features_:
                raise ValueError("x shape does not match training shape")

        if self.spatiotemporal_grid is not None:
            n_samples = self.K

        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)

        # pure integral terms, need to differentiate the weight functions
        library_idx = 0
        library_integrals = empty(
            (n_samples, n_features * self.num_derivatives), dtype=x.dtype
        )

        funcs = reshape(x, concatenate([self.grid_dims, [n_features]]))
        func_final = zeros((self.K, n_features))
        # Note excluding temporal derivatives in the feature library
        for multiindex in self.multiindices:
            derivs = zeros(self.grid_ndim)
            for axis in range(self.grid_ndim - 1):
                if multiindex[axis] > 0:
                    s = [0 for dim in self.grid_dims[:-1]]
                    s[axis] = slice(self.grid_dims[axis])
                    s[-1] = axis
                    derivs[axis] = multiindex[axis]
            integral_weight_derivs = self._smooth_ppoly(derivs) * (-1) ** (
                np_sum(derivs) % 2
            )
            for j in range(n_features):
                func_interp = RegularGridInterpolator(
                    self.grid_pts, take(funcs, j, axis=-1)
                )
                for k in range(self.K):
                    func_new = func_interp(take(self.XT, k, axis=0))
                    func_temp = trapezoid(
                        take(integral_weight_derivs, k, axis=0) * func_new,
                        x=self.xtgrid_k[k, :, 0],
                        axis=0,
                    )
                    for i in range(1, self.grid_ndim):
                        func_temp = trapezoid(
                            func_temp, x=self.xtgrid_k[k, :, i], axis=0
                        )
                    func_final[k, j] = func_temp
            library_integrals[:, library_idx : library_idx + n_features] = func_final
            library_idx += n_features

        # library function terms
        n_library_terms = 0
        for f in self.functions:
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                n_library_terms += 1

        library_functions = empty((n_samples, n_library_terms), dtype=x.dtype)
        library_idx = 0
        integral_weights = self._smooth_ppoly(zeros(self.grid_ndim))
        func_final = zeros(self.K)
        for f in self.functions:
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                func = f(*[x[:, j] for j in c])
                func = reshape(func, self.grid_dims)
                func_interp = RegularGridInterpolator(self.grid_pts, func)
                for k in range(self.K):
                    func_new = func_interp(take(self.XT, k, axis=0))
                    func_temp = trapezoid(
                        take(integral_weights, k, axis=0) * func_new,
                        x=self.xtgrid_k[k, :, 0],
                        axis=0,
                    )
                    for i in range(1, self.grid_ndim):
                        func_temp = trapezoid(
                            func_temp, x=self.xtgrid_k[k, :, i], axis=0
                        )
                    func_final[k] = func_temp
                library_functions[:, library_idx] = func_final
                library_idx += 1

        library_idx = 0

        # Constant term
        if self.include_bias:
            for k in range(self.K):
                func_final[k] = trapezoid(
                    take(integral_weights, k, axis=0), x=self.xtgrid_k[k, :, 0], axis=0
                )
                for i in range(1, self.grid_ndim - 1):
                    func_final[k] = trapezoid(
                        func_final[k], x=self.xtgrid_k[k, :, i], axis=0
                    )
            xp[:, library_idx] = func_final
            library_idx += 1

        # library function terms
        xp[:, library_idx : library_idx + n_library_terms] = library_functions
        library_idx += n_library_terms

        # pure integral terms
        xp[
            :, library_idx : library_idx + self.num_derivatives * n_features
        ] = library_integrals
        library_idx += self.num_derivatives * n_features

        # mixed function integral terms
        # Should include a self.interaction_only case as well?
        if self.include_interaction:
            xp[
                :,
                library_idx : library_idx
                + n_library_terms * self.num_derivatives * n_features,
            ] = reshape(
                library_functions[:, :, newaxis] * library_integrals[:, newaxis, :],
                (n_samples, n_library_terms * self.num_derivatives * n_features),
            )
            library_idx += n_library_terms * self.num_derivatives * n_features

        # Need to recompute derivatives because the training
        # and testing data may have different number of time points
        # Need to compute any derivatives now
        # if self.spatiotemporal_grid is not None:
        #     u = reshape(x, (num_gridx, num_gridy, num_gridz, num_time, n_features))
        #     u_derivatives = self._make_3D_derivatives(u)
        #     self.u_derivatives = u_derivatives
        #     u_derivatives = asarray(
        #         reshape(
        #             u_derivatives,
        #             (
        #                 product(self.grid_dims),
        #                 n_features,
        #                 self.num_derivatives,
        #             ),
        #         )
        #     )
        #
        #     u_integrals = zeros((self.K, n_features, self.num_derivatives))
        #     for kk in range(n_features):
        #         u_interp = RegularGridInterpolator(
        #             (
        #                 self.spatiotemporal_grid[:, 0, 0, 0],
        #                 self.spatiotemporal_grid[0, :, 0, 1],
        #                 self.spatiotemporal_grid[0, 0, :, 2],
        #                 self.temporal_grid,
        #             ),
        #             u[:, :, :, :, kk],
        #         )
        #         for k in range(self.K):
        #             u_new = u_interp(
        #                 array(
        #                     (
        #                         self.X[k, :, :, :, :],
        #                         self.Y[k, :, :, :, :],
        #                         self.Z[k, :, :, :, :],
        #                         self.t[k, :, :, :, :],
        #                     )
        #                 ).T
        #             )
        #             u_new = reshape(
        #                 u_new,
        #                 (
        #                     self.num_pts_per_domain,
        #                     self.num_pts_per_domain,
        #                     self.num_pts_per_domain,
        #                     self.num_pts_per_domain,
        #                     1,
        #                 ),
        #             )
        #             for j in range(self.num_derivatives):
        #                 w_diff = self._smooth_ppoly(
        #                     transpose(
        #                         (
        #                             self.xgrid_k[k, :],
        #                             self.ygrid_k[k, :],
        #                             self.zgrid_k[k, :],
        #                         )
        #                     ),
        #                     self.tgrid_k[k, :],
        #                     k,
        #                     derivative_list[j, 0],
        #                     derivative_list[j, 1],
        #                     derivative_list[j, 2],
        #                     0,
        #                 )
        #                 u_integrals[k, kk, j] = (
        #                     trapezoid(
        #                         trapezoid(
        #                             trapezoid(
        #                                 trapezoid(
        #                                     u_new * w_diff,
        #                                     x=self.xgrid_k[k, :],
        #                                     axis=0,
        #                                 ),
        #                                 x=self.ygrid_k[k, :],
        #                                 axis=0,
        #                             ),
        #                             x=self.zgrid_k[k, :],
        #                             axis=0,
        #                         ),
        #                         x=self.tgrid_k[k, :],
        #                         axis=0,
        #                     )
        #                     * signs_list[j]
        #                 )
        #
        #     # pure integral terms
        #     if self.grid_ndim >= 2:
        #         xp[:, library_idx:library_idx + self.num_derivatives * n_features
        #            ] = library_integrals
        #         library_idx += self.num_derivatives * n_features
        #
        #     # Add all the mixed derivative/non-derivative terms
        #     if self.include_interaction:
        #         for d in range(self.num_derivatives):
        #             for kk in range(n_features):
        #                 for f in self.functions:
        #                     for c in self._combinations(
        #                         n_features,
        #                         f.__code__.co_argcount,
        #                         self.interaction_only,
        #                     ):
        #                         func = f(*[x[:, j] for j in c])
        #                         derivative_term = identity_function(
        #                             u_derivatives[:, kk, d]
        #                         )
        #
        #                         func = reshape(
        #                             func,
        #                             (num_gridx, num_gridy, num_gridz, num_time),
        #                         )
        #                         derivative_term = reshape(
        #                             derivative_term,
        #                             (num_gridx, num_gridy, num_gridz, num_time),
        #                         )
        #                         func_final = zeros((self.K, 1))
        #                         func_interp = RegularGridInterpolator(
        #                             (
        #                                 self.spatiotemporal_grid[:, 0, 0, 0],
        #                                 self.spatiotemporal_grid[0, :, 0, 1],
        #                                 self.spatiotemporal_grid[0, 0, :, 2],
        #                                 self.temporal_grid,
        #                             ),
        #                             func,
        #                         )
        #                         derivative_interp = RegularGridInterpolator(
        #                             (
        #                                 self.spatiotemporal_grid[:, 0, 0, 0],
        #                                 self.spatiotemporal_grid[0, :, 0, 1],
        #                                 self.spatiotemporal_grid[0, 0, :, 2],
        #                                 self.temporal_grid,
        #                             ),
        #                             derivative_term,
        #                         )
        #                         for k in range(self.K):
        #                             XYt = array(
        #                                 (
        #                                     self.X[k, :, :, :, :],
        #                                     self.Y[k, :, :, :, :],
        #                                     self.Z[k, :, :, :, :],
        #                                     self.t[k, :, :, :, :],
        #                                 )
        #                             ).T
        #                             func_new = func_interp(XYt)
        #                             derivative_new = derivative_interp(XYt)
        #                             func_new = reshape(
        #                                 func_new,
        #                                 (
        #                                     self.num_pts_per_domain,
        #                                     self.num_pts_per_domain,
        #                                     self.num_pts_per_domain,
        #                                     self.num_pts_per_domain,
        #                                     1,
        #                                 ),
        #                             )
        #                             derivative_new = reshape(
        #                                 derivative_new,
        #                                 (
        #                                     self.num_pts_per_domain,
        #                                     self.num_pts_per_domain,
        #                                     self.num_pts_per_domain,
        #                                     self.num_pts_per_domain,
        #                                     1,
        #                                 ),
        #                             )
        #                             w = self._smooth_ppoly(
        #                                 transpose(
        #                                     (
        #                                         self.xgrid_k[k, :],
        #                                         self.ygrid_k[k, :],
        #                                         self.zgrid_k[k, :],
        #                                     )
        #                                 ),
        #                                 self.tgrid_k[k, :],
        #                                 k,
        #                                 0,
        #                                 0,
        #                                 0,
        #                                 0,
        #                             )
        #                             func_final[k] = trapezoid(
        #                                 trapezoid(
        #                                     trapezoid(
        #                                         trapezoid(
        #                                             w * func_new * derivative_new,
        #                                             x=self.xgrid_k[k, :],
        #                                             axis=0,
        #                                         ),
        #                                         x=self.ygrid_k[k, :],
        #                                         axis=0,
        #                                     ),
        #                                     x=self.zgrid_k[k, :],
        #                                     axis=0,
        #                                 ),
        #                                 x=self.tgrid_k[k, :],
        #                                 axis=0,
        #                             )
        #                         xp[:, library_idx] = ravel(func_final)
        #                         library_idx += 1
        # If library bagging, return xp missing the terms at ensemble_indices
        return self._ensemble(xp)
