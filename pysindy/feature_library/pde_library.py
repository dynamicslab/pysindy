from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import array
from numpy import asarray
from numpy import empty
from numpy import hstack
from numpy import linspace
from numpy import meshgrid
from numpy import ones
from numpy import ravel
from numpy import reshape
from numpy import shape
from numpy import transpose
from numpy import zeros
from numpy.random import seed
from numpy.random import uniform
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.special import comb as n_choose_k
from scipy.special import hyp2f1
from scipy.special import poch
from sklearn import __version__
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from pysindy.differentiation import FiniteDifference


class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable (but not their derivatives)

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable, max 4

    spatial_grid : np.ndarray, optional (default None)
        The spatial grid for computing derivatives

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

    is_uniform : boolean, optional (default True)
        If True, assume the grid is uniform in all spatial directions, so
        can use uniform grid spacing for the derivative calculations.

    weak_form : boolean, optional (default False)
        If True, uses the weak/integral form of SINDy, requiring some extra
        parameters.

    K : int, optional (default 100)
        Number of domain centers, corresponding to subdomain squares of length
        Hx and height Hy. If weak_form is True but K is not
        specified, defaults to 100.

    Hx : float, optional (default None)
        Half of the length of the square subdomains. If weak_form is True
        but Hx is not specified, defaults to Hx = Lx / 20, where
        Lx is the length of the full domain.

    Hy : float, optional (default None)
        Half of the height of the square subdomains. If weak_form is True
        but Hy is not specified, defaults to Hy = Ly / 20, where
        Ly is the height of the full domain.

    Ht : float, optional (default None)
        Half of the temporal length of the square subdomains.
        If weak_form is True but Ht is not specified, defaults to
        Ht = M / 20, where M is the number of time points.

    p : int, optional (default 4)
        Positive integer to define the polynomial degree of the spatial weights
        used for weak/integral SINDy.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

    rand_seed : integer, optional (default 100)
        A random seed for the random seed generator. This determines the
        selection of the center points in the PDE subdomains.

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
    >>> from pysindy.feature_library import PDELibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = PDELibrary(library_functions=functions).fit(x)
    >>> lib.transform(x)
    array([[ 1.        ,  0.36787944, -0.84147098],
           [ 2.71828183,  1.        ,  0.84147098],
           [ 7.3890561 ,  0.36787944,  0.84147098]])
    >>> lib.get_feature_names()
    ['f0(x0)', 'f0(x1)', 'f1(x0,x1)']
    """

    def __init__(
        self,
        library_functions,
        derivative_order=0,
        spatial_grid=None,
        temporal_grid=None,
        function_names=None,
        interaction_only=True,
        include_bias=False,
        is_uniform=False,
        weak_form=False,
        rand_seed=100,
        K=100,
        num_pts_per_domain=100,
        Hx=None,
        Hy=None,
        Hz=None,
        Ht=None,
        p=4,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(PDELibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.derivative_order = derivative_order
        self.spatial_grid = spatial_grid
        self.temporal_grid = temporal_grid
        self.functions = library_functions
        self.function_names = function_names
        self.include_bias = include_bias
        self.is_uniform = is_uniform
        self.num_pts_per_domain = num_pts_per_domain
        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if derivative_order < 0 or derivative_order > 4:
            raise ValueError("The derivative order must be 1, 2, 3, or 4")
        if (spatial_grid is not None and derivative_order == 0) or (
            spatial_grid is None and derivative_order != 0
        ):
            raise ValueError(
                "Spatial grid and the derivative order must be "
                "defined at the same time"
            )
        if spatial_grid is not None and (
            (len(spatial_grid.shape) != 1)
            and (len(spatial_grid.shape) != 3)
            and (len(spatial_grid.shape) != 4)
        ):
            raise ValueError("Spatial grid size is incorrect")
        self.interaction_only = interaction_only
        if spatial_grid is not None:
            self.s_len = len((self.spatial_grid).shape)
            if self.s_len == 1:
                self.num_derivatives = derivative_order
            elif self.s_len == 3:
                num_derivatives = 2
                for i in range(2, derivative_order + 1):
                    num_derivatives += i + 1
                self.num_derivatives = num_derivatives
            elif self.s_len == 4:
                k = 3
                num_derivatives = 0
                for i in range(1, derivative_order + 1):
                    num_derivatives += int(n_choose_k(i + k - 1, k - 1))
                self.num_derivatives = num_derivatives
        else:
            self.s_len = 0

        # weak form checks now
        if weak_form and temporal_grid is not None:
            if len(shape(temporal_grid)) != 1:
                raise ValueError("Temporal grid must be 1D.")
            self.M = len(temporal_grid)
            t1 = temporal_grid[0]
            t2 = temporal_grid[-1]
            if Ht is not None:
                if Ht <= 0:
                    raise ValueError("Ht must be a positive float")
                if Ht >= (t2 - t1) / 2.0:
                    raise ValueError("2 * Ht is larger than the time domain")
                self.Ht = Ht
            else:
                Lt = t2 - t1
                self.Ht = Lt / 20.0
        if weak_form and (temporal_grid is None):
            raise ValueError("Weak form requires user to pass a temporal grid.")

        self.weak_form = weak_form
        self.num_pts_per_domain = num_pts_per_domain
        self.p = p
        if weak_form and spatial_grid is not None:
            if p < 0:
                raise ValueError("Poly degree of the spatial weights must be > 0")
            if Hx is not None:
                if Hx <= 0:
                    raise ValueError("Hx must be a positive float")
                if self.s_len == 1:
                    x1 = spatial_grid[0]
                    x2 = spatial_grid[-1]
                if self.s_len == 3:
                    x1 = spatial_grid[0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0]
                if self.s_len == 4:
                    x1 = spatial_grid[0, 0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0, 0]
                if Hx >= (x2 - x1) / 2.0:
                    raise ValueError("2 * Hx is bigger than the full domain length")
                self.Hx = Hx
            else:
                if self.s_len == 1:
                    x1 = spatial_grid[0]
                    x2 = spatial_grid[-1]
                if self.s_len == 3:
                    x1 = spatial_grid[0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0]
                if self.s_len == 4:
                    x1 = spatial_grid[0, 0, 0, 0]
                    x2 = spatial_grid[-1, 0, 0, 0]
                Lx = x2 - x1
                self.Hx = Lx / 20.0
            if Hy is not None and self.s_len >= 3:
                if Hy <= 0:
                    raise ValueError("Hy must be a positive float")
                if self.s_len == 3:
                    y1 = spatial_grid[0, 0, 1]
                    y2 = spatial_grid[0, -1, 1]
                if self.s_len == 4:
                    y1 = spatial_grid[0, 0, 0, 1]
                    y2 = spatial_grid[0, -1, 0, 1]
                if Hy >= (y2 - y1) / 2.0:
                    raise ValueError("2 * Hy is bigger than the full domain height")
                self.Hy = Hy
            elif Hy is None and self.s_len >= 3:
                if self.s_len == 3:
                    y1 = spatial_grid[0, 0, 1]
                    y2 = spatial_grid[0, -1, 1]
                if self.s_len == 4:
                    y1 = spatial_grid[0, 0, 0, 1]
                    y2 = spatial_grid[0, -1, 0, 1]
                Ly = y2 - y1
                self.Hy = Ly / 20.0
            if Hz is not None and self.s_len == 4:
                if Hz <= 0:
                    raise ValueError("Hz must be a positive float")
                z1 = spatial_grid[0, 0, 0, 2]
                z2 = spatial_grid[0, 0, -1, 2]
                if Hz >= (z2 - z1) / 2.0:
                    raise ValueError("2 * Hz is bigger than the full domain height")
                self.Hz = Hz
            elif Hz is None and self.s_len == 4:
                z1 = spatial_grid[0, 0, 0, 2]
                z2 = spatial_grid[0, 0, -1, 2]
                Lz = z2 - z1
                self.Hz = Lz / 20.0
        if weak_form:
            if K <= 0:
                raise ValueError("The number of subdomains must be > 0")
            self.K = K
            seed(rand_seed)
            if self.s_len == 0:
                self.domain_centers = uniform(t1 + self.Ht, t2 - self.Ht, size=self.K)
                tgrid_k = zeros((self.K, num_pts_per_domain))
                for k in range(self.K):
                    t1_k = self.domain_centers[k] - self.Ht
                    t2_k = self.domain_centers[k] + self.Ht
                    tgrid_k[k, :] = linspace(t1_k, t2_k, self.num_pts_per_domain)
                self.tgrid_k = tgrid_k
            if self.s_len == 1:
                x1 = spatial_grid[0]
                x2 = spatial_grid[-1]
                domain_centers_x = uniform(x1 + self.Hx, x2 - self.Hx, size=(self.K, 1))
                domain_centers_t = uniform(t1 + self.Ht, t2 - self.Ht, size=(self.K, 1))
                domain_centers = hstack((domain_centers_x, domain_centers_t))
                self.domain_centers = domain_centers
                xgrid_k = zeros((self.K, num_pts_per_domain))
                tgrid_k = zeros((self.K, num_pts_per_domain))
                X = zeros((self.K, num_pts_per_domain, num_pts_per_domain))
                t = zeros((self.K, num_pts_per_domain, num_pts_per_domain))
                for k in range(self.K):
                    x1_k = self.domain_centers[k, 0] - self.Hx
                    x2_k = self.domain_centers[k, 0] + self.Hx
                    t1_k = self.domain_centers[k, -1] - self.Ht
                    t2_k = self.domain_centers[k, -1] + self.Ht
                    xgrid_k[k, :] = linspace(x1_k, x2_k, self.num_pts_per_domain)
                    tgrid_k[k, :] = linspace(t1_k, t2_k, self.num_pts_per_domain)
                    X[k, :, :], t[k, :, :] = meshgrid(
                        xgrid_k[k, :], tgrid_k[k, :], indexing="ij"
                    )
                self.xgrid_k = xgrid_k
                self.tgrid_k = tgrid_k
                self.X = X
                self.t = t
            if self.s_len == 3:
                x1 = spatial_grid[0, 0, 0]
                x2 = spatial_grid[-1, 0, 0]
                y1 = spatial_grid[0, 0, 1]
                y2 = spatial_grid[0, -1, 1]
                domain_centers_x = uniform(x1 + self.Hx, x2 - self.Hx, size=(self.K, 1))
                domain_centers_y = uniform(y1 + self.Hy, y2 - self.Hy, size=(self.K, 1))
                domain_centers_t = uniform(t1 + self.Ht, t2 - self.Ht, size=(self.K, 1))
                domain_centers = hstack((domain_centers_x, domain_centers_y))
                domain_centers = hstack((domain_centers, domain_centers_t))
                self.domain_centers = domain_centers
                xgrid_k = zeros((self.K, num_pts_per_domain))
                ygrid_k = zeros((self.K, num_pts_per_domain))
                tgrid_k = zeros((self.K, num_pts_per_domain))
                X = zeros(
                    (self.K, num_pts_per_domain, num_pts_per_domain, num_pts_per_domain)
                )
                Y = zeros(
                    (self.K, num_pts_per_domain, num_pts_per_domain, num_pts_per_domain)
                )
                t = zeros(
                    (self.K, num_pts_per_domain, num_pts_per_domain, num_pts_per_domain)
                )
                for k in range(self.K):
                    x1_k = self.domain_centers[k, 0] - self.Hx
                    x2_k = self.domain_centers[k, 0] + self.Hx
                    y1_k = self.domain_centers[k, 1] - self.Hy
                    y2_k = self.domain_centers[k, 1] + self.Hy
                    t1_k = self.domain_centers[k, -1] - self.Ht
                    t2_k = self.domain_centers[k, -1] + self.Ht
                    xgrid_k[k, :] = linspace(x1_k, x2_k, self.num_pts_per_domain)
                    ygrid_k[k, :] = linspace(y1_k, y2_k, self.num_pts_per_domain)
                    tgrid_k[k, :] = linspace(t1_k, t2_k, self.num_pts_per_domain)
                    X[k, :, :, :], Y[k, :, :, :], t[k, :, :, :] = meshgrid(
                        xgrid_k[k, :],
                        ygrid_k[k, :],
                        tgrid_k[k, :],
                        indexing="ij",
                    )
                self.xgrid_k = xgrid_k
                self.ygrid_k = ygrid_k
                self.tgrid_k = tgrid_k
                self.X = X
                self.Y = Y
                self.t = t
            if self.s_len == 4:
                x1 = spatial_grid[0, 0, 0, 0]
                x2 = spatial_grid[-1, 0, 0, 0]
                y1 = spatial_grid[0, 0, 0, 1]
                y2 = spatial_grid[0, -1, 0, 1]
                z1 = spatial_grid[0, 0, 0, 2]
                z2 = spatial_grid[0, 0, -1, 2]
                domain_centers_x = uniform(x1 + self.Hx, x2 - self.Hx, size=(self.K, 1))
                domain_centers_y = uniform(y1 + self.Hy, y2 - self.Hy, size=(self.K, 1))
                domain_centers_z = uniform(z1 + self.Hz, z2 - self.Hz, size=(self.K, 1))
                domain_centers_t = uniform(t1 + self.Ht, t2 - self.Ht, size=(self.K, 1))
                domain_centers = hstack((domain_centers_x, domain_centers_y))
                domain_centers = hstack((domain_centers, domain_centers_z))
                domain_centers = hstack((domain_centers, domain_centers_t))
                self.domain_centers = domain_centers
                xgrid_k = zeros((self.K, num_pts_per_domain))
                ygrid_k = zeros((self.K, num_pts_per_domain))
                zgrid_k = zeros((self.K, num_pts_per_domain))
                tgrid_k = zeros((self.K, num_pts_per_domain))
                X = zeros(
                    (
                        self.K,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                    )
                )
                Y = zeros(
                    (
                        self.K,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                    )
                )
                Z = zeros(
                    (
                        self.K,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                    )
                )
                t = zeros(
                    (
                        self.K,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                        num_pts_per_domain,
                    )
                )
                for k in range(self.K):
                    x1_k = self.domain_centers[k, 0] - self.Hx
                    x2_k = self.domain_centers[k, 0] + self.Hx
                    y1_k = self.domain_centers[k, 1] - self.Hy
                    y2_k = self.domain_centers[k, 1] + self.Hy
                    z1_k = self.domain_centers[k, 2] - self.Hz
                    z2_k = self.domain_centers[k, 2] + self.Hz
                    t1_k = self.domain_centers[k, -1] - self.Ht
                    t2_k = self.domain_centers[k, -1] + self.Ht
                    xgrid_k[k, :] = linspace(x1_k, x2_k, self.num_pts_per_domain)
                    ygrid_k[k, :] = linspace(y1_k, y2_k, self.num_pts_per_domain)
                    zgrid_k[k, :] = linspace(z1_k, z2_k, self.num_pts_per_domain)
                    tgrid_k[k, :] = linspace(t1_k, t2_k, self.num_pts_per_domain)
                    (
                        X[k, :, :, :, :],
                        Y[k, :, :, :, :],
                        Z[k, :, :, :, :],
                        t[k, :, :, :, :],
                    ) = meshgrid(
                        xgrid_k[k, :],
                        ygrid_k[k, :],
                        zgrid_k[k, :],
                        tgrid_k[k, :],
                        indexing="ij",
                        # sparse=True,
                    )
                self.xgrid_k = xgrid_k
                self.ygrid_k = ygrid_k
                self.zgrid_k = zgrid_k
                self.tgrid_k = tgrid_k
                self.X = X
                self.Y = Y
                self.Z = Z
                self.t = t

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def _poly_derivative_0D(self, t, d_t):
        """Compute analytic derivatives instead of relying on finite diffs"""
        t_term = (
            (2 * t) ** d_t
            * (t ** 2 - 1) ** (self.p - d_t)
            * hyp2f1((1 - d_t) / 2.0, -d_t / 2.0, self.p + 1 - d_t, 1 - 1 / t ** 2)
            * poch(self.p + 1 - d_t, d_t)
            / self.Ht ** d_t
        )
        return t_term

    def _poly_derivative_1D(self, x, t, d_x, d_t):
        """Compute analytic derivatives instead of relying on finite diffs"""
        x_term = (
            (2 * x) ** d_x
            * (x ** 2 - 1) ** (self.p - d_x)
            * hyp2f1((1 - d_x) / 2.0, -d_x / 2.0, self.p + 1 - d_x, 1 - 1 / x ** 2)
            * poch(self.p + 1 - d_x, d_x)
            / self.Hx ** d_x
        )
        t_term = (
            (2 * t) ** d_t
            * (t ** 2 - 1) ** (self.p - d_t)
            * hyp2f1((1 - d_t) / 2.0, -d_t / 2.0, self.p + 1 - d_t, 1 - 1 / t ** 2)
            * poch(self.p + 1 - d_t, d_t)
            / self.Ht ** d_t
        )
        return x_term * t_term

    def _poly_derivative_2D(self, x, y, t, d_x, d_y, d_t):
        """Compute analytic derivatives instead of relying on finite diffs"""
        x_term = (
            (2 * x) ** d_x
            * (x ** 2 - 1) ** (self.p - d_x)
            * hyp2f1((1 - d_x) / 2.0, -d_x / 2.0, self.p + 1 - d_x, 1 - 1 / x ** 2)
            * poch(self.p + 1 - d_x, d_x)
            / self.Hx ** d_x
        )
        y_term = (
            (2 * y) ** d_y
            * (y ** 2 - 1) ** (self.p - d_y)
            * hyp2f1((1 - d_y) / 2.0, -d_y / 2.0, self.p + 1 - d_y, 1 - 1 / y ** 2)
            * poch(self.p + 1 - d_y, d_y)
            / self.Hy ** d_y
        )
        t_term = (
            (2 * t) ** d_t
            * (t ** 2 - 1) ** (self.p - d_t)
            * hyp2f1((1 - d_t) / 2.0, -d_t / 2.0, self.p + 1 - d_t, 1 - 1 / t ** 2)
            * poch(self.p + 1 - d_t, d_t)
            / self.Ht ** d_t
        )
        return x_term * y_term * t_term

    def _poly_derivative_3D(self, x, y, z, t, d_x, d_y, d_z, d_t):
        """Compute analytic derivatives instead of relying on finite diffs"""
        x_term = (
            (2 * x) ** d_x
            * (x ** 2 - 1) ** (self.p - d_x)
            * hyp2f1((1 - d_x) / 2.0, -d_x / 2.0, self.p + 1 - d_x, 1 - 1 / x ** 2)
            * poch(self.p + 1 - d_x, d_x)
            / self.Hx ** d_x
        )
        y_term = (
            (2 * y) ** d_y
            * (y ** 2 - 1) ** (self.p - d_y)
            * hyp2f1((1 - d_y) / 2.0, -d_y / 2.0, self.p + 1 - d_y, 1 - 1 / y ** 2)
            * poch(self.p + 1 - d_y, d_y)
            / self.Hy ** d_y
        )
        z_term = (
            (2 * z) ** d_z
            * (z ** 2 - 1) ** (self.p - d_z)
            * hyp2f1((1 - d_z) / 2.0, -d_z / 2.0, self.p + 1 - d_z, 1 - 1 / z ** 2)
            * poch(self.p + 1 - d_z, d_z)
            / self.Hz ** d_z
        )
        t_term = (
            (2 * t) ** d_t
            * (t ** 2 - 1) ** (self.p - d_t)
            * hyp2f1((1 - d_t) / 2.0, -d_t / 2.0, self.p + 1 - d_t, 1 - 1 / t ** 2)
            * poch(self.p + 1 - d_t, d_t)
            / self.Ht ** d_t
        )
        return x_term * y_term * z_term * t_term

    def _smooth_ppoly(self, x, t, k, d_x, d_y, d_z, d_t):
        if shape(x)[0] == 0:
            t_tilde = zeros(shape(t)[0])
            weights = zeros((shape(t)[0], 1))
            for i in range(shape(t)[0]):
                t_tilde[i] = (t[i] - self.domain_centers[k]) / self.Ht
            weights = self._poly_derivative_0D(t_tilde, d_t)
        elif shape(x)[1] == 1:
            x_tilde = zeros(shape(x)[0])
            t_tilde = zeros(shape(t)[0])
            weights = zeros((shape(x)[0], shape(t)[0], 1))
            for i in range(shape(x)[0]):
                x_tilde[i] = (x[i] - self.domain_centers[k, 0]) / self.Hx
                t_tilde[i] = (t[i] - self.domain_centers[k, -1]) / self.Ht
            for i in range(shape(x)[0]):
                weights[i, :, 0] = self._poly_derivative_1D(
                    x_tilde[i], t_tilde, d_x, d_t
                )
        elif shape(x)[1] == 2:
            x_tilde = zeros(shape(x)[0])
            y_tilde = zeros(shape(x)[0])
            t_tilde = zeros(shape(t)[0])
            for i in range(shape(x)[0]):
                x_tilde[i] = (x[i, 0] - self.domain_centers[k, 0]) / self.Hx
                y_tilde[i] = (x[i, 1] - self.domain_centers[k, 1]) / self.Hy
                t_tilde[i] = (t[i] - self.domain_centers[k, -1]) / self.Ht
            weights = zeros((shape(x)[0], shape(x)[0], shape(t)[0], 1))
            for i in range(shape(x)[0]):
                for j in range(shape(x)[0]):
                    weights[i, j, :, 0] = self._poly_derivative_2D(
                        x_tilde[i], y_tilde[j], t_tilde, d_x, d_y, d_t
                    )
        elif shape(x)[1] == 3:
            x_tilde = zeros(shape(x)[0])
            y_tilde = zeros(shape(x)[0])
            z_tilde = zeros(shape(x)[0])
            t_tilde = zeros(shape(t)[0])
            for i in range(shape(x)[0]):
                x_tilde[i] = (x[i, 0] - self.domain_centers[k, 0]) / self.Hx
                y_tilde[i] = (x[i, 1] - self.domain_centers[k, 1]) / self.Hy
                z_tilde[i] = (x[i, 2] - self.domain_centers[k, 2]) / self.Hz
                t_tilde[i] = (t[i] - self.domain_centers[k, -1]) / self.Ht
            weights = zeros((shape(x)[0], shape(x)[0], shape(x)[0], shape(t)[0], 1))
            for i in range(shape(x)[0]):
                for j in range(shape(x)[0]):
                    for k in range(shape(x)[0]):
                        weights[i, j, k, :, 0] = self._poly_derivative_3D(
                            x_tilde[i],
                            y_tilde[j],
                            z_tilde[j],
                            t_tilde,
                            d_x,
                            d_y,
                            d_z,
                            d_t,
                        )
        return weights

    def _make_2D_derivatives(self, u):
        (num_gridx, num_gridy, num_time, n_features) = u.shape
        u_derivatives = zeros(
            (num_gridx, num_gridy, num_time, n_features, self.num_derivatives)
        )
        # first derivatives
        for i in range(num_time):
            # u_x
            for kk in range(num_gridy):
                u_derivatives[:, kk, i, :, 0] = FiniteDifference(
                    d=1, is_uniform=self.is_uniform
                )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
            # u_y
            for kk in range(num_gridx):
                u_derivatives[kk, :, i, :, 1] = FiniteDifference(
                    d=1, is_uniform=self.is_uniform
                )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 2:
            # second derivatives
            for i in range(num_time):
                # u_xx
                for kk in range(num_gridy):
                    u_derivatives[:, kk, i, :, 2] = FiniteDifference(
                        d=2, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 3] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivatives[kk, :, i, :, 0], self.spatial_grid[kk, :, 1]
                    )
                # u_yy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 4] = FiniteDifference(
                        d=2, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 3:
            # third derivatives
            for i in range(num_time):
                # u_xxx
                for kk in range(num_gridy):
                    u_derivatives[:, kk, i, :, 5] = FiniteDifference(
                        d=3, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xxy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 6] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivatives[kk, :, i, :, 2], self.spatial_grid[kk, :, 1]
                    )
                # u_xyy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 7] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivatives[kk, :, i, :, 3], self.spatial_grid[kk, :, 1]
                    )
                # u_yyy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 8] = FiniteDifference(
                        d=3, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])

        if self.derivative_order >= 4:
            # fourth derivatives
            for i in range(num_time):
                # u_xxxx
                for kk in range(num_gridy):
                    u_derivatives[:, kk, i, :, 9] = FiniteDifference(
                        d=4, is_uniform=self.is_uniform
                    )._differentiate(u[:, kk, i, :], self.spatial_grid[:, kk, 0])
                # u_xxxy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 10] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivatives[kk, :, i, :, 5], self.spatial_grid[kk, :, 1]
                    )
                # u_xxyy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 11] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivatives[kk, :, i, :, 6], self.spatial_grid[kk, :, 1]
                    )
                # u_xyyy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 12] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u_derivatives[kk, :, i, :, 7], self.spatial_grid[kk, :, 1]
                    )
                # u_yyyy
                for kk in range(num_gridx):
                    u_derivatives[kk, :, i, :, 13] = FiniteDifference(
                        d=4, is_uniform=self.is_uniform
                    )._differentiate(u[kk, :, i, :], self.spatial_grid[kk, :, 1])
        return u_derivatives

    def _make_3D_derivatives(self, u):
        (num_gridx, num_gridy, num_gridz, num_time, n_features) = u.shape
        u_derivatives = zeros(
            (
                num_gridx,
                num_gridy,
                num_gridz,
                num_time,
                n_features,
                self.num_derivatives,
            )
        )
        # first derivatives
        for i in range(num_time):
            # u_x
            for kk in range(num_gridy):
                for jj in range(num_gridz):
                    u_derivatives[:, kk, jj, i, :, 0] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u[:, kk, jj, i, :], self.spatial_grid[:, kk, jj, 0]
                    )
            # u_y
            for kk in range(num_gridx):
                for jj in range(num_gridz):
                    u_derivatives[kk, :, jj, i, :, 1] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u[kk, :, jj, i, :], self.spatial_grid[kk, :, jj, 1]
                    )
            # u_z
            for kk in range(num_gridx):
                for jj in range(num_gridy):
                    u_derivatives[kk, jj, :, i, :, 2] = FiniteDifference(
                        d=1, is_uniform=self.is_uniform
                    )._differentiate(
                        u[kk, jj, :, i, :], self.spatial_grid[kk, jj, :, 2]
                    )

        if self.derivative_order >= 2:
            # second derivatives
            for i in range(num_time):
                # u_xx
                for kk in range(num_gridy):
                    for jj in range(num_gridz):
                        u_derivatives[:, kk, jj, i, :, 3] = FiniteDifference(
                            d=2, is_uniform=self.is_uniform
                        )._differentiate(
                            u[:, kk, jj, i, :], self.spatial_grid[:, kk, jj, 0]
                        )
                # u_xy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 4] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, :, jj, i, :, 0],
                            self.spatial_grid[kk, :, jj, 1],
                        )
                # u_xz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 5] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 0],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_yy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 6] = FiniteDifference(
                            d=2, is_uniform=self.is_uniform
                        )._differentiate(
                            u[kk, :, jj, i, :], self.spatial_grid[kk, :, jj, 1]
                        )
                # u_yz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 7] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 1],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_zz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 8] = FiniteDifference(
                            d=2, is_uniform=self.is_uniform
                        )._differentiate(
                            u[kk, jj, :, i, :], self.spatial_grid[kk, jj, :, 2]
                        )

        if self.derivative_order >= 3:
            # third derivatives
            for i in range(num_time):
                # u_xxx
                for kk in range(num_gridy):
                    for jj in range(num_gridz):
                        u_derivatives[:, kk, jj, i, :, 9] = FiniteDifference(
                            d=3, is_uniform=self.is_uniform
                        )._differentiate(
                            u[:, kk, jj, i, :], self.spatial_grid[:, kk, jj, 0]
                        )
                # u_xxy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 10] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, :, jj, i, :, 3],
                            self.spatial_grid[kk, :, jj, 1],
                        )
                # u_xxz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 11] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 3],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_xyy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 12] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, :, jj, i, :, 4],
                            self.spatial_grid[kk, :, jj, 1],
                        )
                # u_xyz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 13] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 4],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_xzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 14] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 5],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_yyy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 15] = FiniteDifference(
                            d=3, is_uniform=self.is_uniform
                        )._differentiate(
                            u[kk, :, jj, i, :], self.spatial_grid[kk, :, jj, 1]
                        )
                # u_yyz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 16] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 6],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_yzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 17] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 7],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_zzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 18] = FiniteDifference(
                            d=3, is_uniform=self.is_uniform
                        )._differentiate(
                            u[kk, jj, :, i, :], self.spatial_grid[kk, jj, :, 2]
                        )

        if self.derivative_order >= 4:
            # fourth derivatives
            for i in range(num_time):
                # u_xxxx
                for kk in range(num_gridy):
                    for jj in range(num_gridz):
                        u_derivatives[:, kk, jj, i, :, 19] = FiniteDifference(
                            d=4, is_uniform=self.is_uniform
                        )._differentiate(
                            u[:, kk, jj, i, :], self.spatial_grid[:, kk, jj, 0]
                        )
                # u_xxxy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 20] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, :, jj, i, :, 9],
                            self.spatial_grid[kk, :, jj, 1],
                        )
                # u_xxxz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 21] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 9],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_xxyy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 22] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, :, jj, i, :, 10],
                            self.spatial_grid[kk, :, jj, 1],
                        )
                # u_xxyz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 23] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 10],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_xxzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 24] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 11],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_xyyy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 25] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, :, jj, i, :, 12],
                            self.spatial_grid[kk, :, jj, 1],
                        )
                # u_xyyz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 26] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 12],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_xyzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 27] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 13],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_xzzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 28] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 14],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_yyyy
                for kk in range(num_gridx):
                    for jj in range(num_gridz):
                        u_derivatives[kk, :, jj, i, :, 29] = FiniteDifference(
                            d=4, is_uniform=self.is_uniform
                        )._differentiate(
                            u[kk, :, jj, i, :], self.spatial_grid[kk, :, jj, 1]
                        )
                # u_yyyz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 30] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 15],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_yyzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 31] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 16],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_yzzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 32] = FiniteDifference(
                            d=1, is_uniform=self.is_uniform
                        )._differentiate(
                            u_derivatives[kk, jj, :, i, :, 17],
                            self.spatial_grid[kk, jj, :, 2],
                        )
                # u_zzzz
                for kk in range(num_gridx):
                    for jj in range(num_gridy):
                        u_derivatives[kk, jj, :, i, :, 33] = FiniteDifference(
                            d=4, is_uniform=self.is_uniform
                        )._differentiate(
                            u[kk, jj, :, i, :], self.spatial_grid[kk, jj, :, 2]
                        )
        return u_derivatives

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
        feature_names = []
        if self.include_bias:
            feature_names.append("1")
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )
        if self.s_len != 0:
            for k in range(self.num_derivatives):
                for j in range(n_features):
                    feature_names.append(
                        self.function_names[len(self.functions) + k](input_features[j])
                    )
            for k in range(self.num_derivatives):
                for i, f in enumerate(self.functions):
                    for c in self._combinations(
                        n_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        for jj in range(n_features):
                            feature_names.append(
                                self.function_names[i](*[input_features[j] for j in c])
                                + self.function_names[len(self.functions) + k](
                                    input_features[jj]
                                )
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
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )
        if self.s_len != 0:
            # Add the mixed derivative library_terms
            n_output_features += n_output_features * n_features * self.num_derivatives
            # Add the pure derivative library terms
            n_output_features += n_features * self.num_derivatives
        if self.include_bias:
            n_output_features += 1
        self.n_output_features_ = n_output_features
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(n_features),
                )
            )
        if self.spatial_grid is not None:
            if self.s_len == 1:
                # pure derivative terms
                self.function_names = hstack(
                    (
                        self.function_names,
                        list(
                            map(
                                lambda i: (lambda *x: "".join(x) + "_" + "1" * i),
                                range(1, self.num_derivatives + 1),
                            )
                        ),
                    )
                )
            elif self.s_len == 3:
                derivative_strings = [
                    "1",
                    "2",
                    "11",
                    "12",
                    "22",
                    "111",
                    "112",
                    "122",
                    "222",
                    "1111",
                    "1112",
                    "1122",
                    "1222",
                    "2222",
                ]
                # pure derivative terms
                self.function_names = hstack(
                    (
                        self.function_names,
                        list(
                            map(
                                lambda i: (
                                    lambda *x: "".join(x) + "_" + derivative_strings[i]
                                ),
                                range(self.num_derivatives),
                            )
                        ),
                    )
                )
            elif self.s_len == 4:
                derivative_strings = [
                    "1",
                    "2",
                    "3",
                    "11",
                    "12",
                    "13",
                    "22",
                    "23",
                    "33",
                    "111",
                    "112",
                    "113",
                    "122",
                    "123",
                    "133",
                    "222",
                    "223",
                    "233",
                    "333",
                    "1111",
                    "1112",
                    "1113",
                    "1122",
                    "1123",
                    "1133",
                    "1222",
                    "1223",
                    "1233",
                    "1333",
                    "2222",
                    "2223",
                    "2233",
                    "2333",
                    "3333",
                ]
                # pure derivative terms
                self.function_names = hstack(
                    (
                        self.function_names,
                        list(
                            map(
                                lambda i: (
                                    lambda *x: "".join(x) + "_" + derivative_strings[i]
                                ),
                                range(self.num_derivatives),
                            )
                        ),
                    )
                )
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

        if self.spatial_grid is not None:
            if self.s_len == 1:
                num_gridpts = (self.spatial_grid).shape[0]
                num_time = n_samples // num_gridpts
            if self.s_len == 3:
                num_gridx = (self.spatial_grid).shape[0]
                num_gridy = (self.spatial_grid).shape[1]
                num_time = n_samples // num_gridx // num_gridy
            if self.s_len == 4:
                num_gridx = (self.spatial_grid).shape[0]
                num_gridy = (self.spatial_grid).shape[1]
                num_gridz = (self.spatial_grid).shape[2]
                num_time = n_samples // num_gridx // num_gridy // num_gridz
        if self.weak_form:
            n_samples = self.K

        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0
        if self.include_bias:
            if self.weak_form:
                func_final = zeros((self.K, 1))
                if self.s_len == 0:
                    for k in range(self.K):
                        w = self._smooth_ppoly(
                            [],
                            self.tgrid_k[k, :],
                            k,
                            0,
                            0,
                            0,
                            0,
                        )
                        func_final[k] = trapezoid(
                            w,
                            x=self.tgrid_k[k, :],
                            axis=0,
                        )
                if self.s_len == 1:
                    for k in range(self.K):
                        w = self._smooth_ppoly(
                            reshape(self.xgrid_k[k, :], (self.num_pts_per_domain, 1)),
                            self.tgrid_k[k, :],
                            k,
                            0,
                            0,
                            0,
                            0,
                        )
                        func_final[k] = trapezoid(
                            trapezoid(w, x=self.xgrid_k[k, :], axis=0),
                            x=self.tgrid_k[k, :],
                            axis=0,
                        )
                if self.s_len == 3:
                    for k in range(self.K):
                        w = self._smooth_ppoly(
                            transpose((self.xgrid_k[k, :], self.ygrid_k[k, :])),
                            self.tgrid_k[k, :],
                            k,
                            0,
                            0,
                            0,
                            0,
                        )
                        func_final[k] = trapezoid(
                            trapezoid(
                                trapezoid(w, x=self.xgrid_k[k, :], axis=0),
                                x=self.ygrid_k[k, :],
                                axis=0,
                            ),
                            x=self.tgrid_k[k, :],
                            axis=0,
                        )
                if self.s_len == 4:
                    for k in range(self.K):
                        w = self._smooth_ppoly(
                            transpose(
                                (
                                    self.xgrid_k[k, :],
                                    self.ygrid_k[k, :],
                                    self.zgrid_k[k, :],
                                )
                            ),
                            self.tgrid_k[k, :],
                            k,
                            0,
                            0,
                            0,
                            0,
                        )
                        func_final[k] = trapezoid(
                            trapezoid(
                                trapezoid(
                                    trapezoid(w, x=self.xgrid_k[k, :], axis=0),
                                    x=self.ygrid_k[k, :],
                                    axis=0,
                                ),
                                x=self.zgrid_k[k, :],
                                axis=0,
                            ),
                            x=self.tgrid_k[k, :],
                            axis=0,
                        )
                xp[:, library_idx] = ravel(func_final)
            else:
                xp[:, library_idx] = ones(n_samples)
            library_idx += 1
        if self.weak_form:
            if self.s_len == 0:
                for f in self.functions:
                    for c in self._combinations(
                        n_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        func = f(*[x[:, j] for j in c])
                        func_final = zeros((self.K, 1))
                        func_interp = interp1d(
                            self.temporal_grid, func, kind="linear", axis=0
                        )
                        for k in range(self.K):
                            func_new = func_interp(self.tgrid_k[k, :])
                            w = self._smooth_ppoly(
                                [],
                                self.tgrid_k[k, :],
                                k,
                                0,
                                0,
                                0,
                                0,
                            )
                            func_final[k] = trapezoid(
                                func_new * w,
                                x=self.tgrid_k[k, :],
                                axis=0,
                            )
                        xp[:, library_idx] = ravel(func_final)
                        library_idx += 1
            if self.s_len == 1:
                for f in self.functions:
                    for c in self._combinations(
                        n_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        func = f(*[x[:, j] for j in c])
                        func = reshape(func, (num_gridpts, num_time))
                        func_final = zeros((self.K, 1))
                        func_interp = RectBivariateSpline(
                            self.spatial_grid, self.temporal_grid, func
                        )
                        for k in range(self.K):
                            X = ravel(self.X[k, :, :])
                            t = ravel(self.t[k, :, :])
                            func_new = func_interp.ev(X, t)
                            func_new = reshape(
                                func_new,
                                (self.num_pts_per_domain, self.num_pts_per_domain, 1),
                            )
                            w = self._smooth_ppoly(
                                reshape(
                                    self.xgrid_k[k, :], (self.num_pts_per_domain, 1)
                                ),
                                self.tgrid_k[k, :],
                                k,
                                0,
                                0,
                                0,
                                0,
                            )
                            func_final[k] = trapezoid(
                                trapezoid(func_new * w, x=self.xgrid_k[k, :], axis=0),
                                x=self.tgrid_k[k, :],
                                axis=0,
                            )
                        xp[:, library_idx] = ravel(func_final)
                        library_idx += 1
            if self.s_len == 3:
                for f in self.functions:
                    for c in self._combinations(
                        n_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        func = f(*[x[:, j] for j in c])
                        func = reshape(func, (num_gridx, num_gridy, num_time))
                        func_final = zeros((self.K, 1))
                        func_interp = RegularGridInterpolator(
                            (
                                self.spatial_grid[:, 0, 0],
                                self.spatial_grid[0, :, 1],
                                self.temporal_grid,
                            ),
                            func,
                        )
                        for k in range(self.K):
                            X = self.X[k, :, :, :]
                            Y = self.Y[k, :, :, :]
                            t = self.t[k, :, :, :]
                            func_new = func_interp(array((X, Y, t)).T)
                            func_new = reshape(
                                func_new,
                                (
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    1,
                                ),
                            )
                            w = self._smooth_ppoly(
                                transpose((self.xgrid_k[k, :], self.ygrid_k[k, :])),
                                self.tgrid_k[k, :],
                                k,
                                0,
                                0,
                                0,
                                0,
                            )
                            func_final[k] = trapezoid(
                                trapezoid(
                                    trapezoid(
                                        func_new * w, x=self.xgrid_k[k, :], axis=0
                                    ),
                                    x=self.ygrid_k[k, :],
                                    axis=0,
                                ),
                                x=self.tgrid_k[k, :],
                                axis=0,
                            )
                        xp[:, library_idx] = ravel(func_final)
                        library_idx += 1
            if self.s_len == 4:
                for f in self.functions:
                    for c in self._combinations(
                        n_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        func = f(*[x[:, j] for j in c])
                        func = reshape(
                            func, (num_gridx, num_gridy, num_gridz, num_time)
                        )
                        func_final = zeros((self.K, 1))
                        func_interp = RegularGridInterpolator(
                            (
                                self.spatial_grid[:, 0, 0, 0],
                                self.spatial_grid[0, :, 0, 1],
                                self.spatial_grid[0, 0, :, 2],
                                self.temporal_grid,
                            ),
                            func,
                        )
                        for k in range(self.K):
                            func_new = func_interp(
                                array(
                                    (
                                        self.X[k, :, :, :, :],
                                        self.Y[k, :, :, :, :],
                                        self.Z[k, :, :, :, :],
                                        self.t[k, :, :, :, :],
                                    )
                                ).T
                            )
                            func_new = reshape(
                                func_new,
                                (
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    1,
                                ),
                            )
                            w = self._smooth_ppoly(
                                transpose(
                                    (
                                        self.xgrid_k[k, :],
                                        self.ygrid_k[k, :],
                                        self.zgrid_k[k, :],
                                    )
                                ),
                                self.tgrid_k[k, :],
                                k,
                                0,
                                0,
                                0,
                                0,
                            )
                            func_final[k] = trapezoid(
                                trapezoid(
                                    trapezoid(
                                        trapezoid(
                                            func_new * w, x=self.xgrid_k[k, :], axis=0
                                        ),
                                        x=self.ygrid_k[k, :],
                                        axis=0,
                                    ),
                                    x=self.zgrid_k[k, :],
                                    axis=0,
                                ),
                                x=self.tgrid_k[k, :],
                                axis=0,
                            )
                        xp[:, library_idx] = ravel(func_final)
                        library_idx += 1
        else:
            for f in self.functions:
                for c in self._combinations(
                    n_features,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    xp[:, library_idx] = f(*[x[:, j] for j in c])
                    library_idx += 1

        # Need to recompute derivatives because the training
        # and testing data may have different number of time points
        # Need to compute any derivatives now
        if self.spatial_grid is not None:
            if self.s_len == 1:
                u = reshape(x, (num_gridpts, num_time, n_features))
                u_derivatives = zeros(
                    (num_gridpts, num_time, n_features, self.num_derivatives)
                )
                for i in range(num_time):
                    for j in range(self.num_derivatives):
                        u_derivatives[:, i, :, j] = FiniteDifference(
                            d=j + 1, is_uniform=self.is_uniform
                        )._differentiate(u[:, i, :], self.spatial_grid)
                u_derivatives = asarray(
                    reshape(
                        u_derivatives,
                        (num_gridpts * num_time, n_features, self.num_derivatives),
                    )
                )
                if self.weak_form:
                    u_integrals = zeros((self.K, n_features, self.num_derivatives))
                    for kk in range(n_features):
                        u_interp = RectBivariateSpline(
                            self.spatial_grid, self.temporal_grid, u[:, :, kk]
                        )
                        for k in range(self.K):
                            X = ravel(self.X[k, :, :])
                            t = ravel(self.t[k, :, :])
                            u_new = u_interp.ev(X, t)
                            u_new = reshape(
                                u_new,
                                (self.num_pts_per_domain, self.num_pts_per_domain, 1),
                            )
                            for j in range(self.num_derivatives):
                                w_diff = self._smooth_ppoly(
                                    reshape(
                                        self.xgrid_k[k, :], (self.num_pts_per_domain, 1)
                                    ),
                                    self.tgrid_k[k, :],
                                    k,
                                    j + 1,
                                    0,
                                    0,
                                    0,
                                )
                                u_integrals[k, kk, j] = (
                                    trapezoid(
                                        trapezoid(
                                            u_new * w_diff, x=self.xgrid_k[k, :], axis=0
                                        ),
                                        x=self.tgrid_k[k, :],
                                        axis=0,
                                    )
                                    * (-1) ** (j + 1)
                                )

            elif self.s_len == 3:
                u = reshape(x, (num_gridx, num_gridy, num_time, n_features))
                u_derivatives = self._make_2D_derivatives(u)
                u_derivatives = asarray(
                    reshape(
                        u_derivatives,
                        (
                            num_gridx * num_gridy * num_time,
                            n_features,
                            self.num_derivatives,
                        ),
                    )
                )
                if self.weak_form:
                    derivative_list = asarray(
                        [
                            [1, 0],
                            [0, 1],
                            [2, 0],
                            [1, 1],
                            [0, 2],
                            [3, 0],
                            [2, 1],
                            [1, 2],
                            [0, 3],
                            [4, 0],
                            [3, 1],
                            [2, 2],
                            [1, 3],
                            [0, 4],
                        ]
                    )
                    signs_list = asarray(
                        [-1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
                    )
                    u_integrals = zeros((self.K, n_features, self.num_derivatives))
                    for kk in range(n_features):
                        u_interp = RegularGridInterpolator(
                            (
                                self.spatial_grid[:, 0, 0],
                                self.spatial_grid[0, :, 1],
                                self.temporal_grid,
                            ),
                            u[:, :, :, kk],
                        )
                        for k in range(self.K):
                            X = ravel(self.X[k, :, :, :])
                            Y = ravel(self.Y[k, :, :, :])
                            t = ravel(self.t[k, :, :, :])
                            u_new = u_interp(array((X, Y, t)).T)
                            u_new = reshape(
                                u_new,
                                (
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    1,
                                ),
                            )
                            for j in range(self.num_derivatives):
                                w_diff = self._smooth_ppoly(
                                    transpose((self.xgrid_k[k, :], self.ygrid_k[k, :])),
                                    self.tgrid_k[k, :],
                                    k,
                                    derivative_list[j, 0],
                                    derivative_list[j, 1],
                                    0,
                                    0,
                                )
                                u_integrals[k, kk, j] = (
                                    trapezoid(
                                        trapezoid(
                                            trapezoid(
                                                u_new * w_diff,
                                                x=self.xgrid_k[k, :],
                                                axis=0,
                                            ),
                                            x=self.ygrid_k[k, :],
                                            axis=0,
                                        ),
                                        x=self.tgrid_k[k, :],
                                        axis=0,
                                    )
                                    * signs_list[j]
                                )
            elif self.s_len == 4:
                u = reshape(x, (num_gridx, num_gridy, num_gridz, num_time, n_features))
                u_derivatives = self._make_3D_derivatives(u)
                u_derivatives = asarray(
                    reshape(
                        u_derivatives,
                        (
                            num_gridx * num_gridy * num_gridz * num_time,
                            n_features,
                            self.num_derivatives,
                        ),
                    )
                )
                if self.weak_form:
                    derivative_list = asarray(
                        [
                            [1, 0, 0],  # u_x
                            [0, 1, 0],  # u_y
                            [0, 0, 1],  # u_z
                            [2, 0, 0],  # u_xx
                            [1, 1, 0],  # u_xy
                            [1, 0, 1],  # u_xz
                            [0, 2, 0],  # u_yy
                            [0, 1, 1],  # u_yz
                            [0, 0, 2],  # u_zz
                            [3, 0, 0],  # u_xxx
                            [2, 1, 0],  # u_xxy
                            [2, 0, 1],  # u_xxz
                            [1, 2, 0],  # u_xyy
                            [1, 1, 1],  # u_xyz
                            [1, 0, 2],  # u_xzz
                            [0, 3, 0],  # u_yyy
                            [0, 2, 1],  # u_yyz
                            [0, 1, 2],  # u_yzz
                            [0, 0, 3],  # u_zzz
                            [4, 0, 0],  # u_xxxx
                            [3, 1, 0],  # u_xxxy
                            [3, 0, 1],  # u_xxxz
                            [2, 2, 0],  # u_xxyy
                            [2, 1, 1],  # u_xxyz
                            [2, 0, 2],  # u_xxzz
                            [1, 3, 0],  # u_xyyy
                            [1, 2, 1],  # u_xyyz
                            [1, 1, 2],  # u_xyzz
                            [1, 0, 3],  # u_xzzz
                            [0, 4, 0],  # u_yyyy
                            [0, 3, 1],  # u_yyyz
                            [0, 2, 2],  # u_yyzz
                            [0, 1, 3],  # u_yzzz
                            [0, 0, 4],  # u_zzzz
                        ]
                    )
                    signs_list = asarray(
                        [
                            -1,
                            -1,
                            -1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                        ]
                    )
                    u_integrals = zeros((self.K, n_features, self.num_derivatives))
                    for kk in range(n_features):
                        u_interp = RegularGridInterpolator(
                            (
                                self.spatial_grid[:, 0, 0, 0],
                                self.spatial_grid[0, :, 0, 1],
                                self.spatial_grid[0, 0, :, 2],
                                self.temporal_grid,
                            ),
                            u[:, :, :, :, kk],
                        )
                        for k in range(self.K):
                            u_new = u_interp(
                                array(
                                    (
                                        self.X[k, :, :, :, :],
                                        self.Y[k, :, :, :, :],
                                        self.Z[k, :, :, :, :],
                                        self.t[k, :, :, :, :],
                                    )
                                ).T
                            )
                            u_new = reshape(
                                u_new,
                                (
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    self.num_pts_per_domain,
                                    1,
                                ),
                            )
                            for j in range(self.num_derivatives):
                                w_diff = self._smooth_ppoly(
                                    transpose(
                                        (
                                            self.xgrid_k[k, :],
                                            self.ygrid_k[k, :],
                                            self.zgrid_k[k, :],
                                        )
                                    ),
                                    self.tgrid_k[k, :],
                                    k,
                                    derivative_list[j, 0],
                                    derivative_list[j, 1],
                                    derivative_list[j, 2],
                                    0,
                                )
                                u_integrals[k, kk, j] = (
                                    trapezoid(
                                        trapezoid(
                                            trapezoid(
                                                trapezoid(
                                                    u_new * w_diff,
                                                    x=self.xgrid_k[k, :],
                                                    axis=0,
                                                ),
                                                x=self.ygrid_k[k, :],
                                                axis=0,
                                            ),
                                            x=self.zgrid_k[k, :],
                                            axis=0,
                                        ),
                                        x=self.tgrid_k[k, :],
                                        axis=0,
                                    )
                                    * signs_list[j]
                                )

            def identity_function(y):
                return y

            if self.weak_form and self.s_len >= 1:
                for k in range(self.num_derivatives):
                    for kk in range(n_features):
                        xp[:, library_idx] = identity_function(u_integrals[:, kk, k])
                        library_idx += 1
            elif not self.weak_form:
                for k in range(self.num_derivatives):
                    for kk in range(n_features):
                        xp[:, library_idx] = identity_function(u_derivatives[:, kk, k])
                        library_idx += 1
            if self.s_len == 1:
                # All the mixed derivative/non-derivative terms
                for d in range(self.num_derivatives):
                    for kk in range(n_features):
                        if self.weak_form:
                            for f in self.functions:
                                for c in self._combinations(
                                    n_features,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    func = f(*[x[:, j] for j in c])
                                    if d == 0 or d == 1:
                                        derivative_term = u_derivatives[:, kk, 0]
                                    elif d == 2 or d == 3:
                                        derivative_term = u_derivatives[:, kk, 1]
                                    func = reshape(func, (num_gridpts, num_time))
                                    derivative_term = reshape(
                                        derivative_term, (num_gridpts, num_time)
                                    )
                                    func_interp = RectBivariateSpline(
                                        self.spatial_grid, self.temporal_grid, func
                                    )
                                    derivative_interp = RectBivariateSpline(
                                        self.spatial_grid,
                                        self.temporal_grid,
                                        derivative_term,
                                    )
                                    func_final = zeros((self.K, 1))
                                    for k in range(self.K):
                                        X = ravel(self.X[k, :, :])
                                        t = ravel(self.t[k, :, :])
                                        func_new = func_interp.ev(X, t)
                                        derivative_new = derivative_interp.ev(X, t)
                                        func_new = reshape(
                                            func_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        derivative_new = reshape(
                                            derivative_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        w = self._smooth_ppoly(
                                            reshape(
                                                self.xgrid_k[k, :],
                                                (self.num_pts_per_domain, 1),
                                            ),
                                            self.tgrid_k[k, :],
                                            k,
                                            0,
                                            0,
                                            0,
                                            0,
                                        )
                                        if d == 0:
                                            w_func = func_new * w
                                        else:
                                            w_func = zeros(func_new.shape)
                                            for i in range(self.num_pts_per_domain):
                                                if d == 1 or d == 2:
                                                    w_func[:, i, :] = (
                                                        -1
                                                    ) * FiniteDifference(
                                                        d=1, is_uniform=self.is_uniform
                                                    )._differentiate(
                                                        (func_new * w)[:, i, :],
                                                        self.xgrid_k[k, :],
                                                    )
                                                if d == 3:
                                                    w_func[:, i, :] = FiniteDifference(
                                                        d=2, is_uniform=self.is_uniform
                                                    )._differentiate(
                                                        (func_new * w)[:, i, :],
                                                        self.xgrid_k[k, :],
                                                    )
                                        func_final[k] = trapezoid(
                                            trapezoid(
                                                w_func * derivative_new,
                                                x=self.xgrid_k[k, :],
                                                axis=0,
                                            ),
                                            x=self.tgrid_k[k, :],
                                            axis=0,
                                        )
                                    xp[:, library_idx] = ravel(func_final)
                                    library_idx += 1
                        else:
                            for f in self.functions:
                                for c in self._combinations(
                                    n_features,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    xp[:, library_idx] = f(
                                        *[x[:, j] for j in c]
                                    ) * identity_function(u_derivatives[:, kk, d])
                                    library_idx += 1

            if self.s_len == 3:
                # All the mixed derivative/non-derivative terms
                # Note this is really annoying to integrate these mixed terms
                # by parts in 2D so we don't do it for now.
                for d in range(self.num_derivatives):
                    for kk in range(n_features):
                        if self.weak_form:
                            for f in self.functions:
                                for c in self._combinations(
                                    n_features,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    func = f(*[x[:, j] for j in c])
                                    derivative_term = identity_function(
                                        u_derivatives[:, kk, d]
                                    )

                                    func = reshape(
                                        func, (num_gridx, num_gridy, num_time)
                                    )
                                    derivative_term = reshape(
                                        derivative_term,
                                        (num_gridx, num_gridy, num_time),
                                    )
                                    func_final = zeros((self.K, 1))
                                    func_interp = RegularGridInterpolator(
                                        (
                                            self.spatial_grid[:, 0, 0],
                                            self.spatial_grid[0, :, 1],
                                            self.temporal_grid,
                                        ),
                                        func,
                                    )
                                    derivative_interp = RegularGridInterpolator(
                                        (
                                            self.spatial_grid[:, 0, 0],
                                            self.spatial_grid[0, :, 1],
                                            self.temporal_grid,
                                        ),
                                        derivative_term,
                                    )
                                    for k in range(self.K):
                                        X = ravel(self.X[k, :, :, :])
                                        Y = ravel(self.Y[k, :, :, :])
                                        t = ravel(self.t[k, :, :, :])
                                        XYt = array((X, Y, t)).T
                                        func_new = func_interp(XYt)
                                        derivative_new = derivative_interp(XYt)
                                        func_new = reshape(
                                            func_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        derivative_new = reshape(
                                            derivative_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        w = self._smooth_ppoly(
                                            transpose(
                                                (self.xgrid_k[k, :], self.ygrid_k[k, :])
                                            ),
                                            self.tgrid_k[k, :],
                                            k,
                                            0,
                                            0,
                                            0,
                                            0,
                                        )
                                        func_final[k] = trapezoid(
                                            trapezoid(
                                                trapezoid(
                                                    w * func_new * derivative_new,
                                                    x=self.xgrid_k[k, :],
                                                    axis=0,
                                                ),
                                                x=self.ygrid_k[k, :],
                                                axis=0,
                                            ),
                                            x=self.tgrid_k[k, :],
                                            axis=0,
                                        )
                                    xp[:, library_idx] = ravel(func_final)
                                    library_idx += 1
                        else:
                            for f in self.functions:
                                for c in self._combinations(
                                    n_features,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    xp[:, library_idx] = f(
                                        *[x[:, j] for j in c]
                                    ) * identity_function(u_derivatives[:, kk, d])
                                    library_idx += 1
            if self.s_len == 4:
                # All the mixed derivative/non-derivative terms
                # Note this is really annoying to integrate these mixed terms
                # by parts in 3D so we don't do it for now.
                for d in range(self.num_derivatives):
                    for kk in range(n_features):
                        if self.weak_form:
                            for f in self.functions:
                                for c in self._combinations(
                                    n_features,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    func = f(*[x[:, j] for j in c])
                                    derivative_term = identity_function(
                                        u_derivatives[:, kk, d]
                                    )

                                    func = reshape(
                                        func,
                                        (num_gridx, num_gridy, num_gridz, num_time),
                                    )
                                    derivative_term = reshape(
                                        derivative_term,
                                        (num_gridx, num_gridy, num_gridz, num_time),
                                    )
                                    func_final = zeros((self.K, 1))
                                    func_interp = RegularGridInterpolator(
                                        (
                                            self.spatial_grid[:, 0, 0, 0],
                                            self.spatial_grid[0, :, 0, 1],
                                            self.spatial_grid[0, 0, :, 2],
                                            self.temporal_grid,
                                        ),
                                        func,
                                    )
                                    derivative_interp = RegularGridInterpolator(
                                        (
                                            self.spatial_grid[:, 0, 0, 0],
                                            self.spatial_grid[0, :, 0, 1],
                                            self.spatial_grid[0, 0, :, 2],
                                            self.temporal_grid,
                                        ),
                                        derivative_term,
                                    )
                                    for k in range(self.K):
                                        XYt = array(
                                            (
                                                self.X[k, :, :, :, :],
                                                self.Y[k, :, :, :, :],
                                                self.Z[k, :, :, :, :],
                                                self.t[k, :, :, :, :],
                                            )
                                        ).T
                                        func_new = func_interp(XYt)
                                        derivative_new = derivative_interp(XYt)
                                        func_new = reshape(
                                            func_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        derivative_new = reshape(
                                            derivative_new,
                                            (
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                self.num_pts_per_domain,
                                                1,
                                            ),
                                        )
                                        w = self._smooth_ppoly(
                                            transpose(
                                                (
                                                    self.xgrid_k[k, :],
                                                    self.ygrid_k[k, :],
                                                    self.zgrid_k[k, :],
                                                )
                                            ),
                                            self.tgrid_k[k, :],
                                            k,
                                            0,
                                            0,
                                            0,
                                            0,
                                        )
                                        func_final[k] = trapezoid(
                                            trapezoid(
                                                trapezoid(
                                                    trapezoid(
                                                        w * func_new * derivative_new,
                                                        x=self.xgrid_k[k, :],
                                                        axis=0,
                                                    ),
                                                    x=self.ygrid_k[k, :],
                                                    axis=0,
                                                ),
                                                x=self.zgrid_k[k, :],
                                                axis=0,
                                            ),
                                            x=self.tgrid_k[k, :],
                                            axis=0,
                                        )
                                    xp[:, library_idx] = ravel(func_final)
                                    library_idx += 1
                        else:
                            for f in self.functions:
                                for c in self._combinations(
                                    n_features,
                                    f.__code__.co_argcount,
                                    self.interaction_only,
                                ):
                                    xp[:, library_idx] = f(
                                        *[x[:, j] for j in c]
                                    ) * identity_function(u_derivatives[:, kk, d])
                                    library_idx += 1
        # If library bagging, return xp missing the terms at ensemble_indices
        return self._ensemble(xp)
