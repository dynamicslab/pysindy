from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r
from itertools import product as iproduct

from numpy import array
from numpy import concatenate
from numpy import empty
from numpy import newaxis
from numpy import ones
from numpy import product
from numpy import reshape
from numpy import sum
from sklearn import __version__
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from pysindy.differentiation import FiniteDifference


class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions, optional (default None)
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

    include_interaction : boolean, optional (default True)
        This is a different than the use for the PolynomialLibrary. If true,
        it generates all the mixed derivative terms. If false, the library
        will consist of only pure no-derivative terms and pure derivative
        terms, with no mixed terms.

    is_uniform : boolean, optional (default True)
        If True, assume the grid is uniform in all spatial directions, so
        can use uniform grid spacing for the derivative calculations.

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
    >>> from pysindy.feature_library import PDELibrary
    """

    def __init__(
        self,
        library_functions=[],
        derivative_order=0,
        spatial_grid=None,
        interaction_only=True,
        function_names=None,
        include_bias=False,
        include_interaction=True,
        is_uniform=False,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(PDELibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.derivative_order = derivative_order
        self.function_names = function_names
        self.interactions_only = interaction_only
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.is_uniform = is_uniform

        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if derivative_order < 0:
            raise ValueError("The derivative order must be >0")

        if (spatial_grid is not None and derivative_order == 0) or (
            spatial_grid is None and derivative_order != 0
        ):
            raise ValueError(
                "Spatial grid and the derivative order must be "
                "defined at the same time"
            )

        if spatial_grid is None:
            spatial_grid = array([])

        # list of derivatives
        indices = ()
        if len(spatial_grid.shape) == 1:
            spatial_grid = reshape(spatial_grid, (len(spatial_grid), 1))
        dims = spatial_grid.shape[:-1]

        for i in range(0, len(dims)):
            indices = indices + (range(derivative_order + 1),)

        multiindices = []
        for ind in iproduct(*indices):
            current = array(ind)
            if sum(ind) > 0 and sum(ind) <= derivative_order:
                multiindices.append(current)
        multiindices = array(multiindices)
        num_derivatives = len(multiindices)

        self.num_derivatives = num_derivatives
        self.multiindices = multiindices
        self.spatial_grid = spatial_grid

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

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

        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interactions_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )

        def derivative_string(multiindex):
            ret = ""
            for axis in range(len(self.spatial_grid.shape[:-1])):
                for i in range(multiindex[axis]):
                    ret = ret + str(axis + 1)
            return ret

        # Include derivative (integral) terms
        for k in range(self.num_derivatives):
            for j in range(n_features):
                feature_names.append(
                    input_features[j] + "_" + derivative_string(self.multiindices[k])
                )
        # Include mixed non-derivative + derivative (integral) terms
        if self.include_interaction:
            for k in range(self.num_derivatives):
                for i, f in enumerate(self.functions):
                    for c in self._combinations(
                        n_features,
                        f.__code__.co_argcount,
                        self.interactions_only,
                    ):
                        for jj in range(n_features):
                            feature_names.append(
                                self.function_names[i](*[input_features[j] for j in c])
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
        n_output_features = 0
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interactions_only))
            )

        # Add the mixed derivative library_terms
        if self.include_interaction:
            n_output_features += n_output_features * n_features * self.num_derivatives
        # Add the pure derivative library terms
        n_output_features += n_features * self.num_derivatives

        # If there is a constant term, add 1 to n_output_features
        if self.include_bias:
            n_output_features += 1

        self.n_output_features_ = n_output_features

        return self

    def transform(self, x):
        """Transform data to pde features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of
            features generated from the tensor product of the derivative terms
            and the library_functions applied to combinations of the inputs.
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

        dims = self.spatial_grid.shape[:-1]
        if product(dims) > 0:
            num_time = n_samples // product(dims)
        else:
            num_time = n_samples

        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0

        # derivative terms
        library_derivatives = empty(
            (n_samples, n_features * self.num_derivatives), dtype=x.dtype
        )
        library_idx = 0

        for multiindex in self.multiindices:
            derivs = reshape(x, concatenate([dims, [num_time], [n_features]]))
            for axis in range(len(dims)):
                if multiindex[axis] > 0:
                    s = [0 for dim in self.spatial_grid.shape]
                    s[axis] = slice(dims[axis])
                    s[-1] = axis
                    derivs = FiniteDifference(
                        d=multiindex[axis], axis=axis
                    )._differentiate(derivs, self.spatial_grid[tuple(s)])
            library_derivatives[:, library_idx : library_idx + n_features] = reshape(
                derivs, (n_samples, n_features)
            )
            library_idx += n_features

        # library function terms
        n_library_terms = 0
        for f in self.functions:
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interactions_only
            ):
                n_library_terms += 1

        library_functions = empty((n_samples, n_library_terms), dtype=x.dtype)
        library_idx = 0
        for f in self.functions:
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interactions_only
            ):
                library_functions[:, library_idx] = reshape(
                    f(*[x[:, j] for j in c]), (n_samples)
                )
                library_idx += 1

        library_idx = 0

        # constant term
        if self.include_bias:
            xp[:, library_idx] = ones(n_samples, dtype=x.dtype)
            library_idx += 1

        # library function terms
        xp[:, library_idx : library_idx + n_library_terms] = library_functions
        library_idx += n_library_terms

        # pure derivative terms
        xp[
            :, library_idx : library_idx + self.num_derivatives * n_features
        ] = library_derivatives
        library_idx += self.num_derivatives * n_features

        # mixed function derivative terms
        # Should include a self.interactions_only case as well?
        if self.include_interaction:
            xp[
                :,
                library_idx : library_idx
                + n_library_terms * self.num_derivatives * n_features,
            ] = reshape(
                library_functions[:, :, newaxis] * library_derivatives[:, newaxis, :],
                (n_samples, n_library_terms * self.num_derivatives * n_features),
            )
            library_idx += n_library_terms * self.num_derivatives * n_features

        return self._ensemble(xp)
