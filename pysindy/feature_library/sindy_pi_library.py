import warnings
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import empty
from numpy import hstack
from numpy import nan_to_num
from numpy import ones
from sklearn import __version__
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item
from pysindy.differentiation import FiniteDifference


class SINDyPILibrary(BaseFeatureLibrary):
    """
    WARNING: This library is deprecated in PySINDy versions > 1.7. Please
    use the PDE or WeakPDE libraries instead.

    Generate a library with custom functions. The Library takes custom
    libraries for X and Xdot respectively, and then tensor-products them
    together. For a 3D system, a library of constant and linear terms in x_dot,
    i.e. [1, x_dot0, ..., x_dot3], is good
    enough for most problems and implicit terms. The function names list
    should include both X and Xdot functions, without the mixed terms.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable x.

    x_dot_library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable x_dot.

    t : np.ndarray of time slices
        Time base to compute Xdot from X for the implicit terms

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be a class extending
        :class:`pysindy.differentiation_methods.base.BaseDifferentiation` class.
        The default option is centered difference.

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names
        list must be the same length as library_functions. If no list of
        function names is provided, defaults to using
        :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`. For SINDy-PI,
        function_names should include the names of the functions in both the
        x and x_dot libraries (library_functions and x_dot_library_functions),
        but not the mixed terms, which are computed in the code.

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
    >>> from pysindy.feature_library import SINDyPILibrary
    >>> t = np.linspace(0, 1, 5)
    >>> x = np.ones((5, 2))
    >>> functions = [lambda x: 1, lambda x : np.exp(x),
                     lambda x,y : np.sin(x+y)]
    >>> x_dot_functions = [lambda x: 1, lambda x : x]
    >>> function_names = [lambda x: '',
                          lambda x : 'exp(' + x + ')',
                          lambda x, y : 'sin(' + x + y + ')',
                          lambda x: '',
                  lambda x : x]
    >>> lib = ps.SINDyPILibrary(library_functions=functions,
                                x_dot_library_functions=x_dot_functions,
                                function_names=function_names, t=t
                                ).fit(x)
    >>> lib.transform(x)
            [[ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               2.22044605e-16  6.03579815e-16  6.03579815e-16  2.01904588e-16
               2.22044605e-16  6.03579815e-16  6.03579815e-16  2.01904588e-16]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
               0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
             [ 1.00000000e+00  2.71828183e+00  2.71828183e+00  9.09297427e-01
              -2.22044605e-16 -6.03579815e-16 -6.03579815e-16 -2.01904588e-16
              -2.22044605e-16 -6.03579815e-16 -6.03579815e-16 -2.01904588e-16]]
    >>> lib.get_feature_names()
        ['', 'exp(x0)', 'exp(x1)', 'sin(x0x1)', 'x0_dot', 'exp(x0)x0_dot',
         'exp(x1)x0_dot', 'sin(x0x1)x0_dot', 'x1_dot', 'exp(x0)x1_dot',
         'exp(x1)x1_dot', 'sin(x0x1)x1_dot']
    """

    def __init__(
        self,
        library_functions=None,
        t=None,
        x_dot_library_functions=None,
        function_names=None,
        interaction_only=True,
        differentiation_method=None,
        include_bias=False,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(SINDyPILibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.x_functions = library_functions
        self.x_dot_functions = x_dot_library_functions
        self.function_names = function_names

        warnings.warn(
            "This library is deprecated in PySINDy versions > 1.7. Please "
            "use the PDE or WeakPDE libraries instead. "
        )
        if library_functions is None and x_dot_library_functions is None:
            raise ValueError(
                "At least one function library, either for x or x_dot, " "is required."
            )
        if x_dot_library_functions is not None and t is None:
            raise ValueError(
                "If using a library that contains x_dot terms,"
                " you must specify a timebase t"
            )
        if function_names is not None:
            x_library_len = 0 if library_functions is None else len(library_functions)
            x_dot_library_len = (
                0 if x_dot_library_functions is None else len(x_dot_library_functions)
            )
            if x_library_len + x_dot_library_len != len(function_names):
                raise ValueError(
                    "(x_library_functions + x_dot_library_functions) and "
                    " function_names must have the same"
                    " number of elements"
                )
        if differentiation_method is None and x_dot_library_functions is not None:
            differentiation_method = FiniteDifference(drop_endpoints=False)
        self.differentiation_method = differentiation_method
        self.interaction_only = interaction_only
        self.t = t
        self.include_bias = include_bias

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
            n_input_features = self.n_features_in_
        else:
            n_input_features = self.n_input_features_
        if input_features is None:
            input_features = ["x%d" % i for i in range(n_input_features)]
            x_dot_features = ["x%d_dot" % i for i in range(n_input_features)]
        else:
            x_dot_features = [
                input_features[i] + "_dot" for i in range(n_input_features)
            ]

        feature_names = []
        if self.include_bias:
            feature_names.append("1")

        # Put in normal library for x
        if self.x_functions is not None:
            funcs = self.x_functions
            for i, f in enumerate(funcs):
                for c in self._combinations(
                    n_input_features,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    feature_names.append(
                        self.function_names[i](*[input_features[j] for j in c])
                    )

        # Put in normal library for x_dot
        if self.x_dot_functions is not None:
            funcs = self.x_dot_functions
            for i, f in enumerate(funcs):
                for c in self._combinations(
                    n_input_features,
                    f.__code__.co_argcount,
                    self.interaction_only,
                ):
                    feature_names.append(
                        self.function_names[-1 - i](*[x_dot_features[j] for j in c])
                    )

        # Put in all the mixed terms
        if self.x_dot_functions is not None and self.x_functions is not None:
            for k, f_dot in enumerate(self.x_dot_functions):
                for f_dot_combs in self._combinations(
                    n_input_features,
                    f_dot.__code__.co_argcount,
                    self.interaction_only,
                ):
                    for i, f in enumerate(self.x_functions):
                        for f_combs in self._combinations(
                            n_input_features,
                            f.__code__.co_argcount,
                            self.interaction_only,
                        ):
                            feature_names.append(
                                self.function_names[i](
                                    *[input_features[comb] for comb in f_combs]
                                )
                                + self.function_names[-1 - k](
                                    *[x_dot_features[comb] for comb in f_dot_combs]
                                )
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
        n_x_output_features = 0
        n_x_dot_output_features = 0

        # Put in normal x library
        if self.x_functions is not None:
            funcs = self.x_functions
            for f in funcs:
                n_args = f.__code__.co_argcount
                n_x_output_features += len(
                    list(self._combinations(n_features, n_args, self.interaction_only))
                )
            self.n_output_features_ = n_x_output_features

        # Put in normal x_dot library
        if self.x_dot_functions is not None:
            funcs = self.x_dot_functions
            for f in funcs:
                n_args = f.__code__.co_argcount
                n_x_dot_output_features += len(
                    list(self._combinations(n_features, n_args, self.interaction_only))
                )
            self.n_output_features_ += n_x_dot_output_features
            if n_x_output_features != 0:
                self.n_output_features_ += n_x_output_features * n_x_dot_output_features

        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(len(self.x_functions)),
                )
            )
            self.function_names = hstack(
                (
                    self.function_names,
                    list(
                        map(
                            lambda i: (
                                lambda *x_dot: "f_dot"
                                + str(i)
                                + "("
                                + ",".join(x_dot)
                                + ")"
                            ),
                            range(len(self.x_dot_functions)),
                        )
                    ),
                )
            )
        if self.include_bias:
            self.n_output_features_ += 1
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
            The matrix of features, where n_output_features is the number of features
            generated from applying the custom functions to the inputs.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            if self.x_dot_functions is not None:
                x_dot = nan_to_num(self.differentiation_method(x, self.t))

            n_samples, n_features = x.shape

            if float(__version__[:3]) >= 1.0:
                n_input_features = self.n_features_in_
            else:
                n_input_features = self.n_input_features_
            if n_features != n_input_features:
                raise ValueError("x shape does not match training shape")

            xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
            library_idx = 0

            # Put in column of ones in the library
            if self.include_bias:
                xp[:, library_idx] = ones(n_samples)
                library_idx += 1

            # Put in normal x library
            if self.x_functions is not None:
                for i, f in enumerate(self.x_functions):
                    for c in self._combinations(
                        n_input_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        xp[:, library_idx] = f(*[x[:, j] for j in c])
                        library_idx += 1

            # Put in normal x_dot library
            if self.x_dot_functions is not None:
                for i, f in enumerate(self.x_dot_functions):
                    for c in self._combinations(
                        n_input_features,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    ):
                        xp[:, library_idx] = f(*[x_dot[:, j] for j in c])
                        library_idx += 1

            # Put in mixed x, x_dot terms
            if self.x_dot_functions is not None and self.x_functions is not None:
                for k, f_dot in enumerate(self.x_dot_functions):
                    for f_dot_combs in self._combinations(
                        n_input_features,
                        f_dot.__code__.co_argcount,
                        self.interaction_only,
                    ):

                        for i, f in enumerate(self.x_functions):
                            for f_combs in self._combinations(
                                n_input_features,
                                f.__code__.co_argcount,
                                self.interaction_only,
                            ):
                                xp[:, library_idx] = f(
                                    *[x[:, comb] for comb in f_combs]
                                ) * f_dot(*[x_dot[:, comb] for comb in f_dot_combs])
                                library_idx += 1
            xp_full = xp_full + [AxesArray(xp, x.__dict__)]
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
        return xp_full
