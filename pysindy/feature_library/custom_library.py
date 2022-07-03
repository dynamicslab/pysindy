from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

from numpy import empty
from numpy import ones
from numpy import shape
from sklearn import __version__
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from ..utils import comprehend_axes
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item


class CustomLibrary(BaseFeatureLibrary):
    """Generate a library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Default is to use same functions
        for all variables. Can also be used so that each variable has an
        associated library, in this case library_functions is shape
        (n_input_features, num_library_functions)

    function_names : list of functions, optional (default None)
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        :math:`\\sin(x)` given :math:`x` as input. The function_names list
        must be the same length as library_functions. If no list of function
        names is provided, defaults to using
        :math:`[ f_0(x),f_1(x), f_2(x), \\ldots ]`.

    interaction_only : boolean, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaulations of the form :math:`f(x,x)` and :math:`f(x,y,x)`
        will be omitted, but those of the form :math:`f(x,y)` and :math:`f(x,y,z)`
        will be included.
        If False, all combinations will be included.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

    include_bias : boolean, optional (default False)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
        This is hard to do with just lambda functions, because if the system
        is not 1D, lambdas will generate duplicates.

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
    >>> from pysindy.feature_library import CustomLibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib = CustomLibrary(library_functions=functions).fit(x)
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
        function_names=None,
        interaction_only=True,
        library_ensemble=False,
        ensemble_indices=[0],
        include_bias=False,
    ):
        super(CustomLibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.function_names = function_names
        if function_names and (
            shape(library_functions)[-1] != shape(function_names)[-1]
        ):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        self.include_bias = include_bias
        self.interaction_only = interaction_only

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
        feature_names = []
        if self.include_bias:
            feature_names.append("1")
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_input_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
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
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )
        if self.include_bias:
            n_output_features += 1
        self.n_output_features_ = n_output_features
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(len(self.functions)),
                )
            )
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
            n_features = x.shape[x.ax_coord]

            if float(__version__[:3]) >= 1.0:
                n_input_features = self.n_features_in_
            else:
                n_input_features = self.n_input_features_

            if n_features != n_input_features:
                raise ValueError("x shape does not match training shape")

            xp = empty((*x.shape[:-1], self.n_output_features_), dtype=x.dtype)
            library_idx = 0
            if self.include_bias:
                xp[..., library_idx] = ones(x.shape[:-1])
                library_idx += 1
            for f in self.functions:
                for c in self._combinations(
                    n_input_features, f.__code__.co_argcount, self.interaction_only
                ):
                    xp[..., library_idx] = f(*[x[..., j] for j in c])
                    library_idx += 1

            xp = AxesArray(xp, comprehend_axes(xp))
            xp_full.append(xp)
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
        return xp_full
