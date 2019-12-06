from sindy.feature_library import BaseFeatureLibrary

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import numpy as np


class CustomLibrary(BaseFeatureLibrary):
    """
    Generate a library with custom functions.

    Parameters
    ----------
    library_functions : list of mathematical functions
        Functions to include in the library. Each function will be
        applied to each input variable.

    function_names : list of functions
        List of functions used to generate feature names for each library
        function. Each name function must take a string input (representing
        a variable name), and output a string depiction of the respective
        mathematical function applied to that variable. For example, if the
        first library function is sine, the name function might return
        'sin(x)' given 'x' as input. The function_names list must be the
        same length as library_functions.

    Attributes
    ----------
    functions : list of functions
        Mathematical library functions to be applied to each input feature.

    function_names : list of functions
        Functions for generating string representations of each library
        function.

    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.
    """

    def __init__(self, library_functions, function_names):
        super(CustomLibrary, self).__init__()
        self.functions = library_functions
        self.function_names = function_names
        if len(library_functions) != len(function_names):
            raise ValueError(
                "library_functions and function_names must have the same number of elements"
            )

    def get_feature_names(self, input_features=None):
        """
        Return feature names for output features
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
        if input_features is None:
            input_features = ["x%d" % i for i in range(self.n_input_features_)]
        feature_names = []
        for function_name in self.function_names:
            for feature in input_features:
                feature_names.append(function_name(feature))
        return feature_names

    def fit(self, X, y=None):
        """
        Compute number of output features.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(X).shape
        self.n_input_features_ = n_features
        self.n_output_features_ = n_features * len(self.functions)
        return self

    def transform(self, X):
        """Transform data to custom features

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        XP : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.
        """
        check_is_fitted(self)

        X = check_array(X)

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        for i, f in enumerate(self.functions):
            for j in range(self.n_input_features_):
                XP[:, j + i * self.n_input_features_] = f(X[:, j])

        return XP
