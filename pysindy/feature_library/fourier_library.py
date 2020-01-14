from pysindy.feature_library import BaseFeatureLibrary

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import numpy as np


class FourierLibrary(BaseFeatureLibrary):
    """
    Generate a library with custom functions.

    Parameters
    ----------
    n_frequencies : int, optional (default 1)
        Number of frequencies to include in the library. The library will
        include functions sin(x), sin(2*x), ... sin(n_frequencies * x) for
        each input feature x (depending on which of sine and/or cosine
        features are included).

    include_sin : boolean, optional (default True)
        If True, include sine terms in the library.

    include_cos : boolean, optional (default True)
        If True, include cosine terms in the library.

    Attributes
    ----------
    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is 2*n_input_features_*n_frequencies if both sines and cosines
        are included. Otherwise it is n_input_features*n_frequencies.
    """

    def __init__(self, n_frequencies=1, include_sin=True, include_cos=True):
        super(FourierLibrary, self).__init__()
        if not (include_sin or include_cos):
            raise ValueError(
                "include_sin and include_cos cannot both be False"
            )
        if n_frequencies < 1 or not isinstance(n_frequencies, int):
            raise ValueError("n_frequencies must be a positive integer")
        self.n_frequencies = n_frequencies
        self.include_sin = include_sin
        self.include_cos = include_cos

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
        for i in range(self.n_frequencies):
            for feature in input_features:
                if self.include_sin:
                    feature_names.append(
                        "sin(" + str(i + 1) + " " + feature + ")"
                    )
                if self.include_cos:
                    feature_names.append(
                        "cos(" + str(i + 1) + " " + feature + ")"
                    )
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
        if self.include_sin and self.include_cos:
            self.n_output_features_ = n_features * self.n_frequencies * 2
        else:
            self.n_output_features_ = n_features * self.n_frequencies
        return self

    def transform(self, X):
        """Transform data to Fourier features

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        XP : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of Fourier
            features generated from the inputs.
        """
        check_is_fitted(self)

        X = check_array(X)

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        idx = 0
        for i in range(self.n_frequencies):
            for j in range(self.n_input_features_):
                if self.include_sin:
                    XP[:, idx] = np.sin((i + 1) * X[:, j])
                    idx += 1
                if self.include_cos:
                    XP[:, idx] = np.cos((i + 1) * X[:, j])
                    idx += 1
        return XP
