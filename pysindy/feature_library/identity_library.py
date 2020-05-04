from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .feature_library import BaseFeatureLibrary


class IdentityLibrary(BaseFeatureLibrary):
    """
    Generate an identity library which maps all input features to
    themselves.

    Attributes
    ----------
    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is equal to the number of input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import IdentityLibrary
    >>> X = np.array([[0,-1],[0.5,-1.5],[1.,-2.]])
    >>> lib = IdentityLibrary().fit(X)
    >>> lib.transform(X)
    array([[ 0. , -1. ],
           [ 0.5, -1.5],
           [ 1. , -2. ]])
    >>> lib.get_feature_names()
    ['x0', 'x1']
    """

    def __init__(self):
        super(IdentityLibrary, self).__init__()

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
        if input_features:
            if len(input_features) == self.n_input_features_:
                return input_features
            else:
                raise ValueError("input features list is not the right length")
        return ["x%d" % i for i in range(self.n_input_features_)]

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
        self.n_output_features_ = n_features
        return self

    def transform(self, X):
        """Transform data to Fourier features

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        X : np.ndarray, shape [n_samples, n_features]
            The matrix of features, which is just a copy of the input data.
        """
        check_is_fitted(self)

        X = check_array(X)

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        return X.copy()
