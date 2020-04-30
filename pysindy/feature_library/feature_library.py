"""
Base class for feature library classes.
"""
import abc

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BaseFeatureLibrary(TransformerMixin):
    """
    Base class for feature libraries.

    Forces subclasses to implement `fit`, `transform`,
    and `get_feature_names` functions.
    """

    def __init__(self, **kwargs):
        pass

    # Force subclasses to implement this
    @abc.abstractmethod
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
        raise NotImplementedError

    # Force subclasses to implement this
    @abc.abstractmethod
    def transform(self, X):
        """
        Transform data.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        XP : np.ndarray, [n_samples, n_output_features]
            The matrix of features, where n_output_features is the number
            of features generated from the combination of inputs.
        """
        raise NotImplementedError

    # Force subclasses to implement this
    @abc.abstractmethod
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
        raise NotImplementedError

    def __add__(self, other):
        return ConcatLibrary([self, other])

    @property
    def size(self):
        check_is_fitted(self)
        return self.n_output_features_


class ConcatLibrary(BaseFeatureLibrary):
    """Concatenate multiple libraries into one library. All settings
    provided to individual libraries will be applied.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    Attributes
    ----------
    libraries_ : list of libraries
        Library instances to be applied to the input matrix.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary, CustomLibrary
    >>> from pysindy.feature_library import ConcatLibrary
    >>> X = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib_custom = CustomLibrary(library_functions=functions)
    >>> lib_fourier = FourierLibrary()
    >>> lib_concat = ConcatLibrary([lib_custom, lib_fourier])
    >>> lib_concat.fit()
    >>> lib.transform(X)
    """

    def __init__(self, libraries: list):
        super(ConcatLibrary, self).__init__()
        self.libraries_ = libraries

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

        # first fit all libs provided below
        fitted_libs = [lib.fit(X, y) for lib in self.libraries_]

        # calculate the sum of output features
        self.n_output_features_ = sum([lib.n_output_features_ for lib in fitted_libs])

        # save fitted libs
        self.libraries_ = fitted_libs

        return self

    def transform(self, X):
        """Transform data with libs provided below.

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

        n_samples = X.shape[0]

        # preallocate matrix
        XP = np.zeros((n_samples, self.n_output_features_))

        current_feat = 0
        for lib in self.libraries_:

            # retrieve num features from lib
            lib_n_output_features = lib.n_output_features_

            start_feature_index = current_feat
            end_feature_index = start_feature_index + lib_n_output_features

            XP[:, start_feature_index:end_feature_index] = lib.transform(X)

            current_feat += lib_n_output_features

        return XP

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
        feature_names = list()
        for lib in self.libraries_:
            lib_feat_names = lib.get_feature_names(input_features)
            feature_names += lib_feat_names
        return feature_names
