"""
Base class for feature library classes.
"""
import abc

import numpy as np
from sklearn import __version__
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class BaseFeatureLibrary(TransformerMixin):
    """
    Base class for feature libraries.

    Forces subclasses to implement ``fit``, ``transform``,
    and ``get_feature_names`` methods.
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

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library).

    ensemble_indices : integer array, optional (default 0)
        The indices to use for ensembling the library. For instance, if
        ensemble_indices = 0, it chops off the first column of the library.

    Attributes
    ----------
    libraries_ : list of libraries
        Library instances to be applied to the input matrix.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the sum of the numbers of output features for each of the
        concatenated libraries.

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

    def __init__(
        self,
        libraries: list,
        library_ensemble=False,
        ensemble_indices=0,
    ):
        super(ConcatLibrary, self).__init__()
        self.libraries_ = libraries
        self.library_ensemble = library_ensemble
        self.ensemble_indices = ensemble_indices

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
        _, n_features = check_array(X).shape

        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        # First fit all libs provided below
        fitted_libs = [lib.fit(X, y) for lib in self.libraries_]

        # Calculate the sum of output features
        self.n_output_features_ = sum([lib.n_output_features_ for lib in fitted_libs])

        # Save fitted libs
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
        for lib in self.libraries_:
            check_is_fitted(lib)
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

        # If library bagging, return xp missing the terms at ensemble_indices
        if self.library_ensemble:
            if self.n_output_features_ == 1:
                raise ValueError(
                    "Can't use library ensemble methods if your"
                    " library is just one term!"
                )
            inds = range(self.n_output_features_)
            inds = np.delete(inds, self.ensemble_indices)
            return XP[:, inds]
        else:
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
