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

    Parameters
    ----------
    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.
    """

    def __init__(self, library_ensemble=None, ensemble_indices=[0]):
        self.library_ensemble = library_ensemble
        if np.any(np.asarray(ensemble_indices) < 0):
            raise ValueError("Library ensemble indices must be 0 or positive integers.")
        self.ensemble_indices = ensemble_indices

    # Force subclasses to implement this
    @abc.abstractmethod
    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        raise NotImplementedError

    # Force subclasses to implement this
    @abc.abstractmethod
    def transform(self, x):
        """
        Transform data.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, [n_samples, n_output_features]
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

    def _ensemble(self, xp):
        """
        If library bagging, return xp without
        the terms at ensemble_indices
        """
        if self.library_ensemble:
            if self.n_output_features_ <= len(self.ensemble_indices):
                raise ValueError(
                    "Error: you are trying to chop more library terms "
                    "than are available to remove!"
                )
            inds = range(self.n_output_features_)
            inds = np.delete(inds, self.ensemble_indices)
            return xp[:, inds]
        else:
            return xp

    def __add__(self, other):
        return ConcatLibrary([self, other])

    def __mul__(self, other):
        return TensoredLibrary([self, other])

    def __rmul__(self, other):
        return TensoredLibrary([self, other])

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

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library. For instance, if
        ensemble_indices = [0], it chops off the first column of the library.

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
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib_custom = CustomLibrary(library_functions=functions)
    >>> lib_fourier = FourierLibrary()
    >>> lib_concat = ConcatLibrary([lib_custom, lib_fourier])
    >>> lib_concat.fit()
    >>> lib.transform(x)
    """

    def __init__(
        self,
        libraries: list,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(ConcatLibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.libraries_ = libraries

    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        _, n_features = check_array(x).shape

        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        # First fit all libs provided below
        fitted_libs = [lib.fit(x, y) for lib in self.libraries_]

        # Calculate the sum of output features
        self.n_output_features_ = sum([lib.n_output_features_ for lib in fitted_libs])

        # Save fitted libs
        self.libraries_ = fitted_libs

        return self

    def transform(self, x):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        for lib in self.libraries_:
            check_is_fitted(lib)
        n_samples = x.shape[0]

        # preallocate matrix
        xp = np.zeros((n_samples, self.n_output_features_))

        current_feat = 0
        for lib in self.libraries_:

            # retrieve num features from lib
            lib_n_output_features = lib.n_output_features_

            start_feature_index = current_feat
            end_feature_index = start_feature_index + lib_n_output_features

            xp[:, start_feature_index:end_feature_index] = lib.transform(x)

            current_feat += lib_n_output_features

        # If library bagging, return xp missing the terms at ensemble_indices
        return self._ensemble(xp)

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


class TensoredLibrary(BaseFeatureLibrary):
    """Tensor multiple libraries together into one library. All settings
    provided to individual libraries will be applied.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library).

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library. For instance, if
        ensemble_indices = [0], it chops off the first column of the library.

    Attributes
    ----------
    libraries_ : list of libraries
        Library instances to be applied to the input matrix.

    inputs_per_library_ : numpy nd.array
        Array that specifies which inputs should be used for each of the
        libraries you are going to tensor together. Used for building
        GeneralizedLibrary objects.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the numbers of output features for each of the
        libraries that were tensored together.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary, CustomLibrary
    >>> from pysindy.feature_library import TensoredLibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib_custom = CustomLibrary(library_functions=functions)
    >>> lib_fourier = FourierLibrary()
    >>> lib_tensored = lib_custom * lib_fourier
    >>> lib_tensored.fit(x)
    >>> lib_tensored.transform(x)
    """

    def __init__(
        self,
        libraries: list,
        library_ensemble=False,
        inputs_per_library=None,
        ensemble_indices=[0],
    ):
        super(TensoredLibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.libraries_ = libraries
        self.inputs_per_library_ = inputs_per_library

    def _combinations(self, lib_i, lib_j):
        """
        Compute combinations of the numerical libraries.

        Returns
        -------
        lib_full : All combinations of the numerical library terms.
        """
        lib_full = np.reshape(
            lib_i[:, :, np.newaxis] * lib_j[:, np.newaxis, :],
            (lib_i.shape[0], lib_i.shape[-1] * lib_j.shape[-1]),
        )

        return lib_full

    def _name_combinations(self, lib_i, lib_j):
        """
        Compute combinations of the library feature names.

        Returns
        -------
        lib_full : All combinations of the library feature names.
        """
        lib_full = []
        for i in range(len(lib_i)):
            for j in range(len(lib_j)):
                lib_full.append(lib_i[i] + " " + lib_j[j])
        return lib_full

    def _set_inputs_per_library(self, inputs_per_library):
        """
        Extra function to make building a GeneralizedLibrary object easier
        """
        self.inputs_per_library_ = inputs_per_library

    def fit(self, x, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        _, n_features = check_array(x).shape

        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        # If parameter is not set, use all the inputs
        if self.inputs_per_library_ is None:
            temp_inputs = np.tile(range(n_features), len(self.libraries_))
            self.inputs_per_library_ = np.reshape(
                temp_inputs, (len(self.libraries_), n_features)
            )

        # First fit all libs provided below
        fitted_libs = [
            lib.fit(x[:, np.unique(self.inputs_per_library_[i, :])], y)
            for i, lib in enumerate(self.libraries_)
        ]

        # Calculate the sum of output features
        output_sizes = [lib.n_output_features_ for lib in fitted_libs]
        self.n_output_features_ = 1
        for osize in output_sizes:
            self.n_output_features_ *= osize

        # Save fitted libs
        self.libraries_ = fitted_libs

        return self

    def transform(self, x):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        for lib in self.libraries_:
            check_is_fitted(lib)
        n_samples = x.shape[0]

        # preallocate matrix
        xp = np.zeros((n_samples, self.n_output_features_))

        current_feat = 0
        for i in range(len(self.libraries_)):
            lib_i = self.libraries_[i]
            lib_i_n_output_features = lib_i.n_output_features_
            if self.inputs_per_library_ is None:
                xp_i = lib_i.transform(x)
            else:
                xp_i = lib_i.transform(x[:, np.unique(self.inputs_per_library_[i, :])])
            for j in range(i + 1, len(self.libraries_)):
                lib_j = self.libraries_[j]
                lib_j_n_output_features = lib_j.n_output_features_
                xp_j = lib_j.transform(x[:, np.unique(self.inputs_per_library_[j, :])])

                start_feature_index = current_feat
                end_feature_index = (
                    start_feature_index
                    + lib_i_n_output_features * lib_j_n_output_features
                )

                xp[:, start_feature_index:end_feature_index] = self._combinations(
                    xp_i, xp_j
                )

                current_feat += lib_i_n_output_features * lib_j_n_output_features

        # If library bagging, return xp missing the terms at ensemble_indices
        return self._ensemble(xp)

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
        for i in range(len(self.libraries_)):
            lib_i = self.libraries_[i]
            if input_features is None:
                input_features_i = [
                    "x%d" % k for k in np.unique(self.inputs_per_library_[i, :])
                ]
            else:
                input_features_i = np.asarray(input_features)[
                    np.unique(self.inputs_per_library_[i, :])
                ].tolist()
            lib_i_feat_names = lib_i.get_feature_names(input_features_i)
            for j in range(i + 1, len(self.libraries_)):
                lib_j = self.libraries_[j]
                if input_features is None:
                    input_features_j = [
                        "x%d" % k for k in np.unique(self.inputs_per_library_[j, :])
                    ]
                else:
                    input_features_j = np.asarray(input_features)[
                        np.unique(self.inputs_per_library_[j, :])
                    ].tolist()
                lib_j_feat_names = lib_j.get_feature_names(input_features_j)
                feature_names += self._name_combinations(
                    lib_i_feat_names, lib_j_feat_names
                )
        return feature_names
