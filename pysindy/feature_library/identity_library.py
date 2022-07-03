from sklearn import __version__
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from .base import x_sequence_or_item


class IdentityLibrary(BaseFeatureLibrary):
    """
    Generate an identity library which maps all input features to
    themselves.

    Attributes
    ----------
    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is equal to the number of input features.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import IdentityLibrary
    >>> x = np.array([[0,-1],[0.5,-1.5],[1.,-2.]])
    >>> lib = IdentityLibrary().fit(x)
    >>> lib.transform(x)
    array([[ 0. , -1. ],
           [ 0.5, -1.5],
           [ 1. , -2. ]])
    >>> lib.get_feature_names()
    ['x0', 'x1']
    """

    def __init__(
        self,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(IdentityLibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
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
        if float(__version__[:3]) >= 1.0:
            n_input_features = self.n_features_in_
        else:
            n_input_features = self.n_input_features_
        if input_features:
            if len(input_features) == n_input_features:
                return input_features
            else:
                raise ValueError("input features list is not the right length")
        return ["x%d" % i for i in range(n_input_features)]

    @x_sequence_or_item
    def fit(self, x_full, y=None):
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
        n_features = x_full[0].shape[x_full[0].ax_coord]
        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features
        self.n_output_features_ = n_features
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Perform identity transformation (return a copy of the input).

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        x : np.ndarray, shape (n_samples, n_features)
            The matrix of features, which is just a copy of the input data.
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

            xp_full = xp_full + [x.copy()]
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
        return xp_full
