"""
Base class for feature library classes.
"""
import abc

from sklearn.base import TransformerMixin


class BaseFeatureLibrary(TransformerMixin):
    """
    Base class for feature libraries.

    Forces subclasses to implement fit and transform functions.
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

    @property
    def size(self):
        return self._size
