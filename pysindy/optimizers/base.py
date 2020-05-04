"""
Base class for SINDy optimizers.
"""
import abc

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_X_y


def _rescale_data(X, y, sample_weight):
    """Rescale data so as to support sample_weight"""
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y


class ComplexityMixin:
    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)


class BaseOptimizer(LinearRegression, ComplexityMixin):
    """
    Base class for SINDy optimizers. Subclasses must implement
    a _reduce method for carrying out the bulk of the work of
    fitting a model.

    Parameters
    ----------
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out.

    history_ : list
        History of ``coef_`` over iterations of the optimization algorithm.
    """

    def __init__(self, max_iter=20, normalize=False, fit_intercept=False, copy_X=True):
        super(BaseOptimizer, self).__init__(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X
        )

        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.max_iter = max_iter
        self.iters = 0
        self.coef_ = None
        self.ind_ = None
        self.history_ = []

    # Force subclasses to implement this
    @abc.abstractmethod
    def _reduce(self):
        """
        Carry out the bulk of the work of the fit function.

        Subclass implementations MUST update self.coef_.
        """
        raise NotImplementedError

    def fit(self, x_, y, sample_weight=None, **reduce_kws):
        """
        Fit the model.

        Parameters
        ----------
        x_ : array-like, shape (n_samples, n_features)
            Training data

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values

        sample_weight : float or numpy array of shape (n_samples,), optional
            Individual weights for each sample

        reduce_kws : dict
            Optional keyword arguments to pass to the _reduce method
            (implemented by subclasses)

        Returns
        -------
        self : returns an instance of self
        """
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=True)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            x, y = _rescale_data(x, y, sample_weight)

        self.iters = 0
        self.ind_ = np.ones((y.shape[1], x.shape[1]), dtype=bool)
        self.coef_ = np.linalg.lstsq(x, y, rcond=None)[0].T  # initial guess
        self.history_.append(self.coef_)

        self._reduce(x, y, **reduce_kws)
        self.ind_ = np.abs(self.coef_) > 1e-14

        self._set_intercept(X_offset, y_offset, X_scale)
        return self


class _MultiTargetLinearRegressor(MultiOutputRegressor, ComplexityMixin):
    @property
    def coef_(self):
        return np.vstack([est.coef_ for est in self.estimators_])

    @property
    def intercept_(self):
        return np.array([est.intercept_ for est in self.estimators_])
