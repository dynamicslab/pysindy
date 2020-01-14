"""
Base class for SINDy optimizers.
"""

import abc

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y
from sklearn.utils.extmath import safe_sparse_dot


def _rescale_data(X, y, sample_weight):
    """Rescale data so as to support sample_weight"""
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(
            n_samples, sample_weight, dtype=sample_weight.dtype
        )
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix(
        (sample_weight, 0), shape=(n_samples, n_samples)
    )
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y


class BaseOptimizer(LinearRegression):
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
    """

    def __init__(self, normalize=False, fit_intercept=False, copy_X=True):
        super(BaseOptimizer, self).__init__(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X
        )
        # self.fit_intercept = fit_intercept
        # self.normalize = normalize
        # self.copy_X = copy_X
        self.iters = 0
        self.coef_ = []
        self.ind_ = []

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
        x_, y = check_X_y(
            x_, y, accept_sparse=[], y_numeric=True, multi_output=False
        )

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
        self.ind_ = np.ones(x.shape[1], dtype=bool)
        self.coef_ = np.linalg.lstsq(x, y, rcond=None)[0]  # initial guess
        self.history_.append(self.coef_)

        self._reduce(x, y, **reduce_kws)

        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )
