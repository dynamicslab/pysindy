import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from pysindy.optimizers.base import _MultiTargetLinearRegressor
from pysindy.utils.base import supports_multiple_targets


class SINDyOptimizer(BaseEstimator):
    """
    Wrapper class for optimizers/sparse regression methods passed into the SINDy object.

    Enables single target regressors (i.e. those whose predictions are 1-dimensional)
    to perform multi target regression (i.e. predictions are 2-dimensional).
    Also enhances an `_unbias` function to reduce bias when regularization is used.

    Parameters
    ----------
    optimizer: estimator object
        The optimizer/sparse regressor to be wrapped, implementing `fit` and `predict`.
        `optimizer` should also have the attributes `coef_`, `fit_intercept`,
        `normalize`, and `intercept_`.

    unbias : boolean, optional (default True)
        Whether to perform an extra step of unregularized linear regression to unbias
        the coefficients for the identified support.
        For example, if `optimizer=STLSQ(alpha=0.1)` is used then the learned
        coefficients will be biased toward 0 due to the L2 regularization.
        Setting `unbias=True` will trigger an additional step wherein the nonzero
        coefficients learned by the `STLSQ` object will be updated using an
        unregularized least-squares fit.
    """

    def __init__(self, optimizer, unbias=True):
        # TODO: add a check that optimizer has the necessary attributes
        # and methods
        self.optimizer = optimizer
        self.unbias = unbias

    def fit(self, x, y):
        if len(y.shape) > 1 and y.shape[1] > 1:
            if not supports_multiple_targets(self.optimizer):
                self.optimizer = _MultiTargetLinearRegressor(self.optimizer)
        self.optimizer.fit(x, y)
        self.ind_ = np.abs(self.coef_) > 1e-14

        if self.unbias:
            self._unbias(x, y)

        return self

    def _unbias(self, x, y):
        coef = np.zeros((y.shape[1], x.shape[1]))
        for i in range(self.ind_.shape[0]):
            if np.any(self.ind_[i]):
                coef[i, self.ind_[i]] = (
                    LinearRegression(
                        fit_intercept=self.optimizer.fit_intercept,
                        normalize=self.optimizer.normalize,
                    )
                    .fit(x[:, self.ind_[i]], y[:, i])
                    .coef_
                )
        if self.optimizer.coef_.ndim == 1:
            self.optimizer.coef_ = coef[0]
        else:
            self.optimizer.coef_ = coef

    def predict(self, x):
        prediction = self.optimizer.predict(x)
        if prediction.ndim == 1:
            return prediction[:, np.newaxis]
        else:
            return prediction

    @property
    def coef_(self):
        if self.optimizer.coef_.ndim == 1:
            return self.optimizer.coef_[np.newaxis, :]
        else:
            return self.optimizer.coef_

    @property
    def intercept_(self):
        return self.optimizer.intercept_

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
