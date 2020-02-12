import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class SINDyOptimizer(BaseEstimator):
    def __init__(self, optimizer, unbias=True):
        self.optimizer = optimizer
        self.unbias = unbias

    def fit(self, x, y):
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
        return self.optimizer.predict(x)

    @property
    def coef_(self):
        if self.optimizer.coef_.ndim == 1:
            return self.optimizer.coef_[np.newaxis, :]
        else:
            return self.optimizer.coef_

    @property
    def intercept_(self):
        return self.optimizer.intercept_

    # not sure if
    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
