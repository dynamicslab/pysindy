import numpy as np

from .base import BaseOptimizer

COEF_THRESHOLD = 1e-14


class WrappedOptimizer(BaseOptimizer):
    """
    Wrapper class for optimizers/sparse regression methods passed
    into the SINDy object.

    Enables single target regressors
    (i.e. those whose predictions are 1-dimensional)
    to perform multi target regression (i.e. predictions are 2-dimensional).

    Parameters
    ----------
    optimizer: estimator object
        The optimizer/sparse regressor to be wrapped, implementing ``fit`` and
        ``predict``. ``optimizer`` should also have the attributes ``coef_``,
        ``fit_intercept``, and ``intercept_``. Note that attribute
        ``normalize`` is deprecated as of sklearn versions >= 1.0 and will be
        removed in future versions.

    """

    def __init__(self, optimizer, unbias: bool = True):
        super().__init__(unbias=unbias)
        self.optimizer = optimizer

    def _reduce(self, x, y):
        if not hasattr(self.optimizer, "fit") or not callable(
            getattr(self.optimizer, "fit")
        ):
            raise AttributeError("optimizer does not have a callable fit method")
        if not hasattr(self.optimizer, "predict") or not callable(
            getattr(self.optimizer, "predict")
        ):
            raise AttributeError("optimizer does not have a callable predict method")

        coef_shape = (y.shape[1], x.shape[1])
        self.coef_ = np.zeros(coef_shape)
        self.ind_ = np.ones(coef_shape)
        for tgt in range(y.shape[-1]):
            self.optimizer.fit(x, y[..., tgt])
            self.coef_[tgt] = self.optimizer.coef_
        self.ind_ = np.abs(self.coef_) > COEF_THRESHOLD
        if self.unbias:
            self._unbias(x, y)
        if hasattr(self.optimizer, "intercept_"):
            self.intercept_ = self.optimizer.intercept_
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, x):
        prediction = self.optimizer.predict(x)
        if prediction.ndim == 1:
            return prediction[:, np.newaxis]
        else:
            return prediction

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
