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

        self.optimizer.fit(x, y)
        if not hasattr(self.optimizer, "coef_"):
            raise AttributeError("optimizer has no attribute coef_")
        self.ind_ = np.abs(self.coef_) > COEF_THRESHOLD
        self.coef_ = self.optimizer.coef_
        if self.unbias:
            self._unbias(x, y)
        return self

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
        if hasattr(self.optimizer, "intercept_"):
            return self.optimizer.intercept_
        else:
            return 0.0

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
